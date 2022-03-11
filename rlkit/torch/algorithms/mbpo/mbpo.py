import gc
from itertools import count
import traceback
from typing import Callable, Dict, List

import gtimer as gt
import numpy as np
from torch import nn as nn
import torch

from rlkit.core import logger
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.data_management.path_builder import PathBuilder
from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer
from rlkit.envs.vecenvs import BaseVectorEnv
from rlkit.torch.algorithms.mbpo.bnn_trainer import BNNTrainer
from rlkit.torch.algorithms.mbpo.fake_env import FakeEnv
from rlkit.torch.algorithms.sac.sac_alpha import SoftActorCritic
from rlkit.torch.algorithms.torch_rl_algorithm import TorchRLAlgorithm
from rlkit.torch.core import np_to_pytorch_batch


class MBPO(TorchRLAlgorithm):
    
    def __init__(
        self,
        env,
        model: BNNTrainer,
        algo: SoftActorCritic, # model-free algorithm to train policy
        is_terminal: Callable, # for fake_env
        model_replay_buffer: SimpleReplayBuffer = None,
        model_replay_buffer_size: int = 10000,

        # model utilization params
        deterministic: bool = False,
        model_train_freq: int = 250,
        model_retrain_epochs: int = 1,
        rollout_batch_size: int = int(1e5),
        real_ratio: float = 0.1,
        rollout_schedule: List = None,
        max_model_t: float = None,
        **kwargs
    ):
        super().__init__(env=env, trainer=algo, **kwargs)
        self.training_env: BaseVectorEnv
        self.model = model
        self.algo = algo
        self.is_terminal = is_terminal
        self.fake_env = FakeEnv(self.model, self.is_terminal, self.model.get_random_model_index)
        self.model_replay_buffer_size = model_replay_buffer_size
        if model_replay_buffer is None:
            assert self.max_path_length < model_replay_buffer_size
            model_replay_buffer = EnvReplayBuffer(model_replay_buffer_size, self.env, random_seed=np.random.randint(10000))
        else:
            assert self.max_path_length < model_replay_buffer._max_replay_buffer_size
        self.model_replay_buffer = model_replay_buffer
        logger.log(f'MBPO | Target entorpy: {self.algo.target_entropy}')

        self.deterministic = deterministic
        self.model_train_freq = model_train_freq
        self.model_retrain_epochs = model_retrain_epochs
        self.rollout_batch_size = rollout_batch_size
        self.real_ratio = real_ratio
        self.rollout_schedule = rollout_schedule
        self.max_model_t = max_model_t

        self.rollout_length = 1

    def start_training(self, start_epoch: int = 0) -> None:
        self.ready_env_ids = np.arange(self.env_num)
        num_ready_envs = len(self.ready_env_ids)
        obs = self._start_new_rollout(self.ready_env_ids)
        self._current_path_builder = [
            PathBuilder() for _ in range(num_ready_envs)
        ]

        logger.log(f'MBPO | Presampling for {self.min_steps_before_training} steps')
        for epoch in gt.timed_for(
            range(start_epoch, self.num_epochs),
            save_itrs=True,
        ):
            self._start_epoch(epoch)
            total_rews = np.zeros(num_ready_envs)
            self._n_env_steps_this_epoch = 0
            start_env_steps = self._n_env_steps_total
            cur_env_steps = 0
            for i in count():
                if self._can_train():
                    cur_env_steps = self._n_env_steps_total - start_env_steps - 1
                    if cur_env_steps >= self.num_env_steps_per_epoch:
                        break
                    if cur_env_steps % self.model_train_freq == 0 and self.real_ratio < 1.0:
                        logger.log(f'MBPO | Training model: freq {self.model_train_freq} | timestep {cur_env_steps} (total: {self._n_env_steps_total}) | epoch train steps: {self._n_env_steps_this_epoch} (total: {self.num_env_steps_per_epoch})')
                        self._train_model()
                        gt.stamp('model_train')
                        self._set_rollout_length(epoch)        
                        self._extend_model_replay_buffer()
                        self._rollout_stat = self._rollout_model(self.deterministic)
                        gt.stamp('model_rollout')
                else:
                    start_env_steps = self._n_env_steps_total
                
                actions = self._get_action_and_info(obs)
                if isinstance(actions, tuple):
                    actions = actions[0]
                next_obs, rewards, terminals, infos = self.training_env.step(actions, self.ready_env_ids)
                if self.no_terminal:
                    terminals = [False] * num_ready_envs
                self._n_env_steps_total += num_ready_envs
                total_rews += rewards
                rews = rewards
                self._handle_vec_step(
                    obs,
                    actions,
                    rews,
                    next_obs,
                    terminals,
                    absorbings=[np.array([0.0, 0.0]) for _ in range(num_ready_envs)],
                    env_infos=infos
                )
                if np.any(terminals):
                    env_ind_local = np.where(terminals)[0]
                    total_rews[env_ind_local] = 0.0
                    self._handle_vec_rollout_ending(env_ind_local)
                    reset_obs = self._start_new_rollout(env_ind_local)
                    next_obs[env_ind_local] = reset_obs
                elif np.any(env_ind_local := (
                    np.array(
                        [
                            len(self._current_path_builder[i])
                            for i in range(num_ready_envs)
                        ]
                    )
                    >= self.max_path_length)
                ):
                    env_ind_local = np.where(env_ind_local)[0]
                    self._handle_vec_rollout_ending(env_ind_local)
                    reset_obs = self._start_new_rollout(env_ind_local)
                    next_obs[env_ind_local] = reset_obs
                obs = next_obs

                if self._n_env_steps_total - self._n_prev_train_env_steps >= self.num_steps_between_train_calls:
                    gt.stamp('sample')
                    self._try_to_train(epoch)
                    gt.stamp('train')

            gt.stamp("sample")
            self._try_to_eval(epoch)
            gt.stamp("eval")
            self._end_epoch()

    def evaluate(self, epoch: int) -> None:
        self.eval_statistics = self.model.get_eval_statistics()
        self.eval_statistics.update(self.algo.get_eval_statistics())
        self.eval_statistics.update(self._rollout_stat)
        super().evaluate(epoch)

    def get_batch(self) -> torch.Tensor:
        real_ratio = self.real_ratio if self.model_replay_buffer._size > 0 else 1.0
        real_batch_size = int(self.batch_size * real_ratio)
        model_batch_size = self.batch_size - real_batch_size
        real_batch = self.replay_buffer.random_batch(real_batch_size)
        if model_batch_size > 0:
            model_batch = self.model_replay_buffer.random_batch(model_batch_size)
            batch = {k: np.concatenate([real_batch[k], model_batch[k]], axis=0) for k in real_batch.keys()}
        else:
            batch = real_batch
        return np_to_pytorch_batch(batch)

    def _train_model(self) -> None:
        assert self.replay_buffer._size >= self.min_steps_before_training
        batch = self.replay_buffer.get_all()
        self.model.train_step(np_to_pytorch_batch(batch))

    def _do_training(self, epoch: int) -> None:
        for _ in range(self.num_train_steps_per_train_call):
            self.algo.train_step(self.get_batch())

    def _set_rollout_length(self, epoch: int) -> None:
        min_epoch, max_epoch, min_length, max_length = self.rollout_schedule
        if epoch < min_epoch:
            l = min_length
        else:
            dx = (epoch - min_epoch) / (max_epoch - min_epoch)
            dx = min(dx, 1)
            l = dx * (max_length - min_length) + min_length
        self.rollout_length = int(l)
        logger.log(f'Model Rollout | Epoch {epoch} (min: {min_epoch}, max: {max_epoch}) | Length: {self.rollout_length} (min: {min_length}, max: {max_length})')

    def _extend_model_replay_buffer(self) -> bool:
        rollout_per_epoch = self.rollout_batch_size * self.max_path_length / self.model_train_freq
        model_steps_per_epoch = int(self.rollout_length * rollout_per_epoch)
        new_pool_size = self.model_retrain_epochs * model_steps_per_epoch
        if self.model_replay_buffer._max_replay_buffer_size < new_pool_size:
            logger.log(f'Extend model replay buffer | {self.model_replay_buffer._max_replay_buffer_size:.2e} -> {new_pool_size:.2e}')
            try:
                samples = self.model_replay_buffer.get_all()
                new_buffer = EnvReplayBuffer(new_pool_size, self.env, np.random.randint(10000))
                new_buffer.add_path(samples)
            except MemoryError as me:
                print(traceback.format_exc())
                logger.log('Error: extending model replay buffer failed. Out of memory. Retrain the original size.')
                return False
            del self.model_replay_buffer
            gc.collect()
            self.model_replay_buffer = new_buffer
        return True

    def _rollout_model(self, deterministic: bool = False) -> Dict[str, float]:
        logger.log(f'Model Rollout | Rollout length: {self.rollout_length} | Batch size: {self.rollout_batch_size}')
        batch = self.replay_buffer.random_batch(self.rollout_batch_size)
        obs = batch['observations']
        steps = []
        for i in range(self.rollout_length):
            act = self.algo.policy.get_actions(obs)
            next_obs, rew, term, _ = self.fake_env.step(obs, act, deterministic)
            steps.append(len(next_obs))
            samples = {'observations': obs, 'actions': act, 'next_observations': next_obs, 'rewards': rew, 'terminals': term}
            self.model_replay_buffer.add_path(samples)
            terminal = np.squeeze(term, -1)
            if np.all(terminal):
                logger.log(f'Model Rollout | Breaking early at {i}: all episodes terminate')
                break
            obs = next_obs[~terminal]

        total_steps = np.sum(steps)
        mean_len = total_steps / self.rollout_batch_size
        logger.log(f'Model Rollout | Added: {total_steps:.1e} | Model pool: {self.model_replay_buffer._size:.1e} (max {self.model_replay_buffer._max_replay_buffer_size:.1e}) | Mean length: {mean_len} | Train rep: {self.num_train_steps_per_train_call}')
        return {'mean_rollout_length': mean_len}

    @property
    def networks(self) -> List[nn.Module]:
        return self.model.networks + self.algo.networks

    def to(self, device):
        self.model.to(device)
        self.algo.to(device)
