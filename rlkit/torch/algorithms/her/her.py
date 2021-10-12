import numpy as np
from collections import OrderedDict

import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
import gtimer as gt

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.algorithms.torch_rl_algorithm import TorchRLAlgorithm
from rlkit.data_management.relabel_replay_buffer import HindsightReplayBuffer
from rlkit.data_management.path_builder import PathBuilder
from rlkit.samplers import PathSampler

class HER(TorchRLAlgorithm):
    """
    Hindsight Experience Replay. Default using TD3 for RL policy trainer.
    """

    def __init__(
        self,
        replay_buffer=None,
        **kwargs
    ):
        if replay_buffer is None:
            assert kwargs['max_path_length'] < kwargs['replay_buffer_size']
            replay_buffer = HindsightReplayBuffer(
                kwargs['replay_buffer_size'], kwargs['env'], random_seed=np.random.randint(10000)
            )
        super().__init__(replay_buffer=replay_buffer, **kwargs)

    def get_batch(self, relabel=True, keys=None):
        buffer = self.replay_buffer
        batch = buffer.random_batch(self.batch_size, relabel=relabel, keys=keys)
        batch = np_to_pytorch_batch(batch)
        return batch

    def _get_action_and_info(self, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        # print(self._can_train())
        if not self._can_train():
            return [np.random.uniform(-1, 1, size=self.exploration_policy.action_dim) for _ in range(len(self.ready_env_ids))]
        self.exploration_policy.set_num_steps_total(self._n_env_steps_total)
        return self.exploration_policy.get_actions(
            observation,
        )

    # def start_training(self, start_epoch=0, flag=False):
    #     # self._current_path_builder = PathBuilder()
    #     self.ready_env_ids = np.arange(self.env_num)
    #     observations = self._start_new_rollout(
    #         self.ready_env_ids
    #     )  # Do it for support vec env
    #     observations, goals_a, goals_d = observations['observation'], observations['achieved_goal'], observations['desired_goal']

    #     self._current_path_builder = [
    #         PathBuilder() for _ in range(len(self.ready_env_ids))
    #     ]

    #     for epoch in gt.timed_for(
    #         range(start_epoch, self.num_epochs),
    #         save_itrs=True,
    #     ):
    #         self._start_epoch(epoch)
    #         total_rews = np.array([0.0 for _ in range(len(self.ready_env_ids))])
    #         for steps_this_epoch in range(self.num_env_steps_per_epoch // self.env_num):
    #             actions = self._get_action_and_info(observations, goals_d)

    #             if type(actions) is tuple:
    #                 actions = actions[0]

    #             if self.render:
    #                 self.training_env.render()

    #             next_obs, raw_rewards, terminals, env_infos = self.training_env.step(
    #                 actions, self.ready_env_ids
    #             )
    #             if self.no_terminal:
    #                 terminals = [False for _ in range(len(self.ready_env_ids))]
    #             # self._n_env_steps_total += 1
    #             self._n_env_steps_total += len(self.ready_env_ids)

    #             rewards = raw_rewards
    #             total_rews += raw_rewards

    #             self._handle_vec_step(
    #                 observations,
    #                 goals_a,
    #                 goals_d,
    #                 actions,
    #                 rewards,
    #                 next_obs,
    #                 np.array([False for _ in range(len(self.ready_env_ids))])
    #                 if self.no_terminal
    #                 else terminals,
    #                 absorbings=[
    #                     np.array([0.0, 0.0]) for _ in range(len(self.ready_env_ids))
    #                 ],
    #                 env_infos=env_infos,
    #             )
    #             if np.any(terminals):
    #                 env_ind_local = np.where(terminals)[0]
    #                 if flag:
    #                     pass
    #                 total_rews[env_ind_local] = 0.0
    #                 self._handle_vec_rollout_ending(env_ind_local)
    #                 reset_observations = self._start_new_rollout(env_ind_local)
    #                 next_obs[env_ind_local] = reset_observations
    #             elif np.any(
    #                 np.array(
    #                     [
    #                         len(self._current_path_builder[i])
    #                         for i in range(len(self.ready_env_ids))
    #                     ]
    #                 )
    #                 >= self.max_path_length
    #             ):
    #                 env_ind_local = np.where(
    #                     np.array(
    #                         [
    #                             len(self._current_path_builder[i])
    #                             for i in range(len(self.ready_env_ids))
    #                         ]
    #                     )
    #                     >= self.max_path_length
    #                 )[0]
    #                 self._handle_vec_rollout_ending(env_ind_local)
    #                 reset_observations = self._start_new_rollout(env_ind_local)
    #                 next_obs[env_ind_local] = reset_observations

    #             observations = next_obs
    #             observations, goals_a, goals_d = observations['observation'], observations['achieved_goal'], observations['desired_goal']

    #             if (self._n_env_steps_total - self._n_prev_train_env_steps) >= self.num_steps_between_train_calls:
    #                 gt.stamp("sample")
    #                 self._try_to_train(epoch)
    #                 gt.stamp("train")

    #         gt.stamp("sample")
    #         self._try_to_eval(epoch)
    #         gt.stamp("eval")
    #         self._end_epoch()

    # def _handle_vec_step(
    #     self,
    #     observations,
    #     goals_a,
    #     goals_d,
    #     actions,
    #     rewards,
    #     next_observations,
    #     terminals,
    #     absorbings,
    #     env_infos,
    # ):
    #     """
    #     Implement anything that needs to happen after every step under vec envs
    #     :return:
    #     """
    #     for idx, (
    #         ob,
    #         g_a,
    #         g_d,
    #         action,
    #         reward,
    #         next_ob,
    #         terminal,
    #         absorbing,
    #         env_info,
    #     ) in enumerate(
    #         zip(
    #             observations,
    #             goals_a,
    #             goals_d,
    #             actions,
    #             rewards,
    #             next_observations,
    #             terminals,
    #             absorbings,
    #             env_infos,
    #         )
    #     ):
    #         self._handle_step(
    #             ob,
    #             g_a,
    #             g_d,
    #             action,
    #             reward,
    #             next_ob,
    #             terminal,
    #             absorbing=absorbing,
    #             env_info=env_info,
    #             idx=idx,
    #             add_buf=False,
    #         )

    # def _handle_step(
    #     self,
    #     observation,
    #     goal_a,
    #     goal_d,
    #     action,
    #     reward,
    #     next_observation,
    #     terminal,
    #     absorbing,
    #     env_info,
    #     idx=0,
    #     add_buf=True,
    #     path_builder=True,
    # ):
    #     """
    #     Implement anything that needs to happen after every step
    #     :return:
    #     """
    #     if path_builder:
    #         self._current_path_builder[idx].add_all(
    #             observations=observation,
    #             actions=action,
    #             rewards=reward,
    #             next_observations=next_observation,
    #             terminals=terminal,
    #             absorbings=absorbing,
    #             env_infos=env_info,
    #         )
    #     if add_buf:
    #         self.replay_buffer.add_sample(
    #             observation=observation,
    #             achived_goal=goal_a,
    #             desired_goal=goal_d,
    #             action=action,
    #             reward=reward,
    #             terminal=terminal,
    #             next_observation=next_observation,
    #             absorbing=absorbing,
    #             env_info=env_info,
    #         )

    # def _get_action_and_info(self, observation, condition):
    #     """
    #     Get an action to take in the environment.
    #     :param observation:
    #     :return:
    #     """
    #     self.exploration_policy.set_num_steps_total(self._n_env_steps_total)
    #     return self.exploration_policy.get_actions(
    #         observation, condition
    #     )