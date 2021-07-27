import abc
import pickle
import time
from copy import deepcopy

import gtimer as gt
import numpy as np

from rlkit.core import logger
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.data_management.path_builder import PathBuilder
from rlkit.policies.base import ExplorationPolicy
from rlkit.samplers.in_place import InPlacePathSampler
from rlkit.envs.wrapped_absorbing_env import WrappedAbsorbingEnv

from gym.spaces import Dict


class IRLAlgorithm(metaclass=abc.ABCMeta):
    '''
    Generic IRL algorithm class
    Structure:
    while True:
        generate trajectories
        update reward
        fit policy
    '''
    def __init__(
            self,
            env,
            exploration_policy: ExplorationPolicy,
            expert_replay_buffer,
            training_env=None,
            num_epochs=100,
            num_steps_per_epoch=10000,
            num_steps_per_eval=1000,
            num_steps_between_updates=1000,
            min_steps_before_training=1000,
            max_path_length=1000,
            discount=0.99,
            replay_buffer_size=10000,
            render=False,
            save_replay_buffer=False,
            save_algorithm=False,
            save_environment=False,
            save_best=False,
            save_best_starting_from_epoch=0,
            eval_sampler=None,
            eval_policy=None,
            replay_buffer=None,
            policy_uses_pixels=False,
            wrap_absorbing=False,
            freq_saving=1,
            # some environment like halfcheetah_v2 have a timelimit that defines the terminal
            # this is used as a minor hack to turn off time limits
            no_terminal=False,
            policy_uses_task_params=False,
            concat_task_params_to_policy_obs=False
        ):
        """
        Base class for RL Algorithms
        :param env: Environment used to evaluate.
        :param exploration_policy: Policy used to explore
        :param training_env: Environment used by the algorithm. By default, a
        copy of `env` will be made.
        :param num_epochs:
        :param num_steps_per_epoch:
        :param num_steps_per_eval:
        :param num_updates_per_env_step: Used by online training mode.
        :param num_updates_per_epoch: Used by batch training mode.
        :param batch_size:
        :param max_path_length:
        :param discount:
        :param replay_buffer_size:
        :param render:
        :param save_replay_buffer:
        :param save_algorithm:
        :param save_environment:
        :param eval_sampler:
        :param eval_policy: Policy to evaluate with.
        :param replay_buffer:
        """
        self.training_env = training_env or pickle.loads(pickle.dumps(env))
        # self.training_env = training_env or deepcopy(env)
        self.exploration_policy = exploration_policy
        self.expert_replay_buffer = expert_replay_buffer
        self.num_epochs = num_epochs
        self.num_env_steps_per_epoch = num_steps_per_epoch
        self.num_steps_per_eval = num_steps_per_eval
        self.num_steps_between_updates = num_steps_between_updates
        self.min_steps_before_training = min_steps_before_training
        self.max_path_length = max_path_length
        self.discount = discount
        self.replay_buffer_size = replay_buffer_size
        self.render = render
        self.save_replay_buffer = save_replay_buffer
        self.save_algorithm = save_algorithm
        self.save_environment = save_environment
        self.save_best = save_best
        self.save_best_starting_from_epoch = save_best_starting_from_epoch
        self.policy_uses_pixels = policy_uses_pixels
        self.policy_uses_task_params = policy_uses_task_params
        self.concat_task_params_to_policy_obs = concat_task_params_to_policy_obs
        if eval_sampler is None:
            if eval_policy is None:
                eval_policy = exploration_policy
            eval_sampler = InPlacePathSampler(
                env=env,
                policy=eval_policy,
                max_samples=self.num_steps_per_eval + self.max_path_length,
                max_path_length=self.max_path_length, policy_uses_pixels=policy_uses_pixels,
                policy_uses_task_params=policy_uses_task_params,
                concat_task_params_to_policy_obs=concat_task_params_to_policy_obs
            )
        self.eval_policy = eval_policy
        self.eval_sampler = eval_sampler

        self.action_space = env.action_space
        self.obs_space = env.observation_space
        self.env = env
        if replay_buffer is None:
            replay_buffer = EnvReplayBuffer(
                self.replay_buffer_size,
                self.env,
                policy_uses_pixels=self.policy_uses_pixels,
                policy_uses_task_params=self.policy_uses_task_params,
                concat_task_params_to_policy_obs=self.concat_task_params_to_policy_obs
            )
        self.replay_buffer = replay_buffer

        self._n_env_steps_total = 0
        self._n_train_steps_total = 0
        self._n_rollouts_total = 0
        self._do_train_time = 0
        self._epoch_start_time = None
        self._algo_start_time = None
        self._old_table_keys = None
        self._current_path_builder = PathBuilder()
        self._exploration_paths = []
        self.wrap_absorbing = wrap_absorbing
        self.freq_saving = freq_saving
        self.no_terminal = no_terminal


    def train(self, start_epoch=0):
        self.pretrain()
        if start_epoch == 0:
            params = self.get_epoch_snapshot(-1)
            logger.save_itr_params(-1, params)
        self.training_mode(False)
        self._n_env_steps_total = start_epoch * self.num_env_steps_per_epoch
        gt.reset()
        gt.set_def_unique(False)
        self.train_online(start_epoch=start_epoch)


    def pretrain(self):
        """
        Do anything before the main training phase.
        """
        pass


    def train_online(self, start_epoch=0):
        self._current_path_builder = PathBuilder()
        observation = self._start_new_rollout()
        for epoch in gt.timed_for(
                range(start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self._start_epoch(epoch)
            steps_this_epoch = 0
            while steps_this_epoch < self.num_env_steps_per_epoch:
                # print(steps_this_epoch)
                for _ in range(self.num_steps_between_updates):
                    if isinstance(self.obs_space, Dict):
                        if self.policy_uses_pixels:
                            agent_obs = observation['pixels']
                        else:
                            agent_obs = observation['obs']
                    else:
                        agent_obs = observation
                    if self.policy_uses_task_params:
                        task_params = observation['obs_task_params']
                        if self.concat_task_params_to_policy_obs:
                            agent_obs = np.concatenate((agent_obs, task_params), -1)
                        else:
                            agent_obs = {'obs': agent_obs, 'obs_task_params': task_params}
                    action, agent_info = self._get_action_and_info(
                        agent_obs,
                    )
                    if self.render:
                        self.training_env.render()
                    next_ob, raw_reward, terminal, env_info = (
                        self.training_env.step(action)
                    )
                    if self.no_terminal:
                        terminal = False
                    self._n_env_steps_total += 1
                    reward = raw_reward
                    terminal = np.array([terminal])
                    reward = np.array([reward])
                    self._handle_step(
                        observation,
                        action,
                        reward,
                        next_ob,
                        np.array([False]) if self.wrap_absorbing else terminal,
                        absorbing=np.array([0., 0.]),
                        agent_info=agent_info,
                        env_info=env_info,
                    )
                    if terminal:
                        if self.wrap_absorbing:
                            '''
                            If we wrap absorbing states, two additional
                            transitions must be added: (s_T, s_abs) and
                            (s_abs, s_abs). In Disc Actor Critic paper
                            they make s_abs be a vector of 0s with last
                            dim set to 1. Here we are going to add the following:
                            ([next_ob,0], random_action, [next_ob, 1]) and
                            ([next_ob,1], random_action, [next_ob, 1])
                            This way we can handle varying types of terminal states.
                            '''
                            # next_ob is the absorbing state
                            # for now just taking the previous action
                            self._handle_step(
                                next_ob,
                                action,
                                # env.action_space.sample(),
                                # the reward doesn't matter
                                reward,
                                next_ob,
                                np.array([False]),
                                absorbing=np.array([0.0, 1.0]),
                                agent_info=agent_info,
                                env_info=env_info
                            )
                            self._handle_step(
                                next_ob,
                                action,
                                # env.action_space.sample(),
                                # the reward doesn't matter
                                reward,
                                next_ob,
                                np.array([False]),
                                absorbing=np.array([1.0, 1.0]),
                                agent_info=agent_info,
                                env_info=env_info
                            )
                        self._handle_rollout_ending()
                        observation = self._start_new_rollout()
                    elif len(self._current_path_builder) >= self.max_path_length:
                        self._handle_rollout_ending()
                        observation = self._start_new_rollout()
                    else:
                        observation = next_ob

                    steps_this_epoch += 1

                gt.stamp('sample')
                self._try_to_train(epoch)
                gt.stamp('train')

            self._try_to_eval(epoch)
            gt.stamp('eval')
            self._end_epoch()

    def _try_to_train(self, epoch):
        if self._can_train():
            self.training_mode(True)
            self._do_training(epoch)
            self._n_train_steps_total += 1
            self.training_mode(False)

    def _try_to_eval(self, epoch):
        if epoch % self.freq_saving == 0:
            logger.save_extra_data(self.get_extra_data_to_save(epoch))
        if self._can_evaluate():
            self.evaluate(epoch)

            if epoch % self.freq_saving == 0:
                params = self.get_epoch_snapshot(epoch)
                logger.save_itr_params(epoch, params)
            table_keys = logger.get_table_key_set()
            # if self._old_table_keys is not None:
            #     print('$$$$$$$$$$$$$$$')
            #     print(table_keys)
            #     print('\n'*4)
            #     print(self._old_table_keys)
            #     print('$$$$$$$$$$$$$$$')
            #     print(set(table_keys) - set(self._old_table_keys))
            #     print(set(self._old_table_keys) - set(table_keys))
            #     assert table_keys == self._old_table_keys, (
            #         "Table keys cannot change from iteration to iteration."
            #     )
            # self._old_table_keys = table_keys

            logger.record_tabular(
                "Number of train steps total",
                self._n_train_steps_total,
            )
            logger.record_tabular(
                "Number of env steps total",
                self._n_env_steps_total,
            )
            logger.record_tabular(
                "Number of rollouts total",
                self._n_rollouts_total,
            )

            times_itrs = gt.get_times().stamps.itrs
            train_time = times_itrs['train'][-1]
            sample_time = times_itrs['sample'][-1]
            eval_time = times_itrs['eval'][-1] if epoch > 0 else 0
            epoch_time = train_time + sample_time + eval_time
            total_time = gt.get_times().total

            logger.record_tabular('Train Time (s)', train_time)
            logger.record_tabular('(Previous) Eval Time (s)', eval_time)
            logger.record_tabular('Sample Time (s)', sample_time)
            logger.record_tabular('Epoch Time (s)', epoch_time)
            logger.record_tabular('Total Train Time (s)', total_time)

            logger.record_tabular("Epoch", epoch)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
        else:
            logger.log("Skipping eval for now.")

    def _can_evaluate(self):
        """
        One annoying thing about the logger table is that the keys at each
        iteration need to be the exact same. So unless you can compute
        everything, skip evaluation.

        A common example for why you might want to skip evaluation is that at
        the beginning of training, you may not have enough data for a
        validation and training set.

        :return:
        """
        return (
            len(self._exploration_paths) > 0
            and self.replay_buffer.num_steps_can_sample() >= self.min_steps_before_training
        )

    def _can_train(self):
        return self.replay_buffer.num_steps_can_sample() >= self.min_steps_before_training

    def _get_action_and_info(self, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        self.exploration_policy.set_num_steps_total(self._n_env_steps_total)
        return self.exploration_policy.get_action(
            observation,
        )

    def _start_epoch(self, epoch):
        self._epoch_start_time = time.time()
        self._exploration_paths = []
        self._do_train_time = 0
        logger.push_prefix('Iteration #%d | ' % epoch)

    def _end_epoch(self):
        logger.log("Epoch Duration: {0}".format(
            time.time() - self._epoch_start_time
        ))
        logger.log("Started Training: {0}".format(self._can_train()))
        logger.pop_prefix()

    def _start_new_rollout(self):
        self.exploration_policy.reset()
        return self.training_env.reset()

    def _handle_path(self, path):
        raise NotImplementedError('Does not handle absorbing states')
        """
        Naive implementation: just loop through each transition.
        :param path:
        :return:
        """
        for (
            ob,
            action,
            reward,
            next_ob,
            terminal,
            agent_info,
            env_info
        ) in zip(
            path["observations"],
            path["actions"],
            path["rewards"],
            path["next_observations"],
            path["terminals"],
            path["agent_infos"],
            path["env_infos"],
        ):
            self._handle_step(
                ob,
                action,
                reward,
                next_ob,
                terminal,
                agent_info=agent_info,
                env_info=env_info,
            )
        self._handle_rollout_ending()

    def _handle_step(
            self,
            observation,
            action,
            reward,
            next_observation,
            terminal,
            absorbing,
            agent_info,
            env_info,
    ):
        """
        Implement anything that needs to happen after every step
        :return:
        """
        self._current_path_builder.add_all(
            observations=observation,
            actions=action,
            rewards=reward,
            next_observations=next_observation,
            terminals=terminal,
            absorbing=absorbing,
            agent_infos=agent_info,
            env_infos=env_info,
        )
        self.replay_buffer.add_sample(
            observation=observation,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation,
            absorbing=absorbing,
            agent_info=agent_info,
            env_info=env_info,
        )

    def _handle_rollout_ending(self):
        """
        Implement anything that needs to happen after every rollout.
        """
        self.replay_buffer.terminate_episode()
        self._n_rollouts_total += 1
        if len(self._current_path_builder) > 0:
            self._exploration_paths.append(
                self._current_path_builder.get_all_stacked()
            )
            self._current_path_builder = PathBuilder()

    def get_epoch_snapshot(self, epoch):
        data_to_save = dict(
            epoch=epoch,
            exploration_policy=self.exploration_policy,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        return data_to_save

    def get_extra_data_to_save(self, epoch):
        """
        Save things that shouldn't be saved every snapshot but rather
        overwritten every time.
        :param epoch:
        :return:
        """
        if self.render:
            self.training_env.render(close=True)
        data_to_save = dict(
            epoch=epoch,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        if self.save_replay_buffer:
            data_to_save['replay_buffer'] = self.replay_buffer
        if self.save_algorithm:
            data_to_save['algorithm'] = self
        return data_to_save

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass

    @abc.abstractmethod
    def cuda(self):
        """
        Turn cuda on.
        :return:
        """
        pass

    @abc.abstractmethod
    def evaluate(self, epoch):
        """
        Evaluate the policy, e.g. save/print progress.
        :param epoch:
        :return:
        """
        pass

    @abc.abstractmethod
    def _do_training(self):
        """
        Perform some update, e.g. perform one gradient step.
        :return:
        """
        pass
