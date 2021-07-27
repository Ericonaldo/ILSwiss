import abc
import pickle
import time
from collections import OrderedDict
from copy import deepcopy

import gtimer as gt
import numpy as np

from rlkit.core import logger, eval_util
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.data_management.path_builder import PathBuilder
from rlkit.policies.base import ExplorationPolicy
from rlkit.torch.common.policies import MakeDeterministic
from rlkit.samplers import PathSampler

from gym.spaces import Dict


class BaseAlgorithm(metaclass=abc.ABCMeta):
    """
    base algorithm for single task setting
    can be used for RL or Learning from Demonstrations
    """
    def __init__(
            self,
            env,
            exploration_policy: ExplorationPolicy,
            training_env=None,
            eval_policy=None,
            eval_sampler=None,      ##

            num_epochs=100,
            num_steps_per_epoch=10000,
            num_steps_between_train_calls=1000,
            num_steps_per_eval=1000,
            max_path_length=1000,
            min_steps_before_training=0,

            replay_buffer=None,
            replay_buffer_size=10000,

            freq_saving=1,
            save_replay_buffer=False,
            save_environment=False,
            save_algorithm=False,

            save_best=False,
            save_epoch=False,
            save_best_starting_from_epoch=0,
            best_key='AverageReturn', # higher is better
            
            no_terminal=False,
            eval_no_terminal=False,
            wrap_absorbing=False,

            render=False,
            render_kwargs={},

            freq_log_visuals=1,

            eval_deterministic=False
        ):
        self.env = env
        self.training_env = training_env or pickle.loads(pickle.dumps(env))
        self.exploration_policy = exploration_policy

        self.num_epochs = num_epochs
        self.num_env_steps_per_epoch = num_steps_per_epoch
        self.num_steps_between_train_calls = num_steps_between_train_calls
        self.num_steps_per_eval = num_steps_per_eval
        self.max_path_length = max_path_length
        self.min_steps_before_training = min_steps_before_training

        self.render = render

        self.save_replay_buffer = save_replay_buffer
        self.save_algorithm = save_algorithm
        self.save_environment = save_environment
        self.save_best = save_best
        self.save_best_starting_from_epoch = save_best_starting_from_epoch
        self.best_key = best_key
        self.best_statistic_so_far = float('-Inf')
        
        if eval_sampler is None:
            if eval_policy is None:
                eval_policy = exploration_policy
            eval_policy = MakeDeterministic(eval_policy)
            eval_sampler = PathSampler(
                env,
                eval_policy,
                num_steps_per_eval,
                max_path_length,
                no_terminal=eval_no_terminal,
                render=render,
                render_kwargs=render_kwargs
            )
        self.eval_policy = eval_policy
        self.eval_sampler = eval_sampler

        self.action_space = env.action_space
        self.obs_space = env.observation_space
        self.replay_buffer_size = replay_buffer_size
        if replay_buffer is None:
            assert max_path_length < replay_buffer_size
            replay_buffer = EnvReplayBuffer(
                self.replay_buffer_size,
                self.env,
                random_seed=np.random.randint(10000)
            )
        else:
            assert max_path_length < replay_buffer._max_replay_buffer_size
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

        if wrap_absorbing:
            # needs to be properly handled both here and in replay buffer
            raise NotImplementedError()
        self.wrap_absorbing = wrap_absorbing
        self.freq_saving = freq_saving
        self.no_terminal = no_terminal

        self.eval_statistics = None
        self.freq_log_visuals = freq_log_visuals


    def train(self, start_epoch=0, flag=False):
        self.pretrain()
        if start_epoch == 0:
            params = self.get_epoch_snapshot(-1)
            logger.save_itr_params(-1, params)
        self.training_mode(False)
        self._n_env_steps_total = start_epoch * self.num_env_steps_per_epoch
        gt.reset()
        gt.set_def_unique(False)
        self.start_training(start_epoch=start_epoch, flag=flag)


    def pretrain(self):
        """
        Do anything before the main training phase.
        """
        pass

    def start_training(self, start_epoch=0, flag=False):
        self._current_path_builder = PathBuilder()
        observation = self._start_new_rollout()

        for epoch in gt.timed_for(
            range(start_epoch, self.num_epochs),
            save_itrs=True,
        ):
            self._start_epoch(epoch)
            total_rew = 0
            for steps_this_epoch in range(self.num_env_steps_per_epoch):
                action, agent_info = self._get_action_and_info(observation)
                if self.render: self.training_env.render()

                next_ob, raw_reward, terminal, env_info = (
                    self.training_env.step(action)
                )
                if self.no_terminal: terminal = False
                self._n_env_steps_total += 1

                reward = np.array([raw_reward])
                total_rew += reward
                terminal = np.array([terminal])
                self._handle_step(
                    observation,
                    action,
                    reward,
                    next_ob,
                    np.array([False]) if self.no_terminal else terminal,
                    absorbing=np.array([0., 0.]),
                    agent_info=agent_info,
                    env_info=env_info,
                )
                if terminal[0]:
                    if flag:
                        pass
                        #print('reward: ', np.sum(total_rew))
                    total_rew = 0
                    if self.wrap_absorbing:
                        # raise NotImplementedError()
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

                if self._n_env_steps_total % self.num_steps_between_train_calls == 0:
                    gt.stamp('sample')
                    self._try_to_train(epoch)
                    gt.stamp('train')

            gt.stamp('sample')
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

        if self._can_evaluate():
            # save if it's time to save
            if (int(epoch) % self.freq_saving == 0) or (epoch + 1 >= self.num_epochs):
                # if epoch + 1 >= self.num_epochs:
                    # epoch = 'final'
                logger.save_extra_data(self.get_extra_data_to_save(epoch))
                params = self.get_epoch_snapshot(epoch)
                logger.save_itr_params(epoch, params)

            self.evaluate(epoch)

            logger.record_tabular(
                "Number of train calls total",
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
        self.eval_statistics = None
        logger.log("Epoch Duration: {0}".format(
            time.time() - self._epoch_start_time
        ))
        logger.log("Started Training: {0}".format(self._can_train()))
        logger.pop_prefix()

    def _start_new_rollout(self):
        self.exploration_policy.reset()
        return self.training_env.reset()

    def _handle_path(self, path):
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
                self._current_path_builder
            )
            self._current_path_builder = PathBuilder()

    def get_epoch_snapshot(self, epoch):
        """
        Probably will be overridden by each algorithm
        """
        data_to_save = dict(
            epoch=epoch,
            exploration_policy=self.exploration_policy,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        return data_to_save
    
    # @abc.abstractmethod
    # def load_snapshot(self, snapshot):
    #     """
    #     Should be implemented on a per algorithm basis
    #     taking into consideration the particular
    #     get_epoch_snapshot implementation for the algorithm
    #     """
    #     pass

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
    def _do_training(self):
        """
        Perform some update, e.g. perform one gradient step.
        :return:
        """
        pass


    def evaluate(self, epoch):
        """
        Evaluate the policy, e.g. save/print progress.
        :param epoch:
        :return:
        """
        statistics = OrderedDict()
        try:
            statistics.update(self.eval_statistics)
            self.eval_statistics = None
        except:
            print('No Stats to Eval')

        logger.log("Collecting samples for evaluation")
        test_paths = self.eval_sampler.obtain_samples()

        statistics.update(eval_util.get_generic_path_information(
            test_paths, stat_prefix="Test",
        ))
        statistics.update(eval_util.get_generic_path_information(
            self._exploration_paths, stat_prefix="Exploration",
        ))

        if hasattr(self.env, "log_diagnostics"):
            self.env.log_diagnostics(test_paths)
        if hasattr(self.env, "log_statistics"):
            statistics.update(self.env.log_statistics(test_paths))
        if int(epoch) % self.freq_log_visuals == 0:
            if hasattr(self.env, "log_visuals"):
                self.env.log_visuals(test_paths, epoch, logger.get_snapshot_dir())
        
        average_returns = eval_util.get_average_returns(test_paths)
        statistics['AverageReturn'] = average_returns
        for key, value in statistics.items():
            logger.record_tabular(key, np.mean(value))
        
        best_statistic = statistics[self.best_key]
        data_to_save = {
            'epoch': epoch,
            'statistics': statistics
        }
        data_to_save.update(self.get_epoch_snapshot(epoch))
        if self.save_epoch:
            logger.save_extra_data(data_to_save, 'epoch{}.pkl'.format(epoch))
        print('\n\nSAVED MODEL AT EPOCH {}\n\n'.format(epoch))
        if best_statistic > self.best_statistic_so_far:
            self.best_statistic_so_far = best_statistic
            if self.save_best and epoch >= self.save_best_starting_from_epoch:
                data_to_save = {
                    'epoch': epoch,
                    'statistics': statistics
                }
                data_to_save.update(self.get_epoch_snapshot(epoch))
                logger.save_extra_data(data_to_save, 'best.pkl')
                print('\n\nSAVED BEST\n\n')
