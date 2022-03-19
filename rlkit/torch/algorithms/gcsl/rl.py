import numpy as np
import gtimer as gt

from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.algorithms.torch_rl_algorithm import TorchRLAlgorithm
from rlkit.data_management.relabel_horizon_replay_buffer import (
    HindsightHorizonReplayBuffer,
)
from rlkit.data_management.path_builder import PathBuilder


class GoalHorizonRL(TorchRLAlgorithm):
    """
    Hindsight Experience Replay. Default using TD3 for RL policy trainer.
    """

    def __init__(
        self,
        replay_buffer=None,
        her_ratio=0.8,
        relabel_type="future",
        use_horizons=False,
        **kwargs
    ):
        if replay_buffer is None:
            assert kwargs["max_path_length"] < kwargs["replay_buffer_size"]
            replay_buffer = HindsightHorizonReplayBuffer(
                max_replay_buffer_size=kwargs["replay_buffer_size"],
                max_path_length=kwargs["max_path_length"],
                env=kwargs["env"],
                random_seed=np.random.randint(10000),
                relabel_type=relabel_type,
            )

        super().__init__(replay_buffer=replay_buffer, **kwargs)
        self.use_horizons = use_horizons
        if use_horizons:
            self.eval_sampler.horizon = True

    def get_batch(self, keys=None):
        buffer = self.replay_buffer
        batch = buffer.random_batch(self.batch_size, keys=keys)
        batch = np_to_pytorch_batch(batch)
        return batch

    def _get_action_and_info(self, observation, horizon):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        self.exploration_policy.set_num_steps_total(self._n_env_steps_total)
        if isinstance(observation[0], dict):
            observation = np.array(
                [
                    np.concatenate(
                        [
                            observation[i][self.exploration_policy.observation_key],
                            observation[i][self.exploration_policy.desired_goal_key],
                            horizon[i],
                        ],
                        axis=-1,
                    )
                    for i in range(len(observation))
                ]
            )

        return self.exploration_policy.get_actions(
            observation,
        )

    def start_training(self, start_epoch=0, flag=False):
        # self._current_path_builder = PathBuilder()
        self.ready_env_ids = np.arange(self.env_num)
        observations = self._start_new_rollout(
            self.ready_env_ids
        )  # Do it for support vec env

        self._current_path_builder = [
            PathBuilder() for _ in range(len(self.ready_env_ids))
        ]

        for epoch in gt.timed_for(
            range(start_epoch, self.num_epochs),
            save_itrs=True,
        ):
            self._start_epoch(epoch)
            total_rews = np.array([0.0 for _ in range(len(self.ready_env_ids))])
            timesteps = np.array([0 for _ in range(len(self.ready_env_ids))])
            for steps_this_epoch in range(self.num_env_steps_per_epoch // self.env_num):
                horizon = np.array(
                    [
                        np.arange(self.max_path_length)
                        >= (self.max_path_length - 1 - timestep)
                        for timestep in timesteps
                    ]
                )  # Temperature encoding of horizon
                actions = self._get_action_and_info(observations, horizon)

                if type(actions) is tuple:
                    actions = actions[0]

                if self.render:
                    self.training_env.render()

                next_obs, raw_rewards, terminals, env_infos = self.training_env.step(
                    actions, self.ready_env_ids
                )
                timesteps += 1
                if self.no_terminal:
                    terminals = [False for _ in range(len(self.ready_env_ids))]

                self._n_env_steps_total += len(self.ready_env_ids)

                rewards = raw_rewards
                total_rews += raw_rewards

                self._handle_vec_step(
                    observations,
                    actions,
                    rewards,
                    next_obs,
                    np.array([False for _ in range(len(self.ready_env_ids))])
                    if self.no_terminal
                    else terminals,
                    absorbings=[
                        np.array([0.0, 0.0]) for _ in range(len(self.ready_env_ids))
                    ],
                    env_infos=env_infos,
                )
                if np.any(terminals):
                    env_ind_local = np.where(terminals)[0]
                    if flag:
                        pass
                    total_rews[env_ind_local] = 0.0
                    timesteps[env_ind_local] = 0
                    if self.wrap_absorbing:
                        # raise NotImplementedError()
                        """
                        If we wrap absorbing states, two additional
                        transitions must be added: (s_T, s_abs) and
                        (s_abs, s_abs). In Disc Actor Critic paper
                        they make s_abs be a vector of 0s with last
                        dim set to 1. Here we are going to add the following:
                        ([next_ob, 0], random_action, [next_ob, 1]) and
                        ([next_ob, 1], random_action, [next_ob, 1])
                        This way we can handle varying types of terminal states.
                        """
                        # next_ob is the absorbing state
                        # for now just taking the previous action
                        self._handle_vec_step(
                            next_obs,
                            actions,
                            # env.action_space.sample(),
                            # the reward doesn't matter
                            rewards,
                            next_obs,
                            np.array([False for _ in range(len(self.ready_env_ids))]),
                            absorbings=[
                                np.array([0.0, 1.0])
                                for _ in range(len(self.ready_env_ids))
                            ],
                            env_infos=env_infos,
                        )
                        self._handle_vec_step(
                            next_obs,
                            actions,
                            # env.action_space.sample(),
                            # the reward doesn't matter
                            rewards,
                            next_obs,
                            np.array([False for _ in range(len(self.ready_env_ids))]),
                            absorbings=[
                                np.array([1.0, 1.0])
                                for _ in range(len(self.ready_env_ids))
                            ],
                            env_infos=env_infos,
                        )
                    self._handle_vec_rollout_ending(env_ind_local)
                    reset_observations = self._start_new_rollout(env_ind_local)
                    next_obs[env_ind_local] = reset_observations
                elif np.any(
                    np.array(
                        [
                            len(self._current_path_builder[i])
                            for i in range(len(self.ready_env_ids))
                        ]
                    )
                    >= self.max_path_length
                ):
                    env_ind_local = np.where(
                        np.array(
                            [
                                len(self._current_path_builder[i])
                                for i in range(len(self.ready_env_ids))
                            ]
                        )
                        >= self.max_path_length
                    )[0]
                    timesteps[env_ind_local] = 0
                    self._handle_vec_rollout_ending(env_ind_local)
                    reset_observations = self._start_new_rollout(env_ind_local)
                    next_obs[env_ind_local] = reset_observations

                observations = next_obs

                if (
                    self._n_env_steps_total - self._n_prev_train_env_steps
                ) >= self.num_steps_between_train_calls:
                    gt.stamp("sample")
                    self._try_to_train(epoch)
                    gt.stamp("train")

            gt.stamp("sample")
            self._try_to_eval(epoch)
            gt.stamp("eval")
            self._end_epoch()
