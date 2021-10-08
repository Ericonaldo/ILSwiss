import numpy as np
import gtimer as gt
from typing import Dict

from rlkit.data_management.path_builder import PathBuilder
from rlkit.core import dict_list_to_list_dict
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.algorithms.torch_rl_algorithm import TorchRLAlgorithm
from rlkit.torch.algorithms.option.option_ppo import OptionPPO


class TorchOptionRLAlgorithm(TorchRLAlgorithm):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def get_batch_trajs(self, traj_num: int, agent_id: str):
        batch = self.replay_buffer.sample_trajs(traj_num, agent_id)
        batch = [np_to_pytorch_batch(b) for b in batch]
        return batch

    def _init_options_n(self, env_ind_local):
        return {
            a_id: np.ones((len(env_ind_local), 1), dtype=np.float32)
            * self.exploration_policy_n[self.policy_mapping_dict[a_id]].option_dim
            for a_id in self.agent_ids
        }

    def _get_action_and_option(self, observation_n, prev_option_n):
        """
        Get an action to take in the environment.
        :param observation_n:
        :param prev_option_n:
        :return:
        """
        action_n = {}
        option_n = {}
        for agent_id in self.agent_ids:
            policy_id = self.policy_mapping_dict[agent_id]
            observation, prev_option = observation_n[agent_id], prev_option_n[agent_id]
            self.exploration_policy_n[policy_id].set_num_steps_total(
                self._n_env_steps_total
            )
            option = self.exploration_policy_n[policy_id].get_options(
                observation, prev_option
            )
            action = self.exploration_policy_n[policy_id].get_actions(
                observation, option
            )
            action_n[agent_id], option_n[agent_id] = action, option
        return action_n, option_n

    def start_training(self, start_epoch=0, flag=False):
        # self._current_path_builder = PathBuilder()
        self.ready_env_ids = np.arange(self.env_num)
        prev_options_n = self._init_options_n(self.ready_env_ids)
        observations_n = self._start_new_rollout(
            self.ready_env_ids
        )  # Do it for support vec env

        self._current_path_builder = [
            PathBuilder(self.agent_ids) for _ in range(len(self.ready_env_ids))
        ]

        for epoch in gt.timed_for(
            range(start_epoch, self.num_epochs),
            save_itrs=True,
        ):
            self._start_epoch(epoch)
            total_rews_n = dict(
                zip(
                    self.agent_ids,
                    [
                        np.array([0.0 for _ in range(len(self.ready_env_ids))])
                        for _ in range(self.n_agents)
                    ],
                )
            )
            for steps_this_epoch in range(self.num_env_steps_per_epoch // self.env_num):
                actions_n, options_n = self._get_action_and_option(
                    observations_n, prev_options_n
                )

                for a_id in self.agent_ids:
                    if type(actions_n[a_id]) is tuple:
                        actions_n[a_id] = actions_n[a_id][0]

                if self.render:
                    self.training_env.render()

                (
                    next_obs_n,
                    raw_rewards_n,
                    terminals_n,
                    env_infos_n,
                ) = self.training_env.step(actions_n, self.ready_env_ids)
                if self.no_terminal:
                    terminals_n = dict(
                        zip(
                            self.agent_ids,
                            [
                                [False for _ in range(len(self.ready_env_ids))]
                                for _ in range(self.n_agents)
                            ],
                        )
                    )
                # self._n_env_steps_total += 1
                self._n_env_steps_total += len(self.ready_env_ids)

                rewards_n = raw_rewards_n
                for a_id in self.agent_ids:
                    total_rews_n[a_id] += raw_rewards_n[a_id]

                self._handle_vec_step(
                    observations_n,
                    actions_n,
                    rewards_n,
                    next_obs_n,
                    terminals_n,
                    prev_options_n,
                    options_n,
                    env_infos_n=env_infos_n,
                )
                terminals_all = np.ones_like(list(terminals_n.values())[0])
                for terminals in terminals_n.values():
                    terminals_all = np.logical_and(terminals_all, terminals)
                if np.any(terminals_all):
                    env_ind_local = np.where(terminals_all)[0]
                    if flag:
                        pass
                    for a_id in self.agent_ids:
                        total_rews_n[a_id][env_ind_local] = 0.0
                    self._handle_vec_rollout_ending(env_ind_local)
                    reset_prev_options_n = self._init_options_n(env_ind_local)
                    reset_observations_n = self._start_new_rollout(env_ind_local)
                    for a_id in self.agent_ids:
                        next_obs_n[a_id][env_ind_local] = reset_observations_n[a_id]
                        options_n[a_id][env_ind_local] = reset_prev_options_n[a_id]
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
                    self._handle_vec_rollout_ending(env_ind_local)
                    reset_prev_options_n = self._init_options_n(env_ind_local)
                    reset_observations_n = self._start_new_rollout(env_ind_local)
                    for a_id in self.agent_ids:
                        next_obs_n[a_id][env_ind_local] = reset_observations_n[a_id]
                        options_n[a_id][env_ind_local] = reset_prev_options_n[a_id]

                observations_n = next_obs_n
                prev_options_n = options_n

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

    def _handle_path(self, path):
        """
        Naive implementation: just loop through each transition.
        :param path:
        :return:
        """

        for (
            ob_n,
            action_n,
            reward_n,
            next_ob_n,
            terminal_n,
            prev_option_n,
            option_n,
            env_info_n,
        ) in zip(
            *map(
                dict_list_to_list_dict,
                [
                    path.get_all_agent_dict("observations"),
                    path.get_all_agent_dict("actions"),
                    path.get_all_agent_dict("rewards"),
                    path.get_all_agent_dict("next_observations"),
                    path.get_all_agent_dict("terminals"),
                    path.get_all_agent_dict("prev_options"),
                    path.get_all_agent_dict("options"),
                    path.get_all_agent_dict("env_infos"),
                ],
            )
        ):
            self._handle_step(
                ob_n,
                action_n,
                reward_n,
                next_ob_n,
                terminal_n,
                prev_option_n,
                option_n,
                env_info_n=env_info_n,
                path_builder=False,
            )

    def _handle_vec_step(
        self,
        observations_n: Dict,
        actions_n: Dict,
        rewards_n: Dict,
        next_observations_n: Dict,
        terminals_n: Dict,
        prev_options_n: Dict,
        options_n: Dict,
        env_infos_n: Dict,
    ):
        """
        Implement anything that needs to happen after every step under vec envs
        :return:
        """
        for env_idx, (
            ob_n,
            action_n,
            reward_n,
            next_ob_n,
            terminal_n,
            prev_option_n,
            option_n,
            env_info_n,
        ) in enumerate(
            zip(
                *map(
                    dict_list_to_list_dict,
                    [
                        observations_n,
                        actions_n,
                        rewards_n,
                        next_observations_n,
                        terminals_n,
                        prev_options_n,
                        options_n,
                        env_infos_n,
                    ],
                )
            )
        ):
            self._handle_step(
                ob_n,
                action_n,
                reward_n,
                next_ob_n,
                terminal_n,
                prev_option_n,
                option_n,
                env_info_n=env_info_n,
                env_idx=env_idx,
                add_buf=False,
            )

    def _handle_step(
        self,
        observation_n,
        action_n,
        reward_n,
        next_observation_n,
        terminal_n,
        prev_option_n,
        option_n,
        env_info_n,
        env_idx=0,
        add_buf=True,
        path_builder=True,
    ):
        """
        Implement anything that needs to happen after every step
        :return:
        """
        if path_builder:
            for a_id in self.agent_ids:
                self._current_path_builder[env_idx][a_id].add_all(
                    observations=observation_n[a_id],
                    actions=action_n[a_id],
                    rewards=reward_n[a_id],
                    next_observations=next_observation_n[a_id],
                    terminals=terminal_n[a_id],
                    prev_options=prev_option_n[a_id],
                    options=option_n[a_id],
                    env_infos=env_info_n[a_id],
                )
        if add_buf:
            self.replay_buffer.add_sample(
                observation_n=observation_n,
                action_n=action_n,
                reward_n=reward_n,
                terminal_n=terminal_n,
                next_observation_n=next_observation_n,
                prev_option_n=prev_option_n,
                option_n=option_n,
                env_info_n=env_info_n,
            )

    def _do_training(self, epoch):
        for _ in range(self.num_train_steps_per_train_call):
            for a_id in self.agent_ids:
                p_id = self.policy_mapping_dict[a_id]
                if isinstance(self.trainer_n[p_id], OptionPPO):
                    # PPO uses all on-policy data for training
                    self.trainer_n[p_id].train_step(self.get_all_trajs(a_id))
                    self.clear_buffer(a_id)
                else:
                    self.trainer_n[p_id].train_step(self.get_batch(a_id))
