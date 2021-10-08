import numpy as np
from typing import Dict

from rlkit.core import dict_list_to_list_dict
from rlkit.data_management.path_builder import PathBuilder


def rollout(
    env,
    eval_env,
    policy_n: Dict,
    policy_mapping_dict,
    max_path_length,
    no_terminal=False,
    render=False,
    render_kwargs={},
):
    agent_ids = env.agent_ids
    n_agents = env.n_agents
    env_num = len(eval_env)
    path_builder = [PathBuilder(agent_ids) for _ in range(env_num)]

    ready_env_ids = np.arange(env_num)
    observations_n = eval_env.reset(ready_env_ids)

    for _ in range(max_path_length):
        actions_n = {}
        agent_infos_n = {}
        for agent_id, observations in observations_n.items():
            policy_id = policy_mapping_dict[agent_id]
            actions_n[agent_id], agent_infos_n[agent_id] = policy_n[
                policy_id
            ].get_action(observations)
        if render:
            eval_env.render(**render_kwargs)

        next_observations_n, rewards_n, terminals_n, env_infos_n = eval_env.step(
            actions_n, ready_env_ids
        )
        if no_terminal:
            terminals_n = dict(
                zip(
                    agent_ids,
                    [
                        [False for _ in range(len(ready_env_ids))]
                        for _ in range(n_agents)
                    ],
                )
            )

        for agent_id in agent_ids:
            for idx, (
                observation_n,
                action_n,
                reward_n,
                next_observation_n,
                terminal_n,
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
                            env_infos_n,
                        ],
                    )
                )
            ):
                env_idx = ready_env_ids[idx]
                path_builder[env_idx][agent_id].add_all(
                    observations=observation_n[agent_id],
                    actions=action_n[agent_id],
                    rewards=np.array([reward_n[agent_id]]),
                    next_observations=next_observation_n[agent_id],
                    terminals=np.array([terminal_n[agent_id]]),
                    absorbings=np.array([0.0, 0.0]),
                    env_infos=env_info_n[agent_id],
                )

        terminals_all = np.ones_like(list(terminals_n.values())[0])
        for terminals in terminals_n.values():
            terminals_all = np.logical_and(terminals_all, terminals)
        if np.any(terminals_all):
            end_env_ids = ready_env_ids[np.where(terminals_all)[0]]
            ready_env_ids = np.array(list(set(ready_env_ids) - set(end_env_ids)))
            if len(ready_env_ids) == 0:
                break

        observations_n = {}
        for agent_id in agent_ids:
            observations_n[agent_id] = next_observations_n[agent_id][
                np.where(terminals_all == False)
            ]

    return path_builder


class PathSampler:
    def __init__(
        self,
        env,
        eval_env,
        policy_n,
        policy_mapping_dict,
        num_steps,
        max_path_length,
        no_terminal=False,
        render=False,
        render_kwargs={},
    ):
        """
        When obtain_samples is called, the path sampler will generates the
        minimum number of rollouts such that at least num_steps timesteps
        have been sampled
        """
        self.env = env
        self.eval_env = eval_env
        self.policy_n = policy_n
        self.policy_mapping_dict = policy_mapping_dict
        self.num_steps = num_steps
        self.max_path_length = max_path_length
        self.no_terminal = no_terminal
        self.render = render
        self.render_kwargs = render_kwargs

    def obtain_samples(self, num_steps=None):
        paths = []
        total_steps = 0
        if num_steps is None:
            num_steps = self.num_steps
        while total_steps < num_steps:
            new_paths = rollout(
                self.env,
                self.eval_env,
                self.policy_n,
                self.policy_mapping_dict,
                self.max_path_length,
                no_terminal=self.no_terminal,
                render=self.render,
                render_kwargs=self.render_kwargs,
            )
            paths.extend(new_paths)
            total_steps += sum([len(new_path) for new_path in new_paths])
        return paths
