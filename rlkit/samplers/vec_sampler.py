import numpy as np
from rlkit.data_management.path_builder import PathBuilder


def rollout(
    env,
    policy,
    max_path_length,
    no_terminal=False,
    render=False,
    render_kwargs={},
    preprocess_func=None,
):
    env_num = len(env)
    path_builder = [PathBuilder() for _ in range(env_num)]

    ready_env_ids = np.arange(env_num)
    observations = env.reset(ready_env_ids)

    for _ in range(max_path_length):
        if preprocess_func:
            observation = preprocess_func(observation)
            
        actions = policy.get_actions(observations)
        if render:
            env.render(**render_kwargs)

        next_observations, rewards, terminals, env_infos = env.step(actions, ready_env_ids)
        if no_terminal:
            terminals = [False for _ in range(len(ready_env_ids))]

        for idx, (
                observation,
                action,
                reward,
                next_observation,
                terminal,
                env_info,
            ) in enumerate(
                zip(
                    observations,
                    actions,
                    rewards,
                    next_observations,
                    terminals,
                    env_infos,
                )
            ):
                env_idx = ready_env_ids[idx]
                path_builder[env_idx].add_all(
                    observations=observation,
                    actions=action,
                    rewards=np.array([reward]),
                    next_observations=next_observation,
                    terminals=np.array([terminal]),
                    absorbings=np.array([0.0, 0.0]),
                    env_infos=env_info,
                )

        observations = next_observations

        if np.any(terminals):
            end_env_ids = ready_env_ids[np.where(terminals)[0]]
            ready_env_ids = np.array(list(set(ready_env_ids) - set(end_env_ids)))
            if len(ready_env_ids) == 0:
                break
        
    return path_builder


class VecPathSampler:
    def __init__(
        self,
        env,
        policy,
        num_steps,
        max_path_length,
        no_terminal=False,
        render=False,
        render_kwargs={},
        preprocess_func=None,
    ):
        """
        When obtain_samples is called, the path sampler will generates the
        minimum number of rollouts such that at least num_steps timesteps
        have been sampled
        """
        self.env = env
        self.policy = policy
        self.num_steps = num_steps
        self.max_path_length = max_path_length
        self.no_terminal = no_terminal
        self.render = render
        self.render_kwargs = render_kwargs
        self.preprocess_func = preprocess_func

    def obtain_samples(self, num_steps=None):
        paths = []
        total_steps = 0
        if num_steps is None:
            num_steps = self.num_steps
        while total_steps < num_steps:
            new_paths = rollout(
                self.env,
                self.policy,
                self.max_path_length,
                no_terminal=self.no_terminal,
                render=self.render,
                render_kwargs=self.render_kwargs,
                preprocess_func=self.preprocess_func,
            )
            paths.extend(new_paths)
            total_steps += sum([len(new_path) for new_path in new_paths])
        return paths
