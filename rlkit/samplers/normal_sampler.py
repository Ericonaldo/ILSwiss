import numpy as np
from rlkit.data_management.path_builder import PathBuilder


def rollout(
    env,
    policy,
    max_path_length,
    no_terminal=False,
    render=False,
    render_kwargs={},
    render_mode="rgb_array",
    preprocess_func=None,
    use_horizon=False,
):
    path_builder = PathBuilder()
    observation = env.reset()

    images = []
    image = None
    for _ in range(max_path_length):
        if preprocess_func:
            observation = preprocess_func(observation)
        if use_horizon:
            horizon = np.arange(max_path_length) >= (max_path_length - 1 - _)  #
            if isinstance(observation, dict):
                observation = np.concatenate(
                    [
                        observation[policy.stochastic_policy.observation_key],
                        observation[policy.stochastic_policy.desired_goal_key],
                        horizon,
                    ],
                    axis=-1,
                )

        action, agent_info = policy.get_action(observation)
        if render:
            if render_mode == "rgb_array":
                image = env.render(mode=render_mode, **render_kwargs)
                images.append(image)
            else:
                env.render(**render_kwargs)

        next_ob, reward, terminal, env_info = env.step(action)
        if no_terminal:
            terminal = False

        path_builder.add_all(
            observations=observation,
            actions=action,
            rewards=np.array([reward]),
            next_observations=next_ob,
            terminals=np.array([terminal]),
            absorbings=np.array([0.0, 0.0]),
            agent_infos=agent_info,
            env_infos=env_info,
            image=image,
        )

        observation = next_ob
        if terminal:
            break
    return path_builder


class PathSampler:
    def __init__(
        self,
        env,
        policy,
        num_steps,
        max_path_length,
        no_terminal=False,
        render=False,
        render_kwargs={},
        render_mode="rgb_array",
        preprocess_func=None,
        horizon=False,
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
        self.horizon = horizon
        self.render_mode = render_mode

    def obtain_samples(self, num_steps=None):
        paths = []
        total_steps = 0
        if num_steps is None:
            num_steps = self.num_steps
        while total_steps < num_steps:
            new_path = rollout(
                self.env,
                self.policy,
                self.max_path_length,
                no_terminal=self.no_terminal,
                render=self.render,
                render_kwargs=self.render_kwargs,
                preprocess_func=self.preprocess_func,
                use_horizon=self.horizon,
                render_mode=self.render_mode,
            )
            paths.append(new_path)
            total_steps += len(new_path["rewards"])
        return paths
