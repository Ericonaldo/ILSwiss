from rlkit.data_management.simple_replay_buffer import (
    SimpleReplayBuffer,
)
from gym.spaces import Box, Discrete, Tuple, Dict


class EnvReplayBuffer(SimpleReplayBuffer):
    def __init__(self, max_replay_buffer_size, env, random_seed=1995):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self._ob_space = env.observation_space
        self._action_space = env.action_space
        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            random_seed=random_seed,
        )

    def add_sample(
        self, observation, action, reward, terminal, next_observation, **kwargs
    ):
        # if isinstance(self._action_space, Discrete):
        #     new_action = np.zeros(self._action_dim)
        #     new_action[action] = 1
        # else:
        #     new_action = action
        super(EnvReplayBuffer, self).add_sample(
            observation, action, reward, terminal, next_observation, **kwargs
        )


def get_dim(space):
    if isinstance(space, Box):
        if len(space.low.shape) > 1:
            return space.low.shape
        return space.low.size
    elif isinstance(space, Discrete):
        return 1  # space.n
    elif isinstance(space, Tuple):
        return sum(get_dim(subspace) for subspace in space.spaces)
    elif isinstance(space, Dict):
        return {k: get_dim(v) for k, v in space.spaces.items()}
    elif hasattr(space, "flat_dim"):
        return space.flat_dim
    else:
        raise TypeError("Unknown space: {}".format(space))
