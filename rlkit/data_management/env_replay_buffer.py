import numpy as np
from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer, MetaSimpleReplayBuffer
from gym.spaces import Box, Discrete, Tuple, Dict


class EnvReplayBuffer(SimpleReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            random_seed=1995
    ):
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
            random_seed=random_seed
        )

    def add_sample(self, observation, action, reward, terminal,
            next_observation, **kwargs):
        if isinstance(self._action_space, Discrete):
            new_action = np.zeros(self._action_dim)
            new_action[action] = 1
        else:
            new_action = action
        super(EnvReplayBuffer, self).add_sample(
                observation, action, reward, terminal, 
                next_observation, **kwargs)


class MetaEnvReplayBuffer(MetaSimpleReplayBuffer):
    def __init__(
            self,
            max_rb_size_per_task,
            env,
            extra_obs_dim=0,
            policy_uses_pixels=False,
            policy_uses_task_params=False,
            concat_task_params_to_policy_obs=False
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        assert extra_obs_dim == 0, "I removed the extra_obs_dim thing"
        # self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space
        super().__init__(
            max_rb_size_per_task=max_rb_size_per_task,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            policy_uses_pixels=policy_uses_pixels,
            policy_uses_task_params=policy_uses_task_params,
            concat_task_params_to_policy_obs=concat_task_params_to_policy_obs
        )


    def add_path(self, path, task_identifier):
        if isinstance(self._action_space, Discrete):
            action_array = np.eye(self._action_space.n)[path['actions'][:,0]]
            path['actions'] = action_array
        super(MetaEnvReplayBuffer, self).add_path(path, task_identifier)


def get_dim(space):
    if isinstance(space, Box):
        if len(space.low.shape) > 1:
            return space.low.shape
        return space.low.size
    elif isinstance(space, Discrete):
        return space.n
    elif isinstance(space, Tuple):
        return sum(get_dim(subspace) for subspace in space.spaces)
    elif isinstance(space, Dict):
        return {k: get_dim(v) for k,v in space.spaces.items()}
    elif hasattr(space, 'flat_dim'):
        return space.flat_dim
    else:
        raise TypeError("Unknown space: {}".format(space))
