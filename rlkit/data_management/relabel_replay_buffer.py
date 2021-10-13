from inspect import Attribute
import numpy as np
from rlkit.data_management.simple_replay_buffer import (
    SimpleReplayBuffer
)
from rlkit.data_management.env_replay_buffer import get_dim
# from rlkit.envs.goal_env_utils import compute_reward, compute_distance
from gym.spaces import Box, Discrete, Tuple, Dict

import pickle
import copy

def goal_distance(goal_a, goal_b):
    return np.linalg.norm(goal_a - goal_b, ord=2, axis=-1)

def compute_reward(achieved, goal):
    distance_threshold = 0.05
    dis = goal_distance(achieved[0], goal)
    return -(dis > distance_threshold).astype(np.float32)


class HindsightReplayBuffer(SimpleReplayBuffer):
    def __init__(self, max_replay_buffer_size, env, random_seed=1995, relabel_type='final'):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self._ob_space = env.observation_space
        self._action_space = env.action_space
        # self.compute_reward = compute_reward
        # self.compute_distance = compute_distance

        if hasattr(env, 'compute_reward'):
            self.compute_reward = env.compute_reward
        if hasattr(env, 'compute_distance'):
            self.compute_distance = env.compute_distance
        
        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            random_seed=random_seed,
        )
        self._goal_dim = get_dim(self._ob_space)['desired_goal']
        self.relabel_type = relabel_type

    def add_sample(
        self, observation, action, reward, terminal, next_observation, **kwargs
    ):
        assert isinstance(observation, dict), "Observation should be dict!"
        if isinstance(self._action_space, Discrete):
            new_action = np.zeros(self._action_dim)
            new_action[action] = 1
        else:
            new_action = action
        super(HindsightReplayBuffer, self).add_sample(
            observation, action, reward, terminal, next_observation, **kwargs
        )

    def random_batch(
        self, batch_size, keys=None, **kwargs
    ):
        relabel = (self.relabel_type is not None)
        assert (keys is None) or ("observations" in keys)
        if keys is None:
            keys = set(
                [
                    "observations",
                    "actions",
                    "rewards",
                    "terminals",
                    "next_observations"
                ]
            )
        keys_list = list(self._traj_endpoints.keys())
        starts = self._np_choice(
            keys_list, size=len(keys_list), replace=False
        )
        ends = list(map(lambda k: self._traj_endpoints[k], starts))
        
        traj_indice = self._np_randint(0, len(starts), batch_size)
        indices = []
        indices_relabel = []
        for i in traj_indice:
            traj_len = (ends[i]-starts[i]) % self._size
            step = (self._np_randint(0, traj_len, 1)[0] + starts[i]) % self._size
            
            try:
                step_her = {
                    'final': ends[i]-1,
                    'future': np.random.randint(step, (traj_len + starts[i])) % self._size
                }[self.relabel_type]
            except BaseException:
                print(starts[i], ends[i], step, ends[i])
                exit(0)
            
            # print("her:", traj_len, starts[i], ends[i], step, step_her)
            indices.append(step)
            indices_relabel.append(step_her)
        # indices = self._np_randint(0, self._size, batch_size)
        batch_to_return = self._get_batch_using_indices(indices, keys=keys)
        
        # relabel
        if relabel:
            batch_to_relabel = self._get_batch_using_indices(indices_relabel, keys=["observations", "next_observations"])
            batch_to_return["observations"]["desired_goal"] = copy.deepcopy(batch_to_relabel["next_observations"]["achieved_goal"])
            batch_to_return["next_observations"]["desired_goal"] = copy.deepcopy(batch_to_relabel["next_observations"]["achieved_goal"])

        batch_to_return["achieved_goals"] = batch_to_return["observations"]["achieved_goal"]
        batch_to_return["desired_goals"] = batch_to_return["observations"]["desired_goal"]
        batch_to_return["next_achieved_goals"] = batch_to_return["next_observations"]["achieved_goal"]
        batch_to_return["next_desired_goals"] = batch_to_return["next_observations"]["desired_goal"]
        batch_to_return["observations"] = batch_to_return["observations"]["observation"]
        batch_to_return["next_observations"] = batch_to_return["next_observations"]["observation"]
        if relabel:
            # batch_to_return["rewards"] = self.compute_reward(batch_to_return["next_achieved_goals"], batch_to_return["desired_goals"], info=None)
            batch_to_return["rewards"] = compute_reward(batch_to_return["next_achieved_goals"], batch_to_return["desired_goals"])

        return batch_to_return

    def save_data(self, save_name):
        save_dict = {
            "observations": self._observations['observation'][: self._top],
            "achieved_goals": self._observations['achieved_goal'][: self._top],
            "desired_goals": self._observations['desired_goal'][: self._top],
            "actions": self._actions[: self._top],
            "next_observations": self._next_obs['observation'][: self._top],
            "next_achieved_goals": self._next_obs['achieved_goal'][: self._top],
            "next_desired_goals": self._next_obs['desired_goal'][: self._top],
            "terminals": self._terminals[: self._top],
            "timeouts": self._timeouts[: self._top],
            "rewards": self._rewards[: self._top],
            "agent_infos": [None] * len(self._observations[: self._top]),
            "env_infos": [None] * len(self._observations[: self._top]),
        }

        with open(save_name, "wb") as f:
            pickle.dump(save_dict, f)
