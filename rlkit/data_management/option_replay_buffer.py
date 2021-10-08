import numpy as np
import pickle

from rlkit.data_management.env_replay_buffer import get_dim
from rlkit.data_management.simple_replay_buffer import AgentSimpleReplayBuffer


class OptionEnvReplayBuffer:
    def __init__(self, max_replay_buffer_size, env, option_dim, random_seed=1995):
        self._observation_space_n = env.observation_space_n
        self._action_space_n = env.action_space_n
        self.n_agents = env.n_agents
        self.agent_ids = env.agent_ids
        # TODO(zbzhu): MAYBE change agent_buffers to policy_buffers
        self.agent_buffers = {
            a_id: AgentOptionEnvReplayBuffer(
                max_replay_buffer_size,
                self._observation_space_n[a_id],
                self._action_space_n[a_id],
                option_dim,
            )
            for a_id in self.agent_ids
        }
        self._max_replay_buffer_size = max_replay_buffer_size

    def num_steps_can_sample(self):
        return list(self.agent_buffers.values())[0].num_steps_can_sample()

    def random_batch(self, batch_size: int, agent_id: str):
        return self.agent_buffers[agent_id].random_batch(batch_size)

    def sample_trajs(self, num_trajs: int, agent_id: str):
        return self.agent_buffers[agent_id].sample_trajs(num_trajs)

    def terminate_episode(self):
        for a_id in self.agent_ids:
            self.agent_buffers[a_id].terminate_episode()

    def clear(self, agent_id: str):
        self.agent_buffers[agent_id].clear()

    def sample_all_trajs(self, agent_id: str):
        return self.agent_buffers[agent_id].sample_all_trajs()

    def add_path(
        self,
        path_n,
        env,
    ):
        for a_id in path_n.keys():
            self.agent_buffers[a_id].add_path(path_n[a_id], env=env)

    def add_sample(
        self,
        observation_n,
        action_n,
        reward_n,
        terminal_n,
        next_observation_n,
        prev_option_n,
        option_n,
        **kwargs
    ):
        for a_id in observation_n.keys():
            self.agent_buffers[a_id].add_sample(
                observation_n[a_id],
                action_n[a_id],
                reward_n[a_id],
                terminal_n[a_id],
                next_observation_n[a_id],
                prev_option_n[a_id],
                option_n[a_id],
                **{k: v[a_id] if isinstance(v, dict) else v for k, v in kwargs.items()},
            )


class AgentOptionEnvReplayBuffer(AgentSimpleReplayBuffer):
    """
    THE MAX LENGTH OF AN EPISODE SHOULD BE STRICTLY SMALLER THAN THE max_replay_buffer_size
    OTHERWISE THERE IS A BUG IN TERMINATE_EPISODE
    """

    def __init__(
        self,
        max_replay_buffer_size,
        observation_space,
        action_space,
        option_dim,
        random_seed=1995,
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self._ob_space = observation_space
        self._action_space = action_space
        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            random_seed=random_seed,
        )

        self._option_dim = option_dim
        self._prev_options = np.zeros((max_replay_buffer_size, 1))
        self._options = np.zeros((max_replay_buffer_size, 1))

        self.end_list = []

    def clear(self):
        super().clear()
        self._prev_options = np.zeros((self._max_replay_buffer_size, 1))
        self._options = np.zeros((self._max_replay_buffer_size, 1))

        self.end_list = []

    @property
    def num_trajs(self):
        return len(self.end_list)

    def add_sample(
        self,
        observation,
        action,
        reward,
        terminal,
        next_observation,
        prev_option=None,
        option=None,
        timeout=False,
        **kwargs
    ):
        # if option is None:
        #     print("add sample option is None!")
        if (prev_option is not None) or (option is not None):
            assert (prev_option <= self._option_dim) and (
                option <= self._option_dim
            ), "Wrong option dim!"

        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._prev_options[self._top] = prev_option
        self._options[self._top] = option
        self._timeouts[self._top] = timeout

        if terminal:
            next_start = (self._top + 1) % self._max_replay_buffer_size
            self._traj_endpoints[self._cur_start] = next_start
            self._cur_start = next_start

        if isinstance(self._observations, dict):
            for key, obs in observation.items():
                self._observations[key][self._top] = obs
            for key, obs in next_observation.items():
                self._next_obs[key][self._top] = obs
        else:
            self._observations[self._top] = observation
            self._next_obs[self._top] = next_observation
        self._advance()

    def save_data(self, save_name):
        save_dict = {
            "observations": self._observations[: self._top],
            "actions": self._actions[: self._top],
            "next_observations": self._next_obs[: self._top],
            "terminals": self._terminals[: self._top],
            "prev_options": self._prev_options[: self._top],
            "options": self._options[: self._top],
            "timeouts": self._timeouts[: self._top],
            "rewards": self._rewards[: self._top],
            "agent_infos": [None] * len(self._observations[: self._top]),
            "env_infos": [None] * len(self._observations[: self._top]),
        }

        with open(save_name, "wb") as f:
            pickle.dump(save_dict, f)

    def add_path(self, path, env=None):
        if "prev_options" not in path.keys():
            path["prev_options"] = [None] * len(path["observations"])
        if "options" not in path.keys():
            path["options"] = [None] * len(path["observations"])
        for (
            ob,
            action,
            reward,
            next_ob,
            terminal,
            prev_option,
            option,
        ) in zip(
            path["observations"],
            path["actions"],
            path["rewards"],
            path["next_observations"],
            path["terminals"],
            path["prev_options"],
            path["options"],
        ):
            self.add_sample(
                observation=ob,
                action=action,
                reward=reward,
                terminal=terminal,
                next_observation=next_ob,
                prev_option=prev_option,
                option=option,
            )

        self.terminate_episode()
        self._trajs += 1

    def _get_batch_using_indices(self, indices, keys=None, step_num=1, **kwargs):
        if keys is None:
            keys = set(
                [
                    "observations",
                    "actions",
                    "rewards",
                    "terminals",
                    "next_observations",
                    "prev_options",
                    "options",
                ]
            )
        if isinstance(self._observations, dict):
            obs_to_return = {}
            next_obs_to_return = {}
            for k in self._observations:
                if "observations" in keys:
                    obs_to_return[k] = self._observations[k][indices]
                if "next_observations" in keys:
                    next_obs_to_return[k] = self._next_obs[k][indices]
        else:
            obs_to_return = self._observations[indices]
            next_obs_to_return = self._next_obs[indices]

        ret_dict = {}
        if "observations" in keys:
            ret_dict["observations"] = obs_to_return
        if "actions" in keys:
            ret_dict["actions"] = self._actions[indices]
        if "rewards" in keys:
            ret_dict["rewards"] = self._rewards[indices]
        if "terminals" in keys:
            ret_dict["terminals"] = self._terminals[indices]
        if "next_observations" in keys:
            ret_dict["next_observations"] = next_obs_to_return
        if "prev_options" in keys:
            ret_dict["prev_options"] = self._prev_options[indices]
        if "options" in keys:
            ret_dict["options"] = self._options[indices]

        # print(step_num, ret_dict.keys())
        return ret_dict

    def terminate_episode(self):
        super().terminate_episode()
        self.end_list.append(self._top)

    def _advance(self):
        if self._top in self._traj_endpoints:
            # this means that the step in the replay buffer
            # that we just overwrote was the start of a some
            # trajectory, so now the full trajectory is no longer
            # there and we should remove it
            self.end_list.remove(self._traj_endpoints[self._top])
        super()._advance()

    def pop(self):
        keys = set(
            [
                "observations",
                "actions",
                "rewards",
                "terminals",
                "next_observations",
                "prev_options",
                "options",
            ]
        )
        if len(self.end_list) <= 1:
            assert (
                len(self._traj_endpoints.keys()) <= 1
            ), "incosistent of endlist and endpoints!"
            fore_indx = list(self._traj_endpoints.keys())[0]
        else:
            fore_indx = self.end_list[-2]

        if fore_indx > self.end_list[-1]:
            indices = list(range(fore_indx, self._max_replay_buffer_size)) + list(
                range(0, self.end_list[-1])
            )
        else:
            indices = list(range(fore_indx, self.end_list[-1]))

        if isinstance(self._observations, dict):
            obs_to_return = {}
            next_obs_to_return = {}
            for k in self._observations:
                if "observations" in keys:
                    obs_to_return[k] = self._observations[k][indices]
                if "next_observations" in keys:
                    next_obs_to_return[k] = self._next_obs[k][indices]
        else:
            obs_to_return = self._observations[indices]
            next_obs_to_return = self._next_obs[indices]

        ret_dict = {}
        if "observations" in keys:
            ret_dict["observations"] = obs_to_return
        if "actions" in keys:
            ret_dict["actions"] = self._actions[indices]
        if "rewards" in keys:
            ret_dict["rewards"] = self._rewards[indices]
        if "terminals" in keys:
            ret_dict["terminals"] = self._terminals[indices]
        if "next_observations" in keys:
            ret_dict["next_observations"] = next_obs_to_return
        if "prev_options" in keys:
            ret_dict["prev_options"] = self._prev_options[indices]
        if "options" in keys:
            ret_dict["options"] = self._options[indices]

        self._top = fore_indx
        if self._top in self._traj_endpoints.keys():
            del self._traj_endpoints[self._top]

        if self._size < self._max_replay_buffer_size:
            self._size = (
                self._size
                - ((self.end_list[-1] - fore_indx) + self._max_replay_buffer_size)
                % self._max_replay_buffer_size
            ) % self._max_replay_buffer_size

        self.end_list = self.end_list[:-1]
        # print(step_num, ret_dict.keys())
        return ret_dict
