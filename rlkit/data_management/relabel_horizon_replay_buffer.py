from inspect import Attribute
import numpy as np
from rlkit.data_management.relabel_replay_buffer import HindsightReplayBuffer

import pickle
import copy


class HindsightHorizonReplayBuffer(HindsightReplayBuffer):
    def __init__(self, max_path_length=50, *args, **kwargs):
        """
        :param max_replay_buffer_size:
        :param env:
        """

        super().__init__(her_ratio=1.0, *args, **kwargs)  # all relabel

        self.max_path_length = max_path_length
        self.all_data = None

    def add_sample(
        self, observation, action, reward, terminal, next_observation, **kwargs
    ):
        assert isinstance(observation, dict), "Observation should be dict!"
        # if isinstance(self._action_space, Discrete):
        #     new_action = np.zeros(self._action_dim)
        #     new_action[action] = 1
        # else:
        #     new_action = action
        # new_action = action
        super().add_sample(
            observation, action, reward, terminal, next_observation, **kwargs
        )

    def get_all(self, keys=None, relabel=True, **kwargs):
        relabel = (self.relabel_type is not None) & relabel
        assert (keys is None) or ("observations" in keys)
        if keys is None:
            keys = set(
                [
                    "observations",
                    "actions",
                    "rewards",
                    "horizons",
                    "terminals",
                    "next_observations",
                ]
            )
        keys_list = list(self._traj_endpoints.keys())
        starts = self._np_choice(keys_list, size=len(keys_list), replace=False)
        ends = list(map(lambda k: self._traj_endpoints[k], starts))

        traj_indice = range(len(starts))
        indices = []
        indices_relabel = []
        # indices_subgoal = []
        for i in traj_indice:
            traj_len = (ends[i] - starts[i]) % self._size
            for j in range(traj_len):
                step = (j + starts[i]) % self._size

                try:
                    step_her = {
                        "final": ends[i] - 1,
                        "future": np.random.randint(step, (traj_len + starts[i]))
                        % self._size,
                    }[self.relabel_type]
                except Exception as err:
                    print(err, starts[i], ends[i], step, ends[i])
                    exit(0)

                # print("her:", traj_len, starts[i], ends[i], step, step_her)
                indices.append(step)
                indices_relabel.append(step_her)

        batch_to_return = self._get_batch_using_indices(indices, keys=keys)
        # relabel
        if relabel:
            relabel_num = len(indices)
            batch_to_relabel = self._get_batch_using_indices(
                indices_relabel, keys=["observations", "next_observations"]
            )
            if "observations" in keys:
                batch_to_return["observations"][self.desired_goal_key][
                    :relabel_num
                ] = copy.deepcopy(
                    batch_to_relabel["next_observations"][self.achieved_goal_key][
                        :relabel_num
                    ]
                )
            if "next_observations" in keys:
                batch_to_return["next_observations"][self.desired_goal_key][
                    :relabel_num
                ] = copy.deepcopy(
                    batch_to_relabel["next_observations"][self.achieved_goal_key][
                        :relabel_num
                    ]
                )

        if "observations" in keys:
            batch_to_return["achieved_goals"] = batch_to_return["observations"][
                self.achieved_goal_key
            ]
            batch_to_return["desired_goals"] = batch_to_return["observations"][
                self.desired_goal_key
            ]
            batch_to_return["observations"] = batch_to_return["observations"][
                self.observation_key
            ]

        if "horizons" in keys:
            lengths = np.array(indices_relabel) - np.array(indices)
            horizons = np.tile(np.arange(self.max_path_length), (len(indices), 1))
            horizons = horizons >= lengths[..., None]
            batch_to_return["horizons"] = horizons
            # print("encoding horizon", horizons)

        if "next_observations" in keys:
            batch_to_return["next_achieved_goals"] = batch_to_return[
                "next_observations"
            ][self.achieved_goal_key]
            batch_to_return["next_desired_goals"] = batch_to_return[
                "next_observations"
            ][self.desired_goal_key]
            batch_to_return["next_observations"] = batch_to_return["next_observations"][
                self.observation_key
            ]

        if relabel:
            if "next_achieved_goals" in batch_to_return.keys():
                batch_to_return["rewards"] = self.compute_reward(
                    batch_to_return["next_achieved_goals"],
                    batch_to_return["desired_goals"],
                    info=None,
                ).reshape(-1, 1)

        return batch_to_return

    def random_batch(self, batch_size, keys=None, relabel=True, **kwargs):
        relabel = (self.relabel_type is not None) & relabel & (self.her_ratio > 0)
        assert (keys is None) or ("observations" in keys)
        if keys is None:
            keys = set(
                [
                    "observations",
                    "actions",
                    "rewards",
                    "horizons",
                    "terminals",
                    "next_observations",
                ]
            )

        keys_list = list(self._traj_endpoints.keys())
        starts = self._np_choice(keys_list, size=len(keys_list), replace=False)
        ends = list(map(lambda k: self._traj_endpoints[k], starts))

        traj_indice = self._np_randint(0, len(starts), batch_size)
        indices = []
        indices_relabel = []
        # indices_subgoal = []
        for i in traj_indice:
            traj_len = (ends[i] - starts[i]) % self._size
            step = (self._np_randint(0, traj_len, 1)[0] + starts[i]) % self._size
            indices.append(step)

            if relabel:
                try:
                    step_her = {
                        "final": ends[i] - 1,
                        "future": np.random.randint(step, (traj_len + starts[i]))
                        % self._size,
                    }[self.relabel_type]
                except Exception as err:
                    print(err, starts[i], ends[i], step, ends[i])
                    exit(0)

                indices_relabel.append(step_her)

        batch_to_return = self._get_batch_using_indices(indices, keys=keys)

        # relabel
        if relabel:
            relabel_num = int(self.her_ratio * batch_size)
            batch_to_relabel = self._get_batch_using_indices(
                indices_relabel, keys=["observations", "next_observations"]
            )
            if "observations" in keys:
                batch_to_return["observations"][self.desired_goal_key][
                    :relabel_num
                ] = copy.deepcopy(
                    batch_to_relabel["next_observations"][self.achieved_goal_key][
                        :relabel_num
                    ]
                )
            if "next_observations" in keys:
                batch_to_return["next_observations"][self.desired_goal_key][
                    :relabel_num
                ] = copy.deepcopy(
                    batch_to_relabel["next_observations"][self.achieved_goal_key][
                        :relabel_num
                    ]
                )

        if "observations" in keys:
            batch_to_return["achieved_goals"] = batch_to_return["observations"][
                self.achieved_goal_key
            ]
            batch_to_return["desired_goals"] = batch_to_return["observations"][
                self.desired_goal_key
            ]
            batch_to_return["observations"] = batch_to_return["observations"][
                self.observation_key
            ]

        if "horizons" in keys:
            lengths = np.array(indices_relabel) - np.array(indices)
            horizons = np.tile(np.arange(self.max_path_length), (batch_size, 1))
            horizons = horizons >= lengths[..., None]
            batch_to_return["horizons"] = horizons
            # print("encoding horizon", horizons)

        if "next_observations" in keys:
            batch_to_return["next_achieved_goals"] = batch_to_return[
                "next_observations"
            ][self.achieved_goal_key]
            batch_to_return["next_desired_goals"] = batch_to_return[
                "next_observations"
            ][self.desired_goal_key]
            batch_to_return["next_observations"] = batch_to_return["next_observations"][
                self.observation_key
            ]

        if relabel:
            if "next_achieved_goals" in batch_to_return.keys():
                batch_to_return["rewards"] = self.compute_reward(
                    batch_to_return["next_achieved_goals"],
                    batch_to_return["desired_goals"],
                    info=None,
                ).reshape(-1, 1)

        return batch_to_return

    def save_data(self, save_name):
        save_dict = {
            "observations": self._observations[self.observation_key][: self._top],
            "achieved_goals": self._observations[self.achieved_goal_key][: self._top],
            "desired_goals": self._observations[self.desired_goal_key][: self._top],
            "actions": self._actions[: self._top],
            "next_observations": self._next_obs[self.observation_key][: self._top],
            "next_achieved_goals": self._next_obs[self.achieved_goal_key][: self._top],
            "next_desired_goals": self._next_obs[self.desired_goal_key][: self._top],
            "horizons": self._horizons[: self._top],
            "terminals": self._terminals[: self._top],
            "timeouts": self._timeouts[: self._top],
            "rewards": self._rewards[: self._top],
            "agent_infos": [None] * len(self._observations[: self._top]),
            "env_infos": [None] * len(self._observations[: self._top]),
        }

        with open(save_name, "wb") as f:
            pickle.dump(save_dict, f)
