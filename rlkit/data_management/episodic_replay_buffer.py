import numpy as np

from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.data_management.t_step_replay_buffer import TStepReplayBuffer
from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer


class EpisodicReplayBuffer(ReplayBuffer):
    """
    A class used to save and replay data.
    """

    def __init__(
        self,
        max_sub_buf_size,
        observation_dim,
        action_dim,
        random_seed=1995,
        gamma=0.99,
        max_step=5,
    ):
        self._random_seed = random_seed
        self._np_rand_state = np.random.RandomState(random_seed)
        self.gamma = gamma
        self.max_step = max_step

        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_sub_buf_size = max_sub_buf_size

        self.t_step_buffers = []
        for step in range(1, max_step + 1):
            if step == 1:
                self.t_step_buffers.append(
                    SimpleReplayBuffer(
                        self._max_sub_buf_size,
                        self._observation_dim,
                        self._action_dim,
                        self._random_seed,
                    )
                )
            else:
                self.t_step_buffers.append(
                    TStepReplayBuffer(
                        self._max_sub_buf_size,
                        self._observation_dim,
                        self._action_dim,
                        self._random_seed,
                        step,
                    )
                )

        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._data = []
        self._top = 0
        self._size = 0

        self._initial_pairs = []

        self.simple_buffer = None
        self.last_sufficient_length = 0

        self.log_epsilon = None

    def get_size(self):
        return self._size

    def set_data(self, data, **kwargs):
        """
        Set the buffer with a list-like data.
        """
        self._data = data

        self._top = len(self._data) - 1
        self._size = len(self._data)

        self._initial_pairs = {
            "observations": [],
            "actions": [],
            "next_observations": [],
            "rewards": [],
        }
        self._last_pairs = {
            "observations": [],
            "actions": [],
            "next_observations": [],
            "rewards": [],
            "gamma_pow": [],
        }
        for traj in self._data:
            for i in range(self.max_step):
                self.t_step_buffers[i].add_path(traj)
            for key in self._initial_pairs.keys():
                self._initial_pairs[key].append(traj[key][0])
                self._last_pairs[key].append(traj[key][-1])
                self._last_pairs["gamma_pow"].append(
                    self.gamma ** len(traj["observations"])
                )

    def get_initial_pairs(self):
        """
        Return a list of initial pairs of each traj.
        """

        assert self._size > 0

        return self._initial_pairs

    def get_last_pairs(self):
        """
        Return a list of initial pairs of each traj.
        """

        assert self._size > 0

        return self._last_pairs

    def add_sample(
        self, observation, action, reward, next_observation, terminal, **kwargs
    ):
        """
        Add a transition tuple.
        """
        pass

    def terminate_episode(self):
        """
        Let the replay buffer know that the episode has terminated in case some
        special book-keeping has to happen.
        :return:
        """
        pass

    def num_steps_can_sample(self, **kwargs):
        """
        :return: # of unique items that can be sampled.
        """
        res = []
        for i in range(self.max_step):
            res.append(self.t_step_buffers[i].num_steps_can_sample())

        return np.mean(res)

    def random_batch(self, batch_size, keys=None, step=1):
        """
        Return a batch of size `batch_size`.
        :param batch_size:
        :return:
        """

        return self.t_step_buffers[step - 1].random_batch(batch_size, keys=keys)

    def find_batch(self, target_batch, keys=None, step=1):
        """
        Return a batch of size `batch_size`. Find batch which ends with (s,a) in target_batch
        :param batch_size:
        :return:
        """

        return self.t_step_buffers[step - 1].find_batch(target_batch, keys=keys)

    def add_path(self, path):
        for i in range(self.max_step):
            self.t_step_buffers[i].add_path(path)
