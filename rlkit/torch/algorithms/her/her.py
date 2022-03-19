import numpy as np

from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.algorithms.torch_rl_algorithm import TorchRLAlgorithm
from rlkit.data_management.relabel_replay_buffer import HindsightReplayBuffer


class HER(TorchRLAlgorithm):
    """
    Hindsight Experience Replay. Default using TD3 for RL policy trainer.
    """

    def __init__(
        self, replay_buffer=None, her_ratio=0.8, relabel_type="future", **kwargs
    ):
        if replay_buffer is None:
            assert kwargs["max_path_length"] < kwargs["replay_buffer_size"]
            replay_buffer = HindsightReplayBuffer(
                kwargs["replay_buffer_size"],
                kwargs["env"],
                random_seed=np.random.randint(10000),
                relabel_type=relabel_type,
                her_ratio=her_ratio,
            )
        super().__init__(replay_buffer=replay_buffer, **kwargs)

    def get_batch(self, keys=None):
        buffer = self.replay_buffer
        batch = buffer.random_batch(self.batch_size, keys=keys)
        batch = np_to_pytorch_batch(batch)
        return batch

    def _get_action_and_info(self, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        self.exploration_policy.set_num_steps_total(self._n_env_steps_total)
        return self.exploration_policy.get_actions(
            observation,
        )
