from collections import OrderedDict
from typing import Callable, List

import gtimer as gt
import numpy as np
import torch
from torch import nn as nn

from rlkit.core import logger
from rlkit.core.trainer import Trainer
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer
from rlkit.torch.algorithms.mbpo.bnn_trainer import BNNTrainer
from rlkit.torch.algorithms.mbpo.fake_env import FakeEnv
from rlkit.torch.algorithms.sac.sac_alpha import SoftActorCritic
from rlkit.torch.algorithms.torch_rl_algorithm import TorchRLAlgorithm

import rlkit.torch.utils.pytorch_util as ptu


class MBPO(TorchRLAlgorithm):
    
    def __init__(
        self,
        env,
        model: BNNTrainer,
        algo: SoftActorCritic, # model-free algorithm to train policy
        is_terminal: Callable, # for fake_env
        model_replay_buffer: SimpleReplayBuffer = None,
        model_replay_buffer_size: int = 10000,

        # SAC params
        # reward_scale: float = 1.0,
        # discount: float = 0.99,
        # policy_lr: float = 3e-4,
        # qf_lr: float = 3e-4,
        # tau: float = 5e-3,
        # optimizer_class: Type[optim.Optimizer] = optim.Adam,
        # beta_1: float = 0.9,
        # target_entropy: float = None,
        target_update_interval: int = 1,
        action_prior: str = 'uniform',

        # model utilization params
        deterministic: bool = False,
        model_train_freq: int = 250,
        model_retrain_epochs: int = 20,
        rollout_batch_size: int = int(1e5),
        real_ratio: float = 0.1,
        rollout_schedule: List = None,
        max_model_t: float = None,
        **kwargs
    ):
        super().__init__(env=env, trainer=algo, **kwargs)
        self.model = model
        self.algo = algo
        self.is_terminal = is_terminal
        self.fake_env = FakeEnv(self.model, self.is_terminal, self.model.get_random_model_index)
        self.model_replay_buffer_size = model_replay_buffer_size
        if model_replay_buffer is None:
            assert self.max_path_length < model_replay_buffer_size
            model_replay_buffer = EnvReplayBuffer(model_replay_buffer_size, self.env, random_seed=np.random.randint(10000))
        else:
            assert self.max_path_length < model_replay_buffer._max_replay_buffer_size
        self.model_replay_buffer = model_replay_buffer

        # self.reward_scale = reward_scale
        # self.discount = discount
        # self.policy_lr = policy_lr
        # self.qf_lr = qf_lr
        # self.tau = tau
        self.target_update_interval = target_update_interval
        self.action_prior = action_prior
        # self.target_entropy: float = target_entropy if target_entropy else -np.prod(kwargs['env'].action_space.shape) / 2.0
        logger.log(f'MBPO Target entorpy: {self.algo.target_entropy}')

        # self.policy_optimizer = optimizer_class(
        #     self.policy.parameters(), lr=policy_lr, betas=(beta_1, 0.999)
        # )
        # self.qf1_optimizer = optimizer_class(
        #     self.qf1.parameters(), lr=qf_lr, betas=(beta_1, 0.999)
        # )
        # self.qf2_optimizer = optimizer_class(
        #     self.qf2.parameters(), lr=qf_lr, betas=(beta_1, 0.999)
        # )

        self.deterministic = deterministic
        self.model_train_freq = model_train_freq
        self.model_retrain_epochs = model_retrain_epochs
        self.rollout_batch_size = rollout_batch_size
        self.real_ratio = real_ratio
        self.rollout_schedule = rollout_schedule
        self.max_model_t = max_model_t

        self.eval_statistics = None

    def start_training(self, start_epoch=0):
        gt

    def _do_training(self, batch: torch.Tensor) -> None:
        pass