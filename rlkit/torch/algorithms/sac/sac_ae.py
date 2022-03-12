from collections import OrderedDict

import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.torch.utils.normalizer import preprocess_obs
from rlkit.core.trainer import Trainer
from rlkit.core.eval_util import create_stats_ordered_dict


class SoftActorCritic(Trainer):
    """
    version that:
        - uses reparameterization trick
        - has two Q functions
        - has auto-tuned alpha
    """

    def __init__(
        self,
        encoder,
        decoder,
        policy,
        qf1,
        qf2,
        reward_scale=1.0,
        discount=0.99,
        policy_lr=1e-3,
        qf_lr=1e-3,
        alpha_lr=1e-3,
        encdec_lr=1e-3,
        soft_target_tau=0.01,
        enc_soft_target_tau=0.05,
        alpha=0.1,
        train_alpha=True,
        policy_mean_reg_weight=1e-3,
        policy_std_reg_weight=1e-3,
        optimizer_class=optim.Adam,
        beta_1=0.9,
        decoder_latent_lambda=1e-6,
        decoder_weight_lambda=1e-7,  # decoder lr decay, not used by now
        ac_update_freq=2,
        encdec_update_freq=1,
        **kwargs
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.reward_scale = reward_scale
        self.discount = discount
        self.soft_target_tau = soft_target_tau
        self.enc_soft_target_tau = enc_soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.decoder_latent_lambda = decoder_latent_lambda
        self.decoder_weight_lambda = decoder_weight_lambda

        self.ac_update_freq = ac_update_freq
        self.encdec_update_freq = encdec_update_freq

        self.train_alpha = train_alpha
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=train_alpha)
        self.alpha = self.log_alpha.detach().exp()
        assert "env" in kwargs.keys(), "env info should be taken into SAC alpha"
        self.target_entropy = -np.prod(kwargs["env"].action_space.shape)

        self.target_qf1 = qf1.copy()
        self.target_qf2 = qf2.copy()
        self.target_encoder = encoder.copy()

        self.eval_statistics = None

        self.policy_optimizer = optimizer_class(
            list(
                set(self.policy.parameters()).difference(set(self.encoder.parameters()))
            ),
            lr=policy_lr,
            betas=(beta_1, 0.999),
        )
        self.qf_optimizer = optimizer_class(
            list(self.qf1.parameters())
            + list(self.qf2.parameters())
            + list(self.encoder.parameters()),
            lr=qf_lr,
            betas=(beta_1, 0.999),
        )
        self.alpha_optimizer = optimizer_class(
            [self.log_alpha], lr=alpha_lr, betas=(0.5, 0.999)
        )
        self.encdec_optimizer = optimizer_class(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=encdec_lr,
            betas=(beta_1, 0.999),
        )

        self.total_train_step = 0

    def train_encdec(self, batch):
        obs = batch["observations"]

        """
        Enc-Dec Loss
        """
        feature_obs = self.encoder(obs)
        target_obs = obs.detach()
        if target_obs.dim() == 4:
            target_obs = preprocess_obs(target_obs)
        rec_obs = self.decoder(feature_obs)
        rec_loss = F.mse_loss(target_obs, rec_obs)

        # add L2 penalty on latent representation
        # see https://arxiv.org/pdf/1903.12436.pdf
        latent_loss = (0.5 * feature_obs.pow(2).sum(1)).mean()

        encdec_loss = rec_loss + self.decoder_latent_lambda * latent_loss
        self.encdec_optimizer.zero_grad()
        encdec_loss.backward()
        self.encdec_optimizer.step()

        """
        Save some statistics for eval
        """
        if self.eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics = OrderedDict()

        self.eval_statistics["Reward Scale"] = self.reward_scale
        self.eval_statistics["Rec Loss"] = np.mean(ptu.get_numpy(rec_loss))
        self.eval_statistics["Lat Loss"] = np.mean(ptu.get_numpy(latent_loss))

    def train_ac(self, batch):
        rewards = self.reward_scale * batch["rewards"]
        terminals = batch["terminals"]
        obs = batch["observations"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]

        obs = self.encoder(obs)
        target_next_obs = self.target_encoder(next_obs)
        next_obs = self.encoder(next_obs)

        """
        QF Loss
        """
        self.qf_optimizer.zero_grad()
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)

        # Make sure policy accounts for squashing functions like tanh correctly!
        next_policy_outputs = self.policy(
            next_obs.detach(), use_feature=True, return_log_prob=True
        )
        # in this part, we only need new_actions and log_pi with no grad
        (
            next_new_actions,
            next_policy_mean,
            next_policy_log_std,
            next_log_pi,
        ) = next_policy_outputs[:4]
        target_qf1_values = self.target_qf1(
            target_next_obs, next_new_actions
        )  # do not need grad || it's the shared part of two calculation
        target_qf2_values = self.target_qf2(
            target_next_obs, next_new_actions
        )  # do not need grad || it's the shared part of two calculation
        min_target_value = torch.min(target_qf1_values, target_qf2_values)
        q_target = rewards + (1.0 - terminals) * self.discount * (
            min_target_value - self.alpha * next_log_pi
        )  ## original implementation has detach
        q_target = q_target.detach()

        qf1_loss = 0.5 * torch.mean((q1_pred - q_target) ** 2)
        qf2_loss = 0.5 * torch.mean((q2_pred - q_target) ** 2)
        qf_loss = qf1_loss + qf2_loss

        qf_loss.backward()

        self.qf_optimizer.step()

        """
        Policy Loss
        """
        policy_outputs = self.policy(
            obs.detach(), use_feature=True, return_log_prob=True
        )  # policy do not update the encoder
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
        q1_new_acts = self.qf1(obs.detach(), new_actions)
        q2_new_acts = self.qf2(obs.detach(), new_actions)
        q_new_actions = torch.min(q1_new_acts, q2_new_acts)

        self.policy_optimizer.zero_grad()
        policy_loss = torch.mean(self.alpha * log_pi - q_new_actions)  ##
        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        policy_reg_loss = mean_reg_loss + std_reg_loss
        policy_loss = policy_loss + policy_reg_loss
        policy_loss.backward()
        self.policy_optimizer.step()

        """
        Update alpha
        """
        if self.train_alpha:
            log_prob = log_pi.detach() + self.target_entropy
            alpha_loss = -(self.log_alpha * log_prob).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.detach().exp()

        """
        Update networks
        """

        self._update_target_network()

        """
        Save some statistics for eval
        """
        if self.eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics = OrderedDict()
        self.eval_statistics["QF1 Loss"] = np.mean(ptu.get_numpy(qf1_loss))
        self.eval_statistics["QF2 Loss"] = np.mean(ptu.get_numpy(qf2_loss))
        if self.train_alpha:
            self.eval_statistics["Alpha Loss"] = np.mean(ptu.get_numpy(alpha_loss))
        self.eval_statistics["Policy Loss"] = np.mean(ptu.get_numpy(policy_loss))
        self.eval_statistics.update(
            create_stats_ordered_dict(
                "Q1 Predictions",
                ptu.get_numpy(q1_pred),
            )
        )
        self.eval_statistics.update(
            create_stats_ordered_dict(
                "Q2 Predictions",
                ptu.get_numpy(q2_pred),
            )
        )
        self.eval_statistics.update(
            create_stats_ordered_dict(
                "Alpha",
                [ptu.get_numpy(self.alpha)],
            )
        )
        self.eval_statistics.update(
            create_stats_ordered_dict(
                "Log Pis",
                ptu.get_numpy(log_pi),
            )
        )
        self.eval_statistics.update(
            create_stats_ordered_dict(
                "Policy mu",
                ptu.get_numpy(policy_mean),
            )
        )
        self.eval_statistics.update(
            create_stats_ordered_dict(
                "Policy log std",
                ptu.get_numpy(policy_log_std),
            )
        )

    def train_step(self, batch):

        if self.total_train_step % self.ac_update_freq == 0:
            self.train_ac(batch)
        if self.total_train_step % self.encdec_update_freq == 0:
            self.train_encdec(batch)

        self.total_train_step += 1

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
            self.encoder,
            self.target_encoder,
            self.decoder,
        ]

    def _update_target_network(self):
        ptu.soft_update_from_to(self.qf1, self.target_qf1, self.soft_target_tau)
        ptu.soft_update_from_to(self.qf2, self.target_qf2, self.soft_target_tau)
        ptu.soft_update_from_to(self.encoder, self.target_encoder, self.soft_target_tau)

    def get_snapshot(self):
        return dict(
            qf1=self.qf1,
            qf2=self.qf2,
            policy=self.policy,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
            log_alpha=self.log_alpha,
            policy_optimizer=self.policy_optimizer,
            qf_optimizer=self.qf_optimizer,
            alpha_optimizer=self.alpha_optimizer,
            encdec_optimizer=self.encdec_optimizer,
        )

    def load_snapshot(self, snapshot):
        self.qf1 = snapshot["qf1"]
        self.qf2 = snapshot["qf2"]
        self.policy = snapshot["policy"]
        self.target_qf1 = snapshot["target_qf1"]
        self.target_qf2 = snapshot["target_qf2"]
        self.log_alpha = snapshot["log_alpha"]
        self.policy_optimizer = snapshot["policy_optimizer"]
        self.qf_optimizer = snapshot["qf_optimizer"]
        self.alpha_optimizer = snapshot["alpha_optimizer"]
        self.encdec_optimizer = snapshot["encdec_optimizer"]

    def get_eval_statistics(self):
        return self.eval_statistics

    def end_epoch(self):
        self.eval_statistics = None

    def to(self, device):
        self.log_alpha.to(device)
        super.to(device)
