import numpy as np
from numpy.random import choice
import math
import random

import torch
from torch import nn as nn

from rlkit.policies.base import ExplorationPolicy, Policy
from rlkit.torch.common.networks import Mlp, CatagorialMlp
from rlkit.torch.common.distributions import ReparamTanhMultivariateNormal
from rlkit.torch.common.distributions import ReparamMultivariateNormalDiag
from rlkit.torch.core import PyTorchModule

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class MakeDeterministic(Policy):
    def __init__(self, stochastic_policy):
        self.stochastic_policy = stochastic_policy

    def get_action(self, observation):
        return self.stochastic_policy.get_action(observation, deterministic=True)

    def get_actions(self, observations):
        return self.stochastic_policy.get_actions(observations, deterministic=True)

    def train(self, mode):
        pass

    def set_num_steps_total(self, num):
        pass

    def to(self, device):
        self.stochastic_policy.to(device)


class DiscretePolicy(Mlp, ExplorationPolicy):
    """
    Usage:

    ```
    policy = DiscretePolicy(...)
    action, log_prob = policy(obs, return_log_prob=True)
    ```
    """

    def __init__(self, hidden_sizes, obs_dim, action_dim, init_w=1e-3, **kwargs):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            output_activation=nn.LogSoftmax(1),
            **kwargs,
        )

    def get_action(self, obs_np, deterministic=False):
        action = self.get_actions(obs_np[None], deterministic=deterministic)
        return action[0], {}

    def get_actions(self, obs_np, deterministic=False):
        # return self.eval_np(obs_np)[0].reshape(-1)
        return self.eval_np(obs_np, deterministic=deterministic)[0]

    def forward(
        self,
        obs,
        deterministic=False,
        return_log_prob=False,
    ):
        log_probs, pre_act = super().forward(obs, return_preactivations=True)

        if deterministic:
            log_prob, idx = torch.max(log_probs, 1)
            return (idx, None)
        else:
            # Using Gumbel-Max trick to sample from the multinomials
            u = torch.rand(pre_act.size(), requires_grad=False)
            gumbel = -torch.log(-torch.log(u))
            _, idx = torch.max(gumbel + pre_act, 1)

            log_prob = torch.gather(log_probs, 1, idx.unsqueeze(1))

            # # print(log_probs.size(-1))
            # # print(log_probs.data.numpy())
            # # print(np.exp(log_probs.data.numpy()))
            # idx = choice(
            #     log_probs.size(-1),
            #     size=1,
            #     p=np.exp(log_probs.data.numpy())
            # )
            # log_prob = log_probs[0,idx]

            # print(idx)
            # print(log_prob)

            return (idx, log_prob)

    def get_log_pis(self, obs):
        return super().forward(obs)


class MlpPolicy(Mlp, ExplorationPolicy):
    def __init__(self, hidden_sizes, obs_dim, action_dim, init_w=1e-3, **kwargs):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs,
        )

    def get_action(self, obs_np, deterministic=False):
        """
        deterministic=False makes no diff, just doing this for
        consistency in interface for now
        """
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        actions = actions[None]
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return self.eval_np(obs_np)[0]


class MlpGaussianNoisePolicy(Mlp, ExplorationPolicy):
    def __init__(
        self,
        hidden_sizes,
        obs_dim,
        action_dim,
        init_w=1e-3,
        policy_noise=0.1,
        policy_noise_clip=0.5,
        max_act=1.0,
        **kwargs,
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs,
        )
        self.noise = policy_noise
        self.noise_clip = policy_noise_clip
        self.max_act = max_act

    def get_action(self, obs_np, deterministic=False):
        """
        deterministic=False makes no diff, just doing this for
        consistency in interface for now
        """
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        actions = actions[None]
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return self.eval_np(obs_np, deterministic=deterministic)[0]

    def forward(self, obs, deterministic=False):
        h = obs
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm:
                h = self.layer_norms[i](h)
            if self.batch_norm:
                h = self.batch_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        if self.batch_norm_before_output_activation:
            preactivation = self.batch_norms[-1](preactivation)
        action = self.max_act * self.output_activation(preactivation)
        if deterministic:
            pass
        else:
            noise = self.noise * torch.normal(
                torch.zeros_like(action),
            )
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            action += noise

        return (action, preactivation)


class ReparamTanhMultivariateGaussianPolicy(Mlp, ExplorationPolicy):
    """
    Usage:

    ```
    policy = ReparamTanhMultivariateGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """

    def __init__(
        self,
        hidden_sizes,
        obs_dim,
        action_dim,
        init_w=1e-3,
        max_act=1.0,
        conditioned_std: bool = True,
        **kwargs,
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs,
        )
        self.max_act = max_act
        self.conditioned_std = conditioned_std

        if self.conditioned_std:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.action_log_std = nn.Parameter(torch.zeros(1, action_dim))

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return self.eval_np(obs_np, deterministic=deterministic)[0]

    def forward(
        self, obs, deterministic=False, return_log_prob=False, return_tanh_normal=False
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """

        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))

        mean = self.last_fc(h)

        if self.conditioned_std:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        else:
            log_std = self.action_log_std.expand_as(mean)
        std = torch.exp(log_std)

        log_prob = None
        expected_log_prob = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = ReparamTanhMultivariateNormal(mean, log_std)

            if return_log_prob:
                action, pre_tanh_value = tanh_normal.sample(return_pretanh_value=True)
                log_prob = tanh_normal.log_prob(action, pre_tanh_value=pre_tanh_value)
            else:
                action = tanh_normal.sample()

        # doing it like this for now for backwards compatibility
        if return_tanh_normal:
            return (
                action,
                mean,
                log_std,
                log_prob,
                expected_log_prob,
                std,
                mean_action_log_prob,
                pre_tanh_value,
                tanh_normal,
            )
        return (
            action,
            mean,
            log_std,
            log_prob,
            expected_log_prob,
            std,
            mean_action_log_prob,
            pre_tanh_value,
        )

    def get_log_prob_entropy(self, obs, acts):
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.conditioned_std:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        else:
            log_std = self.action_log_std.expand_as(mean)

        tanh_normal = ReparamTanhMultivariateNormal(mean, log_std)
        log_prob = tanh_normal.log_prob(acts)

        entropy = (0.5 + 0.5 * math.log(2 * math.pi) + log_std).sum(
            dim=-1, keepdim=True
        )

        return log_prob, entropy

    def get_log_prob(self, obs, acts, return_normal_params=False):
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.conditioned_std:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        else:
            log_std = self.action_log_std.expand_as(mean)

        tanh_normal = ReparamTanhMultivariateNormal(mean, log_std)
        log_prob = tanh_normal.log_prob(acts)

        if return_normal_params:
            return log_prob, mean, log_std
        return log_prob


class ReparamMultivariateGaussianPolicy(Mlp, ExplorationPolicy):
    def __init__(
        self,
        hidden_sizes,
        obs_dim,
        action_dim,
        conditioned_std=True,
        init_w=1e-3,
        **kwargs,
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs,
        )
        self.conditioned_std = conditioned_std

        if self.conditioned_std:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.action_log_std = nn.Parameter(torch.zeros(1, action_dim))

        self.last_fc.weight.data.mul_(0.1)
        self.last_fc.bias.data.mul_(0.0)

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return self.eval_np(obs_np, deterministic=deterministic)[0]

    def forward(
        self, obs, deterministic=False, return_log_prob=False, return_normal=False
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)

        if self.conditioned_std:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        else:
            log_std = self.action_log_std.expand_as(mean)
        std = torch.exp(log_std)

        log_prob = None
        expected_log_prob = None
        mean_action_log_prob = None
        if deterministic:
            action = mean
        else:
            normal = ReparamMultivariateNormalDiag(mean, log_std)
            action = normal.sample()
            if return_log_prob:
                log_prob = normal.log_prob(action)

        # I'm doing it like this for now for backwards compatibility, sorry!
        if return_normal:
            return (
                action,
                mean,
                log_std,
                log_prob,
                expected_log_prob,
                std,
                mean_action_log_prob,
                normal,
            )
        return (
            action,
            mean,
            log_std,
            log_prob,
            expected_log_prob,
            std,
            mean_action_log_prob,
        )

    def get_log_prob_entropy(self, obs, acts):
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)

        if self.conditioned_std:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        else:
            log_std = self.action_log_std.expand_as(mean)

        normal = ReparamMultivariateNormalDiag(mean, log_std)
        log_prob = normal.log_prob(acts)

        entropy = (0.5 + 0.5 * math.log(2 * math.pi) + log_std).sum(
            dim=-1, keepdim=True
        )
        return log_prob, entropy

    def get_log_prob(self, obs, acts, return_normal_params=False):
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)

        if self.conditioned_std:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        else:
            log_std = self.action_log_std.expand_as(mean)

        normal = ReparamMultivariateNormalDiag(mean, log_std)
        log_prob = normal.log_prob(acts)

        if return_normal_params:
            return log_prob, mean, log_std
        return log_prob


class MlpGaussianAndEpsilonPolicy(Mlp, ExplorationPolicy):
    def __init__(
        self,
        hidden_sizes,
        obs_dim,
        action_dim,
        action_space,
        init_w=1e-3,
        epsilon=0.3,
        max_sigma=0.2,
        min_sigma=0.2,
        decay_period=1000000,
        max_act=1.0,
        min_act=-1.0,
        **kwargs,
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs,
        )
        if min_sigma is None:
            min_sigma = max_sigma
        self.sigma = max_sigma
        self._max_sigma = max_sigma
        self._epsilon = epsilon
        self._min_sigma = min_sigma
        self._decay_period = decay_period
        self._action_space = action_space

        self.max_act = max_act
        self.min_act = min_act
        self.action_dim = action_dim
        self.t = 0

    def set_num_steps_total(self, t):
        self.t = t

    def get_action(self, obs_np, deterministic=False):
        """
        deterministic=False makes no diff, just doing this for
        consistency in interface for now
        """
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return self.eval_np(obs_np, deterministic=deterministic)[0]

    def forward(self, obs, deterministic=False):
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))

        preactivation = self.last_fc(h)
        action = self.max_act * self.output_activation(preactivation)

        batch_size = 0
        if len(obs.shape) >= 2:
            batch_size = np.shape(obs)[0]
        if deterministic:
            pass
        else:
            if random.random() < self._epsilon:
                action = self._action_space.sample()
                if batch_size > 0:
                    action = [self._action_space.sample() for _ in range(batch_size)]
            else:
                self.sigma = self._max_sigma - (
                    self._max_sigma - self._min_sigma
                ) * min(1.0, self.t * 1.0 / self._decay_period)
                action = np.clip(
                    action.detach().cpu().numpy()
                    + np.random.normal(size=np.shape(action)) * self.sigma,
                    self.min_act,
                    self.max_act,
                )

        assert (
            np.shape(action)[-1] == self._action_space.shape[-1]
        ), "action shape mismatch!"

        return (action, preactivation)


class ConditionPolicy(ExplorationPolicy):
    def __init__(
        self,
        obs_dim,
        condition_dim,
        action_dim,
        observation_key="observation",
        desired_goal_key="desired_goal",
        achieved_goal_key="achieved_goal",
    ):
        self.condition_dim = condition_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.observation_key = observation_key
        self.desired_goal_key = desired_goal_key
        self.achieved_goal_key = achieved_goal_key

    def get_action(self, obs_condition_np, deterministic=False):
        """
        deterministic=False makes no diff, just doing this for
        consistency in interface for now
        """

        if isinstance(obs_condition_np, dict):
            obs_condition_np = np.concatenate(
                [
                    obs_condition_np[self.observation_key],
                    obs_condition_np[self.desired_goal_key],
                ],
                axis=-1,
            )
        elif isinstance(obs_condition_np[0], dict):
            obs_condition_np = [
                {k: v for k, v in x.items() if k != self.achieved_goal_key}
                for x in obs_condition_np
            ]
            obs_condition_np = np.array(
                [
                    np.concatenate(
                        [x[self.observation_key], x[self.desired_goal_key]], axis=-1
                    )
                    for x in obs_condition_np
                ]
            )

        actions = self.get_actions(obs_condition_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_condition_np, deterministic=False):

        if isinstance(obs_condition_np, dict):
            obs_condition_np = np.concatenate(
                [
                    obs_condition_np[self.observation_key],
                    obs_condition_np[self.desired_goal_key],
                ],
                axis=-1,
            )
        elif isinstance(obs_condition_np[0], dict):
            obs_condition_np = [
                {k: v for k, v in x.items() if k != self.achieved_goal_key}
                for x in obs_condition_np
            ]
            obs_condition_np = np.array(
                [
                    np.concatenate(
                        [x[self.observation_key], x[self.desired_goal_key]], axis=-1
                    )
                    for x in obs_condition_np
                ]
            )

        return self.eval_np(obs_condition_np, deterministic=deterministic)[0]


class MlpGaussianAndEpsilonConditionPolicy(
    ConditionPolicy, MlpGaussianAndEpsilonPolicy
):
    """
    Custom for Ant Rand Goal
    The only difference is that it linearly embeds the goal into a higher dimension
    """

    def __init__(
        self,
        hidden_sizes,
        obs_dim,
        condition_dim,
        action_dim,
        action_space,
        observation_key="observation",
        desired_goal_key="desired_goal",
        achieved_goal_key="achieved_goal",
        **kwargs,
    ):
        self.save_init_params(locals())
        MlpGaussianAndEpsilonPolicy.__init__(
            self,
            hidden_sizes,
            obs_dim + condition_dim,
            action_dim,
            action_space,
            **kwargs,
        )
        ConditionPolicy.__init__(
            self,
            obs_dim,
            condition_dim,
            action_dim,
            observation_key,
            desired_goal_key,
            achieved_goal_key,
        )

        self.t = 0


class ReparamTanhMultivariateGaussianConditionPolicy(
    ConditionPolicy, ReparamTanhMultivariateGaussianPolicy
):
    """
    Usage:

    ```
    policy = ReparamTanhMultivariateGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """

    def __init__(
        self,
        hidden_sizes,
        obs_dim,
        condition_dim,
        action_dim,
        observation_key="observation",
        desired_goal_key="desired_goal",
        achieved_goal_key="achieved_goal",
        **kwargs,
    ):
        self.save_init_params(locals())
        ReparamTanhMultivariateGaussianPolicy.__init__(
            self, hidden_sizes, obs_dim + condition_dim, action_dim, **kwargs
        )
        ConditionPolicy.__init__(
            self,
            obs_dim,
            condition_dim,
            action_dim,
            observation_key,
            desired_goal_key,
            achieved_goal_key,
        )


class ReparamTanhMultivariateGaussianEncoderPolicy(
    ReparamTanhMultivariateGaussianPolicy
):
    """
    Policy with encoder
    Usage:
    ```
    policy = ReparamTanhMultivariateGaussianEncoderPolicy(...)
    """

    def __init__(self, encoder, **kwargs):
        self.save_init_params(locals())
        super().__init__(**kwargs)
        self.encoder = encoder

    def forward(self, obs, use_feature=False, **kwargs):
        """
        :param obs: Observation
        """
        feature_obs = obs
        if not use_feature:
            feature_obs = self.encoder(obs)
        return super().forward(feature_obs, **kwargs)


class CatagorialPolicy(CatagorialMlp, ExplorationPolicy):
    def __init__(
        self,
        hidden_sizes,
        obs_dim,
        action_dim,
        init_w=1e-3,
        **kwargs,
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes=hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs,
        )
        self.action_dim = action_dim

    def get_action(self, obs_np, deterministic=False):
        """
        deterministic=False makes no diff, just doing this for
        consistency in interface for now
        """
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return self.eval_np(obs_np, deterministic=deterministic)[0]

    def forward(self, obs, deterministic=False, return_log_prob=False):
        action_prob, action_logits = super().forward(obs, return_preactivations=True)

        if deterministic:
            action = torch.argmax(action_prob, axis=-1).unsqueeze(-1)
        else:
            action = torch.multinomial(action_prob, 1)
        assert np.shape(action)[-1] == 1, "action shape mismatch! {}".format(
            np.shape(action)
        )
        log_prob = None
        if return_log_prob:
            log_prob = torch.log(torch.index_select(action_prob, -1, action))

        return action, action_logits, log_prob

    def get_log_prob(self, obs, acts):
        _, action_prob = self.forward(obs)

        log_prob = torch.log(torch.index_select(action_prob, -1, acts))

        return log_prob


class CatagorialConditionPolicy(ConditionPolicy, CatagorialPolicy):
    """
    Usage:

    ```
    policy = ReparamTanhMultivariateGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """

    def __init__(
        self,
        hidden_sizes,
        obs_dim,
        condition_dim,
        action_dim,
        observation_key="observation",
        desired_goal_key="desired_goal",
        achieved_goal_key="achieved_goal",
        **kwargs,
    ):
        self.save_init_params(locals())
        CatagorialPolicy.__init__(
            self, hidden_sizes, obs_dim + condition_dim, action_dim, **kwargs
        )
        ConditionPolicy.__init__(
            self,
            obs_dim,
            condition_dim,
            action_dim,
            observation_key,
            desired_goal_key,
            achieved_goal_key,
        )
