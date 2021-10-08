import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from rlkit.policies.base import ExplorationPolicy
from rlkit.torch.common.networks import Mlp
from rlkit.torch.common.distributions import ReparamMultivariateNormalDiag, ReparamTanhMultivariateNormal
from rlkit.torch.core import PyTorchModule

import rlkit.torch.utils.pytorch_util as ptu

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class OptionPolicy(PyTorchModule, ExplorationPolicy):
    """
    Option base policy.
    """

    def __init__(
        self,
        hidden_sizes,
        obs_dim,
        action_dim,
        option_dim,
        conditioned_std=False,
        share_option_nets=True,
        init_w=1e-3,
        *args,
        **kwargs,
    ):
        self.save_init_params(locals())
        super().__init__()

        assert len(hidden_sizes) == 2
        assert type(hidden_sizes[0]) == list
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.option_dim = option_dim
        self.share_option_nets = share_option_nets

        if self.share_option_nets:
            self.option_policy = Mlp(
                hidden_sizes[0],
                (option_dim + 1) * option_dim,
                obs_dim,
                init_w=init_w,
                **kwargs,
            )
            self.policy = Mlp(
                hidden_sizes[1], option_dim * action_dim, obs_dim, init_w=init_w, **kwargs
            )
            self.action_log_std = nn.Parameter(torch.zeros(1, action_dim))
        else:
            self.option_policy = nn.ModuleList(
                [
                    Mlp(
                        hidden_sizes[0],
                        option_dim,
                        obs_dim,
                        init_w=init_w,
                        **kwargs,
                    )
                    for _ in range(option_dim + 1)
                ]
            )
            self.policy = nn.ModuleList(
                [
                    Mlp(
                        hidden_sizes[1],
                        action_dim,
                        obs_dim,
                        init_w=init_w,
                        **kwargs,
                    )
                    for _ in range(option_dim)
                ]
            )
            self.action_log_std = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.zeros(1, action_dim)
                    )
                    for _ in range(option_dim)
                ]
            )

    def forward(
        self,
        obs,
        option=None,
        deterministic=False,
        return_log_prob=False,
        return_normal=False,
    ):
        # option: None or long(N x 1)
        # option: None for all option, return (N x option_dim x action_dim); else return (N x action_dim)
        # s: N x dim_s, option: N x 1, option should always < option_dim
        if self.share_option_nets:
            mean = self.policy(obs).view(-1, self.option_dim, self.action_dim)
            a_log_std = self.action_log_std.expand_as(mean)
        else:
            mean = torch.stack([m(obs) for m in self.policy], dim=-2)
            a_log_std = torch.stack(
                [m.expand_as(mean[:, 0, :]) for m in self.action_log_std], dim=-2
            )

        a_log_prob = None
        if option is not None:
            ind = option.view(-1, 1, 1).expand(-1, 1, self.action_dim)
            mean = mean.gather(dim=-2, index=ind.type(torch.long)).squeeze(dim=-2)
            a_log_std = a_log_std.gather(dim=-2, index=ind.type(torch.long)).squeeze(
                dim=-2
            )
        a_std = torch.exp(a_log_std)

        if deterministic:
            action = mean
        else:
            normal = ReparamMultivariateNormalDiag(mean, a_log_std)
            action = normal.sample()
            if return_log_prob:
                a_log_prob = normal.log_prob(action)

        if return_normal:
            return (
                action,
                mean,
                a_log_std,
                a_log_prob,
                a_std,
                normal,
            )
        return (
            action,
            mean,
            a_log_std,
            a_log_prob,
            a_std,
        )

    def log_trans(self, obs, prev_option=None):
        # prev_option: long(N x 1) or None
        # prev_option: None: direct output p(option|obs, prev_option): a (N x prev_option x option) array where option is log-normalized
        # print(obs.shape, prev_option.shape)
        unnormed_pcs = self.switcher(obs)
        log_pcs = unnormed_pcs.log_softmax(dim=-1)
        if prev_option is None:
            return log_pcs
        else:
            return log_pcs.gather(
                dim=-2,
                index=prev_option.view(-1, 1, 1)
                .expand(-1, 1, self.option_dim)
                .type(torch.long),
            ).squeeze(dim=-2)

    def get_log_prob_action(self, obs, acts, option, return_normal_params=False):
        # if option is None, return (N x option_dim x 1), else return (N x 1)
        if self.share_option_nets:
            mean = self.policy(obs).view(-1, self.option_dim, self.action_dim)
            a_log_std = self.action_log_std.expand_as(mean)
        else:
            mean = torch.stack([m(obs) for m in self.policy], dim=-2)
            a_log_std = torch.stack(
                [m.expand_as(mean[:, 0, :]) for m in self.action_log_std], dim=-2
            )

        if option is not None:
            ind = option.view(-1, 1, 1).expand(-1, 1, self.action_dim)
            mean = mean.gather(dim=-2, index=ind.type(torch.long)).squeeze(dim=-2)
            a_log_std = a_log_std.gather(dim=-2, index=ind.type(torch.long)).squeeze(
                dim=-2
            )

        if option is None:
            acts = acts.view(-1, 1, self.action_dim)

        normal = ReparamMultivariateNormalDiag(mean, a_log_std)
        a_log_prob = normal.log_prob(acts)

        if return_normal_params:
            return a_log_prob, mean, a_log_std
        return a_log_prob

    def get_log_prob_option(self, obs, prev_option, option):
        log_tr = self.log_trans(obs, prev_option)
        return log_tr.gather(dim=-1, index=option.type(torch.long))

    def switcher(self, obs):
        if self.share_option_nets:
            return self.option_policy(obs).view(-1, self.option_dim + 1, self.option_dim)
        else:
            return torch.stack([m(obs) for m in self.option_policy], dim=-2)

    def get_action(self, observation, prev_options=None, deterministic=False):
        # if prev_options is None:
        #     # initial option
        #     prev_options = np.ones((1, 1), dtype=np.float32)*self.exploration_policy.option_dim
        # else:
        #     if type(prev_options) == torch.tensor:
        #         prev_options = prev_options.numpy()
        options = self.get_options(
            observation,
            prev_options,
            deterministic=deterministic,
        )
        actions = self.get_actions(observation, options, deterministic=deterministic)

        return actions[0, :], options[0, :], {}

    def get_actions(self, obs_np, option_np, deterministic=False):
        return self.eval_np(obs_np, option_np, deterministic=deterministic)[0]

    def get_options(self, obs, prev_option, deterministic=False):
        obs = torch.tensor(obs, dtype=torch.float32, device=ptu.device)
        prev_option = torch.tensor(prev_option, dtype=torch.long, device=ptu.device)
        log_tr = self.log_trans(obs, prev_option)
        if deterministic:
            return log_tr.argmax(dim=-1, keepdim=True).cpu().numpy().astype(np.int64)
        else:
            return (
                F.gumbel_softmax(log_tr, hard=False)
                .multinomial(1)
                .long()
                .cpu()
                .numpy()
                .astype(np.int64)
            )

    def policy_entropy(self, obs, option):
        log_std = self.forward(obs, option)[2]
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + log_std
        return entropy.sum(dim=-1, keepdim=True)

    def option_entropy(self, obs, prev_option):
        log_tr = self.log_trans(obs, prev_option)
        entropy = -(log_tr * log_tr.exp()).sum(dim=-1, keepdim=True)
        return entropy

    def policy_log_prob_entropy(self, obs, option, acts):
        _, mean, logstd, _, _ = self.forward(obs, option)
        log_prob = (
            -(acts - mean).pow(2) / (2 * (logstd * 2).exp())
            - logstd
            - 0.5 * math.log(2 * math.pi)
        ).sum(dim=-1, keepdim=True)
        entropy = (0.5 + 0.5 * math.log(2 * math.pi) + logstd).sum(dim=-1, keepdim=True)
        return log_prob, entropy

    def option_log_prob_entropy(self, obs, prev_option, option):
        # c1 can be option_dim, c2 should always < option_dim
        log_tr = self.log_trans(obs, prev_option)
        log_opt = log_tr.gather(dim=-1, index=option.type(torch.long))
        entropy = -(log_tr * log_tr.exp()).sum(dim=-1, keepdim=True)
        return log_opt, entropy

    def log_alpha_beta(self, obs_array, act_array):
        obs = torch.tensor(obs_array, dtype=torch.float32, device=ptu.device)
        act = torch.tensor(act_array, dtype=torch.float32, device=ptu.device)
        log_pis = self.get_log_prob_action(obs, act, None).view(
            -1, self.option_dim
        )  # demo_len x option
        log_trs = self.log_trans(obs, None)  # demo_len x (prev_option + 1) x option
        log_tr0 = log_trs[0, -1]
        log_trs = log_trs[1:, :-1]  # (demo_len-1) x prev_option x option

        log_alpha = [log_tr0 + log_pis[0]]
        for log_tr, log_pi in zip(log_trs, log_pis[1:]):
            log_alpha_t = (log_alpha[-1].unsqueeze(dim=-1) + log_tr).logsumexp(
                dim=0
            ) + log_pi
            log_alpha.append(log_alpha_t)

        log_beta = [
            torch.zeros(self.option_dim, dtype=torch.float32, device=ptu.device)
        ]
        for log_tr, log_pi in zip(reversed(log_trs), reversed(log_pis[1:])):
            log_beta_t = ((log_beta[-1] + log_pi).unsqueeze(dim=0) + log_tr).logsumexp(
                dim=-1
            )
            log_beta.append(log_beta_t)
        log_beta.reverse()

        log_alpha = torch.stack(log_alpha)
        log_beta = torch.stack(log_beta)
        entropy = -(log_trs * log_trs.exp()).sum(dim=-1)
        return log_alpha, log_beta, log_trs, log_pis, entropy

    def viterbi_path(self, obs_array, act_array):
        obs = torch.tensor(obs_array, dtype=torch.float32, device=ptu.device)
        act = torch.tensor(act_array, dtype=torch.float32, device=ptu.device)
        with torch.no_grad():
            log_pis = self.get_log_prob_action(obs, act, None).view(
                -1, 1, self.option_dim
            )  # demo_len x 1 x option
            log_trs = self.log_trans(obs, None)  # demo_len x (prev_option+1) x option
            log_prob = log_trs[:, :-1] + log_pis
            log_prob0 = log_trs[0, -1] + log_pis[0, 0]
            # forward
            max_path = torch.empty(
                obs.shape[0], self.option_dim, dtype=torch.long, device=ptu.device
            )
            accumulate_logp = log_prob0
            max_path[0] = self.option_dim
            for i in range(1, obs.shape[0]):
                accumulate_logp, max_path[i, :] = (
                    accumulate_logp.unsqueeze(dim=-1) + log_prob[i]
                ).max(dim=-2)
            # backward
            c_array = torch.zeros(
                obs_array.shape[0] + 1, 1, dtype=torch.long, device=ptu.device
            )
            log_prob_traj, c_array[-1] = accumulate_logp.max(dim=-1)

            for i in range(obs_array.shape[0], 0, -1):
                c_array[i - 1] = max_path[i - 1][c_array[i]]
        return c_array.detach(), log_prob_traj.detach()


class OptionTanhPolicy(PyTorchModule, ExplorationPolicy):
    """
    Option base policy.
    """

    def __init__(
        self,
        hidden_sizes,
        obs_dim,
        action_dim,
        option_dim,
        share_option_nets=True,
        init_w=1e-3,
        *args,
        **kwargs,
    ):
        self.save_init_params(locals())
        super().__init__()

        assert len(hidden_sizes) == 2
        assert type(hidden_sizes[0]) == list
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.option_dim = option_dim
        self.share_option_nets = share_option_nets

        if self.share_option_nets:
            self.option_policy = Mlp(
                hidden_sizes[0],
                (option_dim + 1) * option_dim,
                obs_dim,
                init_w=init_w,
                **kwargs,
            )
            self.policy = Mlp(
                hidden_sizes[1], option_dim * action_dim, obs_dim, init_w=init_w, **kwargs
            )
            self.action_log_std = nn.Parameter(torch.zeros(1, action_dim))
        else:
            self.option_policy = nn.ModuleList(
                [
                    Mlp(
                        hidden_sizes[0],
                        option_dim,
                        obs_dim,
                        init_w=init_w,
                        **kwargs,
                    )
                    for _ in range(option_dim + 1)
                ]
            )
            self.policy = nn.ModuleList(
                [
                    Mlp(
                        hidden_sizes[1],
                        action_dim,
                        obs_dim,
                        init_w=init_w,
                        **kwargs,
                    )
                    for _ in range(option_dim)
                ]
            )
            self.action_log_std = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.zeros(1, action_dim)
                    )
                    for _ in range(option_dim)
                ]
            )

    def forward(
        self,
        obs,
        option=None,
        deterministic=False,
        return_log_prob=False,
        return_tanh_normal=False,
    ):
        # option: None or long(N x 1)
        # option: None for all option, return (N x option_dim x action_dim); else return (N x action_dim)
        # s: N x dim_s, option: N x 1, option should always < option_dim
        if self.share_option_nets:
            mean = self.policy(obs).view(-1, self.option_dim, self.action_dim)
            a_log_std = self.action_log_std.expand_as(mean)
        else:
            mean = torch.stack([m(obs) for m in self.policy], dim=-2)
            a_log_std = torch.stack(
                [m.expand_as(mean[:, 0, :]) for m in self.action_log_std], dim=-2
            )

        a_log_prob = None
        pre_tanh_value = None
        if option is not None:
            ind = option.view(-1, 1, 1).expand(-1, 1, self.action_dim)
            mean = mean.gather(dim=-2, index=ind.type(torch.long)).squeeze(dim=-2)
            a_log_std = a_log_std.gather(dim=-2, index=ind.type(torch.long)).squeeze(
                dim=-2
            )
        a_std = torch.exp(a_log_std)

        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = ReparamTanhMultivariateNormal(mean, a_log_std)
            if return_log_prob:
                action, pre_tanh_value = tanh_normal.sample(return_pretanh_value=True)
                a_log_prob = tanh_normal.log_prob(action)
            else:
                action = tanh_normal.sample()

        if return_tanh_normal:
            return (
                action,
                mean,
                a_log_std,
                a_log_prob,
                a_std,
                tanh_normal,
            )
        return (
            action,
            mean,
            a_log_std,
            a_log_prob,
            a_std,
        )

    def log_trans(self, obs, prev_option=None):
        # prev_option: long(N x 1) or None
        # prev_option: None: direct output p(option|obs, prev_option): a (N x prev_option x option) array where option is log-normalized
        # print(obs.shape, prev_option.shape)
        unnormed_pcs = self.switcher(obs)
        log_pcs = unnormed_pcs.log_softmax(dim=-1)
        if prev_option is None:
            return log_pcs
        else:
            return log_pcs.gather(
                dim=-2,
                index=prev_option.view(-1, 1, 1)
                .expand(-1, 1, self.option_dim)
                .type(torch.long),
            ).squeeze(dim=-2)

    def get_log_prob_action(self, obs, acts, option, return_normal_params=False):
        # if option is None, return (N x option_dim x 1), else return (N x 1)
        if self.share_option_nets:
            mean = self.policy(obs).view(-1, self.option_dim, self.action_dim)
            a_log_std = self.action_log_std.expand_as(mean)
        else:
            mean = torch.stack([m(obs) for m in self.policy], dim=-2)
            a_log_std = torch.stack(
                [m.expand_as(mean[:, 0, :]) for m in self.action_log_std], dim=-2
            )

        if option is not None:
            ind = option.view(-1, 1, 1).expand(-1, 1, self.action_dim)
            mean = mean.gather(dim=-2, index=ind.type(torch.long)).squeeze(dim=-2)
            a_log_std = a_log_std.gather(dim=-2, index=ind.type(torch.long)).squeeze(
                dim=-2
            )

        if option is None:
            acts = acts.view(-1, 1, self.action_dim)

        tanh_normal = ReparamTanhMultivariateNormal(mean, a_log_std)
        a_log_prob = tanh_normal.log_prob(acts)

        if return_normal_params:
            return a_log_prob, mean, a_log_std
        return a_log_prob

    def get_log_prob_option(self, obs, prev_option, option):
        log_tr = self.log_trans(obs, prev_option)
        return log_tr.gather(dim=-1, index=option.type(torch.long))

    def switcher(self, obs):
        if self.share_option_nets:
            return self.option_policy(obs).view(-1, self.option_dim + 1, self.option_dim)
        else:
            return torch.stack([m(obs) for m in self.option_policy], dim=-2)

    def get_action(self, observation, prev_options=None, deterministic=False):
        # if prev_options is None:
        #     # initial option
        #     prev_options = np.ones((1, 1), dtype=np.float32)*self.exploration_policy.option_dim
        # else:
        #     if type(prev_options) == torch.tensor:
        #         prev_options = prev_options.numpy()
        options = self.get_options(
            observation,
            prev_options,
            deterministic=deterministic,
        )
        actions = self.get_actions(observation, options, deterministic=deterministic)

        return actions[0, :], options[0, :], {}

    def get_actions(self, obs_np, option_np, deterministic=False):
        return self.eval_np(obs_np, option_np, deterministic=deterministic)[0]

    def get_options(self, obs, prev_option, deterministic=False):
        obs = torch.tensor(obs, dtype=torch.float32, device=ptu.device)
        prev_option = torch.tensor(prev_option, dtype=torch.long, device=ptu.device)
        log_tr = self.log_trans(obs, prev_option)
        if deterministic:
            return log_tr.argmax(dim=-1, keepdim=True).cpu().numpy().astype(np.int64)
        else:
            return (
                F.gumbel_softmax(log_tr, hard=False)
                .multinomial(1)
                .long()
                .cpu()
                .numpy()
                .astype(np.int64)
            )

    def policy_entropy(self, obs, option):
        log_std = self.forward(obs, option)[2]
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + log_std
        return entropy.sum(dim=-1, keepdim=True)

    def option_entropy(self, obs, prev_option):
        log_tr = self.log_trans(obs, prev_option)
        entropy = -(log_tr * log_tr.exp()).sum(dim=-1, keepdim=True)
        return entropy

    def policy_log_prob_entropy(self, obs, option, acts):
        _, mean, a_log_std, _, _ = self.forward(obs, option)

        tanh_normal = ReparamTanhMultivariateNormal(mean, a_log_std)
        a_log_prob = tanh_normal.log_prob(acts)

        entropy = (0.5 + 0.5 * math.log(2 * math.pi) + a_log_std).sum(dim=-1, keepdim=True)
        return a_log_prob, entropy

    def option_log_prob_entropy(self, obs, prev_option, option):
        # c1 can be option_dim, c2 should always < option_dim
        log_tr = self.log_trans(obs, prev_option)
        log_opt = log_tr.gather(dim=-1, index=option.type(torch.long))
        entropy = -(log_tr * log_tr.exp()).sum(dim=-1, keepdim=True)
        return log_opt, entropy

    def log_alpha_beta(self, obs_array, act_array):
        obs = torch.tensor(obs_array, dtype=torch.float32, device=ptu.device)
        act = torch.tensor(act_array, dtype=torch.float32, device=ptu.device)
        log_pis = self.get_log_prob_action(obs, act, None).view(
            -1, self.option_dim
        )  # demo_len x option
        log_trs = self.log_trans(obs, None)  # demo_len x (prev_option + 1) x option
        log_tr0 = log_trs[0, -1]
        log_trs = log_trs[1:, :-1]  # (demo_len-1) x prev_option x option

        log_alpha = [log_tr0 + log_pis[0]]
        for log_tr, log_pi in zip(log_trs, log_pis[1:]):
            log_alpha_t = (log_alpha[-1].unsqueeze(dim=-1) + log_tr).logsumexp(
                dim=0
            ) + log_pi
            log_alpha.append(log_alpha_t)

        log_beta = [
            torch.zeros(self.option_dim, dtype=torch.float32, device=ptu.device)
        ]
        for log_tr, log_pi in zip(reversed(log_trs), reversed(log_pis[1:])):
            log_beta_t = ((log_beta[-1] + log_pi).unsqueeze(dim=0) + log_tr).logsumexp(
                dim=-1
            )
            log_beta.append(log_beta_t)
        log_beta.reverse()

        log_alpha = torch.stack(log_alpha)
        log_beta = torch.stack(log_beta)
        entropy = -(log_trs * log_trs.exp()).sum(dim=-1)
        return log_alpha, log_beta, log_trs, log_pis, entropy

    def viterbi_path(self, obs_array, act_array):
        obs = torch.tensor(obs_array, dtype=torch.float32, device=ptu.device)
        act = torch.tensor(act_array, dtype=torch.float32, device=ptu.device)
        with torch.no_grad():
            log_pis = self.get_log_prob_action(obs, act, None).view(
                -1, 1, self.option_dim
            )  # demo_len x 1 x option
            log_trs = self.log_trans(obs, None)  # demo_len x (prev_option+1) x option
            log_prob = log_trs[:, :-1] + log_pis
            log_prob0 = log_trs[0, -1] + log_pis[0, 0]
            # forward
            max_path = torch.empty(
                obs.shape[0], self.option_dim, dtype=torch.long, device=ptu.device
            )
            accumulate_logp = log_prob0
            max_path[0] = self.option_dim
            for i in range(1, obs.shape[0]):
                accumulate_logp, max_path[i, :] = (
                    accumulate_logp.unsqueeze(dim=-1) + log_prob[i]
                ).max(dim=-2)
            # backward
            c_array = torch.zeros(
                obs_array.shape[0] + 1, 1, dtype=torch.long, device=ptu.device
            )
            log_prob_traj, c_array[-1] = accumulate_logp.max(dim=-1)

            for i in range(obs_array.shape[0], 0, -1):
                c_array[i - 1] = max_path[i - 1][c_array[i]]
        return c_array.detach(), log_prob_traj.detach()
