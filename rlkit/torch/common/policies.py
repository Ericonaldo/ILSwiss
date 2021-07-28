import numpy as np
from numpy.random import choice

import torch
from torch import nn as nn

from rlkit.policies.base import ExplorationPolicy, Policy
from rlkit.torch.common.networks import Mlp, FlattenMlp
from rlkit.torch.common.distributions import ReparamTanhMultivariateNormal
from rlkit.torch.common.distributions import ReparamMultivariateNormalDiag
from rlkit.torch.core import PyTorchModule

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.core.serializable import Serializable

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class MakeDeterministic(Policy):
    def __init__(self, stochastic_policy):
        self.stochastic_policy = stochastic_policy


    def get_action(self, observation):
        return self.stochastic_policy.get_action(observation,
                                                 deterministic=True)

    def get_actions(self, observations):
        return self.stochastic_policy.get_actions(observations,
                                                  deterministic=True)
    
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
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            init_w=1e-3,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            output_activation=nn.LogSoftmax(1),
            **kwargs
        )

    def get_action(self, obs_np, deterministic=False):
        action = self.get_actions(obs_np[None], deterministic=deterministic)
        return action, {}

    def get_actions(self, obs_np, deterministic=False):
        return self.eval_np(obs_np)[0]

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

            idx = torch.unsqueeze(idx, 1)
            log_prob = torch.gather(log_probs, 1, idx)

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
    def __init__(
        self,
        hidden_sizes,
        obs_dim,
        action_dim,
        init_w=1e-3,
        **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
    
    def get_action(self, obs_np, deterministic=False):
        '''
        deterministic=False makes no diff, just doing this for
        consistency in interface for now
        '''
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
        max_act = 1.0,
        **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
        self.noise = policy_noise
        self.noise_clip = policy_noise_clip
        self.max_act = max_act
    
    def get_action(self, obs_np, deterministic=False):
        '''
        deterministic=False makes no diff, just doing this for
        consistency in interface for now
        '''
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        actions = actions[None]
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return self.eval_np(obs_np, deterministic=deterministic)[0]

    def forward(
            self,
            obs,
            deterministic=False):
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
            noise = torch.normal(
                torch.zeros_like(action),
                self.noise,
            )
            noise = torch.clamp(
                noise,
                -self.noise_clip,
                self.noise_clip
            )
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
            std=None,
            init_w=1e-3,
            max_act = 1.0,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
        self.log_std = None
        self.std = std
        self.max_act = max_act
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return self.eval_np(obs_np, deterministic=deterministic)[0]

    def forward(
            self,
            obs,
            deterministic=False,
            return_log_prob=False,
            return_tanh_normal=False
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        
        # print(h)
        mean = self.last_fc(h)
        # print(mean)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        expected_log_prob = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = ReparamTanhMultivariateNormal(mean, log_std)
            # print('mean, std')
            # print(mean)
            # print(log_std)
            if return_log_prob:
                action, pre_tanh_value = tanh_normal.sample(
                    return_pretanh_value=True
                )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
            else:
                action = tanh_normal.sample()

        # doing it like this for now for backwards compatibility
        if return_tanh_normal:
            return (
                action, mean, log_std, log_prob, expected_log_prob, std,
                mean_action_log_prob, pre_tanh_value, tanh_normal,
            )
        return (
            action, mean, log_std, log_prob, expected_log_prob, std,
            mean_action_log_prob, pre_tanh_value,
        )

    def get_log_prob(self, obs, acts, return_normal_params=False):
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std
        
        tanh_normal = ReparamTanhMultivariateNormal(mean, log_std)
        log_prob = tanh_normal.log_prob(acts)

        # print('\n\n\n\n\nGet log prob')
        # print(log_prob)
        # print(mean)
        # print(log_std)

        if return_normal_params:
            return log_prob, mean, log_std
        return log_prob



class ReparamMultivariateGaussianPolicy(Mlp, ExplorationPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            assert LOG_SIG_MIN <= np.log(std) <= LOG_SIG_MAX
            std = std*np.ones((1,action_dim))
            self.log_std = ptu.from_numpy(np.log(std), requires_grad=False)

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return self.eval_np(obs_np, deterministic=deterministic)[0]

    def forward(
            self,
            obs,
            deterministic=False,
            return_log_prob=False,
            return_normal=False
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        # print('forward')
        # print(obs)
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        # print('fuck')
        # print(h)
        mean = self.last_fc(h)

        mean = torch.clamp(mean, min=-1.0, max=1.0)

        # print(mean)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            
            # log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, np.log(0.2))
            
            std = torch.exp(log_std)

        else:
            std = self.std
            log_std = self.log_std

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
                action, mean, log_std, log_prob, expected_log_prob, std,
                mean_action_log_prob, normal
            )
        return (
            action, mean, log_std, log_prob, expected_log_prob, std,
            mean_action_log_prob
        )

    def get_log_prob(self, obs, acts, return_normal_params=False):
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std
        
        normal = ReparamMultivariateNormalDiag(mean, log_std)
        log_prob = normal.log_prob(acts)

        if return_normal_params:
            return log_prob, mean, log_std
        return log_prob



class AntRandGoalCustomReparamTanhMultivariateGaussianPolicy(ReparamTanhMultivariateGaussianPolicy):
    """
    Custom for Ant Rand Goal
    The only difference is that it linearly embeds the goal into a higher dimension
    """
    def __init__(
            self,
            goal_dim,
            goal_embed_dim,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            obs_dim + goal_embed_dim,
            action_dim,
            init_w=init_w,
            **kwargs
        )

        self.goal_embed_fc = nn.Linear(goal_dim, goal_embed_dim)
        self.goal_dim = goal_dim
    
    def forward(
        self,
        obs,
        deterministic=False,
        return_log_prob=False,
        return_tanh_normal=False
    ):
        # obs will actuall be concat of [obs, goal]
        goal = obs[:,-self.goal_dim:]
        goal_embed = self.goal_embed_fc(goal)
        obs = torch.cat([obs[:,:-self.goal_dim], goal_embed], dim=-1)
        return super().forward(
            obs,
            deterministic=deterministic,
            return_log_prob=return_log_prob,
            return_tanh_normal=return_tanh_normal
        )


class ObsPreprocessedReparamTanhMultivariateGaussianPolicy(ReparamTanhMultivariateGaussianPolicy):
    '''
        This is a weird thing and I didn't know what to call.
        Basically I wanted this so that if you need to preprocess
        your inputs somehow (attention, gating, etc.) with an external module
        before passing to the policy you could do so.
        Assumption is that you do not want to update the parameters of the preprocessing
        module so its output is always detached.
    '''
    def __init__(self, preprocess_model, *args, **kwargs):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        # this is a hack so that it is not added as a submodule
        self.preprocess_model_list = [preprocess_model]
    

    @property
    def preprocess_model(self):
        # this is a hack so that it is not added as a submodule
        return self.preprocess_model_list[0]


    def preprocess_fn(self, obs_batch):
        mode = self.preprocess_model.training
        self.preprocess_model.eval()
        processed_obs_batch = self.preprocess_model(obs_batch, False).detach()
        self.preprocess_model.train(mode)
        return processed_obs_batch
    

    def forward(
        self,
        obs,
        deterministic=False,
        return_log_prob=False,
        return_tanh_normal=False
    ):
        obs = self.preprocess_fn(obs).detach()
        return super().forward(
            obs,
            deterministic=deterministic,
            return_log_prob=return_log_prob,
            return_tanh_normal=return_tanh_normal
        )


    def get_log_prob(self, obs, acts):
        obs = self.preprocess_fn(obs).detach()
        return super().get_log_prob(obs, acts)


class WithZObsPreprocessedReparamTanhMultivariateGaussianPolicy(ReparamTanhMultivariateGaussianPolicy):
    '''
        This is a weird thing and I didn't know what to call.
        Basically I wanted this so that if you need to preprocess
        your inputs somehow (attention, gating, etc.) with an external module
        before passing to the policy you could do so.
        Assumption is that you do not want to update the parameters of the preprocessing
        module so its output is always detached.
    '''
    def __init__(self, preprocess_model, z_dim, *args, train_preprocess_model=False, **kwargs):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        # this is a hack so that it is not added as a submodule
        if train_preprocess_model:
            self._preprocess_model = preprocess_model
        else:
            # this is a hack so that it is not added as a submodule
            self.preprocess_model_list = [preprocess_model]
        self.z_dim = z_dim
        self.train_preprocess_model = train_preprocess_model
    

    @property
    def preprocess_model(self):
        if self.train_preprocess_model:
            return self._preprocess_model
        else:
            # this is a hack so that it is not added as a submodule
            return self.preprocess_model_list[0]


    def preprocess_fn(self, obs_batch):
        if self.train_preprocess_model:
            processed_obs_batch = self.preprocess_model(
                obs_batch[:,:-self.z_dim],
                False,
                obs_batch[:,-self.z_dim:]
            )
        else:
            mode = self.preprocess_model.training
            self.preprocess_model.eval()
            processed_obs_batch = self.preprocess_model(
                obs_batch[:,:-self.z_dim],
                False,
                obs_batch[:,-self.z_dim:]
            ).detach()
            self.preprocess_model.train(mode)
        return processed_obs_batch
    

    def forward(
        self,
        obs,
        deterministic=False,
        return_log_prob=False,
        return_tanh_normal=False
    ):
        if self.train_preprocess_model:
            obs = self.preprocess_fn(obs)
        else:
            obs = self.preprocess_fn(obs).detach()
        return super().forward(
            obs,
            deterministic=deterministic,
            return_log_prob=return_log_prob,
            return_tanh_normal=return_tanh_normal
        )


    def get_log_prob(self, obs, acts):
        if self.train_preprocess_model:
            obs = self.preprocess_fn(obs)
        else:
            obs = self.preprocess_fn(obs).detach()
        return super().get_log_prob(obs, acts)


class BaselineContextualPolicy(PyTorchModule):
    def __init__(
        self,
        action_dim
    ):
        self.save_init_params(locals())
        super().__init__()
        
        # # YUKE ARCH : 17x17 out
        self.conv_part = nn.Sequential(
            nn.Conv2d(6, 32, 8, stride=4, padding=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.Conv2d(32, 32, 4, stride=2, padding=p),
            # nn.ReLU(),
        )
        self.fc_part = nn.Sequential(
            nn.Linear(9248, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

        # FINN ARCH: 8x8 out
        # self.conv_part = nn.Sequential(
        #     nn.Conv2d(6, 32, 5, stride=2, padding=2),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 32, 5, stride=2, padding=2),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 32, 5, stride=2, padding=2),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 32, 5, stride=2, padding=2),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     # nn.Conv2d(32, 32, 4, stride=2, padding=p),
        #     # nn.ReLU(),
        # )
        # self.fc_part = nn.Sequential(
        #     nn.Linear(2048, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Linear(256, action_dim),
        # )
        print(self)
    

    def forward(self, obs):
        image = obs['image']
        context_image = obs['z']

        policy_input = torch.cat([image, context_image], dim=1)
        conv_out = self.conv_part(policy_input)
        conv_out = conv_out.view(conv_out.size(0), -1)
        fc_out = self.fc_part(conv_out)
        return fc_out


class CondBaselineContextualPolicy(ExplorationPolicy):
    def __init__(self, policy, z):
        super().__init__()
        self.policy = policy
        self.z = z

    def get_action(self, obs_np):
        new_dict = {k: ptu.from_numpy(obs_np[k][None], requires_grad=False) for k in obs_np}
        new_dict['z'] = self.z
        action = ptu.get_numpy(self.policy.forward(new_dict)[0])
        return action, {}
