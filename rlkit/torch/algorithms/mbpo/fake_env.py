from typing import Callable, Dict, Tuple
import torch
import numpy as np

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.torch.common.networks import BNN


class FakeEnv:
    def __init__(self, model: BNN, is_terminal: Callable, gen_model_idx: Callable):
        self.model = model
        self.is_terminal = is_terminal
        self.gen_model_idx = gen_model_idx

    def _logprob(
        self, x: np.ndarray, means: np.ndarray, vars: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        k = x.shape[-1]
        log_prob = -0.5 * (
            k * np.log(2 * np.pi)
            + np.sum(np.log(vars), axis=-1)
            + np.sum((x - means) ** 2 / vars, axis=-1)
        )
        prob = np.sum(np.exp(log_prob), axis=0)
        log_prob = np.log(prob)
        stds = np.mean(np.std(means, axis=0), axis=-1)
        return log_prob, stds

    def step(
        self, obs: np.ndarray, act: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.array, float, bool, Dict]:
        assert obs.ndim == act.ndim
        if single := (obs.ndim == 1):
            obs = obs[None]
            act = act[None]
        inputs = np.concatenate([obs, act], axis=-1)
        means, vars = self.model.predict(ptu.from_numpy(inputs), factored=True)
        means = ptu.get_numpy(means)
        vars = ptu.get_numpy(vars)
        means[:, :, 1:] += obs
        stds = np.sqrt(vars)
        samples = (
            means
            if deterministic
            else means + np.random.normal(size=means.shape) * stds
        )

        batch_size: int = means.shape[1]
        model_idxs = self.gen_model_idx(batch_size)
        batch_idxs: np.ndarray = np.arange(0, batch_size)
        model_samples: np.ndarray = samples[model_idxs, batch_idxs]
        model_means: np.ndarray = means[model_idxs, batch_idxs]
        model_stds: np.ndarray = stds[model_idxs, batch_idxs]

        log_prob, dev = self._logprob(model_samples, means, vars)
        rewards, next_obs = model_samples[:, :1], model_samples[:, 1:]
        terminals = self.is_terminal(obs, act, next_obs)

        batch_size = model_means.shape[0]
        means = np.concatenate(
            [model_means[:, :1], terminals, model_means[:, 1:]], axis=-1
        )
        stds = np.concatenate(
            [model_stds[:, :1], np.zeros([batch_size, 1]), model_stds[:, 1:]], axis=-1
        )

        if single:
            next_obs = next_obs[0]
            means = means[0]
            stds = stds[0]
            rewards = rewards[0]
            terminals = terminals[0]

        info = {"mean": means, "std": stds, "log_prob": log_prob, "dev": dev}
        return next_obs, rewards, terminals, info

    def step_tensor(
        self, obs: torch.Tensor, act: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        assert obs.ndim == act.ndim
        inputs = torch.cat([obs, act], dim=1)
        means, vars = self.model.predict(inputs, factored=True)
        means = torch.cat([means[:, :, :1], means[:, :, 1:] + obs[None]], dim=-1)
        stds = torch.sqrt(vars)
        samples = (
            means
            if deterministic
            else means + torch.normal(0, 1, size=means.shape) * stds
        )

        samples = samples[0]
        rewards, next_obs = samples[:, :1], samples[:, 1:]
        terminals = self.is_terminal(obs, act, next_obs)
        return next_obs, rewards, terminals, {}

    def close(self):
        pass
