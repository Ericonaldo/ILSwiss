import numpy as np
import gym
from gym import Env
from gym.spaces import Box
from collections import deque

from rlkit.core.serializable import Serializable

EPS = np.finfo(np.float32).eps.item()


class ProxyEnv(Serializable, Env):
    def __init__(self, wrapped_env):
        self._wrapped_env = wrapped_env
        Serializable.quick_init(self, locals())
        super(ProxyEnv, self).__init__()
        
        self.action_space = self._wrapped_env.action_space
        self.observation_space = self._wrapped_env.observation_space

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def reset(self, **kwargs):
        return self._wrapped_env.reset(**kwargs)

    def step(self, action):
        return self._wrapped_env.step(action)

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    def log_diagnostics(self, paths, *args, **kwargs):
        if hasattr(self._wrapped_env, "log_diagnostics"):
            self._wrapped_env.log_diagnostics(paths, *args, **kwargs)

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def terminate(self):
        if hasattr(self._wrapped_env, "terminate"):
            self._wrapped_env.terminate()

    def seed(self, seed):
        return self._wrapped_env.seed(seed)

    def __getattr__(self, attrname):
        return getattr(self._wrapped_env, attrname)


class ScaledEnv(ProxyEnv, Serializable):
    """
    Scale the obs if desired
    Unscale the acts if desired
    """

    def __init__(
        self,
        env,
        obs_mean=None,
        obs_std=None,
        acts_mean=None,
        acts_std=None,
        meta=False,
    ):
        self._wrapped_env = env
        self._serializable_initialized = False
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)

        if obs_mean is not None:
            assert obs_std is not None
            self._scale_obs = True
        else:
            assert obs_std is None
            self._scale_obs = False

        if acts_mean is not None:
            assert acts_std is not None
            self._unscale_acts = True
        else:
            assert acts_std is None
            self._unscale_acts = False

        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.acts_mean = acts_mean
        self.acts_std = acts_std

    def get_unscaled_obs(self, obs):
        if self._scale_obs:
            return obs * (self.obs_std + EPS) + self.obs_mean
        else:
            return obs

    def get_scaled_obs(self, obs):
        if self._scale_obs:
            return (obs - self.obs_mean) / (self.obs_std + EPS)
        else:
            return obs

    def get_unscaled_acts(self, acts):
        if self._unscale_acts:
            return acts * (self.acts_std + EPS) + self.acts_mean
        else:
            return acts

    def get_scaled_acts(self, acts):
        if self._unscale_acts:
            return (acts - self.acts_mean) / (self.acts_std + EPS)
        else:
            return acts

    def step(self, action):
        if self._unscale_acts:
            action = action * (self.acts_std + EPS) + self.acts_mean
        obs, r, done, info = self._wrapped_env.step(action)
        if self._scale_obs:
            obs = (obs - self.obs_mean) / (self.obs_std + EPS)
        return obs, r, done, info

    def reset(self, **kwargs):
        obs = self._wrapped_env.reset(**kwargs)
        if self._scale_obs:
            obs = (obs - self.obs_mean) / (self.obs_std + EPS)
        return obs

    def log_statistics(self, *args, **kwargs):
        if hasattr(self._wrapped_env, "log_statistics"):
            return self._wrapped_env.log_statistics(*args, **kwargs)
        else:
            return {}

    def log_new_ant_multi_statistics(self, paths, epoch, log_dir):
        if hasattr(self._wrapped_env, "log_new_ant_multi_statistics"):
            return self._wrapped_env.log_new_ant_multi_statistics(paths, epoch, log_dir)
        else:
            return {}


class MinmaxEnv(ProxyEnv, Serializable):
    """
    Scale the obs if desired
    """

    def __init__(
        self,
        env,
        obs_min=None,
        obs_max=None,
    ):
        self._wrapped_env = env
        self._serializable_initialized = False
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)

        if obs_min is not None:
            assert obs_max is not None
            self._scale_obs = True
        else:
            assert obs_max is None
            self._scale_obs = False

        self.obs_min = obs_min
        self.obs_max = obs_max

    def get_unscaled_obs(self, obs):
        if self._scale_obs:
            return obs * (self.obs_max - self.obs_min + EPS) + self.obs_min
        else:
            return obs

    def get_scaled_obs(self, obs):
        if self._scale_obs:
            return (obs - self.obs_min) / (self.obs_max - self.obs_min + EPS)
        else:
            return obs

    def step(self, action):
        obs, r, done, info = self._wrapped_env.step(action)
        if self._scale_obs:
            obs = (obs - self.obs_min) / (self.obs_max - self.obs_min + EPS)
        return obs, r, done, info

    def reset(self, **kwargs):
        obs = self._wrapped_env.reset(**kwargs)
        if self._scale_obs:
            obs = (obs - self.obs_min) / (self.obs_max - self.obs_min + EPS)
        return obs

    def log_statistics(self, *args, **kwargs):
        if hasattr(self._wrapped_env, "log_statistics"):
            return self._wrapped_env.log_statistics(*args, **kwargs)
        else:
            return {}

    def log_new_ant_multi_statistics(self, paths, epoch, log_dir):
        if hasattr(self._wrapped_env, "log_new_ant_multi_statistics"):
            return self._wrapped_env.log_new_ant_multi_statistics(paths, epoch, log_dir)
        else:
            return {}


class ScaledMetaEnv(ProxyEnv, Serializable):
    """
    Scale the obs if desired
    Unscale the acts if desired
    """

    def __init__(
        self,
        env,
        obs_mean=None,
        obs_std=None,
        acts_mean=None,
        acts_std=None,
    ):
        self._wrapped_env = env
        self._serializable_initialized = False
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)

        if obs_mean is not None:
            assert obs_std is not None
            self._scale_obs = True
        else:
            assert obs_std is None
            self._scale_obs = False

        if acts_mean is not None:
            assert acts_std is not None
            self._unscale_acts = True
        else:
            assert acts_std is None
            self._unscale_acts = False

        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.acts_mean = acts_mean
        self.acts_std = acts_std

    def step(self, action):
        if self._unscale_acts:
            action = action * (self.acts_std + EPS) + self.acts_mean
        obs, r, done, info = self._wrapped_env.step(action)
        if self._scale_obs:
            obs["obs"] = (obs["obs"] - self.obs_mean) / (self.obs_std + EPS)
            obs["obs"] = obs["obs"][0]
        return obs, r, done, info

    def reset(self, **kwargs):
        obs = self._wrapped_env.reset(**kwargs)
        if self._scale_obs:
            obs["obs"] = (obs["obs"] - self.obs_mean) / (self.obs_std + EPS)
            obs["obs"] = obs["obs"][0]
        return obs

    @property
    def task_identifier(self):
        return self._wrapped_env.task_identifier

    def task_id_to_obs_task_params(self, task_id):
        return self._wrapped_env.task_id_to_obs_task_params(task_id)

    def log_statistics(self, paths):
        return self._wrapped_env.log_statistics(paths)

    def log_diagnostics(self, paths):
        if hasattr(self._wrapped_env, "log_diagnostics"):
            return self._wrapped_env.log_diagnostics(paths)


class NormalizedBoxEnv(ProxyEnv, Serializable):
    """
    Normalize action to in [-1, 1].

    Optionally normalize observations and scale reward.
    """

    def __init__(
        self,
        env,
        reward_scale=1.0,
        obs_mean=None,
        obs_std=None,
    ):
        # self._wrapped_env needs to be called first because
        # Serializable.quick_init calls getattr, on this class. And the
        # implementation of getattr (see below) calls self._wrapped_env.
        # Without setting this first, the call to self._wrapped_env would call
        # getattr again (since it's not set yet) and therefore loop forever.
        self._wrapped_env = env
        # Or else serialization gets delegated to the wrapped_env. Serialize
        # this env separately from the wrapped_env.
        self._serializable_initialized = False
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)
        self._should_normalize = not (obs_mean is None and obs_std is None)
        if self._should_normalize:
            if obs_mean is None:
                obs_mean = np.zeros_like(env.observation_space.low)
            else:
                obs_mean = np.array(obs_mean)
            if obs_std is None:
                obs_std = np.ones_like(env.observation_space.low)
            else:
                obs_std = np.array(obs_std)
        self._reward_scale = reward_scale
        self._obs_mean = obs_mean
        self._obs_std = obs_std
        ub = np.ones(self._wrapped_env.action_space.shape)
        self.action_space = Box(-1 * ub, ub)

    def estimate_obs_stats(self, obs_batch, override_values=False):
        if self._obs_mean is not None and not override_values:
            raise Exception(
                "Observation mean and std already set. To "
                "override, set override_values to True."
            )
        self._obs_mean = np.mean(obs_batch, axis=0)
        self._obs_std = np.std(obs_batch, axis=0)

    def _apply_normalize_obs(self, obs):
        return (obs - self._obs_mean) / (self._obs_std + EPS)

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        # Add these explicitly in case they were modified
        d["_obs_mean"] = self._obs_mean
        d["_obs_std"] = self._obs_std
        d["_reward_scale"] = self._reward_scale
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        self._obs_mean = d["_obs_mean"]
        self._obs_std = d["_obs_std"]
        self._reward_scale = d["_reward_scale"]

    def step(self, action):
        lb = self._wrapped_env.action_space.low
        ub = self._wrapped_env.action_space.high
        scaled_action = lb + (action + 1.0) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)

        wrapped_step = self._wrapped_env.step(scaled_action)
        next_obs, reward, done, info = wrapped_step
        if self._should_normalize:
            next_obs = self._apply_normalize_obs(next_obs)
        return next_obs, reward * self._reward_scale, done, info

    def __str__(self):
        return "Normalized: %s" % self._wrapped_env

    def log_diagnostics(self, paths, **kwargs):
        if hasattr(self._wrapped_env, "log_diagnostics"):
            return self._wrapped_env.log_diagnostics(paths, **kwargs)
        else:
            return None

    def __getattr__(self, attrname):
        return getattr(self._wrapped_env, attrname)


class FrameStackEnv(ProxyEnv, Serializable):
    def __init__(self, env, k):
        self._wrapped_env = env
        # Or else serialization gets delegated to the wrapped_env. Serialize
        # this env separately from the wrapped_env.
        self._serializable_initialized = False
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)
        
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)