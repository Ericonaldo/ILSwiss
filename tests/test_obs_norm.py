import os
import inspect
import sys
import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(sys.path)

from rlkit.envs import get_envs


def test_obs_norm():
    env_specs = dict(
        env_name="hopper",
        env_kwargs={},
        env_num=4,
    )
    vector_envs = get_envs(
        env_specs=env_specs,
        norm_obs=True,
    )
    obs = vector_envs.reset()
    for _ in range(10):
        raw_obs = vector_envs.unnormalize_obs(obs)
        assert np.allclose(vector_envs.normalize_obs(raw_obs), obs)
        obs, _, _, _ = vector_envs.step(
            [action_space.sample() for action_space in vector_envs.action_space]
        )


def test_goal_obs_norm():
    env_specs = dict(
        env_name="fetch-push",
        env_kwargs={},
        env_num=4,
    )
    vector_envs = get_envs(
        env_specs=env_specs,
        norm_obs=True,
    )
    obs = vector_envs.reset()
    for _ in range(10):
        raw_obs = vector_envs.unnormalize_obs(obs)
        norm_obs = vector_envs.normalize_obs(raw_obs)
        for idx in range(len(norm_obs)):
            for k in obs[idx]:
                assert np.allclose(norm_obs[idx][k], obs[idx][k])
        obs, _, _, _ = vector_envs.step(
            [action_space.sample() for action_space in vector_envs.action_space]
        )
