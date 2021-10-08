from rlkit.env_creators.base_env import BaseEnv


class MujocoEnv(BaseEnv):
    """A wrapper for gym Mujoco environments to fit in multi-agent apis."""

    def __init__(self, **configs):
        super().__init__(**configs)

        # create underlying mujoco env
        env_name = configs["env_name"]
        env_kwargs = configs["env_kwargs"]
        if env_name == "halfcheetah":
            from gym.envs.mujoco.half_cheetah import HalfCheetahEnv as Env
        elif env_name == "ant":
            from gym.envs.mujoco.ant import AntEnv as Env
        elif env_name == "hopper":
            from gym.envs.mujoco.hopper import HopperEnv as Env
        elif env_name == "walker":
            from gym.envs.mujoco.walker2d import Walker2dEnv as Env
        elif env_name == "humanoid":
            from gym.envs.mujoco.humanoid import HumanoidEnv as Env
        elif env_name == "swimmer":
            from gym.envs.mujoco.swimmer import SwimmerEnv as Env
        elif env_name == "inverteddoublependulum":
            from gym.envs.mujoco.inverted_double_pendulum import (
                InvertedDoublePendulum2dEnv as Env,
            )
        elif env_name == "invertedpendulum":
            from gym.envs.mujoco.inverted_pendulum import InvertedPendulum as Env
        else:
            raise NotImplementedError(env_name)
        self._env = Env(**env_kwargs)

        self._default_agent_name = "agent_0"
        self.agent_ids = [self._default_agent_name]
        self.n_agents = len(self.agent_ids)
        self.observation_space_n = {
            self._default_agent_name: self._env.observation_space
        }
        self.action_space_n = {self._default_agent_name: self._env.action_space}

    def seed(self, seed):
        return self._env.seed(seed)

    def reset(self):
        return {self._default_agent_name: self._env.reset()}

    def step(self, action_n):
        action = action_n[self._default_agent_name]
        next_obs, rew, done, info = self._env.step(action)
        next_obs_n = {self._default_agent_name: next_obs}
        rew_n = {self._default_agent_name: rew}
        done_n = {self._default_agent_name: done}
        info_n = {self._default_agent_name: info}
        return next_obs_n, rew_n, done_n, info_n

    def render(self, **kwargs):
        return self._env.render(**kwargs)
