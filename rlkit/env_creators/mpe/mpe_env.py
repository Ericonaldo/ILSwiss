import importlib

from rlkit.env_creators.base_env import BaseEnv


class MpeEnv(BaseEnv):
    """A wrapper for gym Mujoco environments to fit in multi-agent apis."""

    def __init__(self, **configs):
        super().__init__(**configs)

        # create underlying mujoco env
        env_name = configs["env_name"]
        env_kwargs = configs["env_kwargs"]

        env_module = importlib.import_module(f"pettingzoo.mpe.{env_name}")
        Env = env_module.parallel_env

        self._env = Env(**env_kwargs)

        self.agent_ids = self._env.possible_agents
        self.n_agents = len(self.agent_ids)
        self.observation_space_n = self._env.observation_spaces
        self.action_space_n = self._env.action_spaces

    def seed(self, seed):
        return self._env.seed(seed)

    def reset(self):
        return self._env.reset()

    def step(self, action_n):
        return self._env.step(action_n)

    def render(self, **kwargs):
        return self._env.render(**kwargs)
