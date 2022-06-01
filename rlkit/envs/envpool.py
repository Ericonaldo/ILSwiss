import envpool


class EnvpoolEnv:
    def __init__(self, env_specs):
        self._envs = envpool.make(
            env_specs["envpool_name"],
            env_type=env_specs["env_type"],
            num_envs=env_specs["env_num"],
            seed=env_specs["training_env_seed"],
        )

    @property
    def envs(self):
        return self._envs

    def step(self, actions, *args, **kwargs):
        obs, rew, done, info_dict = self.envs.step(actions, *args, **kwargs)
        # envpool returns the env_info as dict of list,
        # while ilswiss requires a list of dicts
        info_list = []
        for idx in range(len(obs)):
            # info["players"] is a dict, here just ignore it
            info_list.append(
                {k: v[idx] for k, v in info_dict.items() if k != "players"}
            )
        return obs, rew, done, info_list

    def __getattr__(self, attrname):
        return getattr(self.envs, attrname)

    def __len__(self):
        return len(self.envs)
