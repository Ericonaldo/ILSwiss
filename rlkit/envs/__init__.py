# Inspired by OpenAI gym registration.py
import abc
import importlib

from rlkit.envs.envs_dict import envs_dict
from rlkit.envs.tasks_dict import tasks_dict
from rlkit.envs.wrappers import NormalizedBoxEnv, ProxyEnv
from rlkit.envs.vecenvs import BaseVectorEnv, DummyVectorEnv, SubprocVectorEnv

__all__ = [
    "BaseVectorEnv",
    "DummyVectorEnv",
    "SubprocVectorEnv",
]

import sys

# Overwrite envs
# from rlkit.envs.ant import AntEnv
# from rlkit.envs.humanoid import HumanoidEnv
# from rlkit.envs.swimmer import SwimmerEnv

env_overwrite = {}  # {'Ant': AntEnv, 'Humanoid': HumanoidEnv, 'Swimmer':SwimmerEnv}


def load(name):
    # taken from OpenAI gym registration.py
    print(name)
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


def get_env(env_specs):
    """
    env_specs:
        env_name: 'halfcheetah'
        env_kwargs: {} # kwargs to pass to the env constructor call
    """
    domain = envs_dict[env_specs["env_name"]]

    # Equal to gym.make()
    env_class = load(domain)
    env = env_class(**env_specs["env_kwargs"])

    if domain in env_overwrite:
        print(
            "[ environments/utils ] WARNING: Using overwritten {} environment".format(
                domain
            )
        )
        env = env_overwrite[domain]()

    return env


def get_envs(env_specs, env_wrapper=None, **kwargs):
    """
    env_specs:
        env_name: 'halfcheetah'
        env_kwargs: {} # kwargs to pass to the env constructor call
    """
    domain = envs_dict[env_specs["env_name"]]

    if env_wrapper is None:
        env_wrapper = ProxyEnv

    # Equal to gym.make()
    env_class = load(domain)

    if ("env_num" not in env_specs.keys()) or (env_specs["env_num"] == 1):
        envs = env_wrapper(env_class(**env_specs["env_kwargs"]), **kwargs)

        if domain in env_overwrite:
            print(
                "[ environments/utils ] WARNING: Using overwritten {} environment".format(
                    domain
                )
            )
            envs = env_overwrite[domain]()

        print("\n WARNING: Single environment detected, wrap to DummyVectorEnv.")
        envs = DummyVectorEnv([lambda: envs])

    else:
        envs = SubprocVectorEnv(
            [
                lambda: env_wrapper(env_class(**env_specs["env_kwargs"]), **kwargs)
                for _ in range(env_specs["env_num"])
            ]
        )

        if domain in env_overwrite:
            envs = SubprocVectorEnv(
                [
                    lambda: env_wrapper(env_overwrite[domain](), **kwargs)
                    for _ in range(env_specs["env_num"])
                ]
            )

    return envs


def get_task_params_samplers(task_specs):
    """
    task_specs:
        meta_train_tasks: 'hc_rand_vel_meta_train'
        meta_val_tasks: 'hc_rand_vel_meta_val'
        meta_test_tasks: 'hc_rand_vel_meta_test'
        meta_train_kwargs: {}
        meta_val_kwargs: {}
        meta_test_kwargs: {}
    """
    keys = ["meta_train_tasks", "meta_val_tasks", "meta_test_tasks"]
    d = {}
    for k in keys:
        if k in task_specs:
            task_class = load(task_specs[k])
            d[k] = task_class(**task_specs[k + "_kwargs"])
    return d


class EnvFactory(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __get__(self, task_params):
        """
        Implements returning and environment corresponding to given task params
        """
        pass

    @abc.abstractmethod
    def get_task_identifier(self, task_params):
        """
        Returns a hashable description of task params so it can be used
        as dictionary keys etc.
        """
        pass

    def task_params_to_obs_task_params(self, task_params):
        """
        Sometimes this may be needed. For example if we are training a
        multitask RL algorithm and want to give it the task params as
        part of the state.
        """
        raise NotImplementedError()
