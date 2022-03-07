import yaml
import argparse
import numpy as np
import os, inspect, sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(sys.path)

import gym
from torch import tanh

from rlkit.envs import get_env, get_envs
from rlkit.envs.wrappers import DiscretEnv, ProxyEnv

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.torch.common.networks import FlattenMlp
from rlkit.torch.common.policies import MlpGaussianAndEpsilonConditionPolicy, CatagorialConditionPolicy
from rlkit.torch.algorithms.torch_rl_algorithm import TorchRLAlgorithm
from rlkit.torch.algorithms.gcsl.rl import GoalHorizonRL
from rlkit.torch.algorithms.gcsl.gcsl import GCSL
from rlkit.data_management.relabel_horizon_replay_buffer import HindsightHorizonReplayBuffer


def experiment(variant):
    env_specs = variant["env_specs"]
    env = get_env(env_specs)
    env.seed(env_specs["eval_env_seed"])

    print("\n\nEnv: {}".format(env_specs["env_name"]))
    print("kwargs: {}".format(env_specs["env_kwargs"]))
    print("Obs Space: {}".format(env.observation_space))
    print("Act Space: {}\n\n".format(env.action_space))

    obs_space = env.observation_space
    act_space = env.action_space
    assert isinstance(obs_space, gym.spaces.Dict), "obs is {}".format(obs_space)

    env_wrapper = ProxyEnv  # Identical wrapper
    wrapper_kwargs = {}
    discretize = "discretize" in env_specs and env_specs["discretize"]
    if discretize:
        env_wrapper = DiscretEnv  # Discrete wrapper
        wrapper_kwargs = env_specs["discret_kwargs"]
    
    env = env_wrapper(env, **wrapper_kwargs)
    print("Act Space: {}\n\n".format(env.action_space))
    
    kwargs = {}
    if "vec_env_kwargs" in env_specs:
        kwargs = env_specs["vec_env_kwargs"]
    training_env = get_envs(env_specs, env_wrapper, wrapper_kwargs, **kwargs)
    training_env.seed(env_specs["training_env_seed"])
    
    try:
        obs_dim = obs_space.spaces['observation'].shape[0]
        goal_dim = obs_space.spaces['desired_goal'].shape[0]
    except BaseException:
        tmp = env.reset()
        obs_dim = tmp['observation'].shape[0]
        goal_dim = tmp['desired_goal'].shape[0]
    action_dim = act_space.shape[0]

    net_size = variant["net_size"]
    num_hidden = variant["num_hidden_layers"]
    max_path_length = 0
    use_horizons = variant['rl_alg_params']['use_horizons']
    if use_horizons:
        max_path_length=variant['rl_alg_params']['max_path_length']

    if discretize:
        assert variant["gcsl_params"]["mode"] == "CLASS"
        action_dim = env_specs["discret_kwargs"]["granularity"] ** act_space.shape[0]
        policy = CatagorialConditionPolicy(
            hidden_sizes=num_hidden * [net_size],
            action_dim=action_dim,
            obs_dim=obs_dim,
            condition_dim=goal_dim+max_path_length, 
            batch_norm=True
        )
    else:
        policy = MlpGaussianAndEpsilonConditionPolicy(
            hidden_sizes=num_hidden * [net_size],
            action_space=env.action_space,
            obs_dim=obs_dim,
            condition_dim=goal_dim+max_path_length,
            action_dim=action_dim,
            output_activation=tanh,
            batch_norm=True
        )

    trainer = GCSL(policy=policy, use_horizons=variant["rl_alg_params"]["use_horizons"], **variant["gcsl_params"])
    algorithm = GoalHorizonRL(
        trainer=trainer,
        env=env,
        training_env=training_env,
        exploration_policy=policy,
        **variant["rl_alg_params"]
    )

    if ptu.gpu_enabled():
        algorithm.to(ptu.device)
    algorithm.train()

    return 1


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", help="experiment specification file")
    parser.add_argument("-g", "--gpu", help="gpu id", type=int, default=0)
    args = parser.parse_args()
    with open(args.experiment, "r") as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.safe_load(spec_string)

    # make all seeds the same.
    exp_specs["env_specs"]["eval_env_seed"] = exp_specs["env_specs"][
        "training_env_seed"
    ] = exp_specs["seed"]

    if exp_specs["using_gpus"]:
        print("\n\nUSING GPU\n\n")
        ptu.set_gpu_mode(True, args.gpu)
    exp_id = exp_specs["exp_id"]
    exp_prefix = exp_specs["exp_name"] + "_plr_{}".format(exp_specs["gcsl_params"]["policy_lr"])
    seed = exp_specs["seed"]
    set_seed(seed)
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs)

    experiment(exp_specs)
