import yaml
import argparse
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(sys.path)

import gym
import torch

from rlkit.envs import get_env, get_envs
from rlkit.envs.wrappers import NormalizedBoxActEnv, ProxyEnv

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.torch.common.networks import FlattenMlp
from rlkit.torch.common.policies import ReparamMultivariateGaussianPolicy
from rlkit.torch.common.policies import ReparamTanhMultivariateGaussianPolicy
from rlkit.torch.algorithms.ppo.ppo import PPO
from rlkit.torch.algorithms.torch_rl_algorithm import TorchRLAlgorithm


def experiment(variant):
    env_specs = variant["env_specs"]
    env = get_env(env_specs)
    env.seed(env_specs["eval_env_seed"])

    print("\n\nEnv: {}:{}".format(env_specs["env_creator"], env_specs["env_name"]))
    print("kwargs: {}".format(env_specs["env_kwargs"]))
    print("Obs Space: {}".format(env.observation_space_n))
    print("Act Space: {}\n\n".format(env.action_space_n))

    obs_space_n = env.observation_space_n
    act_space_n = env.action_space_n

    policy_mapping_dict = dict(
        zip(env.agent_ids, ["policy_0" for _ in range(env.n_agents)])
    )

    policy_trainer_n = {}
    policy_n = {}

    # create policies
    for agent_id in env.agent_ids:
        policy_id = policy_mapping_dict.get(agent_id)
        if policy_id not in policy_trainer_n:
            print(f"Create {policy_id} for {agent_id} ...")
            obs_space = obs_space_n[agent_id]
            act_space = act_space_n[agent_id]
            assert isinstance(obs_space, gym.spaces.Box)
            assert isinstance(act_space, gym.spaces.Box)
            assert len(obs_space.shape) == 1
            assert len(act_space.shape) == 1

            obs_dim = obs_space_n[agent_id].shape[0]
            action_dim = act_space_n[agent_id].shape[0]

            net_size = variant["net_size"]
            num_hidden = variant["num_hidden_layers"]
            vf = FlattenMlp(
                hidden_sizes=num_hidden * [net_size],
                input_size=obs_dim,
                output_size=1,
                hidden_activation=torch.tanh,
            )
            policy = ReparamMultivariateGaussianPolicy(
                hidden_sizes=num_hidden * [net_size],
                obs_dim=obs_dim,
                action_dim=action_dim,
                conditioned_std=False,
                hidden_activation=torch.tanh,
            )

            trainer = PPO(
                policy=policy,
                vf=vf,
                **variant["ppo_params"],
            )
            policy_trainer_n[policy_id] = trainer
            policy_n[policy_id] = policy
        else:
            print(f"Use existing {policy_id} for {agent_id} ...")

    env_wrapper = ProxyEnv  # Identical wrapper
    for act_space in act_space_n.values():
        if isinstance(act_space, gym.spaces.Box):
            env_wrapper = NormalizedBoxActEnv
            break

    env = env_wrapper(env)

    print("Creating {} training environments ...".format(env_specs["training_env_num"]))
    training_env = get_envs(
        env_specs,
        env_wrapper,
        env_num=env_specs["training_env_num"],
        norm_obs=True,
    )
    training_env.seed(env_specs["training_env_seed"])

    print("Creating {} evaluation environments ...".format(env_specs["eval_env_num"]))
    eval_env = get_envs(
        env_specs,
        env_wrapper,
        env_num=env_specs["eval_env_num"],
        obs_rms_n=training_env.obs_rms_n,
        norm_obs=True,
        update_obs_rms=False,
    )
    eval_env.seed(env_specs["eval_env_seed"])

    algorithm = TorchRLAlgorithm(
        trainer_n=policy_trainer_n,
        env=env,
        training_env=training_env,
        eval_env=eval_env,
        exploration_policy_n=policy_n,
        policy_mapping_dict=policy_mapping_dict,
        **variant["rl_alg_params"],
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
        exp_specs = yaml.load(spec_string)

    # make all seeds the same.
    exp_specs["env_specs"]["eval_env_seed"] = exp_specs["env_specs"][
        "training_env_seed"
    ] = exp_specs["seed"]

    if exp_specs["using_gpus"] > 0:
        print("\n\nUSING GPU\n\n")
        ptu.set_gpu_mode(True, args.gpu)
    exp_id = exp_specs["exp_id"]
    exp_prefix = exp_specs["exp_name"]
    seed = exp_specs["seed"]
    set_seed(seed)
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs)

    experiment(exp_specs)
