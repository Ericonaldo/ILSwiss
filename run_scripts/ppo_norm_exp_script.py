import yaml
import argparse
import os, inspect, sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(sys.path)

import gym
from torch import tanh

from rlkit.envs import get_env, get_envs
from rlkit.envs.wrappers import NormalizedBoxEnv, ProxyEnv

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.torch.common.networks import FlattenMlp
from rlkit.torch.common.policies import ReparamMultivariateGaussianPolicy
from rlkit.torch.algorithms.ppo.ppo import PPO
from rlkit.torch.algorithms.torch_rl_algorithm import TorchRLAlgorithm


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
    assert not isinstance(obs_space, gym.spaces.Dict)
    assert len(obs_space.shape) == 2
    assert len(act_space.shape) == 1

    env_wrapper = ProxyEnv  # Identical wrapper
    kwargs = {}
    if isinstance(act_space, gym.spaces.Box):
        env_wrapper = NormalizedBoxEnv

    env = env_wrapper(env, **kwargs)
    if "env_num" not in env_specs:
        env_specs["env_num"] = 1

    if "training_env_num" in env_specs:
        env_specs["env_num"] = env_specs["training_env_num"]

    print("Creating {} training environments ...".format(env_specs["env_num"]))
    training_env = get_envs(env_specs, env_wrapper, norm_obs=True, **kwargs)
    training_env.seed(env_specs["training_env_seed"])

    if "eval_env_num" in env_specs:
        env_specs["env_num"] = env_specs["eval_env_num"]

    print("Creating {} evaluation environments ...".format(env_specs["env_num"]))
    eval_env = get_envs(
        env_specs,
        env_wrapper,
        obs_rms=training_env.obs_rms,
        norm_obs=True,
        update_obs_rms=False,
    )
    eval_env.seed(env_specs["eval_env_seed"])

    try:
        obs_dim = obs_space.spaces["observation"].shape[0]
        goal_dim = obs_space.spaces["desired_goal"].shape[0]
    except Exception:
        tmp = env.reset()
        obs_dim = tmp["observation"].shape[0]
        goal_dim = tmp["desired_goal"].shape[0]
    action_dim = act_space.shape[0]

    net_size = variant["net_size"]
    num_hidden = variant["num_hidden_layers"]
    vf = FlattenMlp(
        hidden_sizes=num_hidden * [net_size],
        input_size=obs_dim,
        output_size=1,
        hidden_activation=torch.tanh,
    )
    #  policy = ReparamTanhMultivariateGaussianPolicy(
    policy = ReparamMultivariateGaussianPolicy(
        hidden_sizes=num_hidden * [net_size],
        obs_dim=obs_dim,
        action_dim=action_dim,
        conditioned_std=False,
        hidden_activation=torch.tanh,
    )

    policy = ReparamMultivariateGaussianPolicy(
        hidden_sizes=num_hidden * [net_size],
        obs_dim=obs_dim,
        condition_dim=goal_dim,
        action_dim=action_dim,
        output_activation=tanh,
        policy_noise=variant["policy_noise"],
        policy_noise_clip=variant["policy_noise_clip"],
    )

    trainer = PPO(
        policy=policy,
        vf=vf,
        **variant["ppo_params"],
    )

    algorithm = TorchRLAlgorithm(
        trainer=trainer,
        env=env,
        training_env=training_env,
        exploration_policy=policy,
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
        exp_specs = yaml.safe_load(spec_string)

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

    log_dir = None
    if "load_params" in exp_specs:
        load_path = exp_specs["load_params"]["load_path"]
        if (load_path is not None) and (len(load_path) > 0):
            log_dir = load_path

    setup_logger(
        exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs, log_dir=log_dir
    )

    experiment(exp_specs)
