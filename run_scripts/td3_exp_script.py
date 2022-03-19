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
from rlkit.core.logger import load_from_file
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.torch.common.networks import FlattenMlp
from rlkit.torch.common.policies import MlpGaussianNoisePolicy
from rlkit.torch.algorithms.td3.td3 import TD3
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
    assert len(obs_space.shape) == 1
    assert len(act_space.shape) == 1

    env_wrapper = ProxyEnv  # Identical wrapper
    wrapper_kwargs = {}

    if isinstance(act_space, gym.spaces.Box):
        env_wrapper = NormalizedBoxEnv

    env = env_wrapper(env, **wrapper_kwargs)

    kwargs = {}
    if "vec_env_kwargs" in env_specs:
        kwargs = env_specs["vec_env_kwargs"]

    training_env = get_envs(env_specs, env_wrapper, wrapper_kwargs, **kwargs)
    training_env.seed(env_specs["training_env_seed"])

    obs_dim = obs_space.shape[0]
    action_dim = act_space.shape[0]

    net_size = variant["net_size"]
    num_hidden = variant["num_hidden_layers"]
    qf1 = FlattenMlp(
        hidden_sizes=num_hidden * [net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=num_hidden * [net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    policy = MlpGaussianNoisePolicy(
        hidden_sizes=num_hidden * [net_size],
        obs_dim=obs_dim,
        action_dim=action_dim,
        output_activation=tanh,
        policy_noise=variant["policy_noise"],
        policy_noise_clip=variant["policy_noise_clip"],
    )

    trainer = TD3(policy=policy, qf1=qf1, qf2=qf2, **variant["td3_params"])
    algorithm = TorchRLAlgorithm(
        trainer=trainer,
        env=env,
        training_env=training_env,
        exploration_policy=policy,
        **variant["rl_alg_params"]
    )

    epoch = 0
    if "load_params" in variant:
        algorithm, epoch = load_from_file(algorithm, **variant["load_params"])

    if ptu.gpu_enabled():
        algorithm.to(ptu.device)

    algorithm.train(start_epoch=epoch)

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
        exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs, seed=seed, log_dir=log_dir
    )

    experiment(exp_specs)
