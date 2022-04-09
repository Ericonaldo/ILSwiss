import yaml
import argparse
import numpy as np
import os
import sys
import inspect
import random
import pickle

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(sys.path)

import gym
from rlkit.envs import get_env, get_envs

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.torch.common.networks import FlattenMlp
from rlkit.torch.common.policies import ReparamTanhMultivariateGaussianPolicy
from rlkit.torch.algorithms.opolo.opolo import OPOLO
from rlkit.torch.algorithms.adv_irl.disc_models.simple_disc_models import MLPDisc
from rlkit.envs.wrappers import ProxyEnv, ScaledEnv, MinmaxEnv, NormalizedBoxEnv, EPS


def experiment(variant):
    with open("demos_listing.yaml", "r") as f:
        listings = yaml.safe_load(f.read())

    demos_path = listings[variant["expert_name"]]["file_paths"][variant["expert_idx"]]
    print("demos_path", demos_path)
    with open(demos_path, "rb") as f:
        traj_list = pickle.load(f)
    traj_list = random.sample(traj_list, variant["traj_num"])

    obs = np.vstack([traj_list[i]["observations"] for i in range(len(traj_list))])
    # acts = np.vstack([traj_list[i]["actions"] for i in range(len(traj_list))])
    obs_mean, obs_std = np.mean(obs, axis=0), np.std(obs, axis=0)
    # acts_mean, acts_std = np.mean(acts, axis=0), np.std(acts, axis=0)
    acts_mean, acts_std = None, None
    obs_min, obs_max = np.min(obs, axis=0), np.max(obs, axis=0)

    # print("obs:mean:{}".format(obs_mean))
    # print("obs_std:{}".format(obs_std))
    # print("acts_mean:{}".format(acts_mean))
    # print("acts_std:{}".format(acts_std))

    env_specs = variant["env_specs"]
    env = get_env(env_specs)
    env.seed(env_specs["eval_env_seed"])

    print("\n\nEnv: {}".format(env_specs["env_name"]))
    print("kwargs: {}".format(env_specs["env_kwargs"]))
    print("Obs Space: {}".format(env.observation_space))
    print("Act Space: {}\n\n".format(env.action_space))

    expert_replay_buffer = EnvReplayBuffer(
        variant["opolo_params"]["replay_buffer_size"],
        env,
        random_seed=np.random.randint(10000),
    )

    env_wrapper = ProxyEnv  # Identical wrapper
    kwargs = {}
    wrapper_kwargs = {}

    if variant["scale_env_with_demo_stats"]:
        print("\nWARNING: Using scale env wrapper")
        env_wrapper = ScaledEnv
        wrapper_kwargs = dict(
            obs_mean=obs_mean,
            obs_std=obs_std,
            acts_mean=acts_mean,
            acts_std=acts_std,
        )
        for i in range(len(traj_list)):
            traj_list[i]["observations"] = (traj_list[i]["observations"] - obs_mean) / (
                obs_std + EPS
            )
            traj_list[i]["next_observations"] = (
                traj_list[i]["next_observations"] - obs_mean
            ) / (obs_std + EPS)

    elif variant["minmax_env_with_demo_stats"]:
        print("\nWARNING: Using min max env wrapper")
        env_wrapper = MinmaxEnv
        wrapper_kwargs = dict(obs_min=obs_min, obs_max=obs_max)
        for i in range(len(traj_list)):
            traj_list[i]["observations"] = (traj_list[i]["observations"] - obs_min) / (
                obs_max - obs_min + EPS
            )
            traj_list[i]["next_observations"] = (
                traj_list[i]["next_observations"] - obs_min
            ) / (obs_max - obs_min + EPS)

    obs_space = env.observation_space
    act_space = env.action_space
    assert not isinstance(obs_space, gym.spaces.Dict)
    assert len(obs_space.shape) == 1
    assert len(act_space.shape) == 1

    if isinstance(act_space, gym.spaces.Box) and (
        (acts_mean is None) and (acts_std is None)
    ):
        print("\nWARNING: Using Normalized Box Env wrapper")
        _env_wrapper = env_wrapper
        env_wrapper = lambda *args, **kwargs: NormalizedBoxEnv(
            _env_wrapper(*args, **kwargs)
        )

    env = env_wrapper(env, **wrapper_kwargs)
    training_env = get_envs(
        env_specs, env_wrapper, wrapper_kwargs=wrapper_kwargs, **kwargs
    )
    training_env.seed(env_specs["training_env_seed"])

    for i in range(len(traj_list)):
        expert_replay_buffer.add_path(traj_list[i], env=env)

    obs_dim = obs_space.shape[0]
    action_dim = act_space.shape[0]

    # build the discriminator model
    disc_model = MLPDisc(
        obs_dim * 2,
        num_layer_blocks=variant["disc_num_blocks"],
        hid_dim=variant["disc_hid_dim"],
        hid_act=variant["disc_hid_act"],
        use_bn=variant["disc_use_bn"],
        clamp_magnitude=variant["disc_clamp_magnitude"],
    )

    # build the policy models
    net_size = variant["net_size"]
    num_hidden = variant["num_hidden_layers"]
    nu_net = FlattenMlp(
        hidden_sizes=num_hidden * [net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )

    policy = ReparamTanhMultivariateGaussianPolicy(
        hidden_sizes=num_hidden * [net_size],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )

    # set up the algorithm
    algorithm = OPOLO(
        env=env,
        training_env=training_env,
        exploration_policy=policy,
        discriminator=disc_model,
        nu_net=nu_net,
        expert_replay_buffer=expert_replay_buffer,
        **variant["opolo_params"],
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

    exp_suffix = ""
    exp_suffix = "--gp-{}--trajnum-{}".format(
        exp_specs["opolo_params"]["grad_pen_weight"],
        format(exp_specs["traj_num"]),
    )

    if not exp_specs["opolo_params"]["no_terminal"]:
        exp_suffix = "--terminal" + exp_suffix

    if exp_specs["scale_env_with_demo_stats"]:
        exp_suffix = "--scale" + exp_suffix

    if exp_specs["using_gpus"] > 0:
        print("\n\nUSING GPU\n\n")
        ptu.set_gpu_mode(True, args.gpu)

    exp_id = exp_specs["exp_id"]
    exp_prefix = exp_specs["exp_name"]
    seed = exp_specs["seed"]
    set_seed(seed)

    exp_prefix = exp_prefix + exp_suffix
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs, seed=seed)

    experiment(exp_specs)
