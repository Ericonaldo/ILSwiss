import yaml
import argparse
import joblib
import numpy as np
import os, sys, inspect
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
from rlkit.torch.algorithms.sac.sac_alpha import (
    SoftActorCritic,
)  # SAC Auto alpha version
from rlkit.torch.algorithms.adv_irl.disc_models.simple_disc_models import MLPDisc
from rlkit.torch.algorithms.adv_irl.adv_irl import AdvIRL
from rlkit.envs.wrappers import ProxyEnv, ScaledEnv, MinmaxEnv, NormalizedBoxEnv


def experiment(variant):
    with open("demos_listing.yaml", "r") as f:
        listings = yaml.load(f.read())

    demos_path = listings[variant["expert_name"]]["file_paths"][variant["expert_idx"]]
    """
    Buffer input format
    """
    # buffer_save_dict = joblib.load(expert_demos_path)
    # expert_replay_buffer = buffer_save_dict['train']
    # obs_mean, obs_std = buffer_save_dict['obs_mean'], buffer_save_dict['obs_std']
    # acts_mean, acts_std = buffer_save_dict['acts_mean'], buffer_save_dict['acts_std']
    # obs_min, obs_max = buffer_save_dict['obs_min'], buffer_save_dict['obs_max']
    # if 'minmax_env_with_demo_stats' in variant.keys():
    #     if (variant['minmax_env_with_demo_stats']) and not (variant['scale_env_with_demo_stats']):
    #         assert 'norm_train' in buffer_save_dict.keys()
    #         expert_replay_buffer = buffer_save_dict['norm_train']
    """
    PKL input format
    """
    print("demos_path", demos_path)
    with open(demos_path, "rb") as f:
        traj_list = pickle.load(f)
    traj_list = random.sample(traj_list, variant["traj_num"])

    obs = np.vstack([traj_list[i]["observations"] for i in range(len(traj_list))])
    acts = np.vstack([traj_list[i]["actions"] for i in range(len(traj_list))])
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
        variant["adv_irl_params"]["replay_buffer_size"],
        env,
        random_seed=np.random.randint(10000),
    )

    for i in range(len(traj_list)):
        expert_replay_buffer.add_path(
            traj_list[i], absorbing=variant["adv_irl_params"]["wrap_absorbing"], env=env
        )

    tmp_env_wrapper = env_wrapper = ProxyEnv  # Identical wrapper
    kwargs = {}

    if variant["scale_env_with_demo_stats"]:
        print("\nWARNING: Using scale env wrapper")
        tmp_env_wrapper = env_wrapper = ScaledEnv
        kwargs = dict(
            obs_mean=obs_mean,
            obs_std=obs_std,
            acts_mean=acts_mean,
            acts_std=acts_std,
        )
    elif variant["minmax_env_with_demo_stats"]:
        print("\nWARNING: Using min max env wrapper")
        tmp_env_wrapper = env_wrapper = MinmaxEnv
        kwargs = dict(obs_min=obs_min, obs_max=obs_max)

    obs_space = env.observation_space
    act_space = env.action_space
    assert not isinstance(obs_space, gym.spaces.Dict)
    assert len(obs_space.shape) == 1
    assert len(act_space.shape) == 1

    if isinstance(act_space, gym.spaces.Box) and (
        (acts_mean is None) and (acts_std is None)
    ):
        print("\nWARNING: Using Normalized Box Env wrapper")
        env_wrapper = lambda *args, **kwargs: NormalizedBoxEnv(
            tmp_env_wrapper(*args, **kwargs)
        )

    env = env_wrapper(env, **kwargs)
    training_env = get_envs(env_specs, env_wrapper, **kwargs)
    training_env.seed(env_specs["training_env_seed"])

    obs_dim = obs_space.shape[0]
    action_dim = act_space.shape[0]

    # build the policy models
    net_size = variant["policy_net_size"]
    num_hidden = variant["policy_num_hidden_layers"]
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
    vf = FlattenMlp(
        hidden_sizes=num_hidden * [net_size],
        input_size=obs_dim,
        output_size=1,
    )
    policy = ReparamTanhMultivariateGaussianPolicy(
        hidden_sizes=num_hidden * [net_size],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )

    # build the discriminator model
    disc_model = MLPDisc(
        obs_dim + action_dim
        if not variant["adv_irl_params"]["state_only"]
        else 2 * obs_dim,
        num_layer_blocks=variant["disc_num_blocks"],
        hid_dim=variant["disc_hid_dim"],
        hid_act=variant["disc_hid_act"],
        use_bn=variant["disc_use_bn"],
        clamp_magnitude=variant["disc_clamp_magnitude"],
    )

    # set up the algorithm
    trainer = SoftActorCritic(
        policy=policy, qf1=qf1, qf2=qf2, vf=vf, env=env, **variant["sac_params"]
    )
    algorithm = AdvIRL(
        env=env,
        training_env=training_env,
        exploration_policy=policy,
        discriminator=disc_model,
        policy_trainer=trainer,
        expert_replay_buffer=expert_replay_buffer,
        **variant["adv_irl_params"]
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

    exp_suffix = ""
    exp_suffix = "--gp-{}--rs-{}--trajnum-{}".format(
        exp_specs["adv_irl_params"]["grad_pen_weight"],
        exp_specs["sac_params"]["reward_scale"],
        format(exp_specs["traj_num"]),
    )

    if not exp_specs["adv_irl_params"]["no_terminal"]:
        exp_suffix = "--terminal" + exp_suffix

    if exp_specs["adv_irl_params"]["wrap_absorbing"]:
        exp_suffix = "--absorbing" + exp_suffix

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
