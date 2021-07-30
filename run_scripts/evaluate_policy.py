import yaml
import argparse
import joblib
import numpy as np
import os, sys, inspect
import pickle, random
from pathlib import Path

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(sys.path)

from gym.spaces import Dict
from rlkit.envs import get_env

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.core import eval_util

from rlkit.envs.wrappers import ScaledEnv
from rlkit.samplers import PathSampler
from rlkit.torch.common.policies import (
    MakeDeterministic,
    ReparamTanhMultivariateGaussianLfOPolicy,
)
from .video import save_video


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

    obs_dim = obs_space.shape[0]
    action_dim = act_space.shape[0]

    if variant["scale_env_with_demo_stats"]:
        with open("expert_demos_listing.yaml", "r") as f:
            listings = yaml.load(f.read())
        expert_demos_path = listings[variant["expert_name"]]["file_paths"][
            variant["expert_idx"]
        ]
        buffer_save_dict = joblib.load(expert_demos_path)
        env = ScaledEnv(
            env,
            obs_mean=buffer_save_dict["obs_mean"],
            obs_std=buffer_save_dict["obs_std"],
            acts_mean=buffer_save_dict["acts_mean"],
            acts_std=buffer_save_dict["acts_std"],
        )

    net_size = variant["policy_net_size"]
    num_hidden = variant["policy_num_hidden_layers"]

    policy = joblib.load(variant["policy_checkpoint"])["exploration_policy"][0]

    if variant["eval_deterministic"]:
        policy = MakeDeterministic(policy)
    policy.to(ptu.device)

    eval_sampler = PathSampler(
        env,
        policy,
        variant["num_eval_steps"],
        variant["max_path_length"],
        no_terminal=variant["no_terminal"],
        render=variant["render"],
        render_kwargs=variant["render_kwargs"],
        render_mode=variant["render_mode"],
    )
    test_paths = eval_sampler.obtain_samples()
    average_returns = eval_util.get_average_returns(test_paths)
    std_returns = eval_util.get_std_returns(test_paths)
    print(average_returns, std_returns)

    if variant["render"] and variant["render_mode"] == "rgb_array":
        video_path = variant["video_path"]
        video_path = os.path.join(video_path, variant["env_specs"]["env_name"])

        print("saving videos...")
        for i, test_path in enumerate(test_paths):
            images = np.stack(test_path["image"], axis=0)
            fps = 1 // getattr(env, "dt", 1 / 30)
            video_save_path = os.path.join(video_path, f"episode_{i}.mp4")
            save_video(images, video_save_path, fps=fps)

    return average_returns, std_returns, test_paths


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", help="experiment specification file")
    parser.add_argument("-g", "--gpu", help="gpu id", type=int, default=0)
    parser.add_argument(
        "-s", "--save_res", help="save result to file", type=int, default=1
    )

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
        ptu.set_gpu_mode(True)
    exp_id = exp_specs["exp_id"]
    exp_prefix = exp_specs["exp_name"]

    seed = exp_specs["seed"]
    set_seed(seed)
    # setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs)

    train_file = (
        exp_specs["method"] + "-" + exp_specs["env_specs"]["env_name"] + "-STANDARD-EXP"
    )
    pkl_name = "/best.pkl"

    if "invdoublependulum" in exp_specs["env_specs"]["env_name"]:
        pkl_name = "/params.pkl"

    if "gail" in exp_specs["method"]:
        if "hopper" in exp_specs["env_specs"]["env_name"]:
            train_file = "gail-hopper--rs-2.0"
        elif "walker" in exp_specs["env_specs"]["env_name"]:
            train_file = "gail-walker--rs-2.0"

    train_files = [train_file]
    save_path = "./final_performance/"

    for train_file in train_files:
        res_files = os.listdir("./logs/" + train_file)
        test_paths_all = []
        for file_ in res_files:
            exp_specs["policy_checkpoint"] = (
                "./logs/" + train_file + "/" + file_ + pkl_name
            )
            flag = False
            if "_lfo" in file_:
                flag = True
            average_returns, std_returns, test_paths = experiment(exp_specs, flag)
            test_paths_all.extend(test_paths)

            if args.save_res:
                save_dir = Path(save_path + train_file)
                save_dir.mkdir(exist_ok=True, parents=True)
                file_dir = save_dir.joinpath(
                    exp_specs["method"], exp_specs["env_specs"]["env_name"]
                )
                file_dir.mkdir(exist_ok=True, parents=True)

                if not os.path.exists(file_dir.joinpath("res.csv")):
                    with open(
                        save_dir.joinpath(
                            exp_specs["method"],
                            exp_specs["env_specs"]["env_name"],
                            "res.csv",
                        ),
                        "w",
                    ) as f:
                        f.write("avg,std\n")
                with open(
                    save_dir.joinpath(
                        exp_specs["method"],
                        exp_specs["env_specs"]["env_name"],
                        "res.csv",
                    ),
                    "a",
                ) as f:
                    f.write("{},{}\n".format(average_returns, std_returns))
        if exp_specs["save_samples"]:
            with open(
                Path(save_path).joinpath(
                    exp_specs["method"],
                    exp_specs["env_specs"]["env_name"],
                    "samples.pkl",
                ),
                "wb",
            ) as f:
                pickle.dump(test_paths_all, f)
