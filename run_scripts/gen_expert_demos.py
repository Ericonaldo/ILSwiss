import yaml
import argparse
import joblib
import pickle
from random import randint
import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(sys.path)

import numpy as np
import rlkit.torch.utils.pytorch_util as ptu

from rlkit.core import logger
from rlkit.launchers.launcher_util import setup_logger, set_seed

from rlkit.envs import get_env
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.data_management.path_builder import PathBuilder
from rlkit.torch.common.policies import MakeDeterministic
from rlkit.envs.wrappers import FrameStackEnv, ProxyEnv
import rlkit.data_management.data_augmentation as rad

import torch

# from gym.wrappers.monitor import Monitor


def fill_buffer(
    buffer,
    env,
    expert_policy,
    num_rollouts,
    max_path_length,
    eval_preprocess_func=None,
    no_terminal=False,
    policy_is_scripted=False,
    render=False,
    render_kwargs={},
    check_for_success=False,
    check_for_return=False,
    wrap_absorbing=False,
    subsample_factor=1,
):
    num_rollouts_completed = 0
    total_rewards = 0.0

    res_data = []

    while num_rollouts_completed < num_rollouts:
        print("Rollout %d..." % num_rollouts_completed)

        cur_path_builder = PathBuilder()

        observation = env.reset()
        if policy_is_scripted:
            expert_policy.reset(env)

        # if subsampling what offset do you want to use
        subsample_mod = randint(0, subsample_factor - 1)
        rewards_for_rollout = 0.0
        printed_target_dist = False
        step_num = 0
        terminal = False
        while (not terminal) and step_num < max_path_length:
            if render:
                env.render(**render_kwargs)
            if eval_preprocess_func:
                observation = eval_preprocess_func(observation)

            # get the action
            if policy_is_scripted:
                action, agent_info = expert_policy.get_action(
                    observation, env, len(cur_path_builder)
                )
            else:
                action, agent_info = expert_policy.get_action(observation)

            next_ob, reward, terminal, env_info = env.step(action)
            if "is_success" in env_info.keys():
                terminal = env_info["is_success"]
            if no_terminal:
                terminal = False
            terminal_array = np.array([terminal])

            rewards_for_rollout += reward
            reward = np.array([reward])

            if step_num % subsample_factor == subsample_mod:
                cur_path_builder.add_all(
                    observations=observation,
                    actions=action,
                    rewards=reward,
                    next_observations=next_ob,
                    terminals=terminal_array,
                    absorbing=np.array([0.0, 0.0]),
                    agent_infos=agent_info,
                    env_infos=env_info,
                )
            observation = next_ob
            step_num += 1

        print("\tNum Steps: %d" % step_num)
        print("\tReturns: %.2f" % rewards_for_rollout)

        # if necessary check if it was successful
        if check_for_return:
            if rewards_for_rollout <= 0:
                print("\tFail! Skip")
                continue

        # if necessary check if it was successful
        if check_for_success:
            was_successful = (
                np.sum(
                    [e_info["is_success"] for e_info in cur_path_builder["env_infos"]]
                )
                > 0
            )
            if was_successful:
                print("\tSuccessful")
            else:
                print("\tNot Successful")

        # add the path to the buffer
        if (check_for_success and was_successful) or (not check_for_success):
            res_data.append(cur_path_builder)
            for timestep in range(len(cur_path_builder)):
                buffer.add_sample(
                    cur_path_builder["observations"][timestep],
                    cur_path_builder["actions"][timestep],
                    cur_path_builder["rewards"][timestep],
                    cur_path_builder["terminals"][timestep],
                    cur_path_builder["next_observations"][timestep],
                    agent_info=cur_path_builder["agent_infos"][timestep],
                    env_info=cur_path_builder["env_infos"][timestep],
                    absorbing=cur_path_builder["absorbing"][timestep],
                )
            buffer.terminate_episode()
            num_rollouts_completed += 1
            total_rewards += rewards_for_rollout

    print("\nAverage Episode Return: %f\n" % (total_rewards / num_rollouts_completed))
    return res_data


def experiment(specs):
    if not specs["use_scripted_policy"]:
        policy_is_scripted = False
        policy = joblib.load(specs["expert_path"])["policy"]
    else:
        policy_is_scripted = True
        policy = get_scripted_policy(specs["scripted_policy_name"])

    if specs["use_deterministic_expert"]:
        policy = MakeDeterministic(policy)
    if specs["using_gpus"] > 0:
        policy.to(ptu.device)

    env_specs = specs["env_specs"]
    eval_preprocess_func = None
    if env_specs["env_name"] == "dmc":
        os.environ["LD_LIBRARY_PATH"] = "～/.mujoco/mjpro210/bin"
        os.environ["MUJOCO_GL"] = "egl"

    if "augmentation_params" in specs:  # Use rad augmentation, record important params
        cpc = False
        if "cpc" in specs["augmentation_params"]:
            cpc = True
        data_augs = ""
        if "data_augs" in specs["augmentation_params"]:
            data_augs = specs["augmentation_params"]["data_augs"]
        image_size = specs["augmentation_params"]["image_size"]
        # pre_transform_image_size = (
        #     specs["augmentation_params"]["pre_transform_image_size"]
        #     if "crop" in data_augs
        #     else specs["augmentation_params"]["image_size"]
        # ) # Currently with bugs
        pre_transform_image_size = specs["augmentation_params"]["image_size"]
        pre_image_size = specs["augmentation_params"][
            "pre_transform_image_size"
        ]  # record the pre transform image size for translation
        env_specs["env_kwargs"]["width"] = env_specs["env_kwargs"][
            "height"
        ] = pre_transform_image_size  # The env create as the shape before transformed

        # preprocess obs func for eval
        if "crop" in data_augs:
            eval_preprocess_func = lambda x: rad.center_crop_image(x, image_size)
        if "translate" in data_augs:
            # first crop the center with pre_image_size
            crop_func = lambda x: rad.center_crop_image(x, pre_transform_image_size)
            # then translate cropped to center
            eval_preprocess_func = lambda x: rad.center_translate(
                crop_func(x), image_size
            )

    # make all seeds the same.
    env_specs["env_seed"] = specs["seed"]

    env = get_env(env_specs)
    env.seed(env_specs["env_seed"])

    env_wrapper = ProxyEnv  # Identical wrapper
    wrapper_kwargs = {}
    encoder = None
    if ("frame_stack" in env_specs) and (env_specs["frame_stack"] is not None):
        env_wrapper = FrameStackEnv
        wrapper_kwargs = {"k": env_specs["frame_stack"]}
        # try:
        #     encoder = joblib.load(specs["expert_path"])["encoder"]
        # except:
        #     print("No encoder loaded!")
        #     exit(0)

    env = env_wrapper(env, **wrapper_kwargs)

    # env = Monitor(env, './videos/' + str(time()) + '/')

    # make the replay buffers
    max_path_length = specs["max_path_length"]
    if "wrap_absorbing" in specs and specs["wrap_absorbing"]:
        """
        There was an intial implementation for this in v1.0
        in gen_irl_expert_trajs.py
        """
        _max_buffer_size = (max_path_length + 2) * specs["num_rollouts"]
    else:
        _max_buffer_size = max_path_length * specs["num_rollouts"]
    _max_buffer_size = int(np.ceil(_max_buffer_size / float(specs["subsample_factor"])))

    buffer_constructor = lambda: EnvReplayBuffer(
        _max_buffer_size,
        env,
    )

    train_buffer = buffer_constructor()
    test_buffer = buffer_constructor()

    render = specs["render"]
    render_kwargs = specs["render_kwargs"]
    check_for_success = specs["check_for_success"]
    check_for_return = specs["check_for_return"]
    num_rollouts = specs["num_rollouts"]

    print("\n")
    # fill the train buffer
    res = fill_buffer(
        train_buffer,
        env,
        policy,
        num_rollouts,
        max_path_length,
        eval_preprocess_func=eval_preprocess_func,
        no_terminal=specs["no_terminal"],
        policy_is_scripted=policy_is_scripted,
        render=render,
        render_kwargs=render_kwargs,
        check_for_success=check_for_success,
        wrap_absorbing=False,
        subsample_factor=specs["subsample_factor"],
    )

    if specs["save_buffer"]:
        # fill the test buffer
        fill_buffer(
            test_buffer,
            env,
            policy,
            num_rollouts,
            max_path_length,
            eval_preprocess_func=eval_preprocess_func,
            no_terminal=specs["no_terminal"],
            policy_is_scripted=policy_is_scripted,
            render=render,
            render_kwargs=render_kwargs,
            check_for_success=check_for_success,
            wrap_absorbing=False,
            subsample_factor=specs["subsample_factor"],
        )

    if specs["save_buffer"]:
        # save the replay buffers
        logger.save_extra_data(
            {"train": train_buffer, "test": test_buffer}, name="expert_demos_buffer.pkl"
        )

    env_name = specs["env_specs"]["env_name"]
    if env_name == "dmc":
        env_name = (
            env_specs["env_kwargs"]["domain_name"]
            + "_"
            + env_specs["env_kwargs"]["task_name"]
        )
    if not os.path.exists("./demos/{}/seed-{}".format(env_name, specs["seed"])):
        os.makedirs("./demos/{}/seed-{}".format(env_name, specs["seed"]))
    # save demos directly
    with open(
        "./demos/{}/seed-{}/expert_demos-{}.pkl".format(
            env_name, specs["seed"], num_rollouts
        ),
        "wb",
    ) as f:
        pickle.dump(res, f)

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
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs)

    experiment(exp_specs)
