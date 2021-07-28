import yaml
import argparse
import os
from os import path
import joblib
from random import randint
import os, sys, inspect

import pyglet

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(sys.path)


import numpy as np
from time import time
import rlkit.torch.utils.pytorch_util as ptu

from rlkit.core import logger
from rlkit.launchers.launcher_util import setup_logger, set_seed

from rlkit.envs import get_env
from rlkit.scripted_experts import get_scripted_policy
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.data_management.path_builder import PathBuilder
from rlkit.torch.common.policies import MakeDeterministic

from gym.wrappers.monitor import Monitor


def fill_buffer(
    buffer,
    env,
    expert_policy,
    num_rollouts,
    max_path_length,
    no_terminal=False,
    policy_is_scripted=False,
    render=False,
    render_kwargs={},
    check_for_success=False,
    wrap_absorbing=False,
    subsample_factor=1,
):
    num_rollouts_completed = 0
    total_rewards = 0.0

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

            # get the action
            if policy_is_scripted:
                action, agent_info = expert_policy.get_action(
                    observation, env, len(cur_path_builder)
                )
            else:
                action, agent_info = expert_policy.get_action(observation)

            next_ob, reward, terminal, env_info = env.step(action)
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


def experiment(specs):
    if not specs["use_scripted_policy"]:
        policy_is_scripted = False
        policy = joblib.load(specs["expert_path"])["policy"]
    else:
        policy_is_scripted = True
        policy = get_scripted_policy(specs["scripted_policy_name"])

    if specs["use_deterministic_expert"]:
        policy = MakeDeterministic(policy)
    if ptu.gpu_enabled() and exp_specs["num_gpu_per_worker"] > 0:
        policy.to(ptu.device)

    env = get_env(specs["env_specs"])
    env.seed(specs["env_specs"]["env_seed"])

    # env = Monitor(env, './videos/' + str(time()) + '/')

    # make the replay buffers
    max_path_length = specs["max_path_length"]
    if "wrap_absorbing" in specs and specs["wrap_absorbing"]:
        """
        There was an intial implementation for this in v1.0
        in gen_irl_expert_trajs.py
        """
        raise NotImplementedError()
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

    print("\n")
    # fill the train buffer
    fill_buffer(
        train_buffer,
        env,
        policy,
        specs["num_rollouts"],
        max_path_length,
        no_terminal=specs["no_terminal"],
        policy_is_scripted=policy_is_scripted,
        render=render,
        render_kwargs=render_kwargs,
        check_for_success=check_for_success,
        wrap_absorbing=False,
        subsample_factor=specs["subsample_factor"],
    )

    # fill the test buffer
    fill_buffer(
        test_buffer,
        env,
        policy,
        specs["num_rollouts"],
        max_path_length,
        no_terminal=specs["no_terminal"],
        policy_is_scripted=policy_is_scripted,
        render=render,
        render_kwargs=render_kwargs,
        check_for_success=check_for_success,
        wrap_absorbing=False,
        subsample_factor=specs["subsample_factor"],
    )

    # save the replay buffers
    logger.save_extra_data(
        {"train": train_buffer, "test": test_buffer}, name="expert_demos.pkl"
    )

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
