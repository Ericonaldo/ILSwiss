import yaml
import argparse
import numpy as np
import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(sys.path)

import gym
from rlkit.envs import get_env, get_envs
from rlkit.envs.wrappers import ProxyEnv, FrameStackEnv

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.core.logger import load_from_file
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.torch.common.encoders import make_encoder, make_decoder
from rlkit.torch.common.networks import FlattenMlp
from rlkit.torch.common.policies import ReparamTanhMultivariateGaussianEncoderPolicy
from rlkit.torch.algorithms.sac.sac_ae import SoftActorCritic
from rlkit.torch.algorithms.torch_rl_algorithm import TorchRLAlgorithm
from rlkit.data_management.aug_replay_buffer import AugmentCPCImageEnvReplayBuffer
import rlkit.data_management.data_augmentation as rad

os.environ["LD_LIBRARY_PATH"] = "~/.mujoco/mjpro210/bin"
os.environ["MUJOCO_GL"] = "egl"


def experiment(variant):
    env_specs = variant["env_specs"]
    replay_buffer = None
    eval_preprocess_func = None

    if (
        "augmentation_params" in variant
    ):  # Use rad augmentation, record important params
        cpc = False
        if "cpc" in variant["augmentation_params"]:
            cpc = True
        data_augs = ""
        if "data_augs" in variant["augmentation_params"]:
            data_augs = variant["augmentation_params"]["data_augs"]
        image_size = variant["augmentation_params"]["image_size"]  # this is the size for encoder
        pre_image_size = variant["augmentation_params"]["pre_transform_image_size"] # record the pre transform image size for translation
        pre_transform_image_size = variant["augmentation_params"][
            "pre_transform_image_size"
        ]  if 'crop' in data_augs else image_size # this is the render size and buffer save size, do all transform after this size
        env_specs["env_kwargs"]["width"] = env_specs["env_kwargs"][
            "height"
        ] = pre_transform_image_size  # The env create as the shape before transformed

        # preprocess obs func for eval
        if "crop" in data_augs:
            assert image_size < pre_transform_image_size, "crop need image_size < pre_transform_image_size!"
            eval_preprocess_func = lambda x: rad.center_crop_image(x, image_size) # require image_size < pre_transform_image_size, or it will cause error
        if "translate" in data_augs:
            # first crop the center with pre_image_size
            crop_func = lambda x: rad.center_crop_image(x, pre_image_size)
            # then translate cropped to center
            eval_preprocess_func = lambda x: rad.center_translate(
                crop_func(x), image_size
            )

    env = get_env(env_specs)
    env.seed(env_specs["eval_env_seed"])

    print("\n\nEnv: {}".format(env_specs["env_name"]))
    print("kwargs: {}".format(env_specs["env_kwargs"]))
    print("Obs Space: {}".format(env.observation_space))
    print("Act Space: {}\n\n".format(env.action_space))

    env_wrapper = ProxyEnv  # Identical wrapper
    wrapper_kwargs = {}
    kwargs = {}
    if ("frame_stack" in env_specs) and (env_specs["frame_stack"] is not None):
        env_wrapper = FrameStackEnv
        wrapper_kwargs = {"k": env_specs["frame_stack"]}

    env = env_wrapper(env, **wrapper_kwargs)
    if (
        "augmentation_params" in variant
    ):  # If use rad augmentation, create augmentation env buffer
        replay_buffer = AugmentCPCImageEnvReplayBuffer(
            max_replay_buffer_size=variant["rl_alg_params"]["replay_buffer_size"],
            env=env,
            random_seed=np.random.randint(10000),
            pre_image_size=pre_image_size,
            image_size=image_size,
            data_augs=data_augs,
            cpc=cpc,
        )

    obs_space = env.observation_space
    act_space = env.action_space
    assert not isinstance(obs_space, gym.spaces.Dict)
    assert len(obs_space.shape) == 3
    assert len(act_space.shape) == 1

    training_env = get_envs(
        env_specs, env_wrapper, wrapper_kwargs=wrapper_kwargs, **kwargs
    )
    training_env.seed(env_specs["training_env_seed"])

    obs_shape = obs_space.shape
    action_dim = act_space.shape[0]
    feature_dim = variant["encoder_params"]["encoder_feature_dim"]

    if (
        "augmentation_params" in variant
    ):  # If use rad augmentation, take the after transform size as the shape
        obs_shape = (obs_shape[0], image_size, image_size)

    net_size = variant["net_size"]
    num_hidden = variant["num_hidden_layers"]
    qf1 = FlattenMlp(
        hidden_sizes=num_hidden * [net_size],
        input_size=feature_dim + action_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=num_hidden * [net_size],
        input_size=feature_dim + action_dim,
        output_size=1,
    )
    encoder = make_encoder(
        variant["encoder_params"]["encoder_type"],
        obs_shape,
        feature_dim,
        variant["encoder_params"]["num_layers"],
        variant["encoder_params"]["num_filters"],
        output_logits=True,
    )
    decoder = make_decoder(
        variant["encoder_params"]["encoder_type"],
        obs_shape,
        feature_dim,
        variant["encoder_params"]["num_layers"],
        variant["encoder_params"]["num_filters"],
    )
    policy = ReparamTanhMultivariateGaussianEncoderPolicy(
        encoder=encoder,
        hidden_sizes=num_hidden * [net_size],
        obs_dim=feature_dim,
        action_dim=action_dim,
    )

    trainer = SoftActorCritic(
        encoder=encoder,
        decoder=decoder,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        env=env,
        **variant["sac_params"],
    )

    algorithm = TorchRLAlgorithm(
        trainer=trainer,
        env=env,
        training_env=training_env,
        exploration_policy=policy,
        replay_buffer=replay_buffer,
        eval_preprocess_func=eval_preprocess_func,
        **variant["rl_alg_params"],
    )

    epoch = 0
    if "load_params" in variant:
        algorithm, epoch = load_from_file(algorithm, **variant["load_params"])

    if ptu.gpu_enabled():
        algorithm.to(ptu.device)

    print("Start from epoch", epoch)
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
        exp_prefix=exp_prefix,
        exp_id=exp_id,
        variant=exp_specs,
        seed=seed,
        log_dir=log_dir,
    )

    experiment(exp_specs)
