import yaml
import argparse
import os, sys, inspect
import gym

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(sys.path)

from rlkit.envs import get_env, get_envs
from rlkit.envs.wrappers import NormalizedBoxEnv, ProxyEnv
from rlkit.envs.terminals import get_terminal_func
from rlkit.core.logger import load_from_file
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.torch.common.networks import BNN, FlattenMlp
from rlkit.torch.common.policies import ReparamTanhMultivariateGaussianPolicy
from rlkit.torch.algorithms.sac.sac_alpha import SoftActorCritic
from rlkit.torch.algorithms.mbpo.bnn_trainer import BNNTrainer
from rlkit.torch.algorithms.mbpo.mbpo import MBPO
import rlkit.torch.utils.pytorch_util as ptu


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
        kwargs = env_specs["env_kwargs"]["vec_env_kwargs"]

    training_env = get_envs(env_specs, env_wrapper, wrapper_kwargs, **kwargs)
    training_env.seed(env_specs["training_env_seed"])

    obs_dim = obs_space.shape[0]
    action_dim = act_space.shape[0]

    # SAC params
    sac_params = variant["sac_params"]
    sac_net_size = sac_params["net_size"]
    sac_num_hidden = sac_params["num_hidden_layers"]
    qf1 = FlattenMlp(
        hidden_sizes=sac_num_hidden * [sac_net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=sac_num_hidden * [sac_net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    policy = ReparamTanhMultivariateGaussianPolicy(
        hidden_sizes=sac_num_hidden * [sac_net_size],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    sac_trainer = SoftActorCritic(
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        env=env,
        **sac_params,
    )

    # BNN params
    bnn_params = variant["bnn_params"]
    bnn_net_size = bnn_params["net_size"]
    bnn_num_hidden = bnn_params["num_hidden_layers"]
    bnn_num_nets = bnn_params["num_nets"]
    rew_dim = 1
    bnn = BNN(
        hidden_sizes=bnn_num_hidden * [bnn_net_size],
        input_size=obs_dim + action_dim,
        output_size=obs_dim + rew_dim,
        num_nets=bnn_num_nets,
    )
    bnn_trainer = BNNTrainer(bnn=bnn, **bnn_params)

    # MBPO params
    is_terminal = get_terminal_func(env_specs["env_name"])
    algorithm = MBPO(
        env=env,
        training_env=training_env,
        model=bnn_trainer,
        algo=sac_trainer,
        exploration_policy=policy,
        is_terminal=is_terminal,
        **variant["mbpo_params"],
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
