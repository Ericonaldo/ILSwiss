import argparse
import joblib

from rlkit.torch.common.policies import MakeDeterministic

# assert False, 'I have not successfully used or tested this yet'


def experiment(checkpoint, deterministic=False):
    d = joblib.load(checkpoint)
    print("Epoch = %d" % d["epoch"])

    print(d)

    algorithm = d["algorithm"]
    algorithm.render = True
    print(algorithm.discriminator)

    if deterministic:
        algorithm.exploration_policy = MakeDeterministic(algorithm.exploration_policy)

    # print(algorithm.grad_pen_weight)

    # algorithm.do_not_train = True
    # algorithm.do_not_eval = True
    # for i in range(100):
    #     algorithm.generate_exploration_rollout()

    algorithm.num_steps_between_updates = 1000000
    algorithm.train_online()

    return 1


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", help="experiment specification file")
    parser.add_argument(
        "-d",
        "--deterministic",
        help="make the policy deterministic",
        action="store_true",
    )
    args = parser.parse_args()

    experiment(args.checkpoint, deterministic=args.deterministic)
