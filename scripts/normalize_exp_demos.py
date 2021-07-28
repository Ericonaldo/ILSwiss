"""
The normalization I'm using here is different than the one for the meta version
"""
import numpy as np
import joblib
import yaml
import os
from os import path as osp

from rlkit.core.vistools import plot_histogram
from rlkit.launchers import config


EXPERT_LISTING_YAML_PATH = "expert_demos_listing.yaml"

NORMALIZE_OBS = True
NORMALIZE_ACTS = False


def get_normalized(data, size, mean=None, std=None, return_stats=False):
    print("\n\nSIZE IS: %d" % size)
    if mean is None:
        mean = np.mean(data[:size], axis=0, keepdims=True)
    if std is None:
        std = np.std(data[:size], axis=0, keepdims=True)
        # check for a pathology where some axes are constant
        std = np.where(std == 0, np.ones(std.shape), std)

    if return_stats:
        return (data - mean) / std, mean, std
    return (data - mean) / std


def do_the_thing(data_path, save_path, plot_obs_histogram=False):
    d = joblib.load(data_path)
    d["obs_mean"] = None
    d["obs_std"] = None
    d["acts_mean"] = None
    d["acts_std"] = None
    if NORMALIZE_OBS:
        print(d["train"]._size)
        print(d["train"]._top)
        print(np.max(d["train"]._observations[: d["train"]._size]))
        print(np.min(d["train"]._observations[: d["train"]._size]))
        print(d["train"]._observations.shape)
        if plot_obs_histogram:
            for i in range(d["train"]._observations.shape[1]):
                if i % 4 != 0:
                    continue
                print(i)
                plot_histogram(
                    d["train"]._observations[: d["train"]._size, i],
                    100,
                    "obs %d" % i,
                    "plots/junk_histos/obs_%d.png" % i,
                )
        d["train"]._observations, mean, std = get_normalized(
            d["train"]._observations, d["train"]._size, return_stats=True
        )
        d["train"]._next_obs = get_normalized(
            d["train"]._next_obs, d["train"]._size, mean=mean, std=std
        )
        d["test"]._observations = get_normalized(
            d["test"]._observations, d["test"]._size, mean=mean, std=std
        )
        d["test"]._next_obs = get_normalized(
            d["test"]._next_obs, d["test"]._size, mean=mean, std=std
        )
        d["obs_mean"] = mean
        d["obs_std"] = std
        print(np.max(d["train"]._observations[: d["train"]._size]))
        print(np.min(d["train"]._observations[: d["train"]._size]))
        print("\nObservations:")
        print("Mean:")
        print(mean)
        print("Std:")
        print(std)

        print("\nPost Normalization Check")
        print("Train")
        print("Obs")
        print(np.mean(d["train"]._observations, axis=0))
        print(np.std(d["train"]._observations, axis=0))
        print(np.max(d["train"]._observations, axis=0))
        print(np.min(d["train"]._observations, axis=0))
        print("Next Obs")
        print(np.mean(d["train"]._next_obs, axis=0))
        print(np.std(d["train"]._next_obs, axis=0))
        print(np.max(d["train"]._next_obs, axis=0))
        print(np.min(d["train"]._next_obs, axis=0))
        print("Test")
        print("Obs")
        print(np.mean(d["test"]._observations, axis=0))
        print(np.std(d["test"]._observations, axis=0))
        print(np.max(d["test"]._next_obs, axis=0))
        print(np.min(d["test"]._next_obs, axis=0))
        print("Next Obs")
        print(np.mean(d["test"]._next_obs, axis=0))
        print(np.std(d["test"]._next_obs, axis=0))
        print(np.max(d["test"]._next_obs, axis=0))
        print(np.min(d["test"]._next_obs, axis=0))
    if NORMALIZE_ACTS:
        raise NotImplementedError("Must take into account d['train']._size")
        # d['train']._actions, mean, std = get_normalized(d['train']._actions, return_stats=True)
        # d['test']._actions = get_normalized(d['test']._actions, mean=mean, std=std)
        # d['acts_mean'] = mean
        # d['acts_std'] = std
        # print('\nActions:')
        # print('Mean:')
        # print(mean)
        # print('Std:')
        # print(std)

    print(save_path)
    joblib.dump(d, osp.join(save_path), compress=3)


with open(EXPERT_LISTING_YAML_PATH, "r") as f:
    listings = yaml.load(f.read())

for i, expert in enumerate(["hopper_mul_4_demos_sub_20"]):
    data_path = osp.join(listings[expert]["file_paths"][0])
    save_dir = osp.join(config.LOCAL_LOG_DIR, "norm_" + expert)
    os.makedirs(save_dir, exist_ok=True)
    save_path = osp.join(save_dir, "expert_demos.pkl")
    do_the_thing(data_path, save_path)

print("\n\nRemember to add the new normalized demos to your expert listings!\n\n")
