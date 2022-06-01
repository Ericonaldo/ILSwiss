"""
Common evaluation utilities.
"""

from collections import OrderedDict
from numbers import Number
import os
import json

import numpy as np

from rlkit.core.vistools import plot_returns_on_same_plot, save_plot


def get_generic_path_information(paths, stat_prefix=""):
    """
    Get an OrderedDict with a bunch of statistic names and values.
    """
    statistics = OrderedDict()
    returns = [sum(path["rewards"]) for path in paths]
    # rewards = np.vstack([path["rewards"] for path in paths])
    rewards = np.concatenate([path["rewards"] for path in paths])
    statistics.update(
        create_stats_ordered_dict(
            "Rewards", rewards, stat_prefix=stat_prefix, always_show_all_stats=True
        )
    )
    statistics.update(
        create_stats_ordered_dict(
            "Returns", returns, stat_prefix=stat_prefix, always_show_all_stats=True
        )
    )
    if "is_success" in paths[0]["env_infos"][0].keys():
        acc_sum = [
            (np.sum([x["is_success"] for x in path["env_infos"]]) > 0).astype(float)
            for path in paths
        ]
        acc = np.sum(acc_sum) * 1.0 / len(paths)
        statistics.update(
            create_stats_ordered_dict(
                "Success Num",
                np.sum(acc_sum),
                stat_prefix=stat_prefix,
                always_show_all_stats=True,
            )
        )
        statistics.update(
            create_stats_ordered_dict(
                "Traj Num",
                len(paths),
                stat_prefix=stat_prefix,
                always_show_all_stats=True,
            )
        )
        statistics.update(
            create_stats_ordered_dict(
                "Success Rate", acc, stat_prefix=stat_prefix, always_show_all_stats=True
            )
        )
    actions = [path["actions"] for path in paths]
    # if isinstance(actions[0][0], np.ndarray):
    #     actions = np.vstack([path["actions"] for path in paths])
    # else:
    #     actions = np.hstack([path["actions"] for path in paths])
    statistics.update(
        create_stats_ordered_dict(
            "Actions", actions, stat_prefix=stat_prefix, always_show_all_stats=True
        )
    )
    statistics.update(
        create_stats_ordered_dict(
            "Ep. Len.",
            np.array([len(path["terminals"]) for path in paths]),
            stat_prefix=stat_prefix,
            always_show_all_stats=True,
        )
    )
    statistics["Num Paths"] = len(paths)

    return statistics


def get_average_returns(paths, std=False):
    returns = [sum(path["rewards"]) for path in paths]
    if std:
        return np.mean(returns), np.std(returns)

    return np.mean(returns)


def create_stats_ordered_dict(
    name,
    data,
    stat_prefix=None,
    always_show_all_stats=False,
    exclude_max_min=False,
):
    # print('\n<<<< STAT FOR {} {} >>>>'.format(stat_prefix, name))
    if stat_prefix is not None:
        name = "{} {}".format(stat_prefix, name)
    if isinstance(data, Number):
        # print('was a Number')
        return OrderedDict({name: data})

    if len(data) == 0:
        return OrderedDict()

    if isinstance(data, tuple):
        # print('was a tuple')
        ordered_dict = OrderedDict()
        for number, d in enumerate(data):
            sub_dict = create_stats_ordered_dict(
                "{0}_{1}".format(name, number),
                d,
            )
            ordered_dict.update(sub_dict)
        return ordered_dict

    if isinstance(data, list):
        # print('was a list')
        try:
            iter(data[0])
        except TypeError:
            pass
        else:
            data = np.concatenate(data)

    if isinstance(data, np.ndarray) and data.size == 1 and not always_show_all_stats:
        # print('was a numpy array of data.size==1')
        return OrderedDict({name: float(data)})

    # print('was a numpy array NOT of data.size==1')
    stats = OrderedDict(
        [
            (name + " Mean", np.mean(data)),
            (name + " Std", np.std(data)),
        ]
    )
    if not exclude_max_min:
        stats[name + " Max"] = np.max(data)
        stats[name + " Min"] = np.min(data)
    return stats


def plot_experiment_returns(
    exp_path,
    title,
    save_path,
    column_name="Test_Returns_Mean",
    x_axis_lims=None,
    y_axis_lims=None,
    constraints=None,
    plot_mean=False,
    plot_horizontal_lines_at=None,
    horizontal_lines_names=None,
):
    """
    plots the Test Returns Mean of all the
    """
    arr_list = []
    names = []

    dir_path = os.path.split(save_path)[0]
    os.makedirs(dir_path, exist_ok=True)

    # print(exp_path)

    for sub_exp_dir in os.listdir(exp_path):
        try:
            sub_exp_path = os.path.join(exp_path, sub_exp_dir)
            if not os.path.isdir(sub_exp_path):
                continue
            if constraints is not None:
                constraints_satisfied = True
                with open(os.path.join(sub_exp_path, "variant.json"), "r") as j:
                    d = json.load(j)
                for k, v in constraints.items():
                    k = k.split(".")
                    d_v = d[k[0]]
                    for sub_k in k[1:]:
                        d_v = d_v[sub_k]
                    if d_v != v:
                        constraints_satisfied = False
                        break
                if not constraints_satisfied:
                    # for debugging
                    # print('\nconstraints')
                    # print(constraints)
                    # print('\nthis dict')
                    # print(d)
                    continue

            csv_full_path = os.path.join(sub_exp_path, "progress.csv")
            # print(csv_full_path)
            try:
                progress_csv = np.genfromtxt(
                    csv_full_path, skip_header=0, delimiter=",", names=True
                )
                # print(progress_csv.dtype)
                if isinstance(column_name, str):
                    column_name = [column_name]
                for c_name in column_name:
                    if "+" in c_name:
                        first, second = c_name.split("+")
                        returns = progress_csv[first] + progress_csv[second]
                    elif "-" in c_name:
                        first, second = c_name.split("-")
                        returns = progress_csv[first] - progress_csv[second]
                    else:
                        returns = progress_csv[c_name]
                    arr_list.append(returns)
                    names.append(c_name + "_" + sub_exp_dir)
                # print(csv_full_path)
            except Exception:
                pass
        except Exception:
            pass

    if plot_mean:
        min_len = min(map(lambda a: a.shape[0], arr_list))
        arr_list = list(map(lambda a: a[:min_len], arr_list))
        returns = np.stack(arr_list)
        mean = np.mean(returns, 0)
        std = np.std(returns, 0)
        x = np.arange(min_len)
        save_plot(
            x,
            mean,
            title,
            save_path,
            color="cyan",
            x_axis_lims=x_axis_lims,
            y_axis_lims=y_axis_lims,
        )
        plot_returns_on_same_plot(
            [mean, mean + std, mean - std],
            ["mean", "mean+std", "mean-std"],
            title,
            save_path,
            x_axis_lims=x_axis_lims,
            y_axis_lims=y_axis_lims,
        )
    else:
        if len(arr_list) == 0:
            print(0)
        if plot_horizontal_lines_at is not None:
            max_len = max(map(lambda a: a.shape[0], arr_list))
            arr_list += [np.ones(max_len) * y_val for y_val in plot_horizontal_lines_at]
            names += horizontal_lines_names
        try:
            # print(len(arr_list))
            plot_returns_on_same_plot(
                arr_list,
                names,
                title,
                save_path,
                x_axis_lims=x_axis_lims,
                y_axis_lims=y_axis_lims,
            )
        except Exception as e:
            print("Failed to plot:")
            print(arr_list)
            print(title)
            print(exp_path)
            print(constraints)
            # raise e
