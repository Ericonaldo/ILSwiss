import abc
from collections import OrderedDict
from typing import Iterable

import numpy as np
from torch.autograd import Variable

import rlkit.core.eval_util
from rlkit.core.irl_algorithm import IRLAlgorithm
from rlkit.torch.utils import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule
from rlkit.core import logger, eval_util


class TorchIRLAlgorithm(IRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(self, *args, render_eval_paths=False, plotter=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rewardf_eval_statistics = None
        self.policy_eval_statistics = None
        self.render_eval_paths = render_eval_paths
        self.plotter = plotter
        self.max_returns = np.float("-inf")
        self.best_success_rate = 0.0

    @property
    @abc.abstractmethod
    def networks(self) -> Iterable[PyTorchModule]:
        pass

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def cuda(self):
        for net in self.networks:
            net.cuda()

    def evaluate(self, epoch):
        statistics = OrderedDict()
        if self.rewardf_eval_statistics is not None:
            statistics.update(self.rewardf_eval_statistics)
        # statistics.update(self.policy_optimizer.eval_statistics)
        self.rewardf_eval_statistics = None
        # self.policy_optimizer.eval_statistics = None

        logger.log("Collecting samples for evaluation")
        test_paths = self.eval_sampler.obtain_samples()

        statistics.update(
            eval_util.get_generic_path_information(
                test_paths,
                stat_prefix="Test",
            )
        )
        statistics.update(
            eval_util.get_generic_path_information(
                self._exploration_paths,
                stat_prefix="Exploration",
            )
        )
        # print(statistics.keys())
        if hasattr(self.env, "log_diagnostics"):
            self.env.log_diagnostics(test_paths)
        if hasattr(self.env, "log_statistics"):
            env_log_stats = self.env.log_statistics(test_paths)
            statistics.update(env_log_stats)
        if hasattr(self.env, "log_new_ant_multi_statistics"):
            env_log_stats = self.env.log_new_ant_multi_statistics(
                test_paths, epoch, logger.get_snapshot_dir()
            )
            statistics.update(env_log_stats)

        average_returns = rlkit.core.eval_util.get_average_returns(test_paths)
        statistics["AverageReturn"] = average_returns
        for key, value in statistics.items():
            logger.record_tabular(key, value)

        if self.render_eval_paths:
            self.env.render_paths(test_paths)

        if self.plotter:
            self.plotter.draw()

        # if self.best_success_rate < statistics['Success Rate']:
        #     self.best_success_rate = statistics['Success Rate']
        #     params = self.get_epoch_snapshot(-1)
        #     params['epoch'] = epoch
        #     logger.save_extra_data(params, 'best_params.pkl')

        if average_returns > self.max_returns:
            self.max_returns = average_returns
            if self.save_best and epoch >= self.save_best_starting_from_epoch:
                data_to_save = {
                    "algorithm": self,
                    "epoch": epoch,
                    "average_returns": average_returns,
                    "test_returns_mean": statistics["Test Returns Mean"],
                    "test_returns_std": statistics["Test Returns Std"],
                    "exp_returns_mean": statistics["Exploration Returns Mean"],
                    "exp_returns_std": statistics["Exploration Returns Std"],
                }
                logger.save_extra_data(data_to_save, "best_test.pkl")
                print("\n\nSAVED BEST\n\n")


def _elem_or_tuple_to_variable(elem_or_tuple):
    if isinstance(elem_or_tuple, tuple):
        return tuple(_elem_or_tuple_to_variable(e) for e in elem_or_tuple)
    return Variable(ptu.from_numpy(elem_or_tuple).float(), requires_grad=False)


def _filter_batch(np_batch):
    for k, v in np_batch.items():
        if v.dtype == np.bool:
            yield k, v.astype(int)
        else:
            yield k, v


def np_to_pytorch_batch(np_batch):
    return {
        k: _elem_or_tuple_to_variable(x)
        for k, x in _filter_batch(np_batch)
        if x.dtype != np.dtype("O")  # ignore object (e.g. dictionaries)
    }
