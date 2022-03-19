from collections import OrderedDict

import torch
import torch.optim as optim
import gtimer as gt

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.algorithms.torch_base_algorithm import TorchBaseAlgorithm
from rlkit.core import eval_util
from rlkit.core import logger


class BC(TorchBaseAlgorithm):
    def __init__(
        self,
        mode,  # 'MLE' or 'MSE'
        expert_replay_buffer,
        num_updates_per_train_call=1,
        batch_size=1024,
        lr=1e-3,
        momentum=0.0,
        optimizer_class=optim.Adam,
        **kwargs
    ):
        assert mode in ["MLE", "MSE"], "Invalid mode!"
        if kwargs["wrap_absorbing"]:
            raise NotImplementedError()
        super().__init__(**kwargs)

        self.mode = mode
        self.expert_replay_buffer = expert_replay_buffer

        self.batch_size = batch_size

        self.optimizer = optimizer_class(
            self.exploration_policy.parameters(), lr=lr, betas=(momentum, 0.999)
        )

        self.num_updates_per_train_call = num_updates_per_train_call

    def get_batch(self, batch_size, keys=None, use_expert_buffer=True):
        if use_expert_buffer:
            rb = self.expert_replay_buffer
        else:
            rb = self.replay_buffer
        batch = rb.random_batch(batch_size, keys=keys)
        batch = np_to_pytorch_batch(batch)
        return batch

    def _can_evaluate(self):
        return True

    def _can_train(self):
        return True

    def start_training(self, start_epoch=0):
        # observation = self._start_new_rollout()

        for epoch in gt.timed_for(
            range(start_epoch, self.num_epochs),
            save_itrs=True,
        ):
            self._start_epoch(epoch)
            for steps_this_epoch in range(self.num_env_steps_per_epoch):
                self._n_env_steps_total += 1
                if self._n_env_steps_total % self.num_steps_between_train_calls == 0:
                    gt.stamp("sample")
                    self._try_to_train(epoch)
                    gt.stamp("train")

            gt.stamp("sample")
            self._try_to_eval(epoch)
            gt.stamp("eval")
            self._end_epoch()

    def _do_training(self, epoch):
        for t in range(self.num_updates_per_train_call):
            self._do_update_step(epoch, use_expert_buffer=True)

    def _do_update_step(self, epoch, use_expert_buffer=True):
        batch = self.get_batch(
            self.batch_size,
            keys=["observations", "actions"],
            use_expert_buffer=use_expert_buffer,
        )

        obs = batch["observations"]
        acts = batch["actions"]

        self.optimizer.zero_grad()
        if self.mode == "MLE":
            log_prob = self.exploration_policy.get_log_prob(obs, acts)
            loss = -1.0 * log_prob.mean()
            if self.eval_statistics is None:
                self.eval_statistics = OrderedDict()
                self.eval_statistics["Log-Likelihood"] = ptu.get_numpy(-1.0 * loss)
        else:
            pred_acts = self.exploration_policy(obs)[0]
            squared_diff = (pred_acts - acts) ** 2
            loss = torch.sum(squared_diff, dim=-1).mean()
            if self.eval_statistics is None:
                self.eval_statistics = OrderedDict()
                self.eval_statistics["MSE"] = ptu.get_numpy(loss)
        loss.backward()
        self.optimizer.step()

    @property
    def networks(self):
        return [self.exploration_policy]

    def evaluate(self, epoch):
        """
        Evaluate the policy, e.g. save/print progress.
        :param epoch:
        :return:
        """
        statistics = OrderedDict()
        try:
            statistics.update(self.eval_statistics)
            self.eval_statistics = None
        except:
            print("No Stats to Eval")

        logger.log("Collecting samples for evaluation")
        test_paths = self.eval_sampler.obtain_samples()

        statistics.update(
            eval_util.get_generic_path_information(
                test_paths,
                stat_prefix="Test",
            )
        )

        if hasattr(self.env, "log_diagnostics"):
            self.env.log_diagnostics(test_paths)
        if hasattr(self.env, "log_statistics"):
            statistics.update(self.env.log_statistics(test_paths))
        if int(epoch) % self.freq_log_visuals == 0:
            if hasattr(self.env, "log_visuals"):
                self.env.log_visuals(test_paths, epoch, logger.get_snapshot_dir())

        average_returns = eval_util.get_average_returns(test_paths)
        statistics["AverageReturn"] = average_returns
        for key, value in statistics.items():
            logger.record_tabular(key, value)

        best_statistic = statistics[self.best_key]
        if best_statistic > self.best_statistic_so_far:
            self.best_statistic_so_far = best_statistic
            if self.save_best and epoch >= self.save_best_starting_from_epoch:
                data_to_save = {"epoch": epoch, "statistics": statistics}
                data_to_save.update(self.get_epoch_snapshot(epoch))
                logger.save_extra_data(data_to_save, "best.pkl")
                print("\n\nSAVED BEST\n\n")
