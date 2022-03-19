from collections import OrderedDict

from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.algorithms.torch_base_algorithm import TorchBaseAlgorithm


class TorchRLAlgorithm(TorchBaseAlgorithm):
    def __init__(
        self, trainer, batch_size, num_train_steps_per_train_call, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.trainer = trainer
        self.batch_size = batch_size
        self.num_train_steps_per_train_call = num_train_steps_per_train_call

    def get_batch(self):
        batch = self.replay_buffer.random_batch(self.batch_size)
        return np_to_pytorch_batch(batch)

    @property
    def networks(self):
        return self.trainer.networks

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def _do_training(self, epoch):
        for _ in range(self.num_train_steps_per_train_call):
            if getattr(self.trainer, "on_policy", False):
                self.trainer.train_step(self.get_all_trajs())
                self.clear_buffer()
            else:
                self.trainer.train_step(self.get_batch())

    def get_epoch_snapshot(self, epoch):
        data_to_save = dict(epoch=epoch)
        data_to_save.update(self.trainer.get_snapshot())
        return data_to_save

    def load_snapshot(self, snapshot):
        self.trainer.load_snapshot(snapshot)
        self.exploration_policy = self.trainer.policy
        from rlkit.torch.common.policies import MakeDeterministic

        self.eval_policy = MakeDeterministic(self.exploration_policy)
        self.eval_sampler.policy = self.eval_policy

    def evaluate(self, epoch):
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()
        trainer_stats = self.trainer.get_eval_statistics()
        if trainer_stats is not None:
            self.eval_statistics.update(trainer_stats)
        super().evaluate(epoch)

    def _end_epoch(self):
        self.trainer.end_epoch()
        super()._end_epoch()

    def get_all_trajs(self):
        batch = self.replay_buffer.sample_all_trajs()
        batch = [np_to_pytorch_batch(b) for b in batch]
        return batch

    def clear_buffer(self):
        self.replay_buffer.clear()
