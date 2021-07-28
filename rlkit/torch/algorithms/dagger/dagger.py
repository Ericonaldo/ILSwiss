import numpy as np
from collections import OrderedDict

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.algorithms.bc.bc import BC


class DAgger(BC):
    """
    Note about implementation:
    We will copy over the expert replay buffer into self.replay_buffer
    and when sampling batches, we will sample them from this buffer.
    We also overwrite the handle_step method so that when we do rollouts
    with our policy, we label them with expert actions and then add them
    to the self.replay_buffer.
    """

    def __init__(
        self,
        expert_policy,
        *args,
        unscale_for_expert=True,
        num_initial_train_steps=100,  # how many initial train steps using only expert data
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.unscale_for_expert = unscale_for_expert
        self.expert_policy = expert_policy
        self.num_initial_train_steps = num_initial_train_steps

        erb = self.expert_replay_buffer
        for i in range(erb._size):
            self.replay_buffer.add_sample(
                erb._observations[i],
                erb._actions[i],
                erb._rewards[i],
                erb._terminals[i],
                erb._next_obs[i],
            )

    def _do_training(self, epoch):
        if epoch == 0:
            for t in range(self.num_initial_train_steps):
                self._do_update_step(epoch, use_expert_buffer=True)
        for t in range(self.num_updates_per_train_call):
            self._do_update_step(epoch, use_expert_buffer=False)

    def _handle_step(
        self,
        observation,
        action,
        reward,
        next_observation,
        terminal,
        absorbing,
        agent_info,
        env_info,
    ):
        if self.unscale_for_expert:
            unscaled_obs = self.training_env.get_unscaled_obs(observation)
            action = self.expert_policy.get_action(unscaled_obs)[0]
            action = self.training_env.get_scaled_acts(action)
        else:
            action = self.expert_policy.get_action(observation)[0]
        super()._handle_step(
            observation,
            action,
            reward,
            next_observation,
            terminal,
            absorbing,
            agent_info,
            env_info,
        )

    @property
    def networks(self):
        nets = super().networks
        nets.append(self.expert_policy)
        return nets

    def get_epoch_snapshot(self, epoch):
        d = super().get_epoch_snapshot(epoch)
        d.update(expert_policy=self.expert_policy)
        return d
