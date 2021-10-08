import numpy as np
import gtimer as gt
import copy
from collections import OrderedDict

import torch
import torch.optim as optim
from torch import nn
from torch import autograd
import torch.nn.functional as F

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.algorithms.torch_option_rl_algorithm import (
    TorchOptionRLAlgorithm,
)
from rlkit.data_management.path_builder import PathBuilder


class OptionIRL(TorchOptionRLAlgorithm):
    """
    Depending on choice of reward function and size of replay buffer this will be:
        - AIRL
        - GAIL (without extra entropy term)
        - FAIRL
    """

    def __init__(
        self,
        mode,  # airl, gail, or fairl
        discriminator_n,
        expert_replay_buffer,
        trainer_name="OptionPPO",
        disc_optim_batch_size=1024,
        policy_optim_batch_size=1024,
        num_update_loops_per_train_call=1,
        num_disc_updates_per_loop_iter=100,
        num_policy_updates_per_loop_iter=100,
        disc_lr=1e-3,
        disc_momentum=0.0,
        disc_optimizer_class=optim.Adam,
        use_grad_pen=True,
        grad_pen_weight=10,
        rew_clip_min=None,
        rew_clip_max=None,
        **kwargs,
    ):
        assert mode in [
            "airl",
            "gail",
            "fairl",
            "gail2",
        ], "Invalid adversarial irl algorithm!"
        if kwargs["wrap_absorbing"]:
            raise NotImplementedError()
        super().__init__(**kwargs)

        self.mode = mode

        self.expert_replay_buffer = expert_replay_buffer

        self.trainer_name = trainer_name
        self.policy_optim_batch_size = policy_optim_batch_size

        self.discriminator_n = discriminator_n
        self.disc_optimizer = disc_optimizer_class(
            self.discriminator.parameters(), lr=disc_lr, betas=(disc_momentum, 0.999)
        )
        self.disc_optim_batch_size = disc_optim_batch_size
        print("\n\nDISC MOMENTUM: %f\n\n" % disc_momentum)

        self.bce = nn.BCEWithLogitsLoss()
        self.bce_targets = torch.cat(
            [
                torch.ones(disc_optim_batch_size, 1),
                torch.zeros(disc_optim_batch_size, 1),
            ],
            dim=0,
        )
        self.bce.to(ptu.device)
        self.bce_targets = self.bce_targets.to(ptu.device)

        self.use_grad_pen = use_grad_pen
        self.grad_pen_weight = grad_pen_weight

        self.num_update_loops_per_train_call = num_update_loops_per_train_call
        self.num_disc_updates_per_loop_iter = num_disc_updates_per_loop_iter
        self.num_policy_updates_per_loop_iter = num_policy_updates_per_loop_iter

        self.rew_clip_min = rew_clip_min
        self.rew_clip_max = rew_clip_max
        self.clip_min_rews = rew_clip_min is not None
        self.clip_max_rews = rew_clip_max is not None

        self.disc_eval_statistics = None

    def get_batch(self, batch_size, from_expert, keys=None):
        if from_expert:
            buffer = self.expert_replay_buffer
        else:
            buffer = self.replay_buffer
        batch = buffer.random_batch(batch_size, keys=keys)
        batch = np_to_pytorch_batch(batch)
        return batch

    def get_batch_trajs(self, traj_num=1, from_expert=False, keys=None):
        if from_expert:
            buffer = self.expert_replay_buffer
        else:
            buffer = self.replay_buffer
        batch = buffer.sample_trajs(traj_num, keys=keys)
        batch = [np_to_pytorch_batch(_) for _ in batch]
        return batch

    def _end_epoch(self):
        self.trainer.end_epoch()
        self.disc_eval_statistics = None
        super()._end_epoch()

    def evaluate(self, epoch):
        self.eval_statistics = OrderedDict()
        self.eval_statistics.update(self.disc_eval_statistics)
        self.eval_statistics.update(self.trainer.get_eval_statistics())
        super().evaluate(epoch)

    def _do_training(self, epoch):
        for t in range(self.num_update_loops_per_train_call):
            for _ in range(self.num_disc_updates_per_loop_iter):
                self._do_reward_training(epoch)
            for _ in range(self.num_policy_updates_per_loop_iter):
                self._do_policy_training(epoch)

    def start_training(self, start_epoch=0, flag=False):
        # self._current_path_builder = PathBuilder()
        self.ready_env_ids = np.arange(self.env_num)
        observations, prev_options = self._start_new_rollout(
            self.ready_env_ids
        )  # Do it for support vec env

        self._current_path_builder = [
            PathBuilder() for _ in range(len(self.ready_env_ids))
        ]

        for epoch in gt.timed_for(
            range(start_epoch, self.num_epochs),
            save_itrs=True,
        ):
            self._start_epoch(epoch)
            total_rews = np.array([0.0 for _ in range(len(self.ready_env_ids))])
            for steps_this_epoch in range(self.num_env_steps_per_epoch // self.env_num):
                options, actions = self._get_action_and_option(
                    observations, prev_options
                )

                if type(actions) is tuple:
                    actions = actions[0]

                if self.render:
                    self.training_env.render()

                next_obs, raw_rewards, terminals, env_infos = self.training_env.step(
                    actions, self.ready_env_ids
                )
                if self.no_terminal:
                    terminals = [False for _ in range(len(self.ready_env_ids))]
                # self._n_env_steps_total += 1
                self._n_env_steps_total += len(self.ready_env_ids)

                rewards = raw_rewards
                total_rews += raw_rewards

                self._handle_vec_step(
                    observations,
                    actions,
                    rewards,
                    next_obs,
                    np.array([False for _ in range(len(self.ready_env_ids))])
                    if self.no_terminal
                    else terminals,
                    prev_options,
                    options,
                    env_infos=env_infos,
                )
                if np.any(terminals):
                    env_ind_local = np.where(terminals)[0]
                    if flag:
                        pass
                    total_rews[env_ind_local] = 0.0
                    self._handle_vec_rollout_ending(env_ind_local)
                    reset_observations, reset_prev_options = self._start_new_rollout(
                        env_ind_local
                    )
                    next_obs[env_ind_local] = reset_observations
                    options[env_ind_local] = reset_prev_options
                elif np.any(
                    np.array(
                        [
                            len(self._current_path_builder[i])
                            for i in range(len(self.ready_env_ids))
                        ]
                    )
                    >= self.max_path_length
                ):
                    env_ind_local = np.where(
                        np.array(
                            [
                                len(self._current_path_builder[i])
                                for i in range(len(self.ready_env_ids))
                            ]
                        )
                        >= self.max_path_length
                    )[0]
                    self._handle_vec_rollout_ending(env_ind_local)
                    reset_observations, reset_prev_options = self._start_new_rollout(
                        env_ind_local
                    )
                    next_obs[env_ind_local] = reset_observations
                    options[env_ind_local] = reset_prev_options

                observations = next_obs
                prev_options = options

                if self._n_env_steps_total % self.num_steps_between_train_calls == 0:
                    gt.stamp("sample")
                    self.convert_expert_demos()
                    self._try_to_train(epoch)
                    gt.stamp("train")

            gt.stamp("sample")
            self._try_to_eval(epoch)
            gt.stamp("eval")
            self._end_epoch()

    def convert_expert_demos(self):
        relabeld_traj = []
        for i in range(self.expert_replay_buffer.num_trajs):
            expert_traj = self.expert_replay_buffer.pop()
            expert_obs = expert_traj["observations"]
            expert_acts = expert_traj["actions"]

            induced_option, _ = self.exploration_policy.viterbi_path(
                expert_obs, expert_acts
            )

            expert_traj["prev_options"] = induced_option[:-1]
            expert_traj["options"] = induced_option[1:]
            relabeld_traj.append(copy.deepcopy(expert_traj))

        for expert_traj in relabeld_traj:
            self.expert_replay_buffer.add_path(expert_traj)

    def _do_reward_training(self, epoch):
        """
        Train the discriminator
        """
        self.disc_optimizer.zero_grad()

        keys = ["observations", "actions", "prev_options", "options"]

        expert_batch = self.get_batch(self.disc_optim_batch_size, True, keys)
        policy_batch = self.get_batch(self.disc_optim_batch_size, False, keys)

        expert_obs = expert_batch["observations"]
        policy_obs = policy_batch["observations"]
        expert_acts = expert_batch["actions"]
        policy_acts = policy_batch["actions"]
        policy_c_1 = policy_batch["prev_options"]
        expert_c_1 = expert_batch["prev_options"]
        policy_c = policy_batch["options"]
        expert_c = expert_batch["options"]

        expert_disc_input = torch.cat([expert_obs, expert_acts], dim=1)
        policy_disc_input = torch.cat([policy_obs, policy_acts], dim=1)

        disc_input = torch.cat([expert_disc_input, policy_disc_input], dim=0)
        c_1_disc_input = torch.cat([expert_c_1, policy_c_1], dim=0)
        c_disc_input = torch.cat([expert_c, policy_c], dim=0)

        disc_logits = self.discriminator(disc_input, c_1_disc_input, c_disc_input)
        disc_preds = (disc_logits > 0).type(disc_logits.data.type())

        disc_ce_loss = self.bce(disc_logits, self.bce_targets)
        accuracy = (disc_preds == self.bce_targets).type(torch.FloatTensor).mean()

        if self.use_grad_pen:
            eps = ptu.rand(expert_obs.size(0), 1)
            eps.to(ptu.device)

            interp_obs = eps * expert_disc_input + (1 - eps) * policy_disc_input
            interp_obs = interp_obs.detach()
            interp_obs.requires_grad_(True)

            interp_c_1 = eps * expert_c_1 + (1 - eps) * policy_c_1
            interp_c_1 = interp_c_1.detach()
            interp_c_1.requires_grad_(True)

            interp_c = eps * expert_c + (1 - eps) * policy_c
            interp_c = interp_c.detach()
            interp_c.requires_grad_(True)

            gradients = autograd.grad(
                outputs=self.discriminator(interp_obs, interp_c_1, interp_c).sum(),
                inputs=[interp_obs],
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )
            total_grad = gradients[0]

            # GP from Gulrajani et al.
            gradient_penalty = ((total_grad.norm(2, dim=1) - 1) ** 2).mean()
            disc_grad_pen_loss = gradient_penalty * self.grad_pen_weight

            # # GP from Mescheder et al.
            # gradient_penalty = (total_grad.norm(2, dim=1) ** 2).mean()
            # disc_grad_pen_loss = gradient_penalty * 0.5 * self.grad_pen_weight
        else:
            disc_grad_pen_loss = 0.0

        disc_total_loss = disc_ce_loss + disc_grad_pen_loss
        disc_total_loss.backward()
        self.disc_optimizer.step()

        """
        Save some statistics for eval
        """
        if self.disc_eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.disc_eval_statistics = OrderedDict()

            self.disc_eval_statistics["Disc CE Loss"] = np.mean(
                ptu.get_numpy(disc_ce_loss)
            )
            self.disc_eval_statistics["Disc Acc"] = np.mean(ptu.get_numpy(accuracy))
            if self.use_grad_pen:
                self.disc_eval_statistics["Grad Pen"] = np.mean(
                    ptu.get_numpy(gradient_penalty)
                )
                self.disc_eval_statistics["Grad Pen W"] = np.mean(self.grad_pen_weight)

    def _do_policy_training(self, epoch):
        if self.trainer_name == "OptionPPO":
            policy_trajs = self.get_batch_trajs(1, False)
        else:
            raise NotImplementedError

        for policy_batch in policy_trajs:
            obs = policy_batch["observations"]
            acts = policy_batch["actions"]

            c_1 = policy_batch["prev_options"]
            c = policy_batch["options"]

            self.discriminator.eval()
            disc_input = torch.cat([obs, acts], dim=1)
            disc_logits = self.discriminator(disc_input, c_1, c).detach()
            self.discriminator.train()

            # compute the reward using the algorithm
            if self.mode == "airl":
                # If you compute log(D) - log(1-D) then you just get the logits
                policy_batch["rewards"] = disc_logits
            elif self.mode == "gail":  # -log (1-D) > 0
                policy_batch["rewards"] = F.softplus(
                    disc_logits, beta=1
                )  # F.softplus(disc_logits, beta=-1)
            elif self.mode == "gail2":  # log D < 0
                policy_batch["rewards"] = F.softplus(
                    disc_logits, beta=-1
                )  # F.softplus(disc_logits, beta=-1)
            else:  # fairl
                policy_batch["rewards"] = torch.exp(disc_logits) * (-1.0 * disc_logits)

            if self.clip_max_rews:
                policy_batch["rewards"] = torch.clamp(
                    policy_batch["rewards"], max=self.rew_clip_max
                )
            if self.clip_min_rews:
                policy_batch["rewards"] = torch.clamp(
                    policy_batch["rewards"], min=self.rew_clip_min
                )

        # policy optimization step
        self.trainer.train_step(policy_trajs)

        self.disc_eval_statistics["Disc Rew Mean"] = np.mean(
            ptu.get_numpy(policy_batch["rewards"])
        )
        self.disc_eval_statistics["Disc Rew Std"] = np.std(
            ptu.get_numpy(policy_batch["rewards"])
        )
        self.disc_eval_statistics["Disc Rew Max"] = np.max(
            ptu.get_numpy(policy_batch["rewards"])
        )
        self.disc_eval_statistics["Disc Rew Min"] = np.min(
            ptu.get_numpy(policy_batch["rewards"])
        )

    @property
    def networks(self):
        return [self.discriminator] + self.trainer.networks

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(disc=self.discriminator)
        snapshot.update(self.trainer.get_snapshot())
        return snapshot

    def to(self, device):
        self.bce.to(ptu.device)
        self.bce_targets = self.bce_targets.to(ptu.device)
        super().to(device)
