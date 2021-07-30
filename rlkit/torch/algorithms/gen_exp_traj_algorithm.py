from rlkit.torch.algorithms.torch_rl_algorithm import TorchRLAlgorithm
from rlkit.data_management.expert_replay_buffer import ExpertReplayBuffer
from rlkit.data_management.env_replay_buffer import MetaEnvReplayBuffer


class ExpertTrajGeneratorAlgorithm(TorchRLAlgorithm):
    def __init__(self, *args, **kwargs):
        super(ExpertTrajGeneratorAlgorithm, self).__init__(*args, **kwargs)

        # # replace the replay buffer with an ExpertReplayBuffer
        # # doing it like this i ugly but the easiest modification to make
        # self.replay_buffer = ExpertReplayBuffer(
        #     self.replay_buffer_size,
        #     self.observation_dim,
        #     self.action_dim,
        #     discrete_action_dim=self.discrete_action_dim,
        #     policy_uses_pixels=self.policy_uses_pixels
        # )

        # self.do_not_train = True

    def _do_training(self):
        pass

    @property
    def networks(self):
        return [self.exploration_policy]


class MetaExpertTrajGeneratorAlgorithm(TorchRLAlgorithm):
    def __init__(
        self,
        task_params_list,
        obs_task_params_list,
        num_trajs_per_task,
        *args,
        **kwargs
    ):
        super(MetaExpertTrajGeneratorAlgorithm, self).__init__(*args, **kwargs)
        assert isinstance(self.replay_buffer, MetaEnvReplayBuffer)

        self.task_params_list = task_params_list
        self.obs_task_params_list = obs_task_params_list
        self.num_trajs_per_task = num_trajs_per_task

        self.task_idx = 0
        self.num_trajs_generated_for_cur_task = 0

        self.num_episodes = num_trajs_per_task * len(task_params_list)

    def _start_new_rollout(self):
        self.num_episodes += 1
        self.exploration_policy.reset()

        if self.num_trajs_generated_for_cur_task < self.num_tasks_per_tasks:
            self.num_trajs_generated_for_cur_task += 1
        else:
            self.task_idx += 1
            self.num_trajs_generated_for_cur_task = 0

        task_params, obs_task_params = (
            self.task_params_list[self.task_idx],
            self.obs_task_params_list[self.task_idx],
        )

        return self.training_env.reset(
            task_params=task_params, obs_task_params=obs_task_params
        )
