import time
import random
import numpy as np
import gym
from rlkit.scripted_experts.scripted_policy import ScriptedPolicy

# INV_TEMP = 300
INV_TEMP = 1000
NUM_ACTION_SAMPLES = 1000
ACT_SCALE_DOWN = 0.25


def get_pos_act(cur_pos, reach_pos):
    cur_pos = cur_pos.copy()
    reach_pos = reach_pos.copy()

    possible_actions = np.random.uniform(
        low=-1.0, high=1.0, size=(NUM_ACTION_SAMPLES, 3)
    )
    potentials = possible_actions.copy() * 0.03
    potentials += cur_pos[None, :]
    potentials -= reach_pos[None, :]
    potentials = np.sum(potentials ** 2, axis=1) ** 0.5
    potentials *= -1.0
    potentials = np.exp(potentials * INV_TEMP).flatten()
    potentials = potentials / np.sum(potentials)
    act_idx = np.random.choice(NUM_ACTION_SAMPLES, p=potentials)
    sampled_act = possible_actions[act_idx] * ACT_SCALE_DOWN
    return sampled_act


class ScriptedContFewShotFetchPolicy(ScriptedPolicy):
    def __init__(self):
        super().__init__()

    def reset(self, env):
        # # first make the gripper go slightly above the object
        self.correct_obj_idx = env.correct_obj_idx
        # self.INITIAL_REACH_X_DISP = np.random.uniform(-0.01, 0.01)
        # self.INITIAL_REACH_Y_DISP = np.random.uniform(-0.01, 0.01)
        # self.INITIAL_REACH_HOW_MUCH_ABOVE = np.random.uniform(0.05, 0.06)
        # def cond_0(obs):
        #     correct_obj_rel_pos = obs[
        #         6 + 3*self.correct_obj_idx : 9 + 3*self.correct_obj_idx
        #     ]
        #     object_oriented_goal = correct_obj_rel_pos.copy()
        #     object_oriented_goal[0] += self.INITIAL_REACH_X_DISP
        #     object_oriented_goal[1] += self.INITIAL_REACH_Y_DISP
        #     object_oriented_goal[2] += self.INITIAL_REACH_HOW_MUCH_ABOVE
        #     return 0.005 > np.linalg.norm(object_oriented_goal)
        # self.milestone_0_cond = cond_0

        # then grip the object
        self.GRIP_X_DISP = np.random.uniform(-0.01, 0.01)
        self.GRIP_Y_DISP = np.random.uniform(-0.01, 0.01)
        self.GRIP_Z_DISP = np.random.uniform(0.005, 0.015)
        # self.GRIP_Z_DISP = np.random.uniform(-0.005, 0.015)
        def cond_0(obs):
            correct_obj_rel_pos = obs[
                6 + 3 * self.correct_obj_idx : 9 + 3 * self.correct_obj_idx
            ]
            grip_goal = correct_obj_rel_pos.copy()
            grip_goal[0] += self.GRIP_X_DISP
            grip_goal[1] += self.GRIP_Y_DISP
            grip_goal[2] += self.GRIP_Z_DISP
            return 0.01 > np.linalg.norm(grip_goal)
            # return 0.005 > np.linalg.norm(grip_goal)

        self.milestone_0_cond = cond_0

        # then lift it and take it to the goal
        def cond_1(obs):
            correct_obj_rel_target = obs[
                3 * self.correct_obj_idx : 3 + 3 * self.correct_obj_idx
            ]
            correct_obj_rel_target = correct_obj_rel_target.copy()
            return 0.005 > np.linalg.norm(correct_obj_rel_target)

        self.milestone_1_cond = cond_1

        # reset the milestones
        self.milestone_0_complete = False
        self.milestone_1_complete = False
        # self.milestone_2_complete = False
        self.first_time_all_complete = -1

    def get_action(self, obs, env, timestep):
        # first find out what stage we are in and update milestone info
        cur_stage = -1
        if not self.milestone_0_complete:
            # check if milestone 0 was completed by the last step action
            if self.milestone_0_cond(obs):
                self.milestone_0_complete = True
                cur_stage = 1
            else:
                cur_stage = 0
        elif not self.milestone_1_complete:
            # check if milestone 1 was completed by the last step action
            if self.milestone_1_cond(obs):
                self.milestone_1_complete = True
                cur_stage = 2
            else:
                cur_stage = 1
        # else:
        #     if not self.milestone_2_complete:
        #         # check if milestone 2 was completed by the last step action
        #         if self.milestone_2_cond(obs):
        #             self.milestone_2_complete = True
        #             self.first_time_all_complete = timestep
        #     cur_stage = 2

        # now perform the action corresponding to the current stage
        # if cur_stage == 0:
        #     correct_obj_rel_pos = obs[
        #         6 + 3*self.correct_obj_idx : 9 + 3*self.correct_obj_idx
        #     ]
        #     object_oriented_goal = correct_obj_rel_pos.copy()
        #     object_oriented_goal[0] += self.INITIAL_REACH_X_DISP
        #     object_oriented_goal[1] += self.INITIAL_REACH_Y_DISP
        #     object_oriented_goal[2] += self.INITIAL_REACH_HOW_MUCH_ABOVE

        #     action = [0, 0, 0, 0]
        #     pos_act = get_pos_act(np.zeros(3), object_oriented_goal)
        #     for i in range(len(pos_act)):
        #         action[i] = pos_act[i]
        #     action[len(action)-1] = np.random.uniform(0.005, 0.015) #open
        if cur_stage == 0:
            correct_obj_rel_pos = obs[
                6 + 3 * self.correct_obj_idx : 9 + 3 * self.correct_obj_idx
            ]
            grip_goal = correct_obj_rel_pos.copy()
            grip_goal[0] += self.GRIP_X_DISP
            grip_goal[1] += self.GRIP_Y_DISP
            grip_goal[2] += self.GRIP_Z_DISP

            action = [0, 0, 0, 0]
            pos_act = get_pos_act(np.zeros(3), grip_goal)
            for i in range(len(pos_act)):
                action[i] = pos_act[i]

            OPEN_CLOSE_D = 0.03
            if np.linalg.norm(correct_obj_rel_pos, axis=-1) > OPEN_CLOSE_D:
                action[len(action) - 1] = np.random.uniform(0.005, 0.015)  # open
            else:
                action[len(action) - 1] = np.random.uniform(-0.015, -0.005)  # close
            # action[len(action)-1] = np.random.uniform(-0.015, -0.005) #close
        else:
            correct_obj_rel_target = obs[
                3 * self.correct_obj_idx : 3 + 3 * self.correct_obj_idx
            ]
            correct_obj_rel_target = correct_obj_rel_target.copy()

            action = [0, 0, 0, 0]
            pos_act = get_pos_act(np.zeros(3), correct_obj_rel_target)
            for i in range(len(pos_act)):
                action[i] = pos_act[i]
            action[len(action) - 1] = np.random.uniform(-0.005, -0.015)  # close

        action = np.clip(action, -1.0, 1.0)
        return action, {}
