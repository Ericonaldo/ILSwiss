import time
import random
import numpy as np
import gym
from rlkit.scripted_experts.scripted_policy import ScriptedPolicy

ACT_MAG = 0.4125
# ACT_MAG = 0.275
ACT_NOISE_SCALE = 0.1
ACT_SLOW_NOISE_SCALE = 0.05
SLOW_DOWN_RADIUS = 0.01

def get_linear_pos_act(cur_pos, reach_pos):
    cur_pos = cur_pos.copy()
    reach_pos = reach_pos.copy()
    move_dir = reach_pos - cur_pos
    dist = np.linalg.norm(move_dir, axis=-1)

    # if dist > ACT_MAG:
    # if dist < ACT_MAG:
    #     move_dir = move_dir
    # else:
    move_dir *= (ACT_MAG / dist)
    return move_dir


class ScriptedLinearFewShotFetchPolicy(ScriptedPolicy):
    def __init__(self):
        super().__init__()
    

    def reset(self, env):
        # # first make the gripper go slightly above the object
        self.correct_obj_idx = env.correct_obj_idx
        if self.correct_obj_idx == 0:
            self.correct_obj_abs_pos = env.sim.data.get_site_xpos('object0')
        else:
            self.correct_obj_abs_pos = env.sim.data.get_site_xpos('object1')
        
        self.init_grip_pos = env.sim.data.get_site_xpos('robot0:grip')


        X_Y_FRAC = np.random.uniform(0.7, 0.8)
        Z_FRAC = np.random.uniform(0.2, 0.3)
        self.waypoint = np.zeros(3)
        self.waypoint[:2] = (self.correct_obj_abs_pos[:2] - self.init_grip_pos[:2]) * X_Y_FRAC
        self.waypoint[2] = (self.correct_obj_abs_pos[2] - self.init_grip_pos[2]) * Z_FRAC
        self.waypoint += self.init_grip_pos

        self.waypoint += np.random.uniform(-0.01, 0.01, 3)

        # first go to a way-point
        def cond_0(obs):
            grip_pos = env.sim.data.get_site_xpos('robot0:grip')
            return 0.01 > np.linalg.norm(grip_pos - self.waypoint, axis=-1)
        self.milestone_0_cond = cond_0

        self.GRIP_X_DISP = np.random.uniform(-0.01, 0.01)
        self.GRIP_Y_DISP = np.random.uniform(-0.01, 0.01)
        self.GRIP_Z_DISP = np.random.uniform(0.0, 0.005)
        # now actually go to the object
        def cond_1(obs):
            correct_obj_rel_pos = obs[
                6 + 3*self.correct_obj_idx : 9 + 3*self.correct_obj_idx
            ].copy()
            grip_rel_pos = correct_obj_rel_pos
            grip_rel_pos += np.array([self.GRIP_X_DISP, self.GRIP_Y_DISP, self.GRIP_Z_DISP])
            return 0.01 > np.linalg.norm(grip_rel_pos)
        self.milestone_1_cond = cond_1

        # then lift it and take it to the goal
        def cond_2(obs):
            correct_obj_rel_target = obs[
                3*self.correct_obj_idx : 3 + 3*self.correct_obj_idx
            ].copy()
            return 0.01 > np.linalg.norm(correct_obj_rel_target)
        self.milestone_2_cond = cond_2

        # reset the milestones
        self.milestone_0_complete = False
        self.milestone_1_complete = False
        self.milestone_2_complete = False
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
        else:
            if not self.milestone_2_complete:
                # check if milestone 2 was completed by the last step action
                if self.milestone_2_cond(obs):
                    self.milestone_2_complete = True
                    self.first_time_all_complete = timestep
            cur_stage = 2

        # now perform the action corresponding to the current stage
        if cur_stage == 0:
            grip_pos = env.sim.data.get_site_xpos('robot0:grip')

            action = [0, 0, 0, 0]
            pos_act = get_linear_pos_act(grip_pos, self.waypoint)
            pos_act += np.random.uniform(0.0, ACT_NOISE_SCALE, 3)
            for i in range(len(pos_act)):
                action[i] = pos_act[i]
            action[len(action)-1] = np.random.uniform(0.005, 0.015) #open
        elif cur_stage == 1:
            correct_obj_rel_pos = obs[
                6 + 3*self.correct_obj_idx : 9 + 3*self.correct_obj_idx
            ].copy()
            grip_goal = correct_obj_rel_pos.copy()
            grip_goal += np.array([self.GRIP_X_DISP, self.GRIP_Y_DISP, self.GRIP_Z_DISP])

            action = [0, 0, 0, 0]
            pos_act = get_linear_pos_act(np.zeros(3), grip_goal)
            pos_act += np.random.uniform(0.0, ACT_NOISE_SCALE, 3)
            for i in range(len(pos_act)):
                action[i] = pos_act[i]
            
            OPEN_CLOSE_D = 0.05
            if np.linalg.norm(correct_obj_rel_pos, axis=-1) > OPEN_CLOSE_D:
                action[len(action)-1] = np.random.uniform(0.005, 0.015) #open
            else:
                action[len(action)-1] = np.random.uniform(-0.015, -0.005) #close
        else:
            correct_obj_rel_target = obs[
                3*self.correct_obj_idx : 3 + 3*self.correct_obj_idx
            ]
            correct_obj_rel_target = correct_obj_rel_target.copy()

            action = [0, 0, 0, 0]
            # print(np.linalg.norm(correct_obj_rel_target, axis=-1))
            if np.linalg.norm(correct_obj_rel_target, axis=-1) < SLOW_DOWN_RADIUS:
                # previous version: pos_act = correct_obj_rel_target
                # new version that looks smoother:
                pos_act = 0.25*get_linear_pos_act(np.zeros(3), correct_obj_rel_target)
                pos_act += np.random.uniform(0.0, ACT_SLOW_NOISE_SCALE, 3)
            else:
                pos_act = get_linear_pos_act(np.zeros(3), correct_obj_rel_target)
                pos_act += np.random.uniform(0.0, ACT_NOISE_SCALE, 3)
            # pos_act = get_linear_pos_act(np.zeros(3), correct_obj_rel_target)
            for i in range(len(pos_act)):
                action[i] = pos_act[i]
            action[len(action)-1] = np.random.uniform(-0.005, -0.015) # close
        
        action = np.clip(action, -1.0, 1.0)
        return action, {}
