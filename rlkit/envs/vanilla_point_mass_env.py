import numpy as np
from collections import OrderedDict
from gym import utils
from gym import spaces
import os

# from rlkit.envs.mujoco_env import MujocoEnv
from gym.envs.mujoco.mujoco_env import MujocoEnv

from rlkit.core.vistools import plot_seaborn_heatmap, plot_scatter


class VanillaPointMassEnv():
    def __init__(self, init_pos=np.array([0.0, 0.0])):
        self.cur_pos = np.zeros([2])
        self.init_pos = init_pos.copy()

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype='float32')
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2,), dtype='float32')

    def seed(self, num):
        pass

    def step(self, a):
        if len(a.shape) == 2:
            a = a[0]
        a = np.clip(a, self.action_space.low, self.action_space.high)

        reward = 0.0 # we don't need a reward for what we want to do with this env

        self.cur_pos += a

        # if we want noisy dynamics
        self.cur_pos += np.random.normal(loc=0.0, scale=0.2, size=2)

        done = False
        return self.cur_pos.copy() , reward, done, dict(
            xy_pos=self.cur_pos.copy(),
        )

    def reset(self):
        self.cur_pos = self.init_pos + np.random.normal(loc=0.0, scale=0.5, size=2)
        return self.cur_pos.copy()

    def log_new_ant_multi_statistics(self, paths, epoch, log_dir):
        xy_pos = np.array([d['xy_pos'] for path in paths for d in path['env_infos']])
        
        PLOT_BOUND = 60
        
        plot_scatter(
            xy_pos[:,0],
            xy_pos[:,1],
            30,
            'XY Pos Epoch %d'%epoch,
            os.path.join(log_dir, 'xy_pos_epoch_%d.png'%epoch),
            [[-PLOT_BOUND,PLOT_BOUND], [-PLOT_BOUND,PLOT_BOUND]]
        )

        return {}
