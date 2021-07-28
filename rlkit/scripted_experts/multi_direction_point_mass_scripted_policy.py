import numpy as np
from rlkit.scripted_experts.scripted_policy import ScriptedPolicy


class ScriptedMultiDirectionPointMassPolicy(ScriptedPolicy):
    def __init__(self):
        self._ptr = 0
        self.angles = np.array(
            [
                0,
                np.pi / 4.0,
                np.pi / 2.0,
                3 * np.pi / 4.0,
                np.pi,
                5 * np.pi / 4.0,
                3 * np.pi / 2.0,
                7 * np.pi / 4.0,
            ]
        )

    def reset(self, env):
        # a = np.random.uniform(np.pi/6.0, 5*np.pi/6.0)
        # a = self._ptr*(4*np.pi/6.0)/300 + np.pi/6.0
        a = self.angles[self._ptr]
        self._ptr += 1
        self._ptr %= self.angles.shape[0]
        # self.target = 100.0*np.array([np.cos(a), np.sin(a)])
        self.target = 25.0 * np.array([np.cos(a), np.sin(a)])

    def get_action(self, obs, env, timestep):
        act = self.target - env.cur_pos
        act = np.random.uniform(0.75, 0.95) * act / np.linalg.norm(act)
        return act, {}
