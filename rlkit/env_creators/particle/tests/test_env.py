import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from particle_env import ParticleEnv

env = ParticleEnv()
obs_dict = env.reset()
print(env.observation_space_n)
print(env.action_space_n)
while True:
    env.render()
    act_dict = {
        "agent_0": np.array([3]),
        "agent_1": np.array([3]),
    }
    env.step(act_dict)
    k = input()
    if k == "q":
        break
