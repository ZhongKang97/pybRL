import sys, os
sys.path.append(os.path.realpath('../..'))

import pickle
import pybRL.baselines.mlp_baseline as mlp_baseline
from pybRL.utils.gym_env import GymEnv
from pybRL.utils.gym_env import EnvSpec
import pybRL.policies.gaussian_linear as gaussian_linear
import pybullet_envs
import pybullet_envs.bullet.minitaur_gym_env as e
import pybullet_envs.minitaur.envs.minitaur_trotting_env as e
import pybullet as p
import numpy as np
# p.connect(p.GUI)
env = e.MinitaurTrottingEnv(render=True)
path = '/home/abhik/pybRL/experiments/policy_MinitaurTrottingEnv-v0_20190522-110409.npy'
state = env.reset()

i = 0
policy = np.load(path)
while i<10000:
    action = np.matmul(policy, state)
    state, reward, done, info = env.step(action)
    i =i+1

