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
import pybRL.envs.stoch2_gym_bullet_env as e
import pybullet as p
import numpy as np
# p.connect(p.GUI)
#env = e.MinitaurTrottingEnv(render=True)
env = e.Stoch2Env(render = True)
path = '/home/sashank/mjrl-master/pybRL/experiments/policy_MinitaurTrottingEnv-v0_20190522-110536.npy'
path = '/home/abhik/pybRL/experiments/Stoch2_ARS_1/iterations/best_policy.npy'
state = env.reset()

i = 0
policy = np.load(path)
while i<10000:
    action = np.matmul(policy, state)
    state, reward, done, info = env.step(action)
    i =i+1

