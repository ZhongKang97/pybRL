import sys, os
sys.path.append(os.path.realpath('../..'))

import pickle
import pybRL.baselines.mlp_baseline as mlp_baseline
from pybRL.utils.gym_env import GymEnv
from pybRL.utils.gym_env import EnvSpec
import pybRL.policies.gaussian_linear as gaussian_linear
import pybullet_envs
import pybullet_envs.bullet.minitaur_gym_env as e
import pybullet as p
import numpy as np
# p.connect(p.GUI)
envs_spec = EnvSpec(28, 8,0,1)
env_id = "MinitaurBulletEnv-v0"
# env = GymEnv(env_id)
env = e.MinitaurBulletEnv(render=True)
path = '/home/sashank/mjrl-master/pybRL/experiments/Minitaur_exp1/iterations/best_policy.pickle'
with open(path, 'rb') as f:
    policy = pickle.load(f)
state = env.reset()

i = 0

while i<10000:
    action = policy.get_action(state)
    # env.step(env.action_space.sample())
    action[0] = np.clip(action[0], -1, 1)
    env.step(action[0])

    i = i+1

    # print(action)
    # state, reward, done, info = env.step(action[0])

