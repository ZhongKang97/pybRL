import sys, os
sys.path.append(os.path.realpath('../..'))
# sys.path.append('/home/sashank/stoch2_gym_env')
import pickle
import pybRL.baselines.mlp_baseline as mlp_baseline
from pybRL.utils.gym_env import GymEnv
from pybRL.utils.gym_env import EnvSpec
import pybRL.policies.gaussian_linear as gaussian_linear
from pybRL.utils.logger import DataLog
import pybRL.utils.make_train_plots as plotter 
import pybullet_envs
import pybullet_envs.bullet.minitaur_gym_env as e
import pybullet_envs.minitaur.envs.minitaur_trotting_env as e
import pybRL.envs.stoch2_gym_bullet_env_normal as e
import pybullet as p
import numpy as np
import time
# p.connect(p.GUI)
# env = e.MinitaurTrottingEnv(render=True)
env = e.StochBulletEnv(render = True, gait = 'trot', energy_weight = 10.0 )
# path = '/home/sashank/mjrl-master/pybRL/experiments/policy_MinitaurTrottingEnv-v0_20190522-110536.npy'
# path = '/home/abhik/pybRL/experiments/Stoch2_ARS_1/iterations/best_policy.npy'
path = os.path.realpath('../..') + '/pybRL/experiments/Stoch2_wt_2/iterations/policy_0.npy'
state = env.reset()
logger = DataLog()
i = 0
policy = np.load(path)
print(policy)
total_reward = 0
while i<1000:
    action = np.matmul(policy, state)
    state, reward, done, info = env.step(action)
    total_reward = total_reward + reward
    i =i+1
    logger.log_kv('x_leg1', info['xpos'][0])
    logger.log_kv('x_leg2', info['xpos'][1])
    logger.log_kv('y_leg1', info['ypos'][0])
    logger.log_kv('y_leg2', info['ypos'][1])

    # time.sleep(1./30.)
print(total_reward)

plotter.plot_traj(logger, ['x_leg1', 'x_leg2'], ['y_leg1', 'y_leg2'], ['Leg1 Trajectory, rep:5', 'Leg2 Trajectory, rep:5'], save_loc= './')
