import sys, os
sys.path.append(os.path.realpath('../..'))
sys.path.append('/home/sashank/stoch2_gym_env')
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
env = e.StochBulletEnv(render = True, gait = 'trot')
path = '/home/sashank/mjrl-master/pybRL/experiments/policy_MinitaurTrottingEnv-v0_20190522-110536.npy'
# path = '/home/abhik/pybRL/experiments/Stoch2_ARS_1/iterations/best_policy.npy'
path = '/home/rbccps/pybRL/experiments/Stoch_Test/iterations/best_policy.npy'
state = env.reset()
logger = DataLog()
i = 0
policy = np.load(path)
print(policy)
while i<20:
    action = np.matmul(policy, state)
    state, reward, done, info = env.step(np.clip(action, -1, 1))
    i =i+1
    logger.log_kv('x_leg1', info['xpos'][0])
    logger.log_kv('x_leg2', info['xpos'][1])
    logger.log_kv('y_leg1', info['ypos'][0])
    logger.log_kv('y_leg2', info['ypos'][1])

    # time.sleep(1./30.)

plotter.plot_traj(logger, ['x_leg1', 'x_leg2'], ['y_leg1', 'y_leg2'], ['Leg1 Trajectory, rep:5', 'Leg2 Trajectory, rep:5'], save_loc= './')
