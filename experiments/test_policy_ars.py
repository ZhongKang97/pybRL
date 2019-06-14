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

class Normalizer():

  def __init__(self, nb_inputs):
    self.n = np.zeros(nb_inputs)
    self.mean = np.zeros(nb_inputs)
    self.mean_diff = np.zeros(nb_inputs)
    self.var = np.zeros(nb_inputs)

  def observe(self, x):
    self.n += 1.
    last_mean = self.mean.copy()
    self.mean += (x - self.mean) / self.n
    self.mean_diff += (x - last_mean) * (x - self.mean)
    self.var = (self.mean_diff / self.n).clip(min=1e-2)

  def normalize(self, inputs):
    obs_mean = self.mean
    obs_std = np.sqrt(self.var)
    return (inputs - obs_mean) / obs_std

# p.connect(p.GUI)
# env = e.MinitaurTrottingEnv(render=True)
env = e.StochBulletEnv(render = True, gait = 'trot' )
# path = '/home/sashank/mjrl-master/pybRL/experiments/policy_MinitaurTrottingEnv-v0_20190522-110536.npy'
# path = '/home/abhik/pybRL/experiments/Stoch2_ARS_1/iterations/best_policy.npy'
path = os.path.realpath('../..') + '/pybRL/experiments/Best_till_now/iterations/best_policy.npy'
state = env.reset()
nb_inputs = env.observation_space.sample().shape[0]
normalizer = Normalizer(nb_inputs)
logger = DataLog()
i = 0
policy = np.load(path)
print(policy)
total_reward = 0
while i<1000:
    action = np.clip(policy.dot(state), -1, 1)
    state, reward, done, info = env.step(action)
    total_reward = total_reward + reward
    i =i+1
    logger.log_kv('x_leg1', info['xpos'][0])
    logger.log_kv('x_leg2', info['xpos'][1])
    logger.log_kv('y_leg1', info['ypos'][0])
    logger.log_kv('y_leg2', info['ypos'][1])

    # time.sleep(1./30.)
print(total_reward)

# plotter.plot_traj(logger, ['x_leg1', 'x_leg2'], ['y_leg1', 'y_leg2'], ['Leg1 Trajectory, rep:5', 'Leg2 Trajectory, rep:5'], save_loc= './')
