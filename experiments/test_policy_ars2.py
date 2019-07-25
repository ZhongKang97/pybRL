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
import pybRL.envs.stoch2_gym_bullet_env_bezier as e2
import pybRL.envs.stoch2_gym_bullet_env_bezier_stairs as e3
import pybRL.envs.stoch2_gym_bullet_env_bezier_stairs_kartik as e4


import pybullet as p
import numpy as np
import time
PI = np.pi
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
# env = e.StochBulletEnv(render = True, gait = 'trot', energy_weight= 0.000 )
walk = [0, PI, PI/2, 3*PI/2]
canter = [0, PI, 0, PI]
bound = [0, 0, PI, PI]
trot = [0, PI, PI , 0]
env = e2.Stoch2Env(render = True, phase = walk)
# env = e4.Stoch2Env(render = True)
# path = '/home/sashank/mjrl-master/pybRL/experiments/policy_MinitaurTrottingEnv-v0_20190522-110536.npy'
# path = '/home/abhik/pybRL/experiments/Stoch2_ARS_1/iterations/best_policy.npy'
#'/pybRL/experiments/Stoch2_Jun14_9/iterations/policy_10.npy'
path = '/pybRL/experiments/spline/Jul25_1/iterations/best_policy.npy'
path = os.path.realpath('../..') + path
state = env.reset()
nb_inputs = env.observation_space.sample().shape[0]
normalizer = Normalizer(nb_inputs)
logger = DataLog()
i = 0
policy = np.load(path)
# np.save(os.path.realpath('../..') + '/pybRL/sim2real/ARS_matrix.npy', policy)
# exit()
print(policy)
total_reward = 0
states = []
# action = np.array([ 0.24504616, -0.11582746,  0.71558934, -0.46091432, -0.36284493,  0.00495828,
#  -0.06466855, -0.45247894,  0.72117291, -0.11068088])
while i<10:
    action = np.clip(policy.dot(state), -1, 1)
    action = np.zeros(10)
    state, reward, done, info = env.step(action)
    states.append(state)
    # env.step(env.action_space.sample())
    # print(reward)
    total_reward = total_reward + reward
    i =i+1
    states.append(state)
    # logger.log_kv('x_leg1', info['xpos'][0])
    # logger.log_kv('x_leg2', info['xpos'][1])
    # logger.log_kv('y_leg1', info['ypos'][0])
    # logger.log_kv('y_leg2', info['ypos'][1])

    # time.sleep(1./30.)
print(total_reward)

# plotter.plot_traj(logger, ['x_leg1', 'x_leg2'], ['y_leg1', 'y_leg2'], ['Leg1 Trajectory, rep:5', 'Leg2 Trajectory, rep:5'], save_loc= './')
