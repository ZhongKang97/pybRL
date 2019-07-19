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

import pybullet as p
import numpy as np
import time
from math import ceil
def clean_data(angle_array):
  no_of_rows = angle_array.shape[0]
  final_data = []
  #200 points actually repeat
  for i in range(199):
    current_angle = angle_array[int(i*no_of_rows/200)]
    current_angle = np.delete(current_angle, [0])
    current_angle = np.insert(current_angle,0,i)
    final_data.append(current_angle)
  current_angle = angle_array[-1]
  current_angle = np.delete(current_angle, [0])
  current_angle = np.insert(current_angle,0,199)
  final_data.append(current_angle)
  return np.array(final_data)
env = e2.Stoch2Env(render = True)
path = os.path.realpath('../..') + '/pybRL/experiments/Jul8_2/iterations/best_policy.npy'
state = env.reset()
policy = np.load("ARS_matrix.npy")
action1= np.array([0.10764433, -0.22723689,  0.37480827,  0.01160054, -0.18521147,  0.01536253,
  0.07677522, -0.65195124,  0.0739685,  -0.23965774])
action2 =np.array([-0.05966324,  0.01689064,  0.35201927, -0.10362486, -0.01347287, -0.0224595,
   0.06660015, -0.60092065,  0.08468813, -0.38072799])
i = 0
infos = []
while i<5:
    # action = np.clip(policy.dot(state), -1, 1)
    # print('action: ',action, 'state: ',state)
    state, reward, done, info = env.step(action1)
    if( i >=3  and i <  5):
      infos.append(np.array(info))

    # print(info)
    # env.step(env.action_space.sample())
    # print(reward)
    # exit()
    # total_reward = total_reward + reward
    i =i+1
    # logger.log_kv('x_leg1', info['xpos'][0])
    # logger.log_kv('x_leg2', info['xpos'][1])
    # logger.log_kv('y_leg1', info['ypos'][0])
    # logger.log_kv('y_leg2', info['ypos'][1])

    # time.sleep(1./30.)
final_info = np.concatenate((infos[0], infos[1]))
final_info = clean_data(final_info)
np.savetxt("sim_action1_test.txt", final_info)
# print(states)
# plotter.plot_traj(logger, ['x_leg1', 'x_leg2'], ['y_leg1', 'y_leg2'], ['Leg1 Trajectory, rep:5', 'Leg2 Trajectory, rep:5'], save_loc= './')
