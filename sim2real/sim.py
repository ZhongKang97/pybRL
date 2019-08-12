import sys, os
sys.path.append(os.path.realpath('../..'))
# sys.path.append('/home/sashank/stoch2_gym_env')
from pybRL.utils.logger import DataLog
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
action1= np.array([0.10764433, -0.22723689,  0.37480827,  0.01160054, -0.18521147,  0.01536253,
  0.07677522, -0.65195124,  0.0739685,  -0.23965774])
i = 0
infos = []
while i<5:
    state, reward, done, info = env.step(action1)
    if( i >=3  and i <  5):
      infos.append(np.array(info))
    i =i+1
print(infos)
# final_info = np.concatenate((infos[0], infos[1]))
# final_info = clean_data(final_info)
# np.savetxt("sim_action1_test.txt", final_info)
# ]