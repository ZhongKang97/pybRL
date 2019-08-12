import sys, os
sys.path.append(os.path.realpath('../..'))
# sys.path.append('/home/sashank/stoch2_gym_env')
from pybRL.utils.logger import DataLog
import pybRL.envs.stoch2_gym_bullet_env_bezier as e2
import numpy as np
import pandas as pd
import math

df = pd.DataFrame(columns = ['index','flh', 'flk', 'frh', 'frk','blh', 'blk', 'brh', 'brk'])

def clean_data(info, state, no_of_pts):
  no_of_rows = len(info) 
  x = [int(np.around(x)) for x in np.linspace(0, no_of_rows-1, no_of_pts-1)]
  final_info = []
  counter = 0
  for count in x:
    temp_dict = {'index': counter ,'flh':info[count][1] ,'flk': info[count][2], 'frh': info[count][3], 
    'frk': info[count][4],'blh': info[count][5], 'blk': info[count][6], 'brh': info[count][7], 'brk' : info[count][8]}
    final_info.append(temp_dict)
    counter= counter+1
  final_info.append({'index': counter ,'flh':state[0] ,'flk':state[1], 'frh':state[2],'frk':state[3],'blh':state[4], 'blk':state[5], 'brh':state[6], 'brk' :state[7]})
  return final_info

PI = math.pi
walk = [0, PI, PI/2, 3*PI/2]
pace = [0, PI, 0, PI]
bound = [0, 0, PI, PI]
trot = [0, PI, PI , 0]
custom_phase = [0, PI, PI+0.1 , 0.1]

env = e2.Stoch2Env(render = False, phase = trot, stairs = False)
action = np.array([0.4143, 0.2379, 0.0385, 0.0826, 0.0119, 0.0402, 0.031, 0.063, 0.157, 0.1575, 0.1267, 0.0596, 0.0156, 0.0019, 0.0431, 0.0986, 0.1905, 0.516])

i = 0
infos = []

desired_pt_count = 300

while i<5:
    state, reward, done, info = env.step(action)
    print(state)
    if(i >=3):
      info = clean_data(info,state,desired_pt_count)
      df = pd.concat([df, pd.DataFrame(info)])
    i =i+1
    pass
print(df)
# final_info = clean_data(final_info)
# np.savetxt("sim_action1_test.txt", final_info)