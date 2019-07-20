import sys, os
sys.path.append(os.path.realpath('../..'))

import pybRL.envs.walking_controller as walking_controller
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
import math
print(matplotlib.get_backend())
# p.connect(p.GUI)
# env = e.MinitaurTrottingEnv(render=True)
# env = e.StochBulletEnv(render = True, gait = 'trot', energy_weight= 0.000 )
# env = e2.Stoch2Env(render = False)
# path = '/home/sashank/mjrl-master/pybRL/experiments/policy_MinitaurTrottingEnv-v0_20190522-110536.npy'
# path = '/home/abhik/pybRL/experiments/Stoch2_ARS_1/iterations/best_policy.npy'
#'/pybRL/experiments/Stoch2_Jun14_9/iterations/policy_10.npy'

path = os.path.realpath('../..') + '/pybRL/experiments/Jul8_3/iterations/policy_21.npy'
# logger = DataLog()
i = 0
policy = np.load(path)
total_reward = 0
walkcon = walking_controller.WalkingController(gait_type='trot',spine_enable = False,
                                                planning_space = 'polar_task_space',
                                                left_to_right_switch = True,
                                                frequency=1.)
theta = 0
x_leg_1 = []
x_leg_2 = []
y_leg_1 = []
y_leg_2 = []
# action = np.clip(policy.dot(state), -1, 1)
action = np.array([ 0.24504616, -0.11582746,  0.71558934, -0.46091432, -0.36284493,  0.00495828,
 -0.06466855, -0.45247894,  0.72117291, -0.11068088])
count =0
while count < 1000 :
    if(theta > 2*math.pi):
        theta = 0
    if theta > math.pi:
        tau = (theta - math.pi)/math.pi  # as theta varies from pi to 2 pi, tau varies from 0 to 1    
        stance_leg = 1 # for 0 to pi, stance leg is left leg. (note robot behavior sometimes is erratic, so stance leg is not explicitly seen)
    else:
        tau = theta / math.pi  # as theta varies from 0 to pi, tau varies from 0 to 1    
        stance_leg = 0 # for pi to 2 pi stance leg is right leg.
    action_ref = walkcon._extend_leg_action_space_for_hzd(tau,action)
    rt, drdt = walkcon._transform_action_to_r_and_theta_via_bezier_polynomials(tau, stance_leg, action_ref)

    theta = theta + 0.01*math.pi
    
    x_leg_1.append(rt[0]*math.sin(rt[1]))
    y_leg_1.append(-rt[0]*math.cos(rt[1]))
    x_leg_2.append(rt[2]*math.sin(rt[3]))
    y_leg_2.append(-rt[2]*math.cos(rt[3]))
    count = count + 1
plt.figure()
plt.plot(x_leg_1, y_leg_1)
plt.show()

# with open('leg.txt', 'w') as f1:
#     for i in range(1000):
#         write_str = str(x_leg_1[i])+ ',' +  str(y_leg_1[i]) + ','+str(x_leg_2[i])+ ','+ str(y_leg_2[i]) + '\n'
#         f1.write(write_str)

# f1.close()
# print(total_reward/1000)

# plotter.plot_traj(logger, ['x_leg1', 'x_leg2'], ['y_leg1', 'y_leg2'], ['Leg1 Trajectory, rep:5', 'Leg2 Trajectory, rep:5'], save_loc= './')
