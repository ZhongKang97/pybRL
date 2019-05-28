# Use this to test Stoch2 Environment
import gym
# import mujoco_py
import pickle
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import ik_class as ik

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
       
# Import the RL Algorithm and tools for Multiprocessing
from stable_baselines.ppo2 import PPO2
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder
import envs.stoch2_gym_mjc_env as stoch2_gym_env
# import envs.stoch2_gym_bullet_env as stoch2_gym_env

import warnings
warnings.filterwarnings('ignore')

# Directory for Saved models, Tensorflow log,and  Videos
dir_name = "./tmp"
if not os.path.exists(dir_name):
        os.makedirs(dir_name)

input_file='save.p'
action_file='action.p'
output_file='save.txt'
fig_name = 'GaitSimulation.eps'
###############################################################################
# callbacks are defined here
global i
global qpos, torque
i=0
total_data_length = 1000
total_data_length_for_saving = 200
qpos = np.zeros((total_data_length,10))
qdes = np.zeros((total_data_length,10))
qvel = np.zeros((total_data_length,10))
torque = np.zeros((total_data_length,10))

def render_callback(env_in):
    """
    Callback at each step for rendering
    :env: the mujoco environment
    """
    global i
    global qpos, torque
    
    env_in.render()
#     qpos[i,:] = env_in.sim.data.qpos[[7,8,10,12,14,16,17,19,21,23]]
    qpos[i,:] = env_in.GetMotorAngles()
    qdes[i,[1,2,3,4,6,7,8,9]] = env_in.GetDesiredMotorAngles()
#     qvel[i,:] = env_in.sim.data.qvel[[7,8,10,12,14,16,17,19,21,23]]
    qvel[i,:] = env_in.GetMotorVelocities()
#     torque[i,:] = env_in.sim.data.actuator_force
    torque[i,:] = env_in.GetMotorTorques()
    i = (i + 1) % total_data_length
    return True


###############################################################################
# # Use this code for testing the basic controller
# Create the stoch mujoco environment
# env = stoch2_gym_mjc_env.Stoch2Env()
env = stoch2_gym_env.Stoch2Env(render=True)

model_test = PPO2.load(dir_name+"/tflow_log/model_trot") # model_trot_200kiter_0pen_notermination
obs = env.reset()

print("Render mode...")

for _ in range(15):
    action, _states = model_test.predict(obs,deterministic=True)
    obs, reward, done, _ = env.step(action,callback=render_callback)
#     if done:
#         break

pickle.dump(qpos[0:total_data_length:int(total_data_length/total_data_length_for_saving)], open(input_file, "wb"))  # save it into a file named save.p
pickle.dump(action, open(action_file, "wb"))


plt.plot(np.arange(total_data_length)*env.dt,qpos[:,[1,2,3,4,6,7,8,9]])
# plt.plot(np.arange(total_data_length)*env.dt,qvel[:,[1,2,3,4,6,7,8,9]])
# plt.plot(np.arange(total_data_length)*env.dt,qpos[:,[2,9]])
# plt.plot(np.arange(total_data_length)*env.dt,torque[:,[1,2,3,4,6,7,8,9]])
plt.show()

# data = pickle.load(open('save_gait16.p', "rb"))
# data_np = np.array(data)
# Fs, FLh, FLk, FRh, FRk, Bs, BLh, BLk, BRh, BRk = data_np.T

Fs, FLh, FLk, FRh, FRk, Bs, BLh, BLk, BRh, BRk = qpos[0:total_data_length:int(total_data_length/total_data_length_for_saving)].T

xy = env.GetXYTrajectory(action)

leg = ik.Stoch2Kinematics()

FLx = np.zeros(total_data_length_for_saving)
FLy = np.zeros(total_data_length_for_saving)
for i in range(total_data_length_for_saving):
    valid, [FLx[i], FLy[i]] = leg.forwardKinematics([FLh[i] - 2.35612, FLk[i] - 1.2217]) 
    if not valid:
        print("Invalid data", FLh[i], FLk[i])

FRx = np.zeros(total_data_length_for_saving)
FRy = np.zeros(total_data_length_for_saving)
for i in range(total_data_length_for_saving):
    valid, [FRx[i], FRy[i]] = leg.forwardKinematics([FRh[i] - 2.35612, FRk[i] - 1.2217]) 
    if not valid:
        print("Invalid data", FRh[i], FRk[i])

BLx = np.zeros(total_data_length_for_saving)
BLy = np.zeros(total_data_length_for_saving)
for i in range(total_data_length_for_saving):
    valid, [BLx[i], BLy[i]] = leg.forwardKinematics([BLh[i] - 2.35612, BLk[i] - 1.2217]) 
    if not valid:
        print("Invalid data", BLh[i], BLk[i])

BRx = np.zeros(total_data_length_for_saving)
BRy = np.zeros(total_data_length_for_saving)
for i in range(total_data_length_for_saving):
    valid, [BRx[i], BRy[i]] = leg.forwardKinematics([BRh[i] - 2.35612, BRk[i] - 1.2217]) 
    if not valid:
        print("Invalid data", BRh[i], BRk[i])


plt.subplot(1,2,1)
plt.plot(FLx[-100:], FLy[-100:],xy[2,:],xy[3,:],env.FLx[-100:],env.FLy[-100:])
plt.xlabel('x position, front left leg',fontsize=20)
plt.ylabel('y position',fontsize=20)
plt.gca().legend(('Polynomial','Simulation','Reference'),fontsize=15,loc=4)
# plt.text(0.0, 0.0, 'Front left leg', horizontalalignment='center',verticalalignment='center', fontsize=35, fontname='Times New Roman')
plt.grid()

# plt.subplot(2,2,2)
# plt.plot(FRx[-100:], FRy[-100:],xy[0,:],xy[1,:])
# plt.xlabel('x position, front right leg',fontsize=20, fontname='Times New Roman')
# plt.ylabel('y position',fontsize=20, fontname='Times New Roman')
# plt.gca().legend(('Actual','Desired'),fontsize=15)
# # plt.text(0.0, 0.0, 'Front right leg', horizontalalignment='center',verticalalignment='center', fontsize=35, fontname='Times New Roman')
# plt.grid()

plt.subplot(1,2,2)
plt.plot(BLx[-100:], BLy[-100:],xy[0,:],xy[1,:],env.FLx[-100:],env.FLy[-100:])
plt.xlabel('x position, back left leg',fontsize=20)
plt.ylabel('y position',fontsize=20)
plt.gca().legend(('Polynomial','Simulation','Reference'),fontsize=15,loc=4)
# plt.text(0.0, 0.0, 'Back left leg', horizontalalignment='center',verticalalignment='center', fontsize=35, fontname='Times New Roman')
plt.grid()

# plt.subplot(2,2,4)
# plt.plot(BRx[-100:], BRy[-100:],xy[2,:],xy[3,:])
# plt.xlabel('x position, back right leg',fontsize=20, fontname='Times New Roman')
# plt.ylabel('y position',fontsize=20, fontname='Times New Roman')
# plt.gca().legend(('Actual','Desired'),fontsize=15)
# # plt.text(0.0, 0.0, 'Back right leg', horizontalalignment='center',verticalalignment='center', fontsize=35, fontname='Times New Roman')
# plt.grid()

# plt.subplot(2,2,1)
# plt.plot(FLx[-100:], FLy[-100:],xy[2,:],xy[3,:],env.FLx[-100:],env.FLy[-100:])
# plt.gca().legend(('Front Left Leg Actual','Desired','Default Gait'))
# plt.subplot(2,2,2)
# plt.plot(FRx[-100:], FRy[-100:],xy[0,:],xy[1,:],env.FRx[-100:],env.FRy[-100:])
# plt.gca().legend(('Front Right Leg Actual','Desired','Default Gait'))
# plt.subplot(2,2,3)
# plt.plot(BLx[-100:], BLy[-100:],xy[0,:],xy[1,:],env.FLx[-100:],env.FLy[-100:])
# plt.gca().legend(('Back Left Leg Actual','Desired','Default Gait'))
# plt.subplot(2,2,4)
# plt.plot(BRx[-100:], BRy[-100:],xy[2,:],xy[3,:],env.FRx[-100:],env.FRy[-100:])
# plt.gca().legend(('Back Right Leg Actual','Desired','Default Gait'))

fig = plt.gcf()
fig.set_size_inches(18.5, 5.5) # if using only two subplots then fig.set_size_inches(18.5, 5.5)
# plt.savefig('GaitSimulation.png', format='png', dpi=200, bbox_inches='tight') 

plt.show()


# ###################################### Plotting from the experimental data
# ###################################### Plotting from the experimental data
# # here we plot the end effector trajectories
# # xydata_simulation = np.loadtxt('save.txt', delimiter=' ')
# ############ gather experimental data
# xydata_experiment = np.loadtxt('roman.txt', delimiter=' ')

# xy_s = xydata_experiment[:,[0,1,2,3]]
# xy_r = xydata_experiment[:,[4,5,6,7]]

# #################### Plotting front leg xy trajectory
# plt.subplot(1,2,1)
# plt.plot(xy_s[:,0], xy_s[:,1], xy_r[:,0], xy_r[:,1]);
# plt.xlabel('x position, front left leg',fontsize=20)
# plt.ylabel('y position',fontsize=20, fontname='Times New Roman')
# plt.gca().legend(('Simulation','Experiment'),fontsize=15,loc=4)
# plt.grid()

# #################### Plotting back leg xy trajectory
# plt.subplot(1,2,2)
# plt.plot(xy_s[:,2], xy_s[:,3], xy_r[:,2], xy_r[0:100,3]);
# plt.legend(('Simulation','Experiment'));
# plt.xlabel('x position, back left leg',fontsize=20)
# plt.ylabel('y position',fontsize=20, fontname='Times New Roman')
# plt.gca().legend(('Simulation','Experiment'),fontsize=15,loc=4)
# plt.grid()

# fig = plt.gcf()
# fig.set_size_inches(18.5, 5.5) # if using only two subplots then fig.set_size_inches(18.5, 5.5)
# plt.savefig('sim_v_exp.png', format='png', dpi=200, bbox_inches='tight') 

# ######################################################################################
# plt.plot(np.arange(total_data_length)*env.dt,qpos[:,[1,2]])
# plt.plot(np.arange(total_data_length)*env.dt,qdes[:,[1,2]])
# plt.show()

# pickle.dump(torque, open('torque.p', "wb"))
# pickle.dump(qpos, open('qpos.p', "wb"))
# pickle.dump(qvel, open('qvel.p', "wb"))

# Over-ride spine data
Fs = np.zeros(100)
Bs = np.zeros(100)



# Data ordering is changed here for implementing the gait in hardware
# FLX, BLX ,FRX, BRX ,FLY, BLY ,FRY ,BRY ,SPINE1 ,SPIN2
final_data_sim = np.array([FLx[-100:], BLx[-100:], FRx[-100:], BRx[-100:], FLy[-100:], BLy[-100:], FRy[-100:], BRy[-100:], Fs[-100:], Bs[-100:]]).T
final_data_poly = np.array([xy[2,:], xy[0,:], xy[0,:], xy[2,:], xy[3,:], xy[1,:], xy[1,:], xy[3,:], Fs[-100:], Bs[-100:]]).T

# Append the data
ans = input("Save xy data to file? (y/n):")

output_file_sim = 'save_sim.txt'
output_file_poly = 'save_poly.txt'

if ans=='y':
    # Clear out the file
    txt_file_sim = open(output_file_sim, 'w')
    txt_file_poly = open(output_file_poly, 'w')
    txt_file_sim.close()
    txt_file_poly.close()
    
    # Append the data
    txt_file_sim = open(output_file_sim, 'a')
    txt_file_poly = open(output_file_poly, 'a')
    
    for i in range(len(final_data_sim)):
        print_data_sim = final_data_sim[i]
        print_data_poly = final_data_poly[i]
        
        txt_file_sim.write(' '.join(str(format(j, '.4f')) for j in print_data_sim)+"\n")
        txt_file_poly.write(' '.join(str(format(j, '.4f')) for j in print_data_poly)+"\n")
        
    txt_file_sim.close()
    txt_file_poly.close()
    
    print("Saved to",output_file_sim,"and",output_file_poly)
else:
    print("Not saved")

