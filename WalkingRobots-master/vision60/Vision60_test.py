# Use this to test Stoch2 Environment
import gym
# import mujoco_py
import pickle
import numpy as np
import os
import time
import matplotlib.pyplot as plt

# Import the RL Algorithm and tools for Multiprocessing
from stable_baselines.ppo2 import PPO2
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder
# import envs.vision60_gym_mjc_env as stoch2_gym_env
import envs.vision60_gym_bullet_env as vision60_gym_bullet_env

#from stable_baselines.results_plotter import load_results, ts2xy
import warnings
warnings.filterwarnings('ignore')

# Directory for Saved models, Tensorflow log,and  Videos
dir_name = "./tmp"
if not os.path.exists(dir_name):
        os.makedirs(dir_name)

###############################################################################
# callbacks are defined here
global i
global qpos, torque
i=0
total_data_length = 1000
qpos = np.zeros((total_data_length,10))
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
env = vision60_gym_bullet_env.Vision60BulletEnv(render=True)

model_test = PPO2.load(dir_name+"/model_trot")
obs = env.reset()

print("Render mode...")

for _ in range(10):
    action, _states = model_test.predict(obs,deterministic=True)
    obs, reward, done, _ = env.step(action,callback=render_callback)
#     if done:
#         break


pickle.dump(qpos[0:total_data_length:int(total_data_length/100)], open("save.p", "wb"))  # save it into a file named save.p
# print(np.shape(qpos[0:total_data_length:int(total_data_length/100)]))
# print(np.shape(qpos))


plt.plot(np.arange(total_data_length)*env.dt,qpos[:,[1,2,3,4,6,7,8,9]])
# plt.plot(np.arange(total_data_length)*env.dt,qvel[:,[1,2,3,4,6,7,8,9]])
# plt.plot(np.arange(total_data_length)*env.dt,qpos[:,[2,9]])
# plt.plot(np.arange(total_data_length)*env.dt,torque[:,[1,2,3,4,6,7,8,9]])
# print(qpos[400:500,[1,2,3,4,6,7,8,9]])
plt.show()