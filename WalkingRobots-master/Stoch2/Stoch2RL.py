# Use this to test Stoch2 Environment
import gym
# import mujoco_py
import numpy as np
import os
import time
import tensorflow as tf

# Import the RL Algorithm and tools for Multiprocessing
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.ppo2 import PPO2
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder
from stable_baselines.common import set_global_seeds
import envs.stoch2_gym_mjc_env as stoch2_gym_env
# import envs.stoch2_gym_bullet_env as stoch2_gym_env

#from stable_baselines.results_plotter import load_results, ts2xy
import warnings
warnings.filterwarnings('ignore')

# Directory for Saved models, Tensorflow log,and  Videos
dir_name = "./tmp"
tflow_log = "./tmp/tflow_log/"
# video_dir =  "./tmp/videos"
if not os.path.exists(dir_name):
        os.makedirs(dir_name)
if not os.path.exists(tflow_log):
        os.makedirs(tflow_log)
# if not os.path.exists(video_dir):
#         os.makedirs(video_dir)

###############################################################################
# callbacks are defined here
n_steps = 0
def callback(_locals, _globals):
  """
  Callback called at each step for PPO2
  :param _locals: (dict)
  :param _globals: (dict)
  """
  global n_steps
  # Save the model every ~10^{something} steps and record a video
  print(n_steps)
  if (n_steps + 1) % 5 == 0:
      print('Saving model...')
      _locals['self'].save(tflow_log+"/model_trot")
      # model_test = PPO2.load(dir_name+"/model")
      # print("Entering the render mode...")
      # Record the Video
      # obs = env_test.reset()
#       for j in range(video_length+1):
#           action, _states = model_test.predict(obs)
#           # obs, rewards, done, info = env_test.step(action)
# #           print(action)
#           env_test.render()
#       env_test.reset()
      # del model_test
  n_steps += 1
  return True

###############################################################################
# Function to Create Vectorized Environment
def make_env(rank, seed=0):
    """
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = stoch2_gym_env.Stoch2Env()
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

###############################################################################
# Create Vectorized Environment for Multiprocessing
# Define the number of processes to use
num_cpu = 6
# Create the vectorized environment
env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

# Custom MLP policy of two layers of size 32 each with tanh activation function
policy_kwargs = dict(act_fun=tf.nn.relu, net_arch=[64, 64])
model = PPO2(MlpPolicy, env,policy_kwargs=policy_kwargs, tensorboard_log= tflow_log, verbose=0) #if tf log required add: 

###############################################################################
# Start training
print("RL Training begins....")
model.learn(total_timesteps=3*10**7,tb_log_name="log",callback=callback)
