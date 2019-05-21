import pybullet_envs
import gym
import time
import numpy as np
env = gym.make("MinitaurTrottingEnv-v0")
env.reset()
start_time = time.time()
env.step(np.zeros(8))
print("Time taken to reset: ", time.time() - start_time)