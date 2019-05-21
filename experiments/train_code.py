import sys, os
sys.path.append(os.path.realpath('../..'))

import pybRL.utils.train_agent as train_agent
# import pybRL.algos.batch_reinforce as batch_reinforce
from pybRL.utils.gym_env import GymEnv
import pybRL.samplers.trajectory_sampler as trajectory_sampler

import pybRL.policies.gaussian_linear as linear_policy

import pybRL.baselines.linear_baseline as linear_baseline
import pybRL.baselines.mlp_baseline as MLPBaseline

import pybRL.algos.npg_cg as npg
import pybRL.algos.ppo_clip as ppo

import pybullet_envs
import pybulletgym
SEED = 500
ENV_ID = 'HalfCheetahPyBulletEnv-v0'
lr = 1e-2
env = GymEnv(ENV_ID)
policy = linear_policy.LinearPolicy(env.spec)
# baseline = linear_baseline.LinearBaseline(env.spec)
baseline = MLPBaseline.MLPBaseline(env.spec, reg_coef=1e-3, batch_size=64, epochs=5, learn_rate=1e-3)
# agent = batch_reinforce.BatchREINFORCE(env, policy, baseline, learn_rate=5e-3, seed = None, save_logs=True)
agent = npg.NPG(env, policy, baseline, normalized_step_size=0.01, seed=None, hvp_sample_frac=1.0, save_logs=True, kl_dist=None)
# agent = ppo.PPO(env, policy, baseline, clip_coef=0.2, epochs=10, mb_size=64, learn_rate=3e-4, save_logs=True)
args = dict(
                job_name = 'HalfCheetah_exp1',
                agent = agent,
                seed = 0,
                niter = 500,
                gamma = 0.995,
                gae_lambda = 0.97,
                num_cpu = 1,
                sample_mode = 'trajectories',
                num_traj = 10,
                # num_samples = 50000, # has precedence, used with sample_mode = 'samples'
                save_freq = 5,
                evaluation_rollouts = 5,
                plot_keys = ['stoc_pol_mean','eval_score'],
 )
train_agent.train_agent(**args)

# paths = trajectory_sampler.sample_paths_parallel(10, policy, env_name = ENV_ID)
# print(paths)