import sys, os
sys.path.append(os.path.realpath('../..'))

import pybRL.utils.train_agent as train_agent
# import pybRL.algos.batch_reinforce as batch_reinforce
from pybRL.utils.gym_env import GymEnv
import pybRL.samplers.trajectory_sampler as trajectory_sampler

import pybRL.policies.gaussian_linear as linear_policy
import pybRL.policies.RBF_linear as RBF_linear
import pybRL.baselines.linear_baseline as linear_baseline
import pybRL.baselines.mlp_baseline as MLPBaseline

import pybRL.algos.npg_cg as npg
import pybRL.algos.ppo_clip as ppo

import pybRL.samplers.base_sampler as base_sampler
import pybRL.utils.process_samples as process_samples

import pybullet_envs
# import pybulletgym

SEED = 500
ENV_ID = 'MinitaurTrottingEnv-v0'
lr = 1e-2
env = GymEnv(ENV_ID)
#Getting the average pairwise distance -- You need to only run this once, then just manually save it's value in mean_dist variable for a particular environment
# sample_policy = linear_policy.LinearPolicy(env.spec)
# paths = base_sampler.do_rollout(100, sample_policy,T=100, env = env)
# mean_dist = process_samples.get_avg_step_distance(paths)
mean_dist = 1.8115

policy = linear_policy.LinearPolicy(env.spec)
# policy = RBF_linear.RBFLinearPolicy(env.spec, RBF_number=500, avg_pairwise_distance= mean_dist )
# baseline = linear_baseline.LinearBaseline(env.spec)
baseline = MLPBaseline.MLPBaseline(env.spec, reg_coef=1e-3, batch_size=64, epochs=5, learn_rate=5e-3)
# agent = batch_reinforce.BatchREINFORCE(env, policy, baseline, learn_rate=5e-3, seed = None, save_logs=True)
agent = npg.NPG(env, policy, baseline, normalized_step_size=1e-2, seed=None, hvp_sample_frac=1.0, save_logs=True, kl_dist=None, FIM_invert_args={'iters': 50, 'damping': 1e-4},)
# agent = ppo.PPO(env, policy, baseline, clip_coef=0.2, epochs=10, mb_size=64, learn_rate=3e-4, save_logs=True)
args = dict(
                job_name = 'MinitaurTrot_exp5',
                agent = agent,
                seed = 0,
                niter = 5000,
                gamma = 0.995,
                gae_lambda = 1,
                num_cpu = 1,
                sample_mode = 'trajectories',
                num_traj = 10,
                # num_samples = 50000, # has precedence, used with sample_mode = 'samples'
                save_freq = 20,
                evaluation_rollouts = None,
                plot_keys = ['stoc_pol_mean','eval_score'],
 )
train_agent.train_agent(**args)

# paths = trajectory_sampler.sample_paths_parallel(10, policy, env_name = ENV_ID)
# print(paths)