import sys, os
sys.path.append(os.path.realpath('../..'))

import logging
logging.disable(logging.CRITICAL)
import numpy as np
import scipy as sp
import scipy.sparse.linalg as spLA
import copy
import time as timer
import torch
import torch.nn as nn
from torch.autograd import Variable
import copy

# samplers
import mjrl.samplers.trajectory_sampler as trajectory_sampler
import mjrl.samplers.batch_sampler as batch_sampler

# utility functions
import mjrl.utils.process_samples as process_samples
from mjrl.utils.logger import DataLog

# Import stuff for testing
import mjrl.baselines.linear_baseline as linear_baseline
import mjrl.policies.gaussian_linear as linear_policy
from mjrl.utils.gym_env import GymEnv

class BatchREINFORCE:
    def __init__(self, env, policy, baseline,
                 learn_rate=0.01,
                 seed=None,
                 save_logs=False):

        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.alpha = learn_rate
        self.seed = seed
        self.save_logs = save_logs
        self.running_score = None
        if save_logs: self.logger = DataLog()

    def compute_reinforce_loss(self, observations, actions, advantages):
        """
        Computes the REINFORCE function loss. 
        :param observations : A list containing observations of all paths together
        :param actions : A list containing actions of all paths together
        :param advantages : A list containing advantages of all paths together
        :return : Loss of all the paths combined (Loss according to reinforce algorithm log(policy) * advantage)
        """
        adv_var = Variable(torch.from_numpy(advantages).float(), requires_grad=False)
        old_dist_info = self.policy.old_dist_info(observations, actions)
        new_dist_info = self.policy.new_dist_info(observations, actions)
        log_policy = self.policy.likelihood_ratio(new_dist_info, old_dist_info)
        loss = torch.mean(log_policy*adv_var)
        return loss

    def kl_old_new(self, observations, actions):
        old_dist_info = self.policy.old_dist_info(observations, actions)
        new_dist_info = self.policy.new_dist_info(observations, actions)
        mean_kl = self.policy.mean_kl(new_dist_info, old_dist_info)
        return mean_kl

    def flat_vpg(self, observations, actions, advantages):
        """
        Finds the gradients with respect to the REINFORCE loss
        :param observations : A list containing observations of all paths together
        :param actions : A list containing actions of all paths together
        :param advantages : A list containing advantages of all paths together
        :return : A flat list containing gradients of the weights (as calculated by PyTorch)
        """
        loss = self.compute_reinforce_loss(observations, actions, advantages)
        vpg_grad = torch.autograd.grad(loss, self.policy.trainable_params)
        vpg_grad = np.concatenate([g.contiguous().view(-1).data.numpy() for g in vpg_grad])
        return vpg_grad

    def train_step(self, N,
                   sample_mode='trajectories',
                   env_name=None,
                   T=1e6,
                   gamma=0.995,
                   gae_lambda=0.98,
                   num_cpu='max'):
        """
        The agent performs a single complete step. It samples trajectories, finds gradients and updates it's weights based on those gradients
        :param N : Number of runs to get
        :param sample_mode : If trajectories, it uses trajectory sampler (N= 5, implies 5 different trajectories)
                             If samples, it uses batch smapler (N = 5, implies 5 different samples only)
        :param env_name : Name of env
        :param T : Maximum length of trajectory
        :param gamma : Discount Factor
        :param gae_lambda : Eligibility trace
        :param num_cpu : Number of cores to use (On real systems has to be 1)
        :return : A set of statistics regarding the current step. It returns [Mean_Return, Standard_Return, Min_Return, Max_Return]
        """
        # Clean up input arguments
        if env_name is None: env_name = self.env.env_id
        if sample_mode != 'trajectories' and sample_mode != 'samples':
            print("sample_mode in NPG must be either 'trajectories' or 'samples'")
            quit()

        ts = timer.time()

        if sample_mode == 'trajectories':
            paths = trajectory_sampler.sample_paths_parallel(N, self.policy, T, env_name,
                                                             self.seed, num_cpu)
        elif sample_mode == 'samples':
            paths = batch_sampler.sample_paths(N, self.policy, T, env_name=env_name,
                                               pegasus_seed=self.seed, num_cpu=num_cpu)

        if self.save_logs:
            self.logger.log_kv('time_sampling', timer.time() - ts)

        self.seed = self.seed + N if self.seed is not None else self.seed

        # compute returns
        process_samples.compute_returns(paths, gamma)
        # compute advantages
        process_samples.compute_advantages(paths, self.baseline, gamma, gae_lambda)
        # train from paths
        eval_statistics = self.train_from_paths(paths)
        eval_statistics.append(N)
        # fit baseline
        if self.save_logs:
            ts = timer.time()
            error_before, error_after = self.baseline.fit(paths, return_errors=True)
            self.logger.log_kv('time_VF', timer.time()-ts)
            self.logger.log_kv('VF_error_before', error_before)
            self.logger.log_kv('VF_error_after', error_after)
        else:
            self.baseline.fit(paths)

        return eval_statistics

    # ----------------------------------------------------------
    def train_from_paths(self, paths):
        """ Performs the gradient step for a given set of paths
        :param paths : Paths refers to a list of dictionaries as output by samplers
        :return : Returns a list that contains statistics for the paths. [Mean_Return, Standard_Return, Min_Return, Max_Return]
        The gradient update step is performed internally
        """
        # Concatenate from all the trajectories
        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        advantages = np.concatenate([path["advantages"] for path in paths])
        # Advantage whitening
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-6)

        # cache return distributions for the paths
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        base_stats = [mean_return, std_return, min_return, max_return]
        self.running_score = mean_return if self.running_score is None else \
                             0.9*self.running_score + 0.1*mean_return  # approx avg of last 10 iters
        if self.save_logs: self.log_rollout_statistics(paths)

        # Keep track of times for various computations
        t_gLL = 0.0

        # Optimization algorithm
        # --------------------------
        loss_before_training = self.compute_reinforce_loss(observations, actions, advantages).data.numpy().ravel()[0]

        # VPG
        ts = timer.time()
        vpg_grad = self.flat_vpg(observations, actions, advantages)
        t_gLL += timer.time() - ts

        # Policy update
        # --------------------------
        curr_params = self.policy.get_param_values()
        new_params = curr_params + self.alpha * vpg_grad
        self.policy.set_param_values(new_params, set_new=True, set_old=False)
        loss_after_training = self.compute_reinforce_loss(observations, actions, advantages).data.numpy().ravel()[0]
        kl_dist = self.kl_old_new(observations, actions).data.numpy().ravel()[0]
        self.policy.set_param_values(new_params, set_new=True, set_old=True)

        # Log information
        if self.save_logs:
            self.logger.log_kv('alpha', self.alpha)
            self.logger.log_kv('time_vpg', t_gLL)
            self.logger.log_kv('kl_dist', kl_dist)
            self.logger.log_kv('loss_improvement', loss_after_training - loss_before_training)
            self.logger.log_kv('running_score', self.running_score)
            try:
                self.env.env.env.evaluate_success(paths, self.logger)
            except:
                # nested logic for backwards compatibility. TODO: clean this up.
                try:
                    success_rate = self.env.env.env.evaluate_success(paths)
                    self.logger.log_kv('success_rate', success_rate)
                except:
                    pass

        return base_stats

    def log_rollout_statistics(self, paths):
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        self.logger.log_kv('stoc_pol_mean', mean_return)
        self.logger.log_kv('stoc_pol_std', std_return)
        self.logger.log_kv('stoc_pol_max', max_return)
        self.logger.log_kv('stoc_pol_min', min_return)


if(__name__ == "__main__"):
    env_name = "Pendulum-v0"
    env = GymEnv(env_name)
    baseline = linear_baseline.LinearBaseline(env.spec)
    policy = linear_policy.LinearPolicy(env.spec)
    learner = BatchREINFORCE(env, policy, baseline)
    N = 5
    T = 5
    gamma = 1
    paths = trajectory_sampler.sample_paths(N, policy, T, env)
    process_samples.compute_returns(paths, gamma)
    process_samples.compute_advantages(paths, baseline, gamma)
    baseline.fit(paths)

    observations = np.concatenate([path["observations"] for path in paths])
    actions = np.concatenate([path["actions"] for path in paths])
    advantages = np.concatenate([path["advantages"] for path in paths])

    loss = learner.compute_reinforce_loss(observations, actions, advantages)
    grads = learner.flat_vpg(observations, actions, advantages)
    stats = learner.train_from_paths(paths)
    stats = learner.train_step(1, env_name='Pendulum-v0')
    print('stats: ', stats)
    # print('loss: ',loss)
    # print('grads: ', grads)
    pass