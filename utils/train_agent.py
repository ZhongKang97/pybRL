import sys, os
sys.path.append(os.path.realpath('../..'))

import logging
logging.disable(logging.CRITICAL)

from tabulate import tabulate
from pybRL.utils.make_train_plots import make_train_plots
from pybRL.utils.gym_env import GymEnv
from pybRL.samplers.trajectory_sampler import sample_paths_parallel
import numpy as np
import pickle
import time as timer
import os
import copy

def train_agent(job_name, agent,
                seed = 0,
                niter = 101,
                gamma = 0.995,
                gae_lambda = None,
                num_cpu = 1,
                sample_mode = 'trajectories',
                num_traj = 50,
                num_samples = 50000, # has precedence, used with sample_mode = 'samples'
                save_freq = 10,
                evaluation_rollouts = None,
                plot_keys = ['stoc_pol_mean'],
                ):
    """
    Trains an agent in the given environment, on a given number of iterations
    :params ------------------------------------------------------------------------------------------------------------------------
    job_name : Name of the directory where you want to save the training data
    agent : An object of the class algos. Basically it should have the train_step function ready
    seed : Used for randomizing the env
    niter : Number of times you want to train the agent
    gamma : Discount factor for the environment
    gae_lambda : Eligibility Trace for the environment
    num_cpu : Number of cores you want to use
    sample_mode : Can use 'trajectory' or 'samples', the first will sample trajectories. The second will take samples
    num_traj : If using 'trajectory' mode, this specifies the number of trajectories to be used for each training step
    num_samples : If using 'samples' mode, this specifies the number of samples to be used in each training step
    save_freq : The frequency at which to save the policy, baselines, best policy till now
    evaluation_rollouts : Number of rollouts to perform to just evaluate the policy, at every step (Leace it none if you want 0])
    plot_keys : The values that you want to plot
    ----------------------------------------------------------------------------------------------------------------------------------
    :return : Doesn't return anything, performs all the training on it's own
    """
    np.random.seed(seed)
    if os.path.isdir(job_name) == False:
        os.mkdir(job_name)
    previous_dir = os.getcwd()
    os.chdir(job_name) # important! we are now in the directory to save data
    if os.path.isdir('iterations') == False: os.mkdir('iterations')
    if os.path.isdir('logs') == False and agent.save_logs == True: os.mkdir('logs')
    best_policy = copy.deepcopy(agent.policy)
    best_perf = -1e8
    train_curve = best_perf*np.ones(niter)
    e = GymEnv(agent.env.env_id)
    running_return = 0
    mean_pol_perf = 0
    for i in range(niter):
        print("......................................................................................")
        print("ITERATION : %i " % i)
        if train_curve[i-1] > best_perf:
            best_policy = copy.deepcopy(agent.policy)
            best_perf = train_curve[i-1]
        N = num_traj if sample_mode == 'trajectories' else num_samples
        args = dict(N=N, sample_mode=sample_mode, gamma=gamma, gae_lambda=gae_lambda, num_cpu=num_cpu)
        stats = agent.train_step(**args)
        train_curve[i] = stats[0] #mean return is in train_curve
        if(i == 0):
            running_return = train_curve[i]
        running_return = 0.9*running_return + 0.1*train_curve[i]
        if evaluation_rollouts is not None and evaluation_rollouts > 0:
            print("Performing evaluation rollouts ........")
            eval_paths = sample_paths_parallel(N=evaluation_rollouts, policy=agent.policy, num_cpu=num_cpu,
                                               env_name=e.env_id, mode='evaluation', pegasus_seed=seed)
            mean_pol_perf = np.mean([np.sum(path['rewards']) for path in eval_paths])
            if agent.save_logs:
                agent.logger.log_kv('eval_score', mean_pol_perf)
        if i % save_freq == 0 and i > 0:
            if agent.save_logs:
                agent.logger.save_log('logs/')
                make_train_plots(log=agent.logger.log, keys=plot_keys, save_loc='logs/')
            policy_file = 'policy_%i.pickle' % i
            baseline_file = 'baseline_%i.pickle' % i
            pickle.dump(agent.policy, open('iterations/' + policy_file, 'wb'))
            pickle.dump(agent.baseline, open('iterations/' + baseline_file, 'wb'))
            pickle.dump(best_policy, open('iterations/best_policy.pickle', 'wb'))
        # print results to console
        if i == 0:
            result_file = open('results.txt', 'w')
            print("Iter (all means) | Current Return | Mean Return | Best Return \n")
            result_file.write("Iter | Sampling Pol | Evaluation Pol | Best (Sampled) \n")
            result_file.close()
        print(f"Current Return :{round(train_curve[i],2)} | Mean Return :{round(running_return,2)} | Best Return : {round(best_perf,2)}  ")
        result_file = open('results.txt', 'a')
        result_file.write("%4i %5.2f %5.2f %5.2f \n" % (i, train_curve[i], mean_pol_perf, best_perf))
        result_file.close()
        if agent.save_logs:
            print_data = sorted(filter(lambda v: np.asarray(v[1]).size == 1,
                                       agent.logger.get_current_log().items()))
            # print(tabulate(print_data))

    # final save
    pickle.dump(best_policy, open('iterations/best_policy.pickle', 'wb'))
    if agent.save_logs:
        agent.logger.save_log('logs/')
        make_train_plots(log=agent.logger.log, keys=plot_keys, save_loc='logs/')
    os.chdir(previous_dir)


if(__name__ == "__main__"):
    pass