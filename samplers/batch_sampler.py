import sys, os
sys.path.append(os.path.realpath('../..'))

import logging
logging.disable(logging.CRITICAL)

import numpy as np
import time as timer
import pybRL.samplers.base_sampler as base_sampler
import pybRL.samplers.evaluation_sampler as eval_sampler
import pybRL.samplers.trajectory_sampler as trajectory_sampler
from pybRL.utils.get_environment import get_environment

def sample_paths(N,
    policy,
    T=1e6,
    env=None,
    env_name=None,
    pegasus_seed=None,
    num_cpu='max',
    paths_per_call=5,
    mode='sample'):
    """
    Given the number of sample points desired, it returns a bunch of paths whose total sample points exceed the number desired.
    It tries to play an episode till completion, doesn't stop midway if samples have been collected
    params:
    N               : number of sample points
    policy          : policy to be used to sample the data
    T               : maximum length of trajectory
    env             : env object to sample from
    env_name        : name of env to be sampled from 
                      (one of env or env_name must be specified)
    pegasus_seed    : seed for environment (numpy speed must be set externally)

    :return : paths , a list of dictionaries as defined in base_sampler rollout
    """

    if num_cpu == 1:
        return sample_paths_one_core(N, policy, T, env, env_name, pegasus_seed, mode)
    else:
        start_time = timer.time()
        print("####### Gathering Samples #######")
        sampled_so_far = 0
        paths_so_far = 0
        paths = []
        while sampled_so_far <= N:
            if pegasus_seed is None:
                new_paths = trajectory_sampler.sample_paths_parallel(paths_per_call,
                            policy, T, env_name, pegasus_seed, num_cpu, suppress_print=True, mode=mode)

            else:
                pegasus_seed += paths_so_far
                new_paths = trajectory_sampler.sample_paths_parallel(paths_per_call,
                            policy, T, env_name, pegasus_seed, num_cpu, suppress_print=True, mode=mode)

            for path in new_paths:
                paths.append(path)
            paths_so_far += paths_per_call
            new_samples = np.sum([len(p['rewards']) for p in new_paths])
            sampled_so_far += new_samples
        print("======= Samples Gathered  ======= | >>>> Time taken = %f " % (timer.time()-start_time) )
        print("................................. | >>>> # samples = %i # trajectories = %i " % (sampled_so_far, paths_so_far) )
        return paths

def sample_paths_one_core(N,
    policy,
    T=1e6,
    env=None,
    env_name=None,
    pegasus_seed=None,
    mode='sample'):
    """
    params:
    N               : number of sample points
    policy          : policy to be used to sample the data
    T               : maximum length of trajectory
    env             : env object to sample from
    env_name        : name of env to be sampled from 
                      (one of env or env_name must be specified)
    pegasus_seed    : seed for environment (numpy speed must be set externally)
    """

    if env_name is None and env is None:
        print("No environment specified! Error will be raised")
    if env is None: env = get_environment(env_name)
    if pegasus_seed is not None: env.env._seed(pegasus_seed)
    T = min(T, env.horizon) 

    start_time = timer.time()

    print("####### Gathering Samples #######")
    sampled_so_far = 0
    paths = []
    seed = pegasus_seed if pegasus_seed is not None else 0

    while sampled_so_far < N:
        if mode == 'sample':
            this_path = base_sampler.do_rollout(1, policy, T, env, env_name, seed) # do 1 rollout
        elif mode == 'evaluation':
            this_path = eval_sampler.do_evaluation_rollout(1, policy, env, env_name, seed)
        else:
            print("Mode has to be either 'sample' for training time or 'evaluation' for test time performance")
            break
        paths.append(this_path[0])
        seed += 1
        sampled_so_far += len(this_path[0]["rewards"])

    print("======= Samples Gathered  ======= | >>>> Time taken = %f " % (timer.time()-start_time) )
    print("................................. | >>>> # samples = %i # trajectories = %i " % (sampled_so_far, len(paths)) )
    return paths

if(__name__ == "__main__"):
    # N = 100
    # pol = base_sampler.RandomPolicy(2)
    # T = 5
    # env = None
    # env_name = "CartPole-v0"
    # paths = sample_paths(N,pol, T, env, env_name, num_cpu='max')
    # print(paths)

    # N = 10
    # pol = base_sampler.RandomPolicy((-2,2))
    # T = 5
    # env = None
    # env_name = "Pendulum-v0"
    # paths = sample_paths(N,pol, T, env, env_name, num_cpu=1)
    # print(paths)


    pass