import sys, os
sys.path.append(os.path.realpath('../..'))

import numpy as np
import mjrl.samplers.base_sampler as base_sampler
import mjrl.baselines.linear_baseline as linear_baseline
from mjrl.utils.gym_env import GymEnv
def compute_returns(paths, gamma):
    """
    Computes returns for a given trajectory sample
    :param paths : A list of dictionaries as defined by base_sampler in samplers
    :param gamma : The discount factor
    :return : Doesn't return anything. Modifies the paths dictionary sent inside, adding a return term to each dictionary in the list
    """
    for path in paths:
        path["returns"] = discount_sum(path["rewards"], gamma)

def compute_advantages(paths, baseline, gamma, gae_lambda=None, normalize=False):
    """
    Computes advantages using either TD approximate, or normal. In TD Approximate, eligibility traces has been implemented
    :param paths : This is a list of dictionaries as returned by base_sampler
    :param baseline : This is baseline object from baseline class. It is used to calculate value function of each state
    :param gamma : discount factor to calculate returns
    :param gae_lambda : This is eligibility trace value. (If it is any value not between [0,1], regular value function estimate is used
    in the advantage function. Else TD estimate is used in the advantage function. GAE stands for generalized advantage estimation.)
    :param normalize : Choice of normalizing the advantages by their mean and variance. (USeful for control tasks with different units)
    :return : Will compute advantage (Return -  Baseline) if not using gae or TD approximate if using GAE. Advantages are added to the path dictionary
    Nothing is returned from the function 
    """
    if gae_lambda == None or gae_lambda < 0.0 or gae_lambda > 1.0:
        for path in paths:
            path["baseline"] = baseline.predict(path)
            path["advantages"] = path["returns"] - path["baseline"]
        if normalize:
            alladv = np.concatenate([path["advantages"] for path in paths])
            mean_adv = alladv.mean()
            std_adv = alladv.std()
            for path in paths:
                path["advantages"] = (path["advantages"]-mean_adv)/(std_adv+1e-8)
    # GAE mode
    else:
        for path in paths:
            b = path["baseline"] = baseline.predict(path)
            if b.ndim == 1:
                b1 = np.append(path["baseline"], 0.0 if path["terminated"] else b[-1])
            else:
                b1 = np.vstack((b, np.zeros(b.shape[1]) if path["terminated"] else b[-1]))
            td_deltas = path["rewards"] + gamma*b1[1:] - b1[:-1]
            path["advantages"] = discount_sum(td_deltas, gamma*gae_lambda)
        if normalize:
            alladv = np.concatenate([path["advantages"] for path in paths])
            mean_adv = alladv.mean()
            std_adv = alladv.std()
            for path in paths:
                path["advantages"] = (path["advantages"]-mean_adv)/(std_adv+1e-8)

def discount_sum(x, gamma, terminal=0.0):
    """
    Caculates discounted sum for each timestep of an array
    :param x : An array containing the reward obtained at each timestep
    :param gamma : Discount factor between 0 to 1
    :param terminal : Reward obtained at the final step
    :return : An array containing the discounted reward at each timestep
    """
    y = []
    run_sum = terminal
    for t in range( len(x)-1, -1, -1):
        run_sum = x[t] + gamma*run_sum
        y.append(run_sum)

    return np.array(y[::-1]) # Basically that indexing reverses the array

if(__name__ == "__main__"):

    # x = [0,0,0,0,1]
    # gamma = 0.9
    # y = discount_sum(x, gamma)
    # print(y)
    
    N = 1
    pol = base_sampler.RandomPolicy((-2,2))
    T = 5
    env = None
    env_name = "Pendulum-v0"
    env = GymEnv(env_name)
    gamma = 0.9
    paths = base_sampler.do_rollout(N,pol, T, env)
    compute_returns(paths, gamma)
    baseline = linear_baseline.LinearBaseline(env.spec)
    compute_advantages(paths, baseline, gamma, gae_lambda=0.9 )
    print(paths)