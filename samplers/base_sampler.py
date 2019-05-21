import sys, os
sys.path.append(os.path.realpath('../..'))
import logging
logging.disable(logging.CRITICAL)

import numpy as np
from pybRL.utils.get_environment import get_environment
from pybRL.utils import tensor_utils

#imports for checking optimization
import time

# Single core rollout to sample trajectories
# =======================================================
class RandomPolicy:
    def __init__(self, action_dim):
        self.action_dim = action_dim
    
    def get_action(self,o):
        if(type(self.action_dim) is tuple):
            return [[np.random.uniform(self.action_dim[0], self.action_dim[1])], {'Random Agent':np.ones(1)}]

        else:
            return [np.random.choice(range(self.action_dim)),{'Random Agent' : np.ones(self.action_dim) / self.action_dim}]
def do_rollout(N,
    policy,
    T=1e6,
    env=None,
    env_name=None,
    pegasus_seed=None):
    """
    params:
    N               : number of trajectories
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
    if pegasus_seed is not None: 
        try:
            env.env._seed(pegasus_seed)
        except AttributeError as e:
            env.env.seed(pegasus_seed)
    #Gonna change only for pybullet
    # T = min(T, env.horizon) 
    T=T

    # print("####### Worker started #######")
    
    paths = []

    for ep in range(N):
        start_time = time.time()
        # Set pegasus seed if asked
        # if pegasus_seed is not None:
        #     seed = pegasus_seed + ep
        #     try:
        #         env.env._seed(seed)
        #     except AttributeError as e:
        #         env.env.seed(seed)
        #     np.random.seed(seed)
        # else:
        #     np.random.seed()
        
        observations=[]
        actions=[]
        rewards=[]
        agent_infos = []
        env_infos = []

        o = env.reset()
        done = False
        t = 0
        while t < T and done != True:
            start_time = time.time()
            a, agent_info = policy.get_action(o)
            # print('time to get action: ', time.time() - start_time)
            start_time = time.time()
            next_o, r, done, env_info = env.step(a)
            #observations.append(o.ravel())
            observations.append(o)
            actions.append(a)
            rewards.append(r)
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            # print('time to play action: ', time.time() - start_time)
            o = next_o
            t += 1
        path = dict(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
            env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
            terminated=done
        )

        paths.append(path)

    # print("====== Worker finished ======")
    del(env)
    return paths

def do_rollout_star(args_list):
    """ Performs rollout taking arguments as a list """
    return do_rollout(*args_list)


if(__name__ == "__main__"):
    # pol = RandomPolicy((-2, 2))
    # a = pol.get_action(1)
    # print(a)
    # pol = RandomPolicy(2)
    # a = pol.get_action(1)
    # print(a)

    # N = 1
    # pol = RandomPolicy(2)
    # T = 50
    # env_name = 'CartPole-v1'
    # y = do_rollout(N = N, policy = pol, T = T, env_name=env_name)
    # y = do_rollout_star([2, pol, 5, None, env_name ])
    # print(y)

    N = 1
    pol = RandomPolicy((-2,2))
    T = 5
    env_name = 'Pendulum-v0'
    y = do_rollout(N = N, policy = pol, T = 5, env_name=env_name)
    print(y)
    pass