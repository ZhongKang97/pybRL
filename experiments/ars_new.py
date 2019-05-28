import sys, os
sys.path.append(os.path.realpath('../..'))
from pybRL.utils.gym_env import GymEnv
import argparse
from multiprocessing import Process, Pipe
import multiprocessing as mp
import time
import pybullet_envs
from gym import wrappers
import gym
import numpy as np
import os
import inspect
from pybRL.utils.logger import DataLog
from pybRL.utils.make_train_plots import make_train_plots_ars
from gym.envs.registration import registry, register, make, spec

# currentdir = os.path.dirname(os.path.abspath(
#     inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(os.path.dirname(currentdir))
# os.sys.path.insert(0, parentdir)

# Importing the libraries

# Imports for testing mainly

# Setting the Hyper Parameters


class HyperParameters():
    """
    This class is basically a struct that contains all the hyperparameters that you want to tune
    """
    def __init__(self, nb_steps=10000, episode_length=1000, learning_rate=0.02, nb_directions=16, nb_best_directions=8, noise=0.03, seed=1, env_name='HalfCheetahBulletEnv-v0'):
        self.nb_steps = nb_steps
        self.episode_length = episode_length
        self.learning_rate = learning_rate
        self.nb_directions = nb_directions
        self.nb_best_directions = nb_best_directions
        assert self.nb_best_directions <= self.nb_directions
        self.noise = noise
        self.seed = seed
        self.env_name = env_name
    
    def to_text(self, path):
        res_str = ''
        res_str = res_str + 'learning_rate: ' + str(self.learning_rate) + '\n'
        res_str = res_str + 'noise: ' + str(self.noise) + '\n'
        res_str = res_str + 'env_name: ' + str(self.env_name) + '\n'
        res_str = res_str + 'episode_length: ' + str(self.episode_length) + '\n'
        fileobj = open(path, 'w')
        fileobj.write(res_str)
        fileobj.close()
 
# Normalizing the states
class Normalizer():

    def __init__(self, nb_inputs):
        """
        self.n : Keeps a count of number of steps into the training
        self.mean : numpy array keeps a list of means for each dimension of the state space
        self.mean_diff : numpy array keeps a list of something needed to calculate variance for each dimension of the state space
        self.var : numpy array keeps a list of variances for each dimension of the state space
        """
        self.n = np.zeros(nb_inputs)
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.var = np.zeros(nb_inputs)

    def observe(self, x):
        """ 
        Maintains running average of means and variances for each state dimension.
        :param x: current observation as returned by OpenAI gym environment
        :returns : Nothing. Internally updates the mean and variance for each state dimension
        """
        self.n += 1.
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min=1e-2)

    def normalize(self, inputs):
        """ 
        Normalizing the states by it's mean and std_dev
        :param inputs: state returned by OpenAI gym environment
        :return : numpy array of normalized states
        """
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std


# Building the AI


class Policy():

    def __init__(self, input_size, output_size, env_name, args):
        try:
            self.theta = np.load(args.policy)
        except:
            self.theta = np.zeros((output_size, input_size))
        self.env_name = env_name
        print("Starting policy theta=", self.theta)

    def evaluate(self, input, delta, direction, hp):
        """
        Computes the action given movement in a random direction
        :param input : Observation from the OpenAI gym environment
        :param delta : It is	a random vector in the policy space (as returned by sample_deltas)
        :param direction : Choose action based on forward or backward direction
        :param hp: an object of type hp(Hyper-parameters)
        :returns : an action (clipped between -1 and 1)
        """
        if direction is None:
            return np.clip(self.theta.dot(input), -1.0, 1.0)
        elif direction == "positive":
            return np.clip((self.theta + hp.noise * delta).dot(input), -1.0, 1.0)
        else:
            return np.clip((self.theta - hp.noise * delta).dot(input), -1.0, 1.0)

    def sample_deltas(self, hp):
        """
        Computes a bunch of random directions in the parameter space 
        :param : nothing but does depend on hyperparameters defined globally (not sent into the object)
        :returns : A bunch of random vectors in the parameter space of the policy, they are of the same shape as the policy
        """
        return [np.random.randn(*self.theta.shape) for _ in range(hp.nb_directions)]

    def update(self, rollouts, sigma_r, args):
        """
        Need to test
        Updates policy according to augmented random search algorithm 
        :param rollouts : A 3*n array where each row consists of (reward_positive, reward_negative, delta)
        :param sigma_r : Variance of the rewards
        :args : Mainly for saving data 
        """
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, d in rollouts:
            step += (r_pos - r_neg) * d
        self.theta += hp.learning_rate / \
            (hp.nb_best_directions * sigma_r) * step
        timestr = time.strftime("%Y%m%d-%H%M%S")
        # np.save(args.logdir + "/policy_" + self.env_name +
        #         "_" + timestr + ".npy", self.theta)


def explore(env, normalizer, policy, direction, delta, hp):
    """
    Exploring the policy on one specific direction and over one episode
    :param env : OpenAI Gym environment
    :param normalizer : An object of class Normalizer
    :param policy : An object of class policy
    :param direction : 'positive' or 'negative' or None
    :param delta : Direction in the policy space that you want to explore
    :param hp : object of class Hp
    :return : Returns a python float that indicates the return over one episode (Undiscounted)	
    """
    state = env.reset()
    done = False
    num_plays = 0.
    sum_rewards = 0
    while not done and num_plays < hp.episode_length:
        normalizer.observe(state)
        state = normalizer.normalize(state)
        action = policy.evaluate(state, delta, direction, hp)
        state, reward, done, _ = env.step(action)
        reward = max(min(reward, 1), -1)
        sum_rewards += reward
        num_plays += 1
    return (sum_rewards, num_plays)


def train(env, policy, normalizer, hp, job_name = "default_exp"):
    """
    Training using Augmented Random Search
    :param env : OpenAI gym environment
    :param policy : Object of class Policy
    :param normalizer : Object of class normalizer
    :param hp : Object of class hp
    :param job_name : Name of the directory where you want to save data
    :returns : Nothing, trains the agent  
    """
    logger = DataLog()
    total_steps = 0
    best_return = -99999999
    if os.path.isdir(job_name) == False:
        os.mkdir(job_name)
    previous_dir = os.getcwd()
    os.chdir(job_name)
    if os.path.isdir('iterations') == False: os.mkdir('iterations')
    if os.path.isdir('logs') ==False: os.mkdir('logs')
    hp.to_text('hyperparameters')
    for step in range(hp.nb_steps):
        # Initializing the perturbations deltas and the positive/negative rewards
        deltas = policy.sample_deltas(hp)
        positive_rewards = [0] * hp.nb_directions
        negative_rewards = [0] * hp.nb_directions

    # Getting the positive rewards in the positive directions
        for k in range(hp.nb_directions):
            positive_rewards[k], step_count_positive = explore(
                env, normalizer, policy, "positive", deltas[k], hp)
            # break
            # print('done: ',k)
        # Getting the negative rewards in the negative/opposite directions
        for k in range(hp.nb_directions):
            negative_rewards[k], step_count_negative = explore(
                env, normalizer, policy, "negative", deltas[k], hp)
            # break
            # print('done: ', k)   
        total_steps = total_steps + step_count_positive + step_count_negative

        # Sorting the rollouts by the max(r_pos, r_neg) and selecting the best directions
        scores = {
            k: max(r_pos, r_neg)
            for k, (r_pos, r_neg) in enumerate(zip(positive_rewards, negative_rewards))
        }
        order = sorted(scores.keys(), key=lambda x: -
                        scores[x])[:hp.nb_best_directions]
        rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k])
                    for k in order]
        
        # Gathering all the positive/negative rewards to compute the standard deviation of these rewards
        all_rewards = np.array([x[0] for x in rollouts] + [x[1] for x in rollouts])
        sigma_r = all_rewards.std() # Standard deviation of only rewards in the best directions is what it should be

        # Updating our policy
        policy.update(rollouts, sigma_r, args)
        # Printing the final reward of the policy after the update
        reward_evaluation, _ = explore(
            env, normalizer, policy, None, None, hp)
        logger.log_kv('steps', total_steps)
        logger.log_kv('return', reward_evaluation)
        if(reward_evaluation > best_return):
            best_policy = policy.theta
            best_return = reward_evaluation
            np.save("iterations/best_policy.npy",best_policy )
        print('Step:', step, 'Reward:', reward_evaluation)
        policy_path = "iterations/" + "policy_"+str(step)
        np.save(policy_path, policy.theta)
        logger.save_log('logs/')
        make_train_plots_ars(log = logger.log, keys=['steps', 'return'], save_loc='logs/')



# Running the main code


def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def tests():
    hp = HyperParameters()
    hp.nb_directions = 1
    pol = Policy(3, 5, 'test', None)
    res = pol.sample_deltas(hp)
    print(res)

    state = np.ones(3)
    pol = Policy(3, 5, 'test', None)
    hp = HyperParameters()
    res = pol.sample_deltas(hp)
    direction = 'positive'
    deltas = res[0]
    action = pol.evaluate(state, deltas, direction, hp)
    print(action)

    env = gym.make('Pendulum-v0')
    env = GymEnv('Pendulum-v0')
    pol = Policy(env.spec.observation_dim,
                 env.spec.action_dim, 'Pendulum-v0', None)
    hp = HyperParameters(nb_directions=env.spec.action_dim,
            nb_best_directions=env.spec.action_dim)
    res = pol.sample_deltas(hp)
    delta_test = res[0]
    direction = 'positive'
    normalizer = Normalizer(env.spec.observation_dim)
    returns = explore(env, normalizer, pol, direction, delta_test, hp)
    print(returns)
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
            '--env', help='Gym environment name', type=str, default='MinitaurTrottingEnv-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=1)
    parser.add_argument('--render', help='OpenGL Visualizer', type=int, default=0)
    parser.add_argument('--movie', help='rgb_array gym movie', type=int, default=0)
    parser.add_argument('--steps', help='Number of steps', type=int, default=10000)
    parser.add_argument('--policy', help='Starting policy file (npy)', type=str, default='')
    parser.add_argument('--lr', help='learning rate', type=float, default=0.02)
    parser.add_argument('--noise', help='noise hyperparameter', type=float, default=0.03)
    parser.add_argument(
            '--logdir', help='Directory root to log policy files (npy)', type=str, default='.')
    args = parser.parse_args()
    hp = HyperParameters()
    hp.env_name = args.env
    hp.seed = args.seed
    hp.nb_steps = args.steps
    hp.learning_rate = args.lr
    hp.noise = args.noise
    print("seed = ", hp.seed)
    np.random.seed(hp.seed)

    env = gym.make(hp.env_name)
    if args.render:
        env.render(mode="human")
    if args.movie:
        env = wrappers.Monitor(env, monitor_dir, force=True)
    nb_inputs = env.observation_space.shape[0]
    nb_outputs = env.action_space.shape[0]
    policy = Policy(nb_inputs, nb_outputs, hp.env_name, args)
    normalizer = Normalizer(nb_inputs)

    print("start training")
    train(env, policy, normalizer, hp, args.logdir)

    pass
