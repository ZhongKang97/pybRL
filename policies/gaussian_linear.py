import sys, os
sys.path.append(os.path.realpath('../..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from pybRL.utils.gym_env import GymEnv
from pybRL.samplers import base_sampler
class LinearPolicy:
    def __init__(self, env_spec,
                 min_log_std=-3,
                 init_log_std=0,
                 seed=None):
        """
        :param env_spec: specifications of the env (see utils/gym_env.py)
        :param min_log_std: log_std is clamped at this value and can't go below
        :param init_log_std: initial log standard deviation
        :param seed: random seed
        """
        self.n = env_spec.observation_dim  # number of states
        self.m = env_spec.action_dim  # number of actions
        self.min_log_std = min_log_std

        # Set seed
        # ------------------------
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Policy network
        # ------------------------
        self.model = LinearModel(self.n, self.m)
        # make weights small -- not sure why but he wants to make the last weights very small (for linear and MLP alike)
        for param in list(self.model.parameters())[-2:]:  # only last layer
           param.data = 1e-2 * param.data
        self.log_std = Variable(torch.ones(self.m) * init_log_std, requires_grad=True)
        self.trainable_params = list(self.model.parameters()) + [self.log_std]

        # Old Policy network
        # ------------------------
        self.old_model = LinearModel(self.n, self.m)
        self.old_log_std = Variable(torch.ones(self.m) * init_log_std)
        self.old_params = list(self.old_model.parameters()) + [self.old_log_std]
        for idx, param in enumerate(self.old_params):
            param.data = self.trainable_params[idx].data.clone()

        # Easy access variables
        # -------------------------
        self.log_std_val = np.float64(self.log_std.data.numpy().ravel())
        self.param_shapes = [p.data.numpy().shape for p in self.trainable_params]
        self.param_sizes = [p.data.numpy().size for p in self.trainable_params]
        self.d = np.sum(self.param_sizes)  # total number of params

        # Placeholders
        # ------------------------
        self.obs_var = Variable(torch.randn(self.n), requires_grad=False)

    # Utility functions
    # ============================================
    def get_param_values(self):
        """
        Returns the trainable parameters of this linear policy
        : return : A numpy array containing numerical values of all the trainable parameters
        """
        params = np.concatenate([p.contiguous().view(-1).data.numpy()
                                 for p in self.trainable_params])
        return params.copy()

    def set_param_values(self, new_params, set_new=True, set_old=True):
        """
        Updates params of the model from an array
        :param new_params : This is an contiguous array containing the parameters of the network
        :param set_new : Set the current parameters to given data
        :param set_old : Set the old parameters to given data
        :return : Modifies the parameters (weights, biases, log_std) of the model
        """
        if set_new:
            current_idx = 0
            for idx, param in enumerate(self.trainable_params):
                vals = new_params[current_idx:current_idx + self.param_sizes[idx]]
                vals = vals.reshape(self.param_shapes[idx])
                param.data = torch.from_numpy(vals).float()
                current_idx += self.param_sizes[idx]
            # clip std at minimum value
            self.trainable_params[-1].data = \
                torch.clamp(self.trainable_params[-1], self.min_log_std).data
            # update log_std_val for sampling
            self.log_std_val = np.float64(self.log_std.data.numpy().ravel())
        if set_old:
            current_idx = 0
            for idx, param in enumerate(self.old_params):
                vals = new_params[current_idx:current_idx + self.param_sizes[idx]]
                vals = vals.reshape(self.param_shapes[idx])
                param.data = torch.from_numpy(vals).float()
                current_idx += self.param_sizes[idx]
            # clip std at minimum value
            self.old_params[-1].data = \
                torch.clamp(self.old_params[-1], self.min_log_std).data

    # Main functions
    # ============================================
    def get_action(self, observation):
        """
        Given an observation it returns a random action
        :param observation : The state returned by gym environment
        :return :A list containing the action (not clipped), and a dictionary that contains the mean and log_std 
        """
        o = np.float32(observation.reshape(1, -1))
        self.obs_var.data = torch.from_numpy(o)
        mean = self.model(self.obs_var).data.numpy().ravel()
        noise = np.exp(self.log_std_val) * np.random.randn(self.m)
        action = mean + noise
        return [action, {'mean': mean, 'log_std': self.log_std_val, 'evaluation': mean}]

    def mean_LL(self, observations, actions, model=None, log_std=None):
        """
        Calculates log likelihood function for a normal distribution, formulae https://www.statlect.com/fundamentals-of-statistics/normal-distribution-maximum-likelihood
        likelihood is same as policy
        :param observations : list of states collected by sampler
        :param actions : list of actions collected by sampler
        :model : can provide an external model
        :log_std : can provide an external log_std
        :return : LL = Log Likelihood of Normal distribution (summation of REINFORCE At varies Gaussianly)
                : mean = mean action chosen
        """
        model = self.model if model is None else model
        log_std = self.log_std if log_std is None else log_std
        obs_var = Variable(torch.from_numpy(observations).float(), requires_grad=False)
        act_var = Variable(torch.from_numpy(actions).float(), requires_grad=False)
        mean = model(obs_var)
        # print(mean)
        # print(obs_var)
        # print(act_var)
        zs = (act_var - mean) / torch.exp(log_std)
        LL = - 0.5 * torch.sum(zs ** 2, dim=1) - torch.sum(log_std) - 0.5 * self.m * np.log(2 * np.pi)
        return mean, LL

    def log_likelihood(self, observations, actions, model=None, log_std=None):
        """
        Returns log likelihood for the current model or any other model
        same params as above
        """
        mean, LL = self.mean_LL(observations, actions, model, log_std)
        return LL.data.numpy()

    def old_dist_info(self, observations, actions):
        """
        Returns data regarding the current model, params same as above
        :return LL : Log Likelihood of current dist
        :return mean : Mean of the current dist
        :return log_std : log:std_dev of current distribution
        """
        mean, LL = self.mean_LL(observations, actions, self.old_model, self.old_log_std)
        return [LL, mean, self.old_log_std]

    def new_dist_info(self, observations, actions):
        """
        Returns data regarding the current model, params same as above
        :return LL : Log Likelihood of current dist
        :return mean : Mean of the current dist
        :return log_std : log:std_dev of current distribution
        """
        mean, LL = self.mean_LL(observations, actions, self.model, self.log_std)
        return [LL, mean, self.log_std]

    def likelihood_ratio(self, new_dist_info, old_dist_info):
        """
        Finds the likelihood ratio between 2 distributions
        :param new_dist_info : list containing [Log Likelihood, mean, log_std] for new distribution
        :param old_dist_info : list containing [Log Likelihood, mean, log_std] for old distribution
        :Return : Likelihood ratio
        """
        LL_old = old_dist_info[0]
        LL_new = new_dist_info[0]
        LR = torch.exp(LL_new - LL_old)
        return LR

    def mean_kl(self, new_dist_info, old_dist_info):
        """
        Calculates mean KL Divergence between two distributions
        :param new_dist_info : list containing [Log Likelihood, mean, log_std] for new distribution
        :param old_dist_info : list containing [Log Likelihood, mean, log_std] for new distribution
        :Return : KL Divergence between the 2 distributions
        """
        old_log_std = old_dist_info[2]
        new_log_std = new_dist_info[2]
        old_std = torch.exp(old_log_std)
        new_std = torch.exp(new_log_std)
        old_mean = old_dist_info[1]
        new_mean = new_dist_info[1]
        Nr = (old_mean - new_mean) ** 2 + old_std ** 2 - new_std ** 2
        Dr = 2 * new_std ** 2 + 1e-8
        sample_kl = torch.sum(Nr / Dr + new_log_std - old_log_std, dim=1)
        return torch.mean(sample_kl)


class LinearModel(nn.Module):
    """
    Automatically creates a 1 layer dense neural network, given input and output dimensions. (Ignore scaling for now)
    :param obs_dim : Input dimension
    :param act_dim : Output dimension
    :return : self.fc0, is the neural layer defined using pytorch 
    Input to forward has to be a torch float tensor
    """
    def __init__(self, obs_dim, act_dim,
                #  a = 100,
                 in_shift = None,
                 in_scale = None,
                 out_shift = None,
                 out_scale = None):
        super(LinearModel, self).__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.set_transformations(in_shift, in_scale, out_shift, out_scale)
        # self.feature_layer = nn.Linear(obs_dim,a)
        self.fc0   = nn.Linear(obs_dim, act_dim)

    def set_transformations(self, in_shift=None, in_scale=None, out_shift=None, out_scale=None):
        # store native scales that can be used for resets
        self.transformations = dict(in_shift=in_shift,
                           in_scale=in_scale,
                           out_shift=out_shift,
                           out_scale=out_scale
                          )
        self.in_shift  = torch.from_numpy(np.float32(in_shift)) if in_shift is not None else torch.zeros(self.obs_dim)
        self.in_scale  = torch.from_numpy(np.float32(in_scale)) if in_scale is not None else torch.ones(self.obs_dim)
        self.out_shift = torch.from_numpy(np.float32(out_shift)) if out_shift is not None else torch.zeros(self.act_dim)
        self.out_scale = torch.from_numpy(np.float32(out_scale)) if out_scale is not None else torch.ones(self.act_dim)
        self.in_shift  = Variable(self.in_shift, requires_grad=False)
        self.in_scale  = Variable(self.in_scale, requires_grad=False)
        self.out_shift = Variable(self.out_shift, requires_grad=False)
        self.out_scale = Variable(self.out_scale, requires_grad=False)

    def forward(self, x):
        out = (x - self.in_shift)/(self.in_scale + 1e-8)
        out = self.fc0(out)
        out = out * self.out_scale + self.out_shift
        return out
    

if(__name__ == "__main__"):
    test_linear_model = LinearModel(4,2)
    y = test_linear_model(torch.tensor([1.,1.,1.,1.]))
    print(list(test_linear_model.parameters())[-2])
    print(y)

    # env_name = 'CartPole-v0'
    # env = GymEnv(env_name)
    # lin_pol = LinearPolicy(env.spec)
    # print(lin_pol.get_param_values())

    # new_params = np.random.rand(lin_pol.get_param_values().size)
    # lin_pol.set_param_values(new_params, True, False)
    # print(lin_pol.get_param_values())

    # env_name = 'Pendulum-v0'
    # env = GymEnv(env_name)
    # lin_pol = LinearPolicy(env.spec)
    # print(lin_pol.get_action(env.reset()))

    # env_name = 'Pendulum-v0'
    # env = GymEnv(env_name)
    # lin_pol = LinearPolicy(env.spec)
    # paths =base_sampler.do_rollout(1, lin_pol,T = 5, env_name = env_name)
    # mean , LL = lin_pol.mean_LL(paths[0]['observations'], paths[0]['actions'])
    # new_dist_info = lin_pol.new_dist_info(paths[0]['observations'], paths[0]['actions'])
    # print('new dist info: ',new_dist_info)
    # old_dist_info = lin_pol.old_dist_info(paths[0]['observations'], paths[0]['actions'])
    # print('old dist info: ', old_dist_info)
    # likelihood_ratio = lin_pol.likelihood_ratio(new_dist_info, old_dist_info)
    # print('Likelihood ratio: ', likelihood_ratio)
    # log_likelihood = lin_pol.log_likelihood(paths[0]['observations'], paths[0]['actions'])
    # print('Log likelihood: ', log_likelihood)
    # mean_kl = lin_pol.mean_kl(new_dist_info, old_dist_info)
    # print('mean kl: ', mean_kl)

    # print(mean, LL)

