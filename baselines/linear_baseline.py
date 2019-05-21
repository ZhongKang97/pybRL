import sys, os
sys.path.append(os.path.realpath('../..'))

import numpy as np
import copy
import pybRL.samplers.base_sampler as base_sampler
from pybRL.samplers.base_sampler import RandomPolicy
from pybRL.utils.gym_env import GymEnv
# from pybRL.utils.process_samples import compute_returns ### If you uncomment you will break the 
# import in process_samples.py and an error will appear when running that file Comment the process samples 
# import or comment this import. Either will do.
class LinearBaseline:
    """
    Aims to approximate the Value function of states using a linear function approximator.
    """
    def __init__(self, env_spec, reg_coeff=1e-5):
        n = env_spec.observation_dim       # number of states
        self._reg_coeff = reg_coeff
        self._coeffs = None

    def _features(self, path):
        """
        Computes regression features for the path
        :param path : is a dictionary as returned by base_line sampler
        :return : returns a list containing features corresponding to every observation in a list.
        The features for a state (s1, s2, s3 ... ) are (s1, s2, s3 ..., l, l**2, l**3, 1) where l is the position
        of the state in the trajectory.
        """
        o = np.clip(path["observations"], -10, 10)
        if o.ndim > 2:
            o = o.reshape(o.shape[0], -1)
        l = len(path["rewards"])
        al = np.arange(l).reshape(-1, 1) / 1000.0
        # print('al**2 : ',al**2)
        feat = np.concatenate([o, al, al**2, al**3, np.ones((l, 1))], axis=1)
        return feat

    def fit(self, paths, return_errors=False):
        """
        Computer the value function approximator for a given set of trajectories
        :param paths: Paths is a list of dictionaries as returned by base_line sampler
        :param return_errors: choose to return errors or not
        :return errors : error_before - relative mean square error with the previous set of weights,
        error_after -  relative mean square error with the calculated new set of weights
        """
        featmat = np.concatenate([self._features(path) for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])

        if return_errors:
            predictions = featmat.dot(self._coeffs) if self._coeffs is not None else np.zeros(returns.shape)
            errors = returns - predictions
            error_before = np.sum(errors**2)/np.sum(returns**2)

        reg_coeff = copy.deepcopy(self._reg_coeff)
        for _ in range(10):
            self._coeffs = np.linalg.lstsq(
                featmat.T.dot(featmat) + reg_coeff * np.identity(featmat.shape[1]),
                featmat.T.dot(returns), rcond = None
            )[0]
            if not np.any(np.isnan(self._coeffs)):
                break
            reg_coeff *= 10

        if return_errors:
            predictions = featmat.dot(self._coeffs)
            errors = returns - predictions
            error_after = np.sum(errors**2)/np.sum(returns**2)
            return error_before, error_after

    def predict(self, path):
        """
        Predicts the value function for each state with a set of fit weights
        :param path : it is dictionary as returned by base_sampler
        :return : Returns a list containing the predicted value function for each state in the path
        """
        if self._coeffs is None:
            return np.zeros(len(path["rewards"]))
        return self._features(path).dot(self._coeffs)


if(__name__ == "__main__"):
    #What is this code trying to do?
    N = 1
    pol = RandomPolicy((-2,2))
    T = 5
    env_name = 'Pendulum-v0'
    y = base_sampler.do_rollout(N = N, policy = pol, T = 5, env_name=env_name)
    y2 = base_sampler.do_rollout(N = N, policy = pol, T = 5, env_name=env_name)

    # print(y)
    env = GymEnv(env_name)
    base_line  = LinearBaseline(env.spec)
    features = base_line._features(y[0])
    print(features)
    compute_returns(y, 1)
    compute_returns(y2, 1)
    # print('returns: ', y[0]['returns'])
    # print(base_line.predict(y[0]))
    errors = base_line.fit(y,True)
    # print('y : ', y)
    # print('returns: ', y2[4]['returns'])
    # print('predict: ', base_line.predict(y2[4]))


    # print(errors)


