"""
Some finite state and finite action MDP environments are implemented in this file.

Current available environments:
RandomMDPEnv: cost and transition kernel are random
JumpRiverSwimEnv: Modified RiverSwim with small jump probability
to an arbitrary state. Transition kernel and costs are hard coded.
"""
import numpy as np
from utils import policy_iteration, q_value_iteration


class FiniteMDP(object):
    """
    A superclass for JumpRiverSwimEnv and RandomMDPEnv
    """
    def __init__(self, nb_states, nb_actions, costs, p):
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.states = range(self.nb_states)
        self.actions = range(self.nb_actions)
        self.costs = costs
        self.p = p
        self.rewards = -self.costs
        self.span = np.max(np.max(self.costs, axis=1) - np.min(self.costs, axis=1))

        self.state = None
        self.opt_cost = None
        self.opt_policy = None
        self.opt_q = None

    def optimal_cost(self):
        if self.opt_cost is not None:
            return self.opt_cost
        self.opt_cost, self.opt_policy = policy_iteration(self.costs, self.p)
        return self.opt_cost

    def optimal_reward(self):
        return -self.optimal_cost()

    def optimal_policy(self):
        if self.opt_policy is not None:
            return self.opt_policy
        self.opt_cost, self.opt_policy = policy_iteration(self.costs, self.p)
        return self.opt_policy

    def optimal_q(self):
        if self.opt_q is not None:
            return self.opt_q
        self.opt_q = q_value_iteration(self.costs, self.p)
        return self.opt_q

    def info(self):
        s = ''
        s += 'name = {0}\n'.format(self.__class__.__name__)
        s += 'number of states = {0}, number of actions = {1}\n'.format(self.nb_states, self.nb_actions)
        s += 'optimal cost = {0}\n'.format(self.optimal_cost())
        s += 'optimal policy = {0}\n'.format(self.optimal_policy())
        s += 'optimal q =\n{0}\n'.format(self.optimal_q())
        s += 'cost function =\n{0}\n'.format(self.costs)
        s += 'transition kernel=\n{0}\n'.format(self.p)
        return s

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        reward = self.reward(self.state, action)
        self.state = np.random.choice(self.states, p=self.p[action, self.state])
        return self.state, reward

    def reward(self, state, action):
        return -self.costs[state, action]

    def write_info(self, directory=''):
        import os
        np.savetxt(os.path.join(directory, 'p'), self.p.reshape(self.nb_actions*self.nb_states, self.nb_states))
        np.savetxt(os.path.join(directory, 'r'), -self.costs)


class RandomMDPEnv(FiniteMDP):
    """
    Generates an MDP with randomly chosen transition kernel and costs.
    Note that c(s, a) is uniformly sampled from {0, 0.1, 0.2, ..., 1}
    and p(s'|s,a) is uniformly chosen from unit interval and then normalized.
    """
    def __init__(self, nb_states=6, nb_actions=2):
        costs = self._random_costs(nb_states, nb_actions)
        p = self._random_p(nb_states, nb_actions)
        super(RandomMDPEnv, self).__init__(nb_states=nb_states, nb_actions=nb_actions, costs=costs, p=p)

    def _random_costs(self, nb_states, nb_actions):
        mat = np.random.randint(11, size=(nb_states, nb_actions)) / 10.0
        return mat

    def _random_p(self, nb_states, nb_actions):
        p = np.zeros((nb_actions, nb_states, nb_states))
        for a in range(nb_actions):
            mat = np.random.rand(nb_states, nb_states)
            p[a, :, :] = mat / np.sum(mat, axis=1)[:, None]
        return p


class JumpRiverSwimEnv(FiniteMDP):
    """
    Extension of the RiverSwim environment of https://arxiv.org/abs/1709.04570.
    At each time step there is a tiny probability of jumping to an arbitrary state.
    """
    def __init__(self):
        e = 1e-2 / 6  # 6*e would be probability of jumping to an arbitrary state.
        costs = np.array([[.8, 1],  # c(state=0, a)
                          [1, 1],
                          [1, 1],
                          [1, 1],
                          [1, 1],
                          [1, 0]])

        p = np.array([[[1-5*e, e, e, e, e, e],
                       [1-5*e, e, e, e, e, e],
                       [e, 1-5*e, e, e, e, e],
                       [e, e, 1-5*e, e, e, e],
                       [e, e, e, 1-5*e, e, e],
                       [e, e, e, e, 1-5*e, e]],  # p(s'|s, 0)

                      [[.7+e, .3-5*e, e, e, e, e],
                       [.1+e, .6+e, .3-5*e, e, e, e],
                       [e, .1+e, .6+e, .3-5*e, e, e],
                       [e, e, .1+e, .6+e, .3-5*e, e],
                       [e, e, e, .1+e, .6+e, .3-5*e],
                       [e, e, e, e, .7+e, .3-5*e]]])  # p(s'|s, 1)

        super(JumpRiverSwimEnv, self).__init__(nb_states=costs.shape[0], nb_actions=costs.shape[1], costs=costs, p=p)