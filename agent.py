from __future__ import division
import numpy as np
from collections import defaultdict


class StochasticApproximationAgent(object):
    """
    This agent implements the standard Q-learning algorithm for the infinite-horizon
    average-reward setting with epsilon-greedy exploration.
    """
    def __init__(self, env, epsilon):
        self.REF_STATE = 0
        self.alpha = 1.
        self.env = env
        self.epsilon = epsilon

        self.mu = np.zeros([self.env.nb_states, self.env.nb_actions])  # for consistency called self.mu (it is q).
        self.n = np.zeros([self.env.nb_states, self.env.nb_actions], dtype=int)
        self.n_prime = np.zeros([self.env.nb_states, self.env.nb_actions, self.env.nb_states], dtype=int)

        self.state = None
        self.action = None
        self.t = 1

        self.reset()

    def act(self, state):
        self.state = state

        if np.random.rand() <= self.epsilon:
            self.action = np.random.choice(self.env.actions)
        else:
            self.action = np.argmax(self.mu[self.state])
        return self.action

    def update(self, next_state, reward):
        self.n[self.state, self.action] += 1
        self.alpha = 1.0/self.n[self.state, self.action]
        self.n_prime[self.state, self.action, next_state] += 1
        self.mu[self.state, self.action] = (1 - self.alpha) * self.mu[self.state, self.action] \
                                          + self.alpha * (reward + np.max(self.mu[next_state]) - np.max(self.mu[self.REF_STATE]))
        self.t += 1

    def info(self):
        return 'name = StochasticApproximationAgent\n' + 'epsilon = {0}\n'.format(self.epsilon)

    def reset(self):
        self.state = None
        self.action = None

        self.mu = np.zeros([self.env.nb_states, self.env.nb_actions])
        self.n = np.zeros([self.env.nb_states, self.env.nb_actions], dtype=int)
        self.n_prime = np.zeros([self.env.nb_states, self.env.nb_actions, self.env.nb_states], dtype=int)

        self.alpha = 1.
        self.t = 1


class OptimisticDiscountedAgent(object):
    """
    Implements the Optimistic Q-learning algorithm described in our paper
    """

    def __init__(self, env, gamma=0.99, c=1.0):
        # _________ constants __________
        self.gamma = gamma
        self.H = gamma/(1.0-gamma)
        self.c = c
        # ______________________________

        self.env = env
        self.state = None
        self.action = None

        self.t = 1
        self.n = np.zeros([self.env.nb_states, self.env.nb_actions], dtype=int)
        self.mu = self.H * np.ones([self.env.nb_states, self.env.nb_actions])  # Q in the algorithm
        self.mu_hat = self.H * np.ones([self.env.nb_states, self.env.nb_actions])  # Q_hat in the algorithm
        self.v_hat = self.H * np.ones(self.env.nb_states)

    def act(self, state):
        self.state = state
        self.action = np.argmax(self.mu_hat[self.state, :])
        return self.action

    def update(self, next_state, reward):
        self.n[self.state, self.action] += 1
        self.t += 1
        bonus = self.c * np.sqrt(self.H/self.n[self.state, self.action])
        alpha = (self.H + 1)/(self.H + self.n[self.state, self.action])
        self.mu[self.state, self.action] = (1-alpha)*self.mu[self.state, self.action] + alpha*(reward + self.gamma*self.v_hat[next_state] + bonus)
        self.mu_hat[self.state, self.action] = min(self.mu_hat[self.state, self.action], self.mu[self.state, self.action])
        self.v_hat[self.state] = np.max(self.mu_hat[self.state, :])

    def info(self):
        return 'name = OptimisticDiscountedAgent\n' + 'gamma = {0}\n'.format(self.gamma) + 'c = {0}\n'.format(self.c)

    def reset(self):
        self.t = 1
        self.n = np.zeros([self.env.nb_states, self.env.nb_actions], dtype=int)
        self.mu = self.H * np.ones([self.env.nb_states, self.env.nb_actions])
        self.mu_hat = self.H * np.ones([self.env.nb_states, self.env.nb_actions])
        self.v_hat = self.H * np.ones(self.env.nb_states)


class OOMDAgent(object):
    """
    Implements the MDP-OOMD algorithm described in our paper.
    """
    def __init__(self, env, N, B, eta=0.01):
        # ____________constants____________
        self.eta = eta
        self.N = N
        self.B = B
        if self.B < self.N:
            raise ValueError('B should be larger than N')
        # _________________________________
        self.env = env
        self.state = None
        self.action = None

        self.policy = (1.0/self.env.nb_actions)*np.ones((self.env.nb_states, self.env.nb_actions))
        self.policy_prime = (1.0/self.env.nb_actions)*np.ones((self.env.nb_states, self.env.nb_actions))
        self.episode_trajectory = []  # each element of this list is a tuple of (state, action, reward)
        self.t = 1
        self.n = np.zeros([self.env.nb_states, self.env.nb_actions], dtype=int)

    def act(self, state):
        self.state = state
        if self.t % self.B == 1: # a new episode starts
            self._oomd_update()
        self.action = np.random.choice(self.env.actions, p=self.policy[state])
        return self.action

    def update(self, next_state, reward):
        self.episode_trajectory.append((self.state, self.action, reward))
        self.n[self.state, self.action] += 1
        self.t += 1

    def info(self):
        return 'name = OOMDAgent\n' + 'N = {0}\n'.format(self.N) + 'B = {0}\n'.format(self.B)

    def reset(self):
        self.policy = (1.0/self.env.nb_actions)*np.ones((self.env.nb_states, self.env.nb_actions))
        self.policy_prime = (1.0/self.env.nb_actions)*np.ones((self.env.nb_states, self.env.nb_actions))
        self.episode_trajectory = []
        self.t = 1
        self.n = np.zeros([self.env.nb_states, self.env.nb_actions], dtype=int)

    def _oomd_update(self):

        #  pre-processing the trajectory for fast access
        cum_reward = np.cumsum([item[2] for item in self.episode_trajectory])
        d = defaultdict(list)  # dictionary of {state: [indices]} time indexes of visiting that state
        for i, item in enumerate(self.episode_trajectory):
            d[item[0]].append(i)

        #  estimate q
        beta_hat = np.zeros([self.env.nb_states, self.env.nb_actions])
        for s in self.env.states:
            y = np.zeros(self.env.nb_actions)
            tau = 0
            i = 0
            for j in d[s]:
                if self.B - self.N > j >= tau:
                    R = cum_reward[j + self.N - 1] - cum_reward[j - 1] if j >= 1 else cum_reward[j + self.N - 1]
                    # tau = j + 2 * self.N
                    tau = j+1
                    i += 1
                    y[self.episode_trajectory[j][1]] += float(R)/self.policy[s, self.episode_trajectory[j][1]]
            if i > 0:
                beta_hat[s] = y/i
        self.episode_trajectory = []

        # online mirror descent update
        for s in self.env.states:
            lamda_prime = self._binary_search(self.policy_prime[s], self.eta*beta_hat[s].min(), self.eta*beta_hat[s].max(), beta_hat[s])
            self.policy_prime[s] = self._update_policy(self.policy_prime[s], lamda_prime, beta_hat[s])

            lamda = self._binary_search(self.policy[s], self.eta*beta_hat[s].min(), self.eta*beta_hat[s].max(), beta_hat[s])
            self.policy[s] = self._update_policy(self.policy_prime[s], lamda, beta_hat[s])
            self.policy[s] /= self.policy[s].sum()

    def _binary_search(self, x, low, high, q_a_array):
        tol = 0.0005
        while True:
            lamda = (low+high)/2.0
            y_updated = self._update_policy(x, lamda, q_a_array)
            if abs(high-low) < tol:
                return high
            if (y_updated < 0).any():
                low = lamda
            elif sum(y_updated) < 1 - tol:
                high = lamda
            elif sum(y_updated) > 1 + tol:
                low = lamda
            else:
                return lamda

    def _update_policy(self, pi, lamda, q_a_array):
        new_pi = np.zeros(self.env.nb_actions)
        for a in range(self.env.nb_actions):
            new_pi_reci = 1.0/pi[a] - self.eta*q_a_array[a] + lamda
            new_pi[a] = 1.0/new_pi_reci
        return new_pi


class PolitexAgent(object):
    """
    Implements the Optimistic Q-learning algorithm described in our paper
    """

    def __init__(self, env, N, B, eta=0.2):
        # _________ constants __________
        self.eta = eta
        self.N = N
        self.B = B
        # ______________________________

        self.env = env
        self.state = None
        self.action = None

        self.t = 1
        self.n = np.zeros([self.env.nb_states, self.env.nb_actions], dtype=int)  # total number of visits to (s, a)

        self.ns = np.zeros([self.env.nb_states], dtype=int)  # number of visits to s in phase N of each episode
        self.nss_prime = np.zeros([self.env.nb_states, self.env.nb_states], dtype=int)  # number of visits to ss' in phase N of each episode
        self.policy = np.ones([self.env.nb_states, self.env.nb_actions]) * 1.0/self.env.nb_actions

        self.n_second_phase = np.zeros([self.env.nb_states, self.env.nb_actions], dtype=int)  # number of visits to (s, a) at second phase of each episode
        self.n_episode = np.zeros([self.env.nb_states, self.env.nb_actions], dtype=int)  # number of visits to (s, a) at each episode
        self.q = np.zeros([self.env.nb_states, self.env.nb_actions])  # Q in the algorithm
        self.v = np.zeros([self.env.nb_states])  # v_pi of the algorithm
        self.lamda = 0

    def act(self, state):
        self.state = state
        if self.t % self.B <= self.N: 
            self.action = np.random.choice(self.env.actions, p=self.policy[self.state])
        else:
            for a in self.env.actions:
                if self.n_second_phase[self.state, a]==0:
                   self.action = a
                   return self.action
            self.action = np.random.choice(self.env.actions, p=self.policy[self.state])
        return self.action

    def update(self, next_state, reward):
        if self.t % self.B == self.N:  # beginning of second phase of the episode
            self.lamda = 1.0/self.N * np.sum([self.n_episode[s, a] * self.env.rewards[s, a] for s in self.env.states for a in self.env.actions])
            mat = np.diag(self.ns) - self.nss_prime
            vec = np.sum(self.n_episode * (self.env.rewards - self.lamda), axis=1)
            self.v = np.dot(np.linalg.pinv(mat + 0.05 * np.eye(self.env.nb_states)), vec)
        if self.t % self.B == 0:  # beginning of an episode
            for s in self.env.states:
                for a in self.env.actions:
                    if self.n_second_phase[s, a] > 0:
                        self.q[s,a] /= self.n_second_phase[s, a]
            self.policy = self.policy * np.exp(self.eta * np.maximum(np.minimum(self.q, 100/self.eta),-100/self.eta))
            for s in self.env.states:
                self.policy[s] /= np.sum(self.policy[s])
                #self.policy[s] = 0.999*self.policy[s] + 0.001*np.ones((self.env.nb_actions,))/self.env.nb_actions
                #self.policy[s] /= np.sum(self.policy[s])
                
            self.n_episode = np.zeros([self.env.nb_states, self.env.nb_actions], dtype=int)
            self.n_second_phase = np.zeros([self.env.nb_states, self.env.nb_actions], dtype=int)
            self.ns = np.zeros([self.env.nb_states], dtype=int)
            self.nss_prime = np.zeros([self.env.nb_states, self.env.nb_states], dtype=int)
            self.q = np.zeros([self.env.nb_states, self.env.nb_actions])

        if self.t % self.B > self.N:  # second phase of an episode
            self.n_second_phase[self.state, self.action] += 1
            self.q[self.state, self.action] += self.env.rewards[self.state, self.action] - self.lamda + self.v[next_state]
        self.n[self.state, self.action] += 1
        self.ns[self.state] += 1
        self.nss_prime[self.state, next_state] += 1
        self.n_episode[self.state, self.action] += 1
        self.t += 1

    def info(self):
        return 'name = PolitexAgent\n' + 'N = {0}\n'.format(self.N) + 'B = {0}\n'.format(self.B) + 'eta = {0}\n'.format(self.eta)

    def reset(self):
        self.t = 1
        self.n = np.zeros([self.env.nb_states, self.env.nb_actions], dtype=int)  # total number of visits to (s, a)

        self.ns = np.zeros([self.env.nb_states], dtype=int)  # number of visits to s in phase N of each episode
        self.nss_prime = np.zeros([self.env.nb_states, self.env.nb_states], dtype=int)  # number of visits to ss' in phase N of each episode
        self.policy = np.ones([self.env.nb_states, self.env.nb_actions]) * 1.0/self.env.nb_actions

        self.n_second_phase = np.zeros([self.env.nb_states, self.env.nb_actions], dtype=int)  # number of visits to (s, a) at second phase of each episode
        self.n_episode = np.zeros([self.env.nb_states, self.env.nb_actions], dtype=int)  # number of visits to (s, a) at each episode
        self.q = np.zeros([self.env.nb_states, self.env.nb_actions])  # Q in the algorithm
        self.v = np.zeros([self.env.nb_states])  # v_pi of the algorithm
        self.lamda = 0


