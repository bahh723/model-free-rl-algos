import numpy as np
from copy import deepcopy


def policy_iteration(c, p, v_star=False):
    """
    find optimal policy for a MDP with costs c and transition kernel p using policy iteration method
    :param c: list or numpy array of shape (nb_states, nb_actions) representing costs
    :param p: list or numpy array of shape (nb_actions, nb_states, nb_states) representing transition
    kernel
    :return: (optimal_cost, optimal_policy) where optimal_cost is a scalar and optimal_policy is a
    numpy array of integers with length nb_states that gives the optimal action at each state.
    """
    eps = 0.000001
    c = np.array(c)
    p = np.array(p)
    n, m = c.shape
    g = np.ones(n, dtype='int')
    b = np.zeros(n)
    a = np.zeros([n, n])
    a[:, 0] = 1
    x = np.zeros(n)
    while True:
        for i in range(n):
            b[i] = c[i, g[i]]
            for j in range(1, n):
                if i == j:
                    a[i, j] = 1 - p[g[i], i, j]
                else:
                    a[i, j] = - p[g[i], i, j]
        z = np.linalg.solve(a, b)
        w = np.block([np.array([0]), z[1:]])
        for i in range(n):
            tmp = c[i, :] + np.dot(p[:, i, :], w)
            g[i] = np.argmin(tmp)
            x[i] = tmp[g[i]]
        m = z[0] + w
        if np.linalg.norm(m - x) < eps:
            break
    opt_cost = z[0]
    if v_star:
        return opt_cost, g, w
    return opt_cost, g


def q_value_iteration(c, p):
    """
    find optimal Q function for a MDP with costs c and transition kernel p using value iteration method
    :param c: list or numpy array of shape (nb_states, nb_actions) representing costs
    :param p: list or numpy array of shape (nb_actions, nb_states, nb_states) representing transition
    kernel
    :return: approximately optimal_Q which is a numpy array of size (nb_states, nb_actions).
    """
    eps = 0.000001
    r = -np.array(c)
    p = np.array(p)
    n, m = c.shape
    q = np.zeros([n, m])
    q_new = np.zeros([n, m])
    s_ref = 0
    while True:
        q_new = r + np.dot(p, np.max(q, axis=1)).T - np.max(q[s_ref])
        if np.max(np.abs(q_new - q)) <= eps:
            break
        q = deepcopy(q_new)
    return q_new
