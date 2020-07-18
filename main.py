"""
This file is the main file to be run.

running instructions:
(1) Uncomment the desired environment
(2) Uncomment the desired agent
(3) Set the desired horizon T and nb_runs. In the paper T=5000000, nb_runs=10 (it takes 100 min to complete)
(4) If you want to save the data in a new file change the storage_counter.
The data will be saved in log/environment/agent_'storage_counter'.
To plot previously stored data you can set the corresponding storage_counter in alg_storage
Plots will be saved in plots directory by default.
(5) To reproduce the results of the paper, hyper parameters of each agent should be set as:
    --------
    RandomMDP:      StochasticApproximationAgent(epsilon=0.05)
                    OptimisticDiscountedAgent(gamma=0.99, c=1.0)
                    OOMDAgent(N=2, B=4)

    JumpRiverSwim:  StochasticApproximationAgent(epsilon=0.03)
                    OptimisticDiscountedAgent(gamma=0.99, c=1.0)
                    OOMDAgent(N=10, B=30)
    --------
"""



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from copy import deepcopy
from environment import RandomMDPEnv, JumpRiverSwimEnv
from agent import StochasticApproximationAgent, OptimisticDiscountedAgent, OOMDAgent, PolitexAgent
from plot import regret_plot
import os
import time

seed = 3
np.random.seed(seed)


class Runner(object):

    def __init__(self, agent, env, nb_runs=10, horizon=10000, filename=None, display_mode=1):
        self.agent = agent
        self.env = env
        self.nb_runs = nb_runs
        self.horizon = horizon
        self.display_mode = display_mode

        if filename:
            self.save_directory = os.path.join('log', filename)
            if not os.path.exists(self.save_directory):
                os.makedirs(self.save_directory)
        else:
            self.save_directory = None

    def run(self):
        if self.save_directory:
            with open(os.path.join(self.save_directory, 'info.txt'), 'w') as file:
                file.write('*'*10 + ' General Info ' + '*'*10 + '\n')
                file.write('seed = {0}\n'.format(seed))
                file.write('horizon = {0}\n'.format(self.horizon))
                file.write('number of runs = {0}\n'.format(self.nb_runs))
                file.write('*'*10 + ' Agent Info ' + '*'*10 + '\n')
                file.write(self.agent.info())
                file.write('*'*10 + ' Environment Info ' + '*'*10 + '\n')
                file.write(self.env.info())
        for exp in range(self.nb_runs):
            print('======= Experiment {0} ======='.format(exp))
            state = self.env.reset()
            self.agent.reset()
            regrets = [] # regret buffer
            saving_regrets = []
            saving_times = []
            for t in range(self.horizon):
                action = self.agent.act(state)
                next_state, reward = self.env.step(action)

                if agent.t % 1000 == 1: # samples the data at this time
                    saving_times.append(agent.t)
                    saving_regrets.append(np.sum(regrets))
                    regrets = []
                if self.display_mode == 2 and (agent.t-1) % 1000000 == 0:
                    print('experiment = {0}'.format(exp))
                    print('time = {0}'.format(agent.t))
                    print('number of visits to state action pairs =')
                    print(agent.n)
                    print('________________________')
                self.agent.update(next_state, reward)
                state = deepcopy(next_state)
                regrets.append(self.env.optimal_reward() - reward)
            if self.display_mode == 1:
                print('experiment = {0}'.format(exp))
                print('time = {0}'.format(agent.t))
                print('number of visits to state action pairs =')
                print(agent.n)
                print('________________________')

            if self.save_directory:
                np.savetxt(os.path.join(self.save_directory, str(exp)), np.array([saving_times, np.cumsum(saving_regrets)]).T)
        print('Experiments stored in {0}.'.format(self.save_directory))



if __name__ == '__main__':
    T = 5000000

    # _________________________ Environments ______________________________
    # uncomment the desired environment to run
    env = JumpRiverSwimEnv()
    # env = RandomMDPEnv(nb_states=6, nb_actions=2)

    # ____________________________ Agents __________________________________
    # uncomment the desired agent to run
    # agent = StochasticApproximationAgent(env=env, epsilon=0.03)
    # agent = OptimisticDiscountedAgent(env=env, gamma=.99, c=1.0)
    # agent = OOMDAgent(env=env, N=10, B=30)
    agent = PolitexAgent(env=env, N=3000, B=6000)

    # __________________ Running and storing the results ___________________
    storage_counter = 2 # you may change this for a new filename.
    filename = os.path.join(env.__class__.__name__, agent.__class__.__name__ + '_{0}'.format(storage_counter)) # DO NOT change this. Instead change storage_counter.
    start = time.time()
    runner = Runner(agent=agent, env=env, nb_runs=10, horizon=T, filename=filename, display_mode=2)
    runner.run()
    print('*'*50)
    print("Run time: ", time.time() - start)

    # ___________________________ plots __________________________
    # environments_name = ['JumpRiverSwimEnv', 'RandomMDPEnv']
    # agents = ['OptimisticDiscountedAgent', 'StochasticApproximationAgent', 'OOMDAgent']

    environments_name = ['JumpRiverSwimEnv']
    agents = ['PolitexAgent']

    alg_storage = {'StochasticApproximationAgent': str(storage_counter),
                   'OptimisticDiscountedAgent': str(storage_counter),
                   'OOMDAgent': str(storage_counter),
                   'PolitexAgent': str(storage_counter)}

    legends = {'StochasticApproximationAgent': 'Q-learning with $\epsilon$-greedy',
               'OptimisticDiscountedAgent': 'Optimistic Q-learning',
               'OOMDAgent': 'MDP-OOMD',
               'PolitexAgent': 'Politex'}
               
    

    save_directory = 'plots'
    # save_directory = None

    for env_name in environments_name:
        regret_plot(environment_name=env_name, agents=agents, alg_storage=alg_storage, legends=legends, save_directory=save_directory)








