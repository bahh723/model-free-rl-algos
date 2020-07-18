import os
import matplotlib.pyplot as plt
import numpy as np



def regret_plot(environment_name, agents, alg_storage, legends, save_directory=None):
    plt.figure()
    for agent in agents:
        filename = agent + '_' + alg_storage[agent]
        read_directory = os.path.join('log', environment_name, filename)
        runs = []
        try:
            for name in sorted(os.listdir(read_directory)):
                if name.startswith('info'):
                    pass
                elif name.startswith('.'):
                    pass
                else:
                    runs.append(np.loadtxt(os.path.join(read_directory, name)))
        except FileNotFoundError:
            raise FileNotFoundError('Stored data for {0} on {1} is not found. '
                                    'Need to remove {2} from the environments_name'
                                    ' or run the experiment for it.'.format(agent, environment_name, environment_name))
        t = runs[0][:, 0]

        regrets = []
        for run in runs:
            regrets.append(run[:, 1])

        avg_reg = np.mean(regrets, axis=0)
        error = np.std(regrets, axis=0)
        plt.plot(t, avg_reg, label=legends[agent])
        plt.ylabel('Regret')
        plt.fill_between(t, avg_reg-error, avg_reg+error, alpha=0.2)

    plt.legend()
    plt.title(environment_name[:-3])
    if save_directory:
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        plt.savefig(os.path.join(save_directory, environment_name+'.pdf'))
    else:
        plt.show()
