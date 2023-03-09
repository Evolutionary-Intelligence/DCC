"""This code is used to plot the convergence curves of all optimizers.
"""
import os
import sys
import pickle  # for data storage

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors, rcParams

from pypoplib import optimizer


def read_pickle(s, f, i):
    afile = os.path.join('pypop7_benchmarks_lso', 'Algo-' + s + '_Func-' + f + '_Dim-2000_Exp-' + i + '.pickle')
    with open(afile, 'rb') as handle:
        return pickle.load(handle)


if __name__== '__main__':
    sys.modules['optimizer'] = optimizer  # for `pickle`

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = '12'
    sns.set_theme(style='darkgrid')

    n_trials = 5
    s = ['CMAES', 'OPOC2006', 'OPOC2009', 'CCMAES2009', 'OPOA2010',
        'XNES', 'SNES', 'LMCMAES', 'OPOA2015', 'VDCMA',
        'VKDCMA', 'RPEDA', 'MAES', 'LMCMA', 'R1ES',
        'RMES', 'LMMAES', 'FCMAES', 'MMES', 'DCEM',
        'DDCMA', 'DCC']
    # n_trials, s = 1, ['PRS', 'RHC', 'ARHC', 'SRS', 'CSA',
    #     'SPSO', 'SPSOL', 'CLPSO', 'CCPSO2', 'UMDA',
    #     'AEMNA', 'JADE', 'CDE', 'CODE', 'SCEM', 'DSCEM',
    #     'DSAES', 'CSAES', 'SEPCMAES', 'ENES', 'BES', 'DCC']
    colors, c = [name for name, _ in colors.cnames.items()], []
    indexes = [2, 3, 7] + list(range(9, 18)) + list(range(19, len(colors)))  # for better colors
    for i in indexes:
        c.append(colors[i])
    f = ['sphere', 'cigar', 'discus', 'cigar_discus', 'ellipsoid',
        'different_powers', 'schwefel221', 'step', 'rosenbrock', 'schwefel12']
    for k, ff in enumerate(f):
        time, fitness = [], []
        for j in range(len(s)):
            time.append([])
            fitness.append([])
            for i in range(n_trials):
                time[j].append([])
                fitness[j].append([])
        for i in range(n_trials):
            b = []
            for j, ss in enumerate(s):
                results = read_pickle(ss, ff, str(i + 1))
                b.append(results['best_so_far_y'])
                time[j][i] = results['fitness'][:, 0]*results['runtime']/results['n_function_evaluations']
                y = results['fitness'][:, 1]
                for i_y in range(1, len(y)):
                    if y[i_y] > y[i_y - 1]:
                         y[i_y] = y[i_y - 1]
                fitness[j][i] = y
            for j, b in enumerate(b):
                print('{:s} - {:s}: {:5.2e} '.format(ff, s[j], b), end='')
            print()

        plt.figure(figsize=(5, 5))
        plt.yscale('log')
        # for i in range(n_trials):
        #     for j, ss in enumerate(s):
        #         plt.plot(time[j][i], fitness[j][i], label=ss, color=c[j])
        top_ranked = []
        for j, ss in enumerate(s):
            end_runtime = [time[j][i][-1] for i in range(len(time[j]))]
            end_fit = [fitness[j][i][-1] for i in range(len(fitness[j]))]
            order = np.argsort(end_runtime)[2]  # 2-> median
            _r = end_runtime[order] if end_runtime[order] <= 3600 else 3600
            _f = end_fit[order] if end_fit[order] >= 1e-10 else 1e-10
            top_ranked.append([_r, _f, ss])
        top_ranked.sort(key= lambda x: (x[0], x[1]))
        top_ranked = [fi[2] for fi in top_ranked]
        top_ranked = [fi for fi in top_ranked]
        print(top_ranked)
        for j, ss in enumerate(s):
            end_runtime = [time[j][i][-1] for i in range(len(time[j]))]
            order = np.argsort(end_runtime)[2]  # 2-> median
            if ss in top_ranked[:3]:
                plt.plot(time[j][order], fitness[j][order], label=ss, color=c[j])
            else:
                plt.plot(time[j][order], fitness[j][order], color=c[j])
        plt.xlabel('Running Time (Seconds)')
        plt.ylabel('Fitness (to be Minimized)')
        plt.title(ff)
        plt.legend(loc='best')
        plt.show()
