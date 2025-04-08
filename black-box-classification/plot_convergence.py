"""This script is used to plot all the convergence curves
    (median) of all considered optimizers.

    Chinese: 此脚本程序已经被段琦琦的博士论文所使用。
"""
import os
import sys
import pickle5 as pickle

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pypoplib import optimizer
sys.modules['optimizer'] = optimizer  # for `pickle`


sns.set_theme(style='darkgrid', rc={'figure.figsize':(5, 5)})
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = '10'


def read_pickle(s, f, i):
    file_name = 'Algo-' + s + '_Func-' + f + '_Dim-1025_Exp-' + i + '.pickle'
    with open(os.path.join(file_name), 'rb') as handle:
        return pickle.load(handle)


if __name__== '__main__':
    n_trials = 5
    s = ['DCC',
         'SCEM', 'COCMA', 'RPEDA', 'SHADE',
         'VKDCMA', 'R1NES', 'LMCMA', 'CLPSO']

    time, fitness = [], []
    for i in range(len(s)):
        time.append([])
        fitness.append([])
        for j in range(n_trials):
            time[i].append([])
            fitness[i].append([])
    for i in range(n_trials):
        for j, ss in enumerate(s):
            results = read_pickle(ss, 'loss_svm_qar', str(i + 1))
            time[j][i] = results['fitness'][:, 0]*results['runtime']/results['n_function_evaluations']
            y = results['fitness'][:, 1]
            for i_y in range(1, len(y)):
                if y[i_y] > y[i_y - 1]:
                        y[i_y] = y[i_y - 1]
            fitness[j][i] = y
    order = []
    for i in range(len(s)):
        fit = []
        for j in range(n_trials):
            fit.append(fitness[i][j][-1])
        order.append(np.argsort(fit)[2])  # 2-> median
    plt.figure(figsize=(6, 6))
    plt.yscale('log')
    for i, ss in enumerate(s):
        plt.plot(time[i][order[i]], fitness[i][order[i]])
    plt.xlabel('Running Time (Seconds)')
    plt.ylabel('Fitness (to be Minimized)')
    plt.title('SVM')
    plt.legend(s, loc='best')
    plt.savefig('SVM.png', dpi=700, bbox_inches='tight')
    plt.show()
