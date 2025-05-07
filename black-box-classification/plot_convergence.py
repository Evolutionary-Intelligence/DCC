"""This script is used to plot all the convergence curves
    (median) of all considered optimizers.

    Chinese: 此脚本程序已经被段琦琦的博士论文所使用。
"""
import os
import sys
import pickle5 as pickle

import matplotlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pypoplib import optimizer
sys.modules['optimizer'] = optimizer  # for `pickle`


sns.set_theme(style='darkgrid')
font_size = 10
font_family = 'Times New Roman'
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'SimSun'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = font_size  # 对应5号字体
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


def read_pickle(s, f, i):
    file_name = 'Algo-' + s + '_Func-' + f + '_Dim-1025_Exp-' + i + '.pickle'
    with open(os.path.join(file_name), 'rb') as handle:
        return pickle.load(handle)


if __name__== '__main__':
    n_trials = 5
    s = ['DCC',
         'SCEM', 'COCMA', 'RPEDA', 'SHADE',
         'VKDCMA', 'R1NES', 'LMCMA', 'CLPSO']
    markers = ['o', 'v', '^', '<', '>',
               'd', 's', 'p', 'P', '*',
               'h', 'H', '+', 'x', 'X',
               'D', '8']
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
    plt.figure(figsize=(4, 4))
    plt.yscale('log')
    for i, ss in enumerate(s):
        plt.plot(time[i][order[i]], fitness[i][order[i]],
                 c=colors[i])
        plt.plot(time[i][order[i]][int(len(time[i][order[i]])/5)],
                 fitness[i][order[i]][int(len(time[i][order[i]])/5)],
                 c=colors[i], marker=markers[i], markersize=7)
        plt.plot(time[i][order[i]][int(len(time[i][order[i]])*2/5)],
                 fitness[i][order[i]][int(len(time[i][order[i]])*2/5)],
                 c=colors[i], marker=markers[i], markersize=7)
        plt.plot(time[i][order[i]][int(len(time[i][order[i]])*3/5)],
                 fitness[i][order[i]][int(len(time[i][order[i]])*3/5)],
                 c=colors[i], marker=markers[i], markersize=7)
        plt.plot(time[i][order[i]][int(len(time[i][order[i]])*4/5)],
                 fitness[i][order[i]][int(len(time[i][order[i]])*4/5)],
                 c=colors[i], label=ss, marker=markers[i], markersize=7)
    plt.xlabel('运行时间（单位：秒）', fontsize=font_size)
    plt.ylabel('适应值（最小化）', fontsize=font_size)
    plt.xticks(fontsize=font_size, fontfamily=font_family)
    plt.yticks(fontsize=font_size, fontfamily=font_family)
    plt.title('高维黑箱分类任务', fontsize=font_size)
    plt.legend(loc='best', prop = {'size' : font_size,
                                   'family': font_family})
    plt.savefig('SVM.png', dpi=700, bbox_inches='tight')
    plt.show()
