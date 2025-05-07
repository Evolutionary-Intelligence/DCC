"""This script is used to plot all the convergence
    curves (median) of all considered optimizers.

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
font_size, font_family = 10, 'Times New Roman'
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'SimSun'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = font_size  # 对应5号字体
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


def read_pickle(s, f, i):
    file_name = 'Algo-' + s + '_Func-' + f + '_Dim-2000_Exp-' + i + '.pickle'
    file_name = os.path.join(s, file_name)
    with open(os.path.join('data', file_name), 'rb') as handle:
        return pickle.load(handle)


if __name__== '__main__':
    fs = ['sphere', 'cigar',
          'discus', 'cigar_discus',
          'ellipsoid', 'different_powers',
          'schwefel221', 'step',
          'rosenbrock', 'schwefel12']
    for fun_name in fs:
        n_trials = 5
        s = ['CMAES', 'OPOC2006', 'OPOC2009', 'CCMAES2009', 'OPOA2010',
            'XNES', 'SNES', 'LMCMAES', 'OPOA2015', 'VDCMA',
            'VKDCMA', 'RPEDA', 'MAES', 'LMCMA', 'R1ES',
            'RMES', 'LMMAES', 'FCMAES', 'MMES', 'DCEM',
            'DDCMA', 'DCC']
        # s = ['PRS', 'RHC', 'ARHC', 'SRS', 'CSA',
        #     'SPSO', 'SPSOL', 'CLPSO', 'CCPSO2', 'UMDA',
        #     'AEMNA', 'JADE', 'CDE', 'CODE', 'SCEM', 'DSCEM',
        #     'DSAES', 'CSAES', 'SEPCMAES', 'ENES', 'BES', 'DCC']
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
                results = read_pickle(ss, fun_name, str(i + 1))
                time[j][i] = results['fitness'][:, 0] *\
                    results['runtime'] /\
                        results['n_function_evaluations']
                y = results['fitness'][:, 1]
                for i_y in range(1, len(y)):
                    if y[i_y] > y[i_y - 1]:
                            y[i_y] = y[i_y - 1]
                fitness[j][i] = y
        top_ranked = []
        for j, ss in enumerate(s):
            # first: runtime 
            end_runtime = [time[j][i][-1] for i in range(len(time[j]))]
            # second: fitness
            end_fitness = [fitness[j][i][-1] for i in range(len(fitness[j]))]
            # median (non-standard): int(n_trials / 2)
            order = np.argsort(end_runtime)[int(n_trials / 2)]
            _r = end_runtime[order] if end_runtime[order] <= 3600 else 3600
            _f = end_fitness[order] if end_fitness[order] >= 1e-10 else 1e-10
            top_ranked.append([_r, _f, ss])
        top_ranked.sort(key= lambda x: (x[0], x[1]))
        top_ranked_s = [fi[2] for fi in top_ranked]
        print(top_ranked)
        print(top_ranked_s)
        plt.figure(figsize=(2.3, 2.3))
        plt.yscale('log')
        id = 0
        top = top_ranked_s[:5]
        top.append('DCC')
        top = set(top)
        print(top)
        for i, ss in enumerate(s):
            if ss not in top:
                continue
            end_runtime = [time[j][k][-1] for k in range(len(time[j]))]
            order = np.argsort(end_runtime)[int(n_trials / 2)]
            plt.plot(time[i][order], fitness[i][order],
                    c=colors[id])
            plt.plot(time[i][order][int(len(time[i][order])/5)],
                    fitness[i][order][int(len(time[i][order])/5)],
                    c=colors[id], marker=markers[id], markersize=7)
            plt.plot(time[i][order][int(len(time[i][order])*2/5)],
                    fitness[i][order][int(len(time[i][order])*2/5)],
                    c=colors[id], marker=markers[id], markersize=7)
            plt.plot(time[i][order][int(len(time[i][order])*3/5)],
                    fitness[i][order][int(len(time[i][order])*3/5)],
                    c=colors[id], marker=markers[id], markersize=7)
            plt.plot(time[i][order][int(len(time[i][order])*4/5)],
                    fitness[i][order][int(len(time[i][order])*4/5)],
                    c=colors[id], label=ss, marker=markers[id], markersize=7)
            id += 1
        plt.xlabel('运行时间（单位：秒）', fontsize=font_size)
        plt.ylabel('适应值（最小化）', fontsize=font_size)
        plt.xticks(fontsize=font_size, fontfamily=font_family)
        plt.yticks(fontsize=font_size, fontfamily=font_family)
        plt.title(fun_name, fontsize=font_size, fontfamily=font_family)
        plt.legend(loc='best', prop = {'size' : font_size,
                                    'family': font_family})
        plt.savefig('{0}.png'.format(fun_name),
                    dpi=700, bbox_inches='tight')
        plt.show()
