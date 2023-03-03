import os
import time
import pickle  # all data is saved in pickle form
import argparse

import numpy as np  # computing engine for numerical computing

import pypoplib.continuous_functions as cf  # all rotated and shifted functions


class Experiment(object):
    def __init__(self, index, function, seed, ndim_problem):
        self.index = index
        self.function = function
        self.seed = seed
        self.ndim_problem = ndim_problem
        self._folder = 'pypop7_benchmarks_lso'  # data folder in the working space (which will be automatically made)
        if not os.path.exists(self._folder):
            os.makedirs(self._folder)
        self._file = os.path.join(self._folder, 'Algo-{}_Func-{}_Dim-{}_Exp-{}.pickle')

    def run(self, optimizer):
        problem = {'fitness_function': self.function,
                   'ndim_problem': self.ndim_problem,
                   'upper_boundary': 10.0*np.ones((self.ndim_problem,)),
                   'lower_boundary': -10.0*np.ones((self.ndim_problem,))}
        options = {'max_function_evaluations': np.Inf,
                   'max_runtime': 3600*3,  # seconds (3 hours)
                   'fitness_threshold': 1e-10,
                   'seed_rng': self.seed,
                   'saving_fitness': 2000,
                   'verbose': 0}
        if optimizer.__name__ in ['DCC']:
            options['sigma'] = 20.0/3.0  # common setting
        if optimizer.__name__ in ['DCC']:
            options['n_islands'] = 380  # need to be reset according to the number of available cores
        solver = optimizer(problem, options)
        results = solver.optimize()
        file = self._file.format(solver.__class__.__name__,
                                 solver.fitness_function.__name__,
                                 solver.ndim_problem,
                                 self.index)
        with open(file, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


class Experiments(object):
    def __init__(self, start, end, ndim_problem):
        self.start = start
        self.end = end
        self.ndim_problem = ndim_problem
        self.indices = range(self.start, self.end + 1)
        self.functions = [cf.sphere, cf.cigar, cf.discus, cf.cigar_discus, cf.ellipsoid,
                          cf.different_powers, cf.schwefel221, cf.step, cf.rosenbrock, cf.schwefel12]
        self.seeds = np.random.default_rng(2022).integers(  # explicitly control randomness for repeatability
            np.iinfo(np.int64).max, size=(len(self.functions), 50))

    def run(self, optimizer):
        for index in self.indices:
            print('* experiment: {:d} ***:'.format(index))
            for d, f in enumerate(self.functions):
                start_time = time.time()
                print('  * function: {:s}:'.format(f.__name__))
                experiment = Experiment(index, f, self.seeds[d, index], self.ndim_problem)
                experiment.run(optimizer)
                print('    runtime: {:7.5e}.'.format(time.time() - start_time))


if __name__ == '__main__':
    start_runtime = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', '-s', type=int)  # starting number of independent experiments
    parser.add_argument('--end', '-e', type=int)  # ending number of independent experiments
    parser.add_argument('--optimizer', '-o', type=str)  # name of the used optimizer
    parser.add_argument('--ndim_problem', '-d', type=int, default=2000)  # dimension of function
    args = parser.parse_args()
    params = vars(args)
    if params['optimizer'] == 'DCC':  # Distributed
        from dcc import DCC as Optimizer
    experiments = Experiments(params['start'], params['end'], params['ndim_problem'])
    experiments.run(Optimizer)
    print('*** Total runtime: {:7.5e} ***.'.format(time.time() - start_runtime))
