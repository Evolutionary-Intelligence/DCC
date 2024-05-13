import os
import time
import pickle  # for data storage
import argparse

import numpy as np  # engine for numerical computing
import ray  # engine for distributed computing
from sklearn.preprocessing import Normalizer

# for loading large datasets involved in fitness evaluations
import pypop7.benchmarks.data_science as ds
from dcc import DCC as Optimizer


def tanh_loss_lr(x, args):
    return ds.tanh_loss_lr(x, args['x'], args['y'])


def square_loss_lr(x, args):
    return ds.square_loss_lr(x, args['x'], args['y'])


def loss_svm(x, args):
    return ds.loss_svm(x, args['x'], args['y'])


def loss_margin_perceptron(x, args):
    return ds.loss_margin_perceptron(x, args['x'], args['y'])


class Experiment(object):
    def __init__(self, index, function, seed):
        self.index = index  # index of each experiment (trial)
        assert self.index > 0
        self.function = function
        self.seed = seed  # random seed of each experiment
        assert self.seed >= 0
        self._folder = 'pypop7_benchmarks_lso'  # folder to save all data
        if not os.path.exists(self._folder):
            os.makedirs(self._folder)
        # to set file name for each experiment
        self._file = os.path.join(self._folder, 'Algo-{}_Func-{}_Dim-{}_Exp-{}.pickle')
        # https://docs.ray.io/en/latest/ray-core/scheduling/ray-oom-prevention.html
        ray.init(
            runtime_env={'py_modules': ['./pypoplib'],  # this local folder is shared across all nodes
                'env_vars': {'OPENBLAS_NUM_THREADS': '1',  # to close *multi-thread* for avoiding possible conflicts
                    'MKL_NUM_THREADS': '1',  # to close *multi-thread* for avoiding possible conflicts
                    'OMP_NUM_THREADS': '1',  # to close *multi-thread* for avoiding possible conflicts
                    'NUMEXPR_NUM_THREADS': '1',  # to close *multi-thread* for avoiding possible conflicts
                    'RAY_memory_monitor_refresh_ms': '0'}})  # to avoid Out-Of-Memory Prevention

    def run(self, optimizer):
        x, y = None, None
        if self.function == 'tanh_loss_lr_qar':
            x, y = ds.read_qsar_androgen_receptor(is_10=False)
            self.ndim_problem = x.shape[1] + 1
            transformer = Normalizer().fit(x)
            x, y = transformer.transform(x), y
            function_name = 'tanh_loss_lr_qar'
        elif self.function == 'tanh_loss_lr_c':
            x, y = ds.read_cnae9(is_10=False)
            self.ndim_problem = x.shape[1] + 1
            transformer = Normalizer().fit(x)
            x, y = transformer.transform(x), y
            function_name = 'tanh_loss_lr_c'
        elif self.function == 'tanh_loss_lr_shd':
            x, y = ds.read_semeion_handwritten_digit(is_10=False)
            self.ndim_problem = x.shape[1] + 1
            transformer = Normalizer().fit(x)
            x, y = transformer.transform(x), y
            function_name = 'tanh_loss_lr_shd'
        elif self.function == 'square_loss_lr_qar':
            x, y = ds.read_qsar_androgen_receptor(is_10=True)
            self.ndim_problem = x.shape[1] + 1
            transformer = Normalizer().fit(x)
            x, y = transformer.transform(x), y
            function_name = 'square_loss_lr_qar'
        elif self.function == 'square_loss_lr_c':
            x, y = ds.read_cnae9(is_10=True)
            self.ndim_problem = x.shape[1] + 1
            transformer = Normalizer().fit(x)
            x, y = transformer.transform(x), y
            function_name = 'square_loss_lr_c'
        elif self.function == 'square_loss_lr_shd':
            x, y = ds.read_semeion_handwritten_digit(is_10=True)
            self.ndim_problem = x.shape[1] + 1
            transformer = Normalizer().fit(x)
            x, y = transformer.transform(x), y
            function_name = 'square_loss_lr_shd'
        elif self.function == 'loss_svm_qar':
            x, y = ds.read_qsar_androgen_receptor(is_10=False)
            self.ndim_problem = x.shape[1] + 1
            transformer = Normalizer().fit(x)
            x, y = transformer.transform(x), y
            function_name = 'loss_svm_qar'
        elif self.function == 'loss_margin_perceptron_qar':
            x, y = ds.read_qsar_androgen_receptor(is_10=False)
            self.ndim_problem = x.shape[1] + 1
            transformer = Normalizer().fit(x)
            x, y = transformer.transform(x), y
            function_name = 'loss_margin_perceptron_qar'
        # to define all the necessary properties of the objective/cost function to be minimized
        problem = {'fitness_function': tanh_loss_lr,  # cost function
                   'ndim_problem': self.ndim_problem,  # dimension
                   'upper_boundary': 10.0*np.ones((self.ndim_problem,)),  # search boundary
                   'lower_boundary': -10.0*np.ones((self.ndim_problem,))}
        # to define all the necessary properties of the black-box optimizer considered
        options = {'max_function_evaluations': np.Inf,  # here we focus on the *wall-clock* time
                   'max_runtime': 60*60*3,  # maximal runtime to be allowed (seconds)
                   'fitness_threshold': 1e-10,  # fitness threshold to stop the optimization process
                   'seed_rng': self.seed,  # seed for random number generation (RNG)
                   'saving_fitness': 2000,  # to compress the convergence data (for saving storage space)
                   'verbose': 0,  # to not print verbose information (for simplicity)
                   'sigma': 20.0/3.0}  # note that not all optimizers will use this setting (for e.g., ESs)
        solver = optimizer(problem, options)  # to initialize the optimizer object
        results = solver.optimize(problem['fitness_function'], ray.put({'x': x, 'y': y}))  # to run the optimization/search/evolution process
        file = self._file.format(solver.__class__.__name__,
                                 function_name,
                                 solver.ndim_problem,
                                 self.index)
        with open(file, 'wb') as handle:  # to save all data in .pickle format
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


class Experiments(object):
    """A set of *independent* experiments starting and ending in the given index range."""
    def __init__(self, start, end):
        self.start = start  # starting index of independent experiments
        assert self.start > 0
        self.end = end  # ending index of independent experiments
        assert self.end > 0 and self.end >= self.start
        self.indices = range(self.start, self.end + 1)  # index range (1-based rather 0-based)
        self.functions = ['loss_svm_qar']
        self.seeds = np.random.default_rng(2023).integers(  # to generate all random seeds *in advances*
            np.iinfo(np.int64).max, size=(len(self.functions), 20))

    def run(self, optimizer):
        for index in self.indices:  # independent experiments
            print('* experiment: {:d} ***:'.format(index))
            for d, f in enumerate(self.functions):  # for each function
                start_time = time.time()
                print('  * function: {:s}:'.format(f))
                experiment = Experiment(index, f, self.seeds[d, index])
                experiment.run(optimizer)
                print('    runtime: {:7.5e}.'.format(time.time() - start_time))


if __name__ == '__main__':
    start_runtime = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', '-s', type=int)  # starting index
    parser.add_argument('--end', '-e', type=int)  # ending index
    args = parser.parse_args()
    params = vars(args)
    experiments = Experiments(params['start'], params['end'])
    experiments.run(Optimizer)
    print('*** Total runtime: {:7.5e} ***.'.format(time.time() - start_runtime))
