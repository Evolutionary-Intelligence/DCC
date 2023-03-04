import time
from enum import IntEnum

import numpy as np


class Terminations(IntEnum):
    """Helper class used by all optimizer classes."""
    NO_TERMINATION = 0
    MAX_FUNCTION_EVALUATIONS = 1  # maximum of function evaluations
    MAX_RUNTIME = 2  # maximal runtime to be allowed
    FITNESS_THRESHOLD = 3  # threshold of fitness (when the best-so-far fitness is below it, the optimizer will stop)


class Optimizer(object):
    """Base (abstract) class of all optimizers for continuous black-box **minimization**.

    References
    ----------
    Kochenderfer, M.J. and Wheeler, T.A., 2019.
    Algorithms for optimization.
    MIT Press.
    https://algorithmsbook.com/optimization/
    (See Chapter 7: Direct Methods for details.)

    Miikkulainen, R. and Forrest, S., 2021.
    A biological perspective on evolutionary computation.
    Nature Machine Intelligence, 3(1), pp.9-15.
    https://www.nature.com/articles/s42256-020-00278-8

    Nesterov, Y., 2018.
    Lectures on convex optimization.
    Berlin: Springer International Publishing.
    https://link.springer.com/book/10.1007/978-3-319-91578-4

    Nesterov, Y. and Spokoiny, V., 2017.
    Random gradient-free minimization of convex functions.
    Foundations of Computational Mathematics, 17(2), pp.527-566.
    https://link.springer.com/article/10.1007/s10208-015-9296-2

    Audet, C. and Hare, W., 2017.
    Derivative-free and blackbox optimization.
    Berlin: Springer International Publishing.
    https://link.springer.com/book/10.1007/978-3-319-68913-5

    Eiben, A.E. and Smith, J., 2015.
    From evolutionary computation to the evolution of things.
    Nature, 521(7553), pp.476-482.
    https://www.nature.com/articles/nature14544

    Forrest, S., 1993.
    Genetic algorithms: Principles of natural selection applied to computation.
    Science, 261(5123), pp.872-878.
    https://www.science.org/doi/10.1126/science.8346439
    """
    def __init__(self, problem, options):
        # problem-related settings
        self.problem = problem
        self.fitness_function = problem.get('fitness_function')
        self.ndim_problem = problem['ndim_problem']
        self.upper_boundary = problem.get('upper_boundary')
        self.lower_boundary = problem.get('lower_boundary')
        self.initial_upper_boundary = problem.get('initial_upper_boundary', self.upper_boundary)
        self.initial_lower_boundary = problem.get('initial_lower_boundary', self.lower_boundary)
        self.problem_name = problem.get('problem_name')
        if (self.problem_name is None) and hasattr(self.fitness_function, '__name__'):
            self.problem_name = self.fitness_function.__name__

        # optimizer-related options
        self.max_function_evaluations = options.get('max_function_evaluations', np.Inf)
        self.max_runtime = options.get('max_runtime', np.Inf)
        self.fitness_threshold = options.get('fitness_threshold', -np.Inf)
        self.n_individuals = options.get('n_individuals')  # offspring population size
        self.n_parents = options.get('n_parents')  # parent population size
        self.seed_rng = options.get('seed_rng')
        if self.seed_rng is None:  # it is highly recommended to explicitly set *seed_rng*
            self.rng = np.random.default_rng()  # NOT use it, if possible
        else:
            self.rng = np.random.default_rng(self.seed_rng)
        self.seed_initialization = options.get('seed_initialization', self.rng.integers(np.iinfo(np.int64).max))
        self.rng_initialization = np.random.default_rng(self.seed_initialization)
        self.seed_optimization = options.get('seed_optimization', self.rng.integers(np.iinfo(np.int64).max))
        self.rng_optimization = np.random.default_rng(self.seed_optimization)
        self.saving_fitness = options.get('saving_fitness', 0)
        self.verbose = options.get('verbose', 10)

        # auxiliary members
        self.Terminations = Terminations
        self.n_function_evaluations = options.get('n_function_evaluations', 0)
        self.start_function_evaluations = None
        self.time_function_evaluations = options.get('time_function_evaluations', 0)
        self.runtime = options.get('runtime', 0)
        self.start_time = None
        self.best_so_far_y = options.get('best_so_far_y', np.Inf)
        self.best_so_far_x = None
        self.termination_signal = 0  # NO_TERMINATION
        self.fitness = None
        self.is_restart = options.get('is_restart', True)

    def _evaluate_fitness(self, x, args=None):
        self.start_function_evaluations = time.time()
        if args is None:
            y = self.fitness_function(x)
        else:
            y = self.fitness_function(x, args['shift_vector'], args['rotation_matrix'])
        self.time_function_evaluations += time.time() - self.start_function_evaluations
        self.n_function_evaluations += 1
        # update best-so-far solution (x) and fitness (y)
        if y < self.best_so_far_y:
            self.best_so_far_x, self.best_so_far_y = np.copy(x), y
        return float(y)

    def _check_terminations(self):
        self.runtime = time.time() - self.start_time
        if self.n_function_evaluations >= self.max_function_evaluations:
            termination_signal = True, Terminations.MAX_FUNCTION_EVALUATIONS
        elif self.runtime >= self.max_runtime:
            termination_signal = True, Terminations.MAX_RUNTIME
        elif self.best_so_far_y <= self.fitness_threshold:
            termination_signal = True, Terminations.FITNESS_THRESHOLD
        else:
            termination_signal = False, Terminations.NO_TERMINATION
        self.termination_signal = termination_signal[1]
        return termination_signal[0]

    def _compress_fitness(self, fitness):
        fitness = np.array(fitness)
        # arrange in non-increasing order
        for i in range(len(fitness) - 1):
            if fitness[i] < fitness[i + 1]:
                fitness[i + 1] = fitness[i]
        if self.saving_fitness == 1:
            self.fitness = np.stack((np.arange(1, len(fitness) + 1), fitness), 1)
        elif self.saving_fitness > 1:
            # use 1-based index
            index = np.arange(1, len(fitness), self.saving_fitness)
            # recover 0-based index via - 1
            index = np.append(index, len(fitness)) - 1
            self.fitness = np.stack((index, fitness[index]), 1)
            # recover 1-based index
            self.fitness[0, 0], self.fitness[-1, 0] = 1, len(fitness)

    def _collect(self, fitness):
        if self.saving_fitness:
            self._compress_fitness(fitness[:self.n_function_evaluations])
        return {'best_so_far_x': self.best_so_far_x,
                'best_so_far_y': self.best_so_far_y,
                'n_function_evaluations': self.n_function_evaluations,
                'runtime': time.time() - self.start_time,
                'termination_signal': self.termination_signal,
                'time_function_evaluations': self.time_function_evaluations,
                'fitness': self.fitness}

    def initialize(self):
        raise NotImplementedError

    def iterate(self):
        raise NotImplementedError

    def optimize(self, fitness_function=None):
        self.start_time = time.time()
        if fitness_function is not None:
            self.fitness_function = fitness_function
        fitness = []  # to store all fitness generated during evolution/optimization
        return fitness
