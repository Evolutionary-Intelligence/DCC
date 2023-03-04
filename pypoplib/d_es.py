import time

import numpy as np
import ray

from pypoplib.continuous_functions import load_shift_and_rotation as load_sr
from pypoplib.es import ES


class DES(ES):
    def __init__(self, problem, options):
        ES.__init__(self, problem, options)
        self.inner_max_runtime = options.get('inner_max_runtime', 3*60)  # for innerES
        # to make sure actual runtime does not exceed given 'max_runtime' AMAP
        self.max_runtime -= self.inner_max_runtime
        self.sigmas = np.array([1.0/130.0, 1.0/13.0, 1.0/1.3, 1.0, 1.3, 13.0, 130.0])
        self.sigmas_prob = np.array([0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05])
        self.sub_optimizer = None  # which needs to be set by the wrapper
        self.n_islands = options.get('n_islands')  # for innerES
        assert self.n_islands is not None and self.n_islands >= 40, 'there are at least 40 (CPU) cores.'
        self.n_es = int(np.ceil(self.n_islands/2))
        self.n_cc = self.n_islands - self.n_es
        self.nb_es, self.nb_cc = int(self.n_es/5), int(self.n_cc/5)
        w_base, w = np.log((self.nb_es*2 + 1.0)/2.0), np.log(np.arange(self.nb_es) + 1.0)
        self._dw_es = (w_base - w)/(self.nb_es*w_base - np.sum(w))
        w_base, w = np.log((self.nb_cc*2 + 1.0)/2.0), np.log(np.arange(self.nb_cc) + 1.0)
        self._dw_cc = (w_base - w)/(self.nb_cc*w_base - np.sum(w))

    def optimize(self, fitness_function=None, args=None):  # for all iterations (generations)
        super(ES, self).optimize(fitness_function)
        ro = self.rng_optimization
        ray.init(address='auto',
            runtime_env={'py_modules': ['./pypoplib'],
                'env_vars': {'OPENBLAS_NUM_THREADS': '1',  # to close *multi-thread* for avoiding possible conflicts
                'MKL_NUM_THREADS': '1',
                'OMP_NUM_THREADS': '1',
                'NUMEXPR_NUM_THREADS': '1',
                'RAY_memory_monitor_refresh_ms': '0'}})
        ray_problem = ray.put(self.problem)
        # to avoid repeated coping and communication of the same data over network,
        #   use *ray.put* to upload the shared memory in each worker node only once
        sv, rm = load_sr(self.fitness_function, np.empty((self.ndim_problem,)))
        ray_args = ray.put({'shift_vector': sv, 'rotation_matrix': rm})
        ray_es, ray_cc = ray.remote(self.sub_optimizer[0]), ray.remote(self.sub_optimizer[1])
        fitness = []  # to store all fitness generated during search
        is_first = True  # whether is the first generation or not
        # for both ES and CC
        x, y = np.empty((self.n_islands, self.ndim_problem)), np.empty((self.n_islands,))
        # for ES
        p = np.zeros((self.n_es, self.ndim_problem))
        w = np.zeros((self.n_es,))
        q = np.zeros((self.n_es, 2*int(np.ceil(np.sqrt(self.ndim_problem))), self.ndim_problem))
        s = np.ones((self.n_es,))*self.sigma
        # for CC
        ee = np.zeros((self.n_cc, self.ndim_problem))
        while not self._check_terminations():
            ray_islands, ray_results = [], []
            for i in range(self.n_islands):
                if i < self.n_es:  # for ES
                    if is_first:
                        _xx = ro.uniform(self.initial_lower_boundary, self.initial_upper_boundary)
                        _pp, _ww, _qq = p[i], w[i], q[i]
                        _ss = ro.choice(self.sigmas, p=self.sigmas_prob)*ro.uniform()*self.sigma
                    else:
                        if i < self.nb_es:  # based on elitists
                            ii = o_es[i]
                            _xx, _pp, _ww, _qq, _ss = x[ii], p[ii], w[ii], q[ii], s[ii]
                        else:
                            if ro.uniform() <= 0.5:  # based on weighted recombination
                                _xx, _pp, _qq = xx, pp, qq
                            else:  # based on diveristy-perserving recombination
                                _xx, _pp, _qq = xxx, ppp, (qq + ee[ro.choice(o_cc)])/2.0
                            _ss, _ww = ro.choice(self.sigmas, p=self.sigmas_prob)*ro.uniform()*ss, ww
                    options = {'mean': _xx,
                        'p': _pp,
                        'w': _ww,
                        'q': _qq,
                        'sigma': _ss,
                        'max_runtime': self.inner_max_runtime,
                        'fitness_threshold': self.fitness_threshold,
                        'seed_rng': ro.integers(0, np.iinfo(np.int64).max),
                        'verbose': False,
                        'saving_fitness': 100}
                    ray_islands.append(ray_es.remote(ray_problem, options))
                else:  # for CC
                    if is_first:
                        _xx = ro.uniform(self.initial_lower_boundary, self.initial_upper_boundary)
                        _ss = ro.choice(self.sigmas, p=self.sigmas_prob)*ro.uniform()*self.sigma
                    else:
                        _xx = xx if ro.uniform() < 0.5 else xxx
                        _ss = ro.choice(self.sigmas, p=self.sigmas_prob)*ro.uniform()*ss
                    options = {'mean': _xx,
                        'sigma': _ss,
                        'max_runtime': self.inner_max_runtime,
                        'fitness_threshold': self.fitness_threshold,
                        'seed_rng': ro.integers(0, np.iinfo(np.int64).max),
                        'verbose': False,
                        'saving_fitness': 100}
                    ray_islands.append(ray_cc.remote(ray_problem, options))
                ray_results.append(ray_islands[i].optimize.remote(self.fitness_function, ray_args))
            results = ray.get(ray_results)  # to synchronize (for simplicity)
            for i, r in enumerate(results):
                if self.best_so_far_y > r['best_so_far_y']:
                    self.best_so_far_x, self.best_so_far_y = r['best_so_far_x'], r['best_so_far_y']
                x[i], y[i] = r['best_so_far_x'], r['best_so_far_y']
                sub_fitness = np.copy(r['fitness'])
                sub_fitness[0, 0] += self.n_function_evaluations
                sub_fitness[-1, 0] += self.n_function_evaluations
                self.n_function_evaluations += r['n_function_evaluations']
                self.time_function_evaluations += r['time_function_evaluations']
                fitness.extend([sub_fitness[0], sub_fitness[-1]])
                if i < self.n_es:
                    p[i], w[i], q[i], s[i] = r['p'], r['w'], r['q'], r['sigma']
                else:
                    ee[i - self.n_es] = r['ee']
            # for ES
            o_es = np.argsort(y[:self.n_es])[:self.nb_es]
            ss, ww = np.dot(self._dw_es, s[o_es]), np.dot(self._dw_es, w[o_es])
            xx = np.dot(self._dw_es, x[o_es])
            pp = np.dot(self._dw_es, p[o_es]), 
            qq = np.zeros((q.shape[1], q.shape[2]))
            for i in range(self.nb_es):
                qq += self._dw_es[i]*q[o_es[i]]
            # for CC (-> ES)
            o_cc = np.argsort(y[self.n_es:])[:self.nb_cc]
            xxx = (xx + np.dot(self._dw_cc, x[self.n_es + o_cc]))/2.0
            ppp = (pp + np.dot(self._dw_cc, ee[o_cc]))/2.0
            is_first = False
        ray.shutdown()
        return self._collect(fitness)

    def _collect(self, fitness=None):
        results = {'best_so_far_x': self.best_so_far_x,
                'best_so_far_y': self.best_so_far_y,
                'n_function_evaluations': self.n_function_evaluations,
                'runtime': time.time() - self.start_time,
                'termination_signal': self.termination_signal,
                'time_function_evaluations': self.time_function_evaluations,
                'fitness': np.array(fitness)}
        return results
