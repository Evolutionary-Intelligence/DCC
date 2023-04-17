# DCC: Distributed Cooperative Coevolution (powered by LM-CMA/CMA-ES under the recently proposed [Multi-Level Learning](https://www.pnas.org/doi/10.1073/pnas.2120037119) framework).

This is a companion website for the paper **Cooperative Coevolution for Non-Separable Large-Scale Black-Box Optimization: Convergence Analyses and Distributed Accelerations**, which has been submitted to [IEEE-TNNLS](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385) (*under review*).

All the source code and data (as well as [online Supplementary Materials](https://github.com/Evolutionary-Intelligence/DCC/blob/main/SupplementaryMaterials.pdf)) involved in this paper are presented here to ensure **repeatability**.

## All Baseline Optimizers

The below optimizers are given in **alphabetical** order. See their references according to their **open-source** code files.

Optimizer | Source Code
--------- | -----------
Adaptive Estimation of Multivariate Normal Algorithm | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/eda/aemna.py
Annealed Random Hill Climber | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/rs/arhc.py
BErnoulli Smoothing | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/rs/bes.py
Cholesky-CMA-ES 2009 | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/ccmaes2009.py
Cooperative Coevolving Particle Swarm Optimizer | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/pso/ccpso2.py
Classic Differential Evolution | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/de/cde.py
Comprehensive Learning Particle Swarm Optimizer | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/pso/clpso.py
Covariance Matrix Adaptation Evolution Strategy | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/cmaes.py
CoOperative CO-evolutionary Covariance Matrix Adaptation | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/cc/cocma.py
COmposite Differential Evolution | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/de/code.py
CoOperative co-Evolutionary Algorithm | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/cc/coea.py
CoOperative SYnapse NEuroevolution | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/cc/cosyne.py
Corana et al.' Simulated Annealing | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/sa/csa.py
Cumulative Step-size self-Adaptation Evolution Strategy | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/csaes.py
Differentiable Cross-Entropy Method | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/cem/dcem.py
Diagonal Decoding Covariance Matrix Adaptation | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/ddcma.py
Derandomized Self-Adaptation Evolution Strategy | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/dsaes.py
Dynamic Smoothing Cross-Entropy Method | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/cem/dscem.py
Exact Natural Evolution Strategy | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/nes/enes.py
Enhanced Simulated Annealing | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/sa/esa.py
Fast Covariance Matrix Adaptation Evolution Strategy | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/fcmaes.py
Adaptive Differential Evolution | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/de/jade.py
Limited-Memory Covariance Matrix Adaptation | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/lmcma.py
Limited-Memory Covariance Matrix Adaptation Evolution Strategy | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/lmcmaes.py
Limited-Memory Matrix Adaptation Evolution Strategy | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/lmmaes.py
Matrix Adaptation Evolution Strategy | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/maes.py
Mixture Model-based Evolution Strategy | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/mmes.py
(1+1)-Active-CMA-ES 2010 | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/opoa2010.py
(1+1)-Active-CMA-ES 2015 | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/opoa2015.py
(1+1)-Cholesky-CMA-ES 2006 | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/opoc2006.py
(1+1)-Cholesky-CMA-ES 2009 | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/opoc2009.py
Pure Random Search | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/rs/prs.py
Rank-One Evolution Strategy | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/r1es.py
Rank-One Natural Evolution Strategies | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/nes/r1nes.py
Rechenberg's (1+1)-Evolution Strategy with 1/5th success rule | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/res.py
Random (stochastic) Hill Climber | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/rs/rhc.py
Rank-M Evolution Strategy | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/rmes.py
Random-Projection Estimation of Distribution Algorithm | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/eda/rpeda.py
Standard Cross-Entropy Method | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/cem/scem.py
Separable Covariance Matrix Adaptation Evolution Strategy | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/sepcmaes.py
Separable Natural Evolution Strategies | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/nes/snes.py
Standard Particle Swarm Optimizer with a global topology | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/pso/spso.py
Standard Particle Swarm Optimizer with a Local (ring) topology | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/pso/spsol.py
Simple Random Search | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/rs/srs.py
Univariate Marginal Distribution Algorithm for normal models | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/eda/umda.py
Linear Covariance Matrix Adaptation | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/vdcma.py
Projection-based Covariance Matrix Adaptation | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/vkdcma.py
Exponential Natural Evolution Strategies | https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/nes/xnes.py

# Configurations of Clustering Computing Platform

Use the following three `bash` commands to obtain the information of each slave node in the clustering computing platform:

1. Use `lsb_release -a` to obtain **operation system (OS)** information,
2. Use `cat /proc/cpuinfo | grep name | cut -f2 -d: | uniq -c` to obtain **CPU** information,
3. Use `free -mh` to obtain **memory** information.

OS | CPU | Memory
-- | --- | ------
Ubuntu 16.04 LTS | 40  Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz | 62G
Ubuntu 16.04 LTS | 40  Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz | 62G
Ubuntu 16.04 LTS | 40  Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz | 62G
Ubuntu 16.04 LTS | 40  Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz | 62G
Ubuntu 16.04 LTS | 40  Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz | 62G
Ubuntu 16.04 LTS | 40  Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz | 62G
Ubuntu 16.04 LTS | 40  Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz | 62G
Ubuntu 16.04 LTS | 40  Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz | 32G
Ubuntu 16.04 LTS | 40  Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz | 32G
Ubuntu 16.04 LTS | 40  Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz | 32G

* Increase the limits of open files (by default, 1024) to e.g. 500000 for successful runs:

```bash
$ ulimit -n
$ sudo vi /etc/security/limits.conf
```

* Make sure that all slave nodes use the same versions of all three Python libraries (NumPy, SciPy, and Ray):

```bash
$ pip install numpy  # to use the virtual environment to install depended libraries
$ pip install scipy
$ pip install ray
$ export RAY_memory_monitor_refresh_ms=0
```

* Choose one node as the master node:

```bash
$ ray start --head
$ ray status

$ ssh -L 8265:0.0.0.0:8265 <NAME>@<IP-ADDRESS>  # for local access to the remote clustering computing platform
# <NAME> and <IP-ADDRESS> are replaced by user name and IP of the master node
$ python  # local
>>> import ray
>>> ray.init(include_dashboard=True)
# open http://localhost:8265/ in the browser for status of the used clustering computing platform
```
