# MLMC-MIMC-SGD

**WARNING** work in progress

Code for the numerical experiments in the paper [Multi-Index Antithetic Stochastic Gradient Algorithm](https://arxiv.org/abs/2006.06102?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%253A+arxiv%252FQSXk+%2528ExcitingAds%2521+cs+updates+on+arXiv.org%2529)


This repository contains Stochastic MCMC methods for Bayesian regression with variance reduction (MLMC with antithetic samples, MIMC with antithetic samples, control variate using MAP estimate) using Pytorch:
- MASGA (Multi-index Antithetic Stochastic Gradient Algorithm). Multi-index and antithetic samples at the level of the time discretisation and subsampling on the Stochastic Langevin SDE. 
For mathematical details, see

        @misc{majka2020multiindex,
            title={Multi-index Antithetic Stochastic Gradient Algorithm},
            author={Mateusz B. Majka and Marc Sabate-Vidales and Łukasz Szpruch},
            year={2020},
            eprint={2006.06102},
            archivePrefix={arXiv},
            primaryClass={stat.ML}
        }

- Stochastic Langevin Dynamics using Control Variate to reduce the variance a MAP estimator. This algorithm is taken from the paper [Control Variates for Stochastic Gradient MCMC](https://arxiv.org/abs/1706.05439) and is used for benchmarking purposes. 

## Acknowledgements
MLMC and MIMC part of the code initially based on https://bitbucket.org/pefarrell/pymlmc/src/master/.

## Files

## Running the code
- Bayesian logistic regression with Gaussian prior on [covertype dataset](https://archive.ics.uci.edu/ml/datasets/covertype) (the code is already prepared to download it). The code samples from the posterior using antithetic MLMC on the discretised Stochastic Langeving SDE and approximates E(F(X)), with F(X) = |X|^2. The code returns various plots specifying computational costs necessary to achieve different Mean Squared Errors. 
```
python logistic_masga_mlmc_subsampling.py --prior Gaussian --T 5 --n_steps 100 --device 1 --s0 32 --type_data covtype --Lmin 0 --N 10000
```
- Bayesian logistic regression with Mixture of two Gaussians as the prior on a synthetic dataset:
```
python logistic_masga_mlmc_subsampling.py --prior MixtureGaussians --T 5 --n_steps 100 --device 1 --s0 2 --dim 2 --data_size 512 --type_data synthetic --Lmin 0 --N 10000
```

