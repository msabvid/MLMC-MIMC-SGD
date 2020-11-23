
# MLMC-MIMC-SGD
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


For the MLMC and MIMC part of the code, code initially based on https://bitbucket.org/pefarrell/pymlmc/src/master/.

## Files

## Running the code

- SGLD with and without control variate
```
python logistic_sgld_cv.py --device 2 --subsample_size 10000 --dim 10 --type_data covtype --N 100
```
