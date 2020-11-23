import sys
import os
from abc import ABC, abstractmethod
import math
import torch
import torch.nn as nn
import copy
import argparse
import numpy as np
import math
from sklearn.datasets import make_blobs
import pickle

from mlmc_mimc import MLMC 
from lib.data import get_dataset
from lib.priors import Gaussian, MixtureGaussians
from lib.models import LogisticNets
from lib.hyperparameters import config_priors



class Bayesian_logistic(MLMC):

    def __init__(self, Lmin, Lmax, N0, M, T, s0, n0, data_X, data_Y, prior, device):
        super().__init__(Lmin, Lmax, N0)
        self.data_X = data_X
        self.data_Y = data_Y
        self.dim = data_X.shape[1]
        self.M = M # refinement factor
        self.T = T  # horizon time
        self.s0 = s0 # data batch size
        self.n0 = n0 # number of timesteps at level 0
        self.device=device
        self.data_size = self.data_X.shape[0]
        self.prior = prior

    @staticmethod
    def init_weights(net, mu=0, std=1):
        """ Init weights with prior

        """
        #net.params.data.copy_(mu + std * torch.randn_like(net.params))
        net.params.data.copy_(torch.zeros_like(net.params))
        #net.params.data.copy_(torch.ones_like(net.params)*0.5)


    def euler_step(self, nets, U, sigma, h, dW):
        """Perform a step of Euler scheme in-place on the parameters of nets

        Parameters
        ----------
        nets : LogisticNets
            logistics networks. The parameters of each logistic network participate in the SDE
        U : np.ndarray
            random indexes of data for subsampling
        sigma : float
            vol
        h : float
            size of timestep
        dW : Brownian
            Brownian motion
        """
        nets.zero_grad()
        subsample_size = U.shape[1]
        drift_langevin = 1/self.data_size * self.prior.logprob(nets.params) + 1/subsample_size * nets.loglik(self.data_X, self.data_Y, U)
        drift_langevin.backward(torch.ones_like(drift_langevin))
        nets.params.data.copy_(nets.params.data + h/2*(nets.params.grad) + sigma * dW)
        if torch.isnan(nets.params.mean()):
            raise ValueError("getting nans in SDE step")
        return 0

    
    
    def Func(self, nets):
        """Function of X. 
        Recall we want to approximate E(F(X)) where X is a random vector
        
        Parameters
        ----------
        nets : LogisticNets 

        """
        with torch.no_grad():
            F = torch.norm(nets.params, p=2, dim=1)**2
        #F = nets
        return F.cpu().numpy()

    

    def mlmc_fn(self, l, N):
        """SDE to train Bayesian logistic regression
        We fix sampling parameter. We do antithetic approach on Brownian motion

        """
        dim = self.data_X.shape[1]-1
        
        nf = self.n0 # n steps discretisation level
        hf = self.T/nf # step size in discretisation level
        
        sf = self.s0 * self.M ** l
        sc = int(sf/self.M)
        sigma_f = 1/math.sqrt(self.data_size)

        sums_level_l = np.zeros(6) # this will store level l sum  and higher order momentums 

        for N1 in range(0, N, 1000):
            N2 = min(1000, N-N1) # we will do batches of paths of size N2
            
            X_f = LogisticNets(dim, N2).to(device=self.device)
            self.init_weights(X_f)

            X_c1 = copy.deepcopy(X_f) # 1st coarse process for antithetics
            X_c2 = copy.deepcopy(X_f) # 2nd coarse process for antithetics
            
            dW = torch.zeros_like(X_f.params)

            for n in range(int(nf)):
                dW = math.sqrt(hf) * torch.randn_like(dW)
                U = np.random.choice(self.data_size, (N2,sf), replace=True)
                self.euler_step(X_f, U, sigma_f, hf, dW)
                
                if l>0:
                    self.euler_step(X_c1, U[:,:sc], sigma_f, hf, dW)
                    self.euler_step(X_c2, U[:,sc:], sigma_f, hf, dW)

            F_fine = self.Func(X_f)
            F_coarse_antithetic = 0.5 * (self.Func(X_c1)+self.Func(X_c2)) if l>0 else 0
            
            # sums level l
            sums_level_l[0] += np.sum(F_fine - F_coarse_antithetic)      
            sums_level_l[1] += np.sum((F_fine - F_coarse_antithetic)**2)  
            sums_level_l[2] += np.sum((F_fine - F_coarse_antithetic)**3)  
            sums_level_l[3] += np.sum((F_fine - F_coarse_antithetic)**4)  
            sums_level_l[4] += np.sum(F_fine)
            sums_level_l[5] += np.sum(F_fine**2)  
        return sums_level_l


    def get_cost(self, l):
        """
        Number of random numbers generated:
            - Number of random numbers generated for Brownian motion: self.n0
            - In each step in time discretisation, number of random numbers generated for subsampling: self.n0*self.s0*self.M**l
        """
        cost = self.n0 * (1+self.s0 * self.M ** l)
        return cost
    
    
    def get_cost_std_MC(self, eps, Nl):
        """Cost of standard Monte Carlo
        
        Note
        ----
        We are assuming that self.var_Pf[-1] created during the approximation of alpha, beta, gamma
        is a good approximation of the variance at the finest level 

        Parameters
        ----------
        eps : float
            desired accuracy
        Nl : np.ndarray
            Number of samples per level
        """
        
        L = len(Nl)
        CL = self.n0 *  (1 + self.s0 * self.M ** (L-1))
        #cost = np.ceil(2/eps**2 * self.var_Pf[min(L-1, len(self.var_Pf)-1)]) * CL
        cost = np.ceil(2/eps**2 * self.var_Pf[0]) * CL
        return cost
    
    def get_cost_MLMC(self, eps, Nl):
        """Cost of MLMC
        
        Parameters
        ----------
        eps : float
            desired accuracy
        Nl : np.ndarray
            Number of samples per level
        """
        #cost = sum(Nl * self.n0 * self.M ** np.arange(len(Nl)))
        L = len(Nl)
        cost = 0
        for idx, nl in enumerate(Nl):
            cost += nl * self.n0 * (1+self.s0 * self.M**idx)
        return cost
    
    
    def get_target(self, logfile):

        self.write(logfile, "\n***************************\n")
        self.write(logfile, "***  Calculating target ***\n")
        self.write(logfile, "***************************\n")
        L = self.Lmax
        sums_level_l = self.mlmc_fn(L, 50000)
        avg_Pf = sums_level_l[4]/50000
        self.target = avg_Pf 
        self.write(logfile, "target = {:.4f}\n\n".format(self.target))
        return 1

    
    def get_weak_error_from_target(self, P):
        weak_error = np.abs(P-self.target)
        return weak_error

    def get_weak_error(self, ml):
        """Get weak error of MLMC approximation
        See http://people.maths.ox.ac.uk/~gilesm/files/acta15.pdf p. 21
        """
        weak_error = ml[-1]/(2**self.alpha-1)
        return weak_error
        

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

if __name__ == '__main__':
    
    #CONFIGURATION
    parser = argparse.ArgumentParser()
    parser.add_argument('--M', type=int, default=2, help='refinement value')
    parser.add_argument('--N', type=int, default=5000, help='samples for convergence tests')
    parser.add_argument('--L', type=int, default=5, help='levels for convergence tests')
    parser.add_argument('--s0', type=int, default=256, help='initial value of data batch size')
    parser.add_argument('--N0', type=int, default=2, help='initial number of samples for MLMC algorithm')
    parser.add_argument('--Lmin', type=int, default=0, help='minimum refinement level')
    parser.add_argument('--Lmax', type=int, default=8, help='maximum refinement level')
    parser.add_argument('--device', default='cpu', help='device')
    parser.add_argument('--dim', type=int, default=2, help='dimension of data if type_data==synthetic')
    parser.add_argument('--data_size', type=int, default=512, help="data_size if type_data==synthetic")
    parser.add_argument('--T', type=int, default=10, help='horizon time')
    parser.add_argument('--n_steps', type=int, default=10000, help='number of steps in time discretisation')
    parser.add_argument('--seed', type=int, default=1, help='seed')
    parser.add_argument('--type_data', type=str, default="synthetic", help="type of data", choices=["synthetic", "covtype"])
    parser.add_argument('--prior', type=str, default="Gaussian", help="type of prior", choices=["Gaussian", "MixtureGaussians"])
    
    args = parser.parse_args()
    
    if args.device=='cpu' or (not torch.cuda.is_available()):
        device='cpu'
    else:
        device = 'cuda:'+str(args.device)
    
    
    # Target Logistic regression, and synthetic data
    data_X, data_Y = get_dataset(m=args.data_size, d=args.dim,
        type_regression="logistic", type_data=args.type_data, data_dir="./data/")

    data_X = data_X.to(device=device)
    data_Y = data_Y.to(device=device)
    
    # path numerical results
    dim = data_X.shape[1]
    data_size = data_X.shape[0]
    path_results = "./numerical_results/mlmc_subsampling/logistic/prior_{}/{}_d{}_m{}".format(args.prior, args.type_data, dim, data_size)
    if not os.path.exists(path_results):
        os.makedirs(path_results)
    
    # prior configuration
    PRIORS = {"Gaussian":Gaussian, "MixtureGaussians":MixtureGaussians}
    CONFIG_PRIORS = config_priors(dim, device)
    prior = PRIORS[args.prior](**CONFIG_PRIORS[args.prior])
    
    set_seed(args.seed)
    MLMC_CONFIG = {'Lmin':args.Lmin,
            'Lmax':args.Lmax,
            'N0':args.N0,
            'M':args.M,
            'T':args.T,
            's0':args.s0,
            'n0':args.n_steps, # number of steps in time discretisation
            'data_X':data_X,
            'data_Y':data_Y,
            'device':device,
            'prior':prior
            }
    
    # save config
    with open(os.path.join(path_results, "mlmc_config.pickle"), "wb") as f:
        pickle.dump(MLMC_CONFIG, f)
    
    # Bayesian log regressor
    bayesian_logregress = Bayesian_logistic(**MLMC_CONFIG)
    
    # 1.a Convergence tests
    bayesian_logregress.estimate_alpha_beta_gamma(args.L, args.N, 
            os.path.join(path_results,"log.txt"))
    #bayesian_logregress.save_convergence_test(os.path.join(path_results, "logistic_level_s_data.txt"))


    # 2. get complexities
    Eps = [0.005, 0.001, 0.0005, 0.0001, 0.00005]#, 0.0005]
    Nl_list, mlmc_cost, std_cost = bayesian_logregress.get_complexities(Eps, 
            os.path.join(path_results, "log.txt"))

    # 3. plot
    bayesian_logregress.plot(Eps, Nl_list, mlmc_cost, std_cost, 
            os.path.join(path_results,"masga_plots.pdf"))
    
    # 4. save
    bayesian_logregress.save(Eps, Nl_list, mlmc_cost, std_cost, 
            os.path.join(path_results, "masga_results.pickle"))

