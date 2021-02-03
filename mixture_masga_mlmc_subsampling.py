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
import tqdm

from mlmc_mimc import MLMC 
from lib.data import get_dataset
from lib.priors import Gaussian, MixtureGaussians
from lib.models import MixtureGaussianNets
from lib.hyperparameters import config_priors



class Bayesian_Inference(MLMC):

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
        net.params.data.copy_(0.5*torch.ones_like(net.params))


    def euler_step(self, nets, U, sigma, h, dW):
        """Perform a step of Euler scheme in-place on the parameters of nets

        Parameters
        ----------
        nets : MixtureGaussianNets
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
        drift_langevin = self.prior.logprob(nets.params) + self.data_size/subsample_size * nets.loglik(self.data_X, U)
        drift_langevin.backward(torch.ones_like(drift_langevin))
        nets.params.data.copy_(nets.params.data + h*(nets.params.grad) + sigma * dW)
        if torch.isnan(nets.params.mean()):
            raise ValueError("getting nans in SDE step")
        return 0

    
    
    def Func(self, nets):
        """Function of X. 
        Recall we want to approximate E(F(X)) where X is a random vector
        
        Parameters
        ----------
        nets : MixtureGaussianNets 

        """
        with torch.no_grad():
            F = torch.norm(nets.params, p=2, dim=1)**2
        #F = nets
        return F.cpu().numpy()

    

    def mlmc_fn(self, l, N):
        """SDE to train Bayesian logistic regression
        We fix sampling parameter. We do antithetic approach on Brownian motion

        """
        dim = self.data_X.shape[1]#-1 # we don't have an intercept in this model
        
        nf = self.n0 # n steps discretisation level
        hf = 1/self.data_size#self.T/nf # step size in discretisation level
        
        sf = self.s0 * self.M ** l
        sc = int(sf/self.M)
        sigma_f = math.sqrt(2)#1/math.sqrt(self.data_size)

        sums_level_l = np.zeros(6) # this will store level l sum  and higher order momentums 
        for N1 in range(0, N, 4000):
            N2 = min(4000, N-N1) # we will do batches of paths of size N2
            
            X_f = MixtureGaussianNets(dim, N2, sigma_x=math.sqrt(5)).to(device=self.device)
            self.init_weights(X_f)

            X_c1 = copy.deepcopy(X_f) # 1st coarse process for antithetics
            X_c2 = copy.deepcopy(X_f) # 2nd coarse process for antithetics
            
            pbar = tqdm.tqdm(total=self.n0)
            for n in range(int(nf)):
                dW = math.sqrt(hf) * torch.randn_like(X_f.params)
                U = np.random.choice(self.data_size, (N2,sf), replace=True)
                self.euler_step(X_f, U, sigma_f, hf, dW)
                if l>0:
                    self.euler_step(X_c1, U[:,:sc], sigma_f, hf, dW)
                    self.euler_step(X_c2, U[:,sc:], sigma_f, hf, dW)
                if n%100==0:
                    pbar.update(100)

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
        CL = self.n0 *  (1 + self.s0 * self.M ** L)
        #cost = 2/eps**2 * self.var_Pf[min(L-1, len(self.var_Pf)-1)] * CL
        cost = 2/eps**2 * self.var_Pf[0] * CL
                
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
        #sums_level_l = self.mlmc_fn(L, 10000)
        #avg_Pf = sums_level_l[4]/10000
        #self.target = avg_Pf 
        self.target = sum(self.avg_Pf_Pc)
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
        


if __name__ == '__main__':
    
    #CONFIGURATION
    parser = argparse.ArgumentParser()
    parser.add_argument('--M', type=int, default=2, help='refinement value')
    parser.add_argument('--N', type=int, default=1000, help='samples for convergence tests')
    parser.add_argument('--L', type=int, default=5, help='levels for convergence tests')
    parser.add_argument('--s0', type=int, default=256, help='initial value of data batch size')
    parser.add_argument('--N0', type=int, default=10, help='initial number of samples for MLMC algorithm')
    parser.add_argument('--Lmin', type=int, default=1, help='minimum refinement level')
    parser.add_argument('--Lmax', type=int, default=10, help='maximum refinement level')
    parser.add_argument('--device', default='cpu', help='device')
    parser.add_argument('--dim', type=int, default=2, help='dimension of data if type_data==synthetic')
    parser.add_argument('--data_size', type=int, default=512, help="data_size if type_data==synthetic")
    parser.add_argument('--T', type=int, default=10, help='horizon time')
    parser.add_argument('--n_steps', type=int, default=10000, help='number of steps in time discretisation')
    parser.add_argument('--seed', type=int, default=1, help='seed')
    parser.add_argument('--type_data', type=str, default="synthetic", help="type of data")
    parser.add_argument('--prior', type=str, default="Gaussian", help="type of prior")
    
    args = parser.parse_args()
    
    if args.device=='cpu' or (not torch.cuda.is_available()):
        device='cpu'
    else:
        device = 'cuda:'+str(args.device)
    
    
    # Target Logistic regression, and synthetic data
    data_X, data_Y = get_dataset(m=args.data_size, d=args.dim,
        type_regression="MixtureGaussians", type_data=args.type_data, data_dir="./data/")

    data_X = data_X.to(device=device)
    try:
        data_Y = data_Y.to(device=device)
    except:
        pass
    
    # path numerical results
    dim = data_X.shape[1]
    data_size = data_X.shape[0]
    path_results = "./numerical_results/mlmc_subsampling/mixture/prior_{}/{}_d{}_m{}".format(args.prior, args.type_data, dim, data_size)
    if not os.path.exists(path_results):
        os.makedirs(path_results)
    with open(os.path.join(path_results, 'commandline_args.json'), 'w') as f:
        json.dump(vars(args), f)
    
    # prior configuration
    prior = Gaussian(mu=torch.zeros(dim, device=device),
            diagSigma=torch.tensor([10.,1.], device=device, dtype=torch.float32))
    # likelihood configuration
    
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
    
    # Bayesian log regressor
    bayesian_mixture = Bayesian_Inference(**MLMC_CONFIG)
    
    # 1.a Convergence tests
    bayesian_mixture.estimate_alpha_beta_gamma(args.L, args.N, 
            os.path.join(path_results,"log.txt"))
    bayesian_mixture.N_samples_convergence = args.N
    bayesian_mixture.save(Eps=None, Nl_list=None, mlmc_cost=None, std_cost=None, 
            filename=os.path.join(path_results, "masga_results.pickle"))
    #bayesian_logregress.save_convergence_test(os.path.join(path_results, "{}_level_s_data.txt".format()))

    # estimate target
    #bayesian_mixture.get_target(os.path.join(path_results, "log.txt"))

    ## 2. get complexities
    #Eps = [0.1,0.01,0.001,0.0005,0.0001]#,0.00005]#[0.01, 0.001,0.0005, 0.0001]#, 0.0005]
    #Nl_list, mlmc_cost, std_cost = bayesian_mixture.get_complexities(Eps, 
    #        os.path.join(path_results, "log.txt"))

    ## 3. plot
    #bayesian_mixture.plot(Eps, Nl_list, mlmc_cost, std_cost, 
    #        os.path.join(path_results,"masga_plots.pdf"))
    #
    ## 4. save
    #bayesian_mixture.save(Eps, Nl_list, mlmc_cost, std_cost, 
    #        os.path.join(path_results, "masga_results.pickle"))
