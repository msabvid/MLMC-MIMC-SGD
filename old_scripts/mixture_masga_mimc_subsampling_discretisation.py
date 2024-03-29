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
from itertools import product
import matplotlib.pyplot as plt
import json
import tqdm

from mlmc_mimc import MIMC 
from lib.data import get_dataset
from lib.priors import Gaussian, MixtureGaussians
from lib.models import MixtureGaussianNets
from lib.config import config_priors


        

class Bayesian_Inference(MIMC):

    def __init__(self, Lmin, Lmax, N0, M, h0, s0, n0, data_X, data_Y, prior, device):
        super().__init__(Lmin, Lmax, N0)
        self.data_X = data_X
        self.data_Y = data_Y
        self.dim = data_X.shape[1]
        self.M = M # refinement factor
        self.h0 = h0  # initial time step size 
        self.s0 = s0 # data batch size
        self.n0 = n0 # number of timesteps at level 0
        self.T = self.n0 * self.h0
        self.device=device
        self.data_size = self.data_X.shape[0]
        self.target = 0
        self.prior = prior

    @staticmethod
    def init_weights(net, mu=0, std=1):
        """ Init weights with prior

        """
        #net.params.data.copy_(mu + std * torch.randn_like(net.params))
        net.params.data.copy_(torch.zeros_like(net.params))


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
        drift_langevin = self.prior.logprob(nets.params) + self.data_size/subsample_size * nets.loglik(self.data_X, U)
        drift_langevin.backward(torch.ones_like(drift_langevin))
        nets.params.data.copy_(nets.params.data + h*(nets.params.grad) + sigma * dW)
        if torch.isnan(nets.params.mean()):
            raise ValueError('nans!')
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
        return F.cpu().numpy()

    

    def mlmc_fn(self, l, N):
        """SDE to train Bayesian logistic regression
        We fix sampling parameter. We do antithetic approach on Brownian motion

        """
        dim = self.data_X.shape[1]#-1
        lh, ls = l[0],l[1] # level h and level s
        
        # discretisation level
        nf = self.n0 * self.M ** lh # n steps in fine time discretisation
        hf = self.T/nf # step size in coarse time discretisation
        nc = int(nf/self.M)
        hc = self.T/nc
        
        # drift estimation level
        sf = self.s0 * self.M ** ls
        sc = int(sf/self.M)
        sigma_f = math.sqrt(2)
        
        
        sums_level_l = np.zeros(6) # this will store level l sum  and higher order momentums 

        for N1 in range(0, N, 4000):
            N2 = min(4000, N-N1) # we will do batches of paths of size N2
            
            X_hf_sf = MixtureGaussianNets(dim, N2, sigma_x=math.sqrt(5)).to(device=self.device)
            self.init_weights(X_hf_sf)

            X_hf_sc1 = copy.deepcopy(X_hf_sf)
            X_hf_sc2 = copy.deepcopy(X_hf_sf)

            X_hc1_sf = copy.deepcopy(X_hf_sf)
            X_hc2_sf = copy.deepcopy(X_hf_sf)

            X_hc1_sc1 = copy.deepcopy(X_hf_sf)
            X_hc1_sc2 = copy.deepcopy(X_hf_sf)
            X_hc2_sc1 = copy.deepcopy(X_hf_sf)
            X_hc2_sc2 = copy.deepcopy(X_hf_sf)

            dWf = torch.zeros_like(X_hf_sf.params).requires_grad_(False)
            dWc = torch.zeros_like(X_hf_sf.params).requires_grad_(False)

            if lh==0 and ls==0:
                pbar = tqdm.tqdm(total=int(nf))
                for n in range(int(nf)):
                    dWf = math.sqrt(hf) * torch.randn_like(dWf)
                    U = np.random.choice(self.data_size, (N2,sf))
                    self.euler_step(X_hf_sf, U, sigma_f, hf, dWf)
                    if n%100==0:
                        pbar.update(100)
            else:
                pbar = tqdm.tqdm(total=int(nc))
                for n in range(int(nc)):
                    dWc = dWc * 0
                    U_list = []
                    for m in range(self.M):
                        U = np.random.choice(self.data_size, (N2,sf))
                        U_list.append(U)
                        dWf = math.sqrt(hf) * torch.randn_like(dWf)
                        dWc += dWf

                        self.euler_step(X_hf_sf, U, sigma_f, hf, dWf)
                        
                        self.euler_step(X_hf_sc1, U[:,:sc], sigma_f, hf, dWf)
                        self.euler_step(X_hf_sc2, U[:,sc:], sigma_f, hf, dWf)
                
                    self.euler_step(X_hc1_sf, U_list[0], sigma_f, hc, dWc)
                    self.euler_step(X_hc2_sf, U_list[1], sigma_f, hc, dWc)
                    
                    self.euler_step(X_hc1_sc1, U_list[0][:,:sc], sigma_f, hc, dWc)
                    self.euler_step(X_hc1_sc2, U_list[0][:,sc:], sigma_f, hc, dWc)
                    self.euler_step(X_hc2_sc1, U_list[1][:,:sc], sigma_f, hc, dWc)
                    self.euler_step(X_hc2_sc2, U_list[1][:,sc:], sigma_f, hc, dWc)
                    
                    if n%100 == 0:
                        pbar.update(100)
                
            F_fine = self.Func(X_hf_sf)
            if lh>0 and ls>0:
                F_coarse_antithetic = -0.5 * (self.Func(X_hc1_sf) + self.Func(X_hc2_sf))
                F_coarse_antithetic = F_coarse_antithetic - 0.5 * (self.Func(X_hf_sc1) - \
                        0.5 * (self.Func(X_hc1_sc1) + self.Func(X_hc2_sc1)))
                F_coarse_antithetic = F_coarse_antithetic - 0.5 * (self.Func(X_hf_sc2) - \
                        0.5 * (self.Func(X_hc1_sc2) + self.Func(X_hc2_sc2)))
            elif lh>0 and ls==0:
                F_coarse_antithetic = -0.5 * (self.Func(X_hc1_sf) + self.Func(X_hc2_sf))
            elif lh==0 and ls>0:
                F_coarse_antithetic = -0.5 * (self.Func(X_hf_sc1) + self.Func(X_hf_sc2))
            else:
                F_coarse_antithetic=0
            
            # sums level l
            sums_level_l[0] += np.sum(F_fine + F_coarse_antithetic)      
            sums_level_l[1] += np.sum((F_fine + F_coarse_antithetic)**2)  
            sums_level_l[2] += np.sum((F_fine + F_coarse_antithetic)**3)  
            sums_level_l[3] += np.sum((F_fine + F_coarse_antithetic)**4)  
            sums_level_l[4] += np.sum(F_fine)
            sums_level_l[5] += np.sum(F_fine**2)  
        return sums_level_l


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
        x = np.array(Nl.shape) - 1 
        Cl = self.n0 * (self.M ** x[0]) * (1 + self.s0 * self.M ** x[1])
        #try:
        #    cost = np.ceil(2/eps**2 * self.var_Pf[x[0],x[1]]) * Cl
        #except:
        #    cost = np.ceil(2/eps**2 * self.var_Pf[-1,-1]) * Cl
        cost = np.ceil(2/eps**2 * self.var_Pf[0,0]) * Cl
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
        Lh,Ls = Nl.shape
        cost = 0
        for lh,ls in product(range(Lh),range(Ls)):
            cl = self.n0 * (self.M ** lh) * (1 + self.s0 * self.M ** ls)
            cost += Nl[lh, ls] * cl

        return cost
        
    def get_target(self, logfile):
        self.write(logfile, "\n***************************\n")
        self.write(logfile, "***  Calculating target ***\n")
        self.write(logfile, "***************************\n")
        Lh, Ls = 7,7
        #sums_level_l = self.mlmc_fn((Lh,Ls), 5000)
        #self.target = sums_level_l[4]/5000 + sum(self.avg_Pf_Pc)
        self.target = np.sum(self.avg_Pf_Pc)
        #avg_Pf_Pc = np.zeros([Lh+1,Ls+1])
        #for lh,ls in product(range(Lh+1), range(Ls+1)):
        #    sums_level_l = self.mlmc_fn((lh,ls), 5000)
        #    avg_Pf_Pc[lh,ls] = sums_level_l[0]/5000
        #    avg_Pf = sums_level_l[4]/5000
        #    format_string = "{:<5}{:<5}{:<15.4e}{:<15.4e}\n"
        #    self.write(logfile,format_string.format(lh,ls,avg_Pf_Pc[lh,ls], avg_Pf))
        #self.target = np.sum(avg_Pf_Pc)
        self.write(logfile, "target = {:.4f}\n\n".format(self.target))
        return 0
    
    def get_weak_error_from_target(self, P):
        weak_error = np.abs(P-self.target)
        return weak_error
    
    def get_weak_error(self, L):
        """Get weak error of MLMC approximation
        See https://link.springer.com/content/pdf/10.1007/s00211-015-0734-5.pdf (53)
        """
        weak_error = 0

        #weak_error = np.abs(P-self.target) 
        for l1, l2 in zip([L+1]*(L+1), range(L+1)):
            sums_level_l = self.mlmc_fn((l1,l2),10000)
            weak_error += sums_level_l[0]/10000
        for l1, l2 in zip(range(L+1), [L+1]*(L+1)):
            sums_level_l = self.mlmc_fn((l1,l2),10000)
            weak_error += sums_level_l[0]/10000
        sums_level_l = self.mlmc_fn((L+1,L+1),10000)
        weak_error += sums_level_l[0]/10000
        return abs(weak_error)
        
            

if __name__ == '__main__':
    
    #CONFIGURATION
    parser = argparse.ArgumentParser()
    parser.add_argument('--M', type=int, default=2, help='refinement value')
    parser.add_argument('--N', type=int, default=1000, help='samples for convergence tests')
    parser.add_argument('--L', type=int, default=5, help='levels for convergence tests')
    parser.add_argument('--s0', type=int, default=2, help='initial value of data batch size')
    parser.add_argument('--N0', type=int, default=2, help='initial number of samples for MLMC algorithm')
    parser.add_argument('--Lmin', type=int, default=1, help='minimum refinement level')
    parser.add_argument('--Lmax', type=int, default=10, help='maximum refinement level')
    parser.add_argument('--device', default='cpu', help='device')
    parser.add_argument('--dim', type=int, default=2, help='dimension of data if type_data==synthetic')
    parser.add_argument('--data_size', type=int, default=512, help="data_size if type_data==synthetic")
    parser.add_argument('--h0', type=float, default=0.005, help='initial timestep size')
    parser.add_argument('--n_steps', type=int, default=10, help='inital number of steps in time discretisation')
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
    path_results = "./numerical_results/mimc/mixture/prior_{}/{}_d{}_m{}".format(args.prior, args.type_data, dim, data_size)
    if not os.path.exists(path_results):
        os.makedirs(path_results)
    with open(os.path.join(path_results, 'commandline_args.json'), 'w') as f:
        json.dump(vars(args), f)
    
    # prior configuration
    PRIORS = {"Gaussian":Gaussian, "MixtureGaussians":MixtureGaussians}
    CONFIG_PRIORS = config_priors(dim, device)
    prior = PRIORS[args.prior](**CONFIG_PRIORS[args.prior])
    # MIMC config
    MIMC_CONFIG = {'Lmin':args.Lmin,
            'Lmax':args.Lmax,
            'N0':args.N0,
            'M':args.M,
            'h0':args.h0,
            's0':args.s0,
            'n0':args.n_steps, # initial number of steps at level 0
            'data_X':data_X,
            'data_Y':data_Y,
            'device':device,
            'prior':prior
            }
    
    # Bayesian log regressor
    bayesian_mixture = Bayesian_Inference(**MIMC_CONFIG)
    
    # 1. Convergence tests
    bayesian_mixture.estimate_alpha_beta_gamma(args.L, args.N, 
            os.path.join(path_results, "log.txt"))
    bayesian_mixture.save(Eps=None, Nl_list=None, mlmc_cost=None, std_cost=None, 
            filename=os.path.join(path_results, "masga_results.pickle"))
