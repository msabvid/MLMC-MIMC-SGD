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

from mlmc_mimc import MIMC 
from lib.priors import Gaussian, MixtureGaussians
from lib.models import LogisticNets
from lib.config import config_priors

class Bayesian_MASGA(MIMC):

    def __init__(self, Lmin: int, Lmax: int, N0: int, M: int, h0: float, 
            s0: int, n0: int, data_X: torch.Tensor, data_Y: torch.Tensor, prior, model, func, device):
        super().__init__(Lmin, Lmax, N0)
        self.data_X = data_X
        self.data_Y = data_Y
        self.dim = data_X.shape[1]
        self.M = M # refinement factor
        self.h0 = h0  # horizon time
        self.s0 = s0 # data batch size at level 0
        self.n0 = n0 # number of timesteps at level 0
        self.device=device
        self.data_size = self.data_X.shape[0]
        self.target = 0
        self.prior = prior
        self.model = model
        self.func = func

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
        drift_langevin = self.prior.logprob(nets.params) + self.data_size/subsample_size * nets.loglik(self.data_X, self.data_Y, U)
        drift_langevin.backward(torch.ones_like(drift_langevin))
        nets.params.data.copy_(nets.params.data + h*(nets.params.grad) + sigma * dW)
        if torch.isnan(nets.params.mean()):
            raise ValueError('nans in Euler scheme!')
        return 0

    
    
    def Func(self, nets):
        """Function of X. 
        Recall we want to approximate E(F(X)) where X is a random vector
        
        Parameters
        ----------
        nets : LogisticNets 

        """
        with torch.no_grad():
            F = self.func(nets.params)#torch.norm(nets.params, p=2, dim=1)**2
        return F.cpu().numpy()

    

    def mlmc_fn(self, l, N, **kwargs):
        """SDE to train Bayesian logistic regression
        We fix sampling parameter. We do antithetic approach on Brownian motion
        
        Parameters
        ----------
        l: Tuple[int]
            Tuple (lh, ls) of discretisation level and subsampling level
        N: int
            Number of paths to run for Monte Carlo simulation
        
        Returns
        -------
        sums_level_l : np.ndarray of size (6)
            sums_level_l[0] := \sum F(X_L^i) - F(X_{L-1}^i)
            sums_level_l[1] := \sum (F(X_L^i) - F(X_{L-1}^i))^2
            sums_level_l[2] := \sum (F(X_L^i) - F(X_{L-1}^i))^3
            sums_level_l[3] := \sum (F(X_L^i) - F(X_{L-1}^i))^4
            sums_level_l[4] := \sum F(X_L^i)
            sums_level_l[5] := \sum (F(X_L^i)^2) 
        """
        dim = self.data_X.shape[1]-1
        lh, ls = l[0],l[1] # level h and level s
        
        # discretisation level
        nf = self.n0 * self.M ** lh # n steps in fine time discretisation
        hf = self.h0 / (self.M ** lh) #self.T/nf # step size in coarse time discretisation
        nc = int(nf/self.M)
        hc = hf * self.M #self.T/nc

        # we re-scale timestep size: h:=h/(2m)
        hf = hf/(2*self.data_size)
        hc = hc/(2*self.data_size)
        
        # drift estimation level
        sf = self.s0 * self.M ** ls
        sc = int(sf/self.M)
        sigma_f = math.sqrt(2)#5/math.sqrt(self.data_size)
        
        
        sums_level_l = np.zeros(6) # this will store level l sum  and higher order momentums 

        for N1 in range(0, N, 5000):
            N2 = min(5000, N-N1) # we will do batches of paths of size N2
            
            X_hf_sf = self.model(dim, N2, **kwargs).to(device=self.device)
            self.init_weights(X_hf_sf)

            X_hf_sc1 = copy.deepcopy(X_hf_sf)
            X_hf_sc2 = copy.deepcopy(X_hf_sf)

            X_hc1_sf = copy.deepcopy(X_hf_sf)
            X_hc2_sf = copy.deepcopy(X_hf_sf)

            X_hc1_sc1 = copy.deepcopy(X_hf_sf)
            X_hc1_sc2 = copy.deepcopy(X_hf_sf)
            X_hc2_sc1 = copy.deepcopy(X_hf_sf)
            X_hc2_sc2 = copy.deepcopy(X_hf_sf)

            dWf = torch.zeros_like(X_hf_sf.params)
            dWc = torch.zeros_like(X_hf_sf.params)

            if lh==0 and ls==0:
                for n in range(int(nf)):
                    dWf = math.sqrt(hf) * torch.randn_like(dWf)
                    U = np.random.choice(self.data_size, (N2,sf))
                    self.euler_step(X_hf_sf, U, sigma_f, hf, dWf)
            else:
                for n in range(int(nc)):
                    dWc = dWc * 0
                    U_list = []
                    for m in range(self.M):
                        U = np.random.choice(self.data_size, (N2,sf)) # subsampling with replacement
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
        #cost = sum(Nl * self.n0 * self.M ** np.arange(len(Nl)))
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
        self.target = np.sum(self.avg_Pf_Pc)
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
        
        if (L+1)<self.avg_Pf_Pc.shape[0]:
            # use simulations we used used in the calculation of the rates.
            for l1, l2 in zip([L+1]*(L+1), range(L+1)):
                weak_error += self.avg_Pf_Pc[l1,l2]
            for l1, l2 in zip(range(L+1), [L+1]*(L+1)):
                weak_error += self.avg_Pf_Pc[l1,l2]
            weak_error += self.avg_Pf_Pc[L+1,L+1]
        else:
            for l1, l2 in zip([L+1]*(L+1), range(L+1)):
                sums_level_l = self.mlmc_fn((l1,l2),10000)
                weak_error += sums_level_l[0]/10000
            for l1, l2 in zip(range(L+1), [L+1]*(L+1)):
                sums_level_l = self.mlmc_fn((l1,l2),10000)
                weak_error += sums_level_l[0]/10000
            sums_level_l = self.mlmc_fn((L+1,L+1),10000)
            weak_error += sums_level_l[0]/10000
        return abs(weak_error)
    

        
            
