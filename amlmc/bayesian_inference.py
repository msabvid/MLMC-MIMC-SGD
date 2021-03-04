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
import pickle
import json

from mlmc_mimc import MLMC 
from ..lib.data import get_dataset
from ..nn.priors import Gaussian, MixtureGaussians
from ..nn.models import LogisticNets



class Bayesian_AMLMC(MLMC):
    """
    Multi-level Monte Carlo at the subsampling level with antithetic samples 
    """

    def __init__(self, Lmin: int, Lmax: int, N0: int, M: int, h0: float, 
            s0: int, n0: int, data_X: torch.Tensor, data_Y: torch.Tensor, prior, model, func, init_func, device):
        """
        Parameters
        ----------
        Lmin: int, minimum refinement level in MIMC algorithm
        LMax: int, maximum refinement level in MIMC algorithm
        N0: int, initial number of samples in MIMC algorithm
        M: int, refinement value
        h0: float, initial step size at discretisation level 0
        s0: int, initial subsample size at subsampling level 0
        n0: int, number of steps at discretisation level 0
        data_X: torch.Tensor, 
        data_Y: torch.Tensor
        prior: BasePrior
        model: BaseModel
        func: function. f in E(f(X))
        device: str
        """
        super().__init__(Lmin, Lmax, N0)
        self.data_X = data_X
        self.data_Y = data_Y
        self.dim = data_X.shape[1]
        self.M = M # refinement factor
        self.h0 = h0  # initial timestep
        self.s0 = s0 # data batch size
        self.n0 = n0 # number of timesteps at level 0
        self.device=device
        self.data_size = self.data_X.shape[0]
        self.prior = prior
        self.model = model
        self.func = func
        self.init_func = init_func

    def init_weights(self, net, mu=0, std=1, **kwargs):
        """ Init weights with prior

        """
        #net.params.data.copy_(torch.zeros_like(net.params))
        net.params.data.copy_(self.init_func(params=net.params.data))


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
        #drift_langevin = 1/self.data_size * self.prior.logprob(nets.params) + 1/subsample_size * nets.loglik(self.data_X, self.data_Y, U)
        drift_langevin = self.prior.logprob(nets.params) + self.data_size/subsample_size * nets.loglik(self.data_X, self.data_Y, U)
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
        nets : LogisticNets 

        """
        with torch.no_grad():
            F = self.func(nets.params) #torch.norm(nets.params, p=2, dim=1)**2
        return F.cpu().numpy()

    

    def mlmc_fn(self, l, N, **kwargs):
        """SDE to train Bayesian logistic regression
        We fix sampling parameter. We do antithetic approach on Brownian motion

        """
        dim = self.data_X.shape[1]-1
        
        hf = self.h0 #5e-6#0.01/self.data_size#1/self.data_size#self.T/self.n0 # step size in discretisation level
        # we re-scale timestep size: h:=h/(2m)
        hf = hf/(2*self.data_size)
        
        sf = self.s0 * self.M ** l
        sc = int(sf/self.M)
        sigma_f = math.sqrt(2)

        sums_level_l = np.zeros(6) # this will store level l sum  and higher order momentums 

        for N1 in range(0, N, 5000):
            N2 = min(5000, N-N1) # we will do batches of paths of size N2
            
            X_f = self.model(dim, N2, **kwargs).to(device=self.device)
            self.init_weights(X_f)

            X_c1 = copy.deepcopy(X_f) # 1st coarse process for antithetics
            X_c2 = copy.deepcopy(X_f) # 2nd coarse process for antithetics
            
            dW = torch.zeros_like(X_f.params)
            pbar = tqdm.tqdm(total=self.n0)
            for n in range(self.n0):
                dW = math.sqrt(hf) * torch.randn_like(dW)
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
        cost = self.n0 * (self.s0 * self.M ** l)
        return cost
    
    def get_cost_path(self):
        L = len(self.var_Pf_Pc)
        Cl = self.n0 * (self.s0 * self.M ** np.arange(L))
        return Cl
    
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
        CL = self.n0 *  (self.s0 * self.M ** (L-1))
        cost = np.ceil(2/eps**2 * self.var_Pf[min(L-1, len(self.var_Pf)-1)]) * CL
        return cost
    
    def get_cost_MLMC(self, Nl):
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
        

