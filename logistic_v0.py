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
from utils import MLMC 


class LogisticNet(nn.Module):
    """
    Logistic regression
    """

    def __init__(self, dim):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

    def forward(self, data):
        return self.layer(data)


class LogisticNets(nn.Module):
    """
    List of logistic regressions
    """
    
    def __init__(self, dim, N):
        """List of logistic networks. 
        We aim to sample from the  posterior of the parameters of the logistic network
        by solving the Langevin sde process

        Parameters
        ----------
        dim : int
            Dimension of input data
        N : int
            Number of copies of the logistic regression necessary for Monte Carlo

        """
        
        super().__init__()
        
        self.N = N
        self.nets = nn.ModuleList([LogisticNet(dim) for i in range(N)])
        self.loss_fn = nn.BCELoss(reduction = 'mean')

    def get_loss(self, data_X, data_Y):
        batch_size = data_X.shape[0]
        loss = torch.zeros(self.N, 1, device=data_X.device)
        for idx, net in enumerate(self.nets):
            loss[idx] = self.loss_fn(net(data_X), data_Y.float())
        return -loss # we return minus the loss because we want to maximise the log-likelihood 


class Brownian():
    
    """
    Brownian Motion class. 
    This class 
    """
    
    def __init__(self, net):
        """Init Brownian motion with the same dimension as the number of parameters in the network
        
        Parameters
        ----------
        net : torch.nn.Module
            Neural Network. 
        """
        self.dW = copy.deepcopy([p for p in net.parameters()])
        for p in self.dW:
            p.requires_grad_(False)
    
    def reset_dW(self):
        """
        Resets dW to zero
        """
        for p in self.dW:
            p.data.zero_()

    
    def update_dW(self, h):
        """Updates dW

        Parameters
        h : float
            time step size

        """
        for p in self.dW:
            p.copy_(torch.randn_like(p) * math.sqrt(h))
    
    def sum_brownian(self, dW2):
        """Sum of brownian motions

        Parameters
        ----------
        dW2 : list(Tensor)
            Brownian Motion 2
        """
        
        for p1,p2 in zip(self.dW, dW2.dW):
            p1.copy_(p1.data + p2.data)
        

class Bayesian_logistic(MLMC):

    def __init__(self, Lmin, Lmax, N0, M, T, s0, n0, data_X, data_Y, device):
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

    @staticmethod
    def init_weights(net):
        if type(net)==nn.Linear:
            nn.init.normal_(net.weight.data)
            nn.init.normal_(net.bias.data)
    
    def _grad_logprior(self, x):
        """
        Prior is d-dimensional N(0,1)
        f(x) = 1/sqrt(2pi) * exp(-x^2/2)
        log f(x) = Const - x^2/2
        d/dx log(f(x)) = -x
        """
        return -x



    def _euler_step(self, nets, U, sigma, h, dW):
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
        loss = nets.get_loss(self.data_X[U,:], self.data_Y[U,:])
        loss.backward(torch.ones_like(loss))
        
        for p,dB in zip(nets.parameters(), dW.dW):
            p.data.copy_(p.data + h*(1/self.data_size * self._grad_logprior(p.data) + p.grad) + \
                    sigma * dB)

        return 0

    
    
    def Func(self, nets):
        """Function of X. 
        Recall we want to approximate E(F(X)) where X is a random vector
        
        Parameters
        ----------
        nets : LogisticNets 

        """
        N2 = len(nets.nets)
        F = np.zeros(N2)
        for idx, net in enumerate(nets.nets):
            X = torch.cat([p.flatten() for p in net.parameters()])
            F[idx] = torch.norm(X, 1).item()
        return F

    

    def mlmc_fn(self, l, N):
        """SDE to train Bayesian logistic regression
        We fix sampling parameter. We do antithetic approach on Brownian motion

        """
        dim = self.data_X.shape[1]
        sigma = 1/math.sqrt(self.s0)
        
        nf = self.n0 * self.M ** l # n steps in fine time discretisation
        hf = self.T/nf # step size in coarse time discretisation
        
        nc = nf/self.M
        hc = self.T/nc

        sums_level_l = np.zeros(6) # this will store level l sum  and higher order momentums 

        for N1 in range(0, N, 1000):
            N2 = min(1000, N-N1) # we will do batches of paths of size N2
            
            logistics_f = LogisticNets(dim, N2).to(self.device)
            logistics_f.apply(self.init_weights) # we initialise logistics weights with prior
            
            logistics_c1 = copy.deepcopy(logistics_f) # 1st coarse process for antithetics
            logistics_c2 = copy.deepcopy(logistics_f) # 2nd coarse process for antithetics
            
            dWf = Brownian(logistics_f)
            dWc = Brownian(logistics_c1)

            if l==0:
                dWf.update_dW(hf)
                U = np.random.choice(self.data_size, self.s0)
                self._euler_step(logistics_f, U, sigma, hf, dWf)
            else:
                for n in range(int(nc)):
                    dWc.reset_dW()
                    U_list = []
                    for m in range(self.M):
                        U = np.random.choice(self.data_size, self.s0)
                        U_list.append(U)
                        dWc.update_dW(hf)
                        dWc.sum_brownian(dWf)
                        self._euler_step(logistics_f, U, sigma, hf, dWf)

                    self._euler_step(logistics_c1, U_list[0], sigma, hc, dWc)
                    self._euler_step(logistics_c2, U_list[1], sigma, hc, dWc)

            F_fine = self.Func(logistics_f)
            F_coarse_antithetic = 0.5 * (self.Func(logistics_c1)+self.Func(logistics_c2)) if l>0 else 0
            
            # sums level l
            sums_level_l[0] += np.sum(F_fine - F_coarse_antithetic)      
            sums_level_l[1] += np.sum((F_fine - F_coarse_antithetic)**2)  
            sums_level_l[2] += np.sum((F_fine - F_coarse_antithetic)**3)  
            sums_level_l[3] += np.sum((F_fine - F_coarse_antithetic)**4)  
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
        
        L = len(Nl)
        cost = 2/eps**2 * self.var_Pf[-1] * (self.n0 * self.M**(L-1)) 
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
        cost = sum(Nl * self.n0 * self.M ** np.arange(len(Nl)))
        return cost

    def get_weak_error(self, ml):
        """Get weak error of MLMC approximation
        See http://people.maths.ox.ac.uk/~gilesm/files/acta15.pdf p. 21
        """
        weak_error = ml[-1]/(2**self.alpha-1)
        return weak_error
        


def synthetic(dim : int, data_size : int):
    """Creates synthetic dataset for logistic regression

    Parameters
    ----------
    dim : int
        Dimension of the dataset
    data_size : int
        Number of points

    Returns
    -------
    model : LogisticNet

    data_x : torch.Tensor of size (data_size, dim)
    
    data_y : torch.Tensor of size (data_size, 1)
    """
    model = LogisticNet(dim)
    def init_prior(m):
        if type(m)==nn.Linear:
            m.weight.data.normal_(mean=0.4,std=1)
            m.bias.data.normal_(mean=0.4, std=1)
    model.apply(init_prior)

    data_x = torch.randn(data_size, dim)
    data_y = model(data_x)
    data_y = (data_y>=0.5).int()
    
    return model, data_x, data_y


if __name__ == '__main__':
    
    #CONFIGURATION
    parser = argparse.ArgumentParser()
    parser.add_argument('--M', type=int, default=2, \
            help='refinement value')
    parser.add_argument('--N', type=int, default=5000, \
            help='samples for convergence tests')
    parser.add_argument('--L', type=int, default=5, \
            help='levels for convergence tests')
    parser.add_argument('--s0', type=int, default=500, \
            help='initial value of data batch size')
    parser.add_argument('--N0', type=int, default=10, \
            help='initial number of samples')
    parser.add_argument('--Lmin', type=int, default=0, \
            help='minimum refinement level')
    parser.add_argument('--Lmax', type=int, default=15, \
            help='maximum refinement level')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dim', type=int, default=2,
            help='dimension of data')
    args = parser.parse_args()
    
    if args.device=='cpu' or (not torch.cuda.is_available()):
        device='cpu'
    else:
        device = 'cuda:{}'+str(args.device)
    

    # Target Logistic regression, and synthetic data
    model, data_X, data_Y = synthetic(args.dim, data_size = 5000)
    data_X = data_X.to(device)
    data_Y = data_Y.to(device)
    
    MLMC_CONFIG = {'Lmin':args.Lmin,
            'Lmax':args.Lmax,
            'N0':args.N0,
            'M':args.M,
            'T':10,
            's0':args.s0,
            'n0':10, # initial number of steps at level 0
            'data_X':data_X,
            'data_Y':data_Y,
            'device':device,
            }
    
    # Bayesian log regressor
    bayesian_logregress = Bayesian_logistic(**MLMC_CONFIG)
    
    # 1. Convergence tests
    bayesian_logregress.estimate_alpha_beta_gamma(args.L, args.N, "convergence_test.txt")

    # 2. get complexities
    Eps = [0.1,0.01, 0.005, 0.001, 0.0005]
    Nl_list, mlmc_cost, std_cost = bayesian_logregress.get_complexities(Eps)

    # 3. plot
    
