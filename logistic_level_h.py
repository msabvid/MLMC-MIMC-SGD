import sys
import os
path = os.path.abspath(__file__)
sys.path.append(path)


from abc import ABC, abstractmethod
import math
import torch
import torch.nn as nn
import copy
import argparse
import numpy as np
import math
from utils import MLMC 




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
        self.params = nn.Parameter(torch.zeros(N, dim+1))

    def forward(self, data_X):
        y = torch.matmul(data_X, self.params.T)
        y = nn.Sigmoid()(y)
        return y
    
    def forward2(self, data_X, data_Y):
        z = torch.matmul(data_X, self.params.T) * data_Y
        output = nn.Sigmoid()(z)
        return output

    def get_loglikelihood(self, data_X, data_Y):
        """Get the loglikelihood of the data
        y\in {0,1}
        loglik = \sum y * log(f(x)) + (1-y) * log(1-f(x))
        
        Parameters
        ----------
        data_X : np.ndarray of size (batch_size, dim+1)
            covariates
        data_Y : np.ndarray of size (batch_size, 1)
            binary data
        """
        pred = self.forward(data_X)
        # we add 1e-5 to avoid numerical problems
        loglik = data_Y * torch.log(pred) + (1-data_Y) * torch.log(1 - pred)
        loglik = loglik.mean(0, keepdim=True)
        return loglik

    def get_gradloglikelihood_explicitely(self, data_X, data_Y):
        """Get gradient of the loglikelihood of the data explicitely
        without using automatic differentiation
        If y\in{-1,1}
        f(x) = \sigmoid(z) where z = matmul(x, params.T) * y
        f'(z) = 1/sigmoid(z) * sigmoid(z) * (1-sigmoid(z))

        """
        data_Y = 2*data_Y - 1
        pred = self.forward2(data_X, data_Y) 
        pred = pred.unsqueeze(2)
        data_X = data_X.unsqueeze(1)
        data_Y = data_Y.unsqueeze(1)
        grad_loglik = (1-pred) * data_X * data_Y
        grad_loglik = grad_loglik.mean(0)
        return grad_loglik



        

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
    def init_weights(net, mu=0, std=1):
        """ Init weights with prior

        """
        net.params.data.copy_(mu + std * torch.randn_like(net.params))

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
        with torch.no_grad():
            pred = nets(data_X[U,:])

        nets.zero_grad()
        #loglik = nets.get_loglikelihood(self.data_X[U,:], self.data_Y[U,:])
        #loglik.backward(torch.ones_like(loglik))
        
        grad_loglik = nets.get_gradloglikelihood_explicitely(self.data_X[U,:], self.data_Y[U, :])
        nets.params.data.copy_(nets.params.data + h*(1/self.data_size * self._grad_logprior(nets.params.data) + grad_loglik) + \
                sigma * dW)
        #nets.params.data.copy_(nets.params.data + h*(1/self.data_size * self._grad_logprior(nets.params.data) + nets.params.grad) + \
        #        sigma * dW)
        if torch.isnan(nets.params.mean()):
            raise ValueError
        return 0

    
    
    def Func(self, nets):
        """Function of X. 
        Recall we want to approximate E(F(X)) where X is a random vector
        
        Parameters
        ----------
        nets : LogisticNets 

        """
        #with torch.no_grad():
        #    F = torch.norm(nets.params, p=1, dim=1)
        with torch.no_grad():
            F = nets(self.data_X[-1,:].unsqueeze(0))
        #F = nets
        return F.cpu().numpy()

    

    def mlmc_fn(self, l, N):
        """SDE to train Bayesian logistic regression
        We fix sampling parameter. We do antithetic approach on Brownian motion

        """
        dim = self.data_X.shape[1]-1
        sigma = 1/math.sqrt(self.s0)
        
        nf = self.n0 * self.M ** l # n steps in fine time discretisation
        hf = self.T/nf # step size in coarse time discretisation
        
        nc = nf/self.M
        hc = self.T/nc

        sums_level_l = np.zeros(6) # this will store level l sum  and higher order momentums 

        for N1 in range(0, N, 1000):
            N2 = min(1000, N-N1) # we will do batches of paths of size N2
            
            X_f = LogisticNets(dim, N2).to(device=self.device)
            self.init_weights(X_f)

            X_c1 = copy.deepcopy(X_f) # 1st coarse process for antithetics
            X_c2 = copy.deepcopy(X_f) # 2nd coarse process for antithetics
            
            dWf = torch.zeros_like(X_f.params)
            dWc = torch.zeros_like(X_f.params)

            if l==0:
                for n in range(int(nf)):
                    dWf = math.sqrt(hf) * torch.randn_like(dWf)
                    U = np.random.choice(self.data_size, self.s0)
                    self._euler_step(X_f, U, sigma, hf, dWf)
            else:
                for n in range(int(nc)):
                    dWc = dWc * 0
                    U_list = []
                    for m in range(self.M):
                        U = np.random.choice(self.data_size, self.s0)
                        U_list.append(U)
                        dWf = math.sqrt(hf) * torch.randn_like(dWf)
                        dWc += dWf
                        self._euler_step(X_f, U, sigma, hf, dWf)

                    self._euler_step(X_c1, U_list[0], sigma, hc, dWc)
                    self._euler_step(X_c2, U_list[1], sigma, hc, dWc)

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
        #cost = 2/eps**2 * self.var_Pf[-1] * (self.n0 * self.M**(L)) 
        cost = 2/eps**2 * self.var_Pf[-1] * (self.n0 * 2**(self.gamma*L)) 
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
        cost = sum(Nl * self.n0 * (2**self.gamma) ** np.arange(len(Nl)))
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
    data_x = torch.randn(data_size, dim)
    data_x = torch.cat([torch.ones(data_size, 1), data_x], 1)
    
    params = torch.randn(dim+1, 1) - 0.4
    data_y = torch.matmul(data_x, params)
    data_y = torch.sign(torch.clamp(data_y, 0))

    
    return params, data_x, data_y


if __name__ == '__main__':
    
    #CONFIGURATION
    parser = argparse.ArgumentParser()
    parser.add_argument('--M', type=int, default=2, \
            help='refinement value')
    parser.add_argument('--N', type=int, default=5000, \
            help='samples for convergence tests')
    parser.add_argument('--L', type=int, default=4, \
            help='levels for convergence tests')
    parser.add_argument('--s0', type=int, default=1000, \
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
    params, data_X, data_Y = synthetic(args.dim, data_size = 5000)
    data_X = data_X.to(device=device)
    data_Y = data_Y.to(device=device)
    
    MLMC_CONFIG = {'Lmin':args.Lmin,
            'Lmax':args.Lmax,
            'N0':args.N0,
            'M':args.M,
            'T':2,
            's0':args.s0,
            'n0':2, # initial number of steps at level 0
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
    Nl_list, mlmc_cost, std_cost = bayesian_logregress.get_complexities(Eps, "convergence_test.txt")

    # 3. plot
    
