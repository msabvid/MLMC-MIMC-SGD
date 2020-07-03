import sys
import os
path = os.path.join(os.path.dirname(__file__),'..')
sys.path.append(path)


from abc import ABC, abstractmethod
import math
import torch
import torch.nn as nn
import copy
import argparse
import numpy as np
import math
from sklearn.datasets import make_blobs
from mlmc_mimc import MLMC 




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
        self.activation = nn.Sigmoid()
        #self.nets = nn.ModuleList([LogisticNet(dim) for n in range(N)])


    def forward(self, idx, data_X):
        y = torch.matmul(data_X, self.params[idx, :].view(-1,1))
        y = nn.Sigmoid()(y)
        return y
    
    
    def forward_backward_pass(self, data_X, data_Y, U):
        loss_fn = nn.BCELoss(reduction='none')
        x = data_X[U,:] # x has shape (N, subsample_size, (dim+1))
        target = data_Y[U,:]
        
        y = torch.bmm(x,self.params.unsqueeze(2))
        y = self.activation(y)
        
        loss = loss_fn(y,target).squeeze(2)
        loss = loss.mean(1)

        loss.backward(torch.ones_like(loss))
        return 0 


        

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
        #net.params.data.copy_(mu + std * torch.randn_like(net.params))
        net.params.data.copy_(torch.zeros_like(net.params))

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
        nets.forward_backward_pass(self.data_X, self.data_Y, U)
        
        #nets.params.data.copy_(nets.params.data + h*(1/self.data_size * self._grad_logprior(nets.params.data) + grad_loglik) + \
        #        sigma * dW)
        nets.params.data.copy_(nets.params.data - h/2*(-1/self.data_size * self._grad_logprior(nets.params.data) + nets.params.grad) + \
                sigma * dW)
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
        sigma_c = 1/math.sqrt(sc)

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
                #U = np.array([np.random.choice(self.data_size, sf, replace=False) for i in range(N2)])
                self._euler_step(X_f, U, sigma_f, hf, dW)
                
                if l>0:
                    self._euler_step(X_c1, U[:,:sc], sigma_f, hf, dW)
                    self._euler_step(X_c2, U[:,sc:], sigma_f, hf, dW)

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
        #cost = 2/eps**2 * self.var_Pf[-1] * (self.n0 * self.M**(L)) 
        #cost = 2/eps**2 * self.var_Pf[-1] * (self.n0 * 2**(self.gamma*L)) 
        CL = self.n0 *  (1 + self.s0 * self.M ** L)
        cost = 2/eps**2 * self.var_Pf[min(L, len(self.var_Pf)-1)] * CL
                
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
    data_x = 2*torch.randn(data_size, dim)
    data_x = torch.cat([torch.ones(data_size, 1), data_x], 1)
    #
    params = torch.randn(dim+1, 1)-0.2 
    data_y = torch.matmul(data_x, params)
    data_y = torch.sign(torch.clamp(data_y, 0))

    #data_x, data_y = make_blobs(n_samples = np.array([int(0.2*data_size), data_size - int(0.2*data_size)]), \
    #                 n_features=2, centers=np.array([[-10,-10],[5,30]]), \
    #                 cluster_std = [3,5])
    #data_y = data_y.reshape(-1,1)
    #data_x = torch.tensor(data_x).float()
    #data_x = torch.cat([torch.ones(data_size, 1), data_x], 1)
    #data_y = torch.tensor(data_y).float()
    return data_x, data_y
    #return params, data_x, data_y
    


if __name__ == '__main__':
    
    #CONFIGURATION
    parser = argparse.ArgumentParser()
    parser.add_argument('--M', type=int, default=2, \
            help='refinement value')
    parser.add_argument('--N', type=int, default=10000, \
            help='samples for convergence tests')
    parser.add_argument('--L', type=int, default=6, \
            help='levels for convergence tests')
    parser.add_argument('--s0', type=int, default=2, \
            help='initial value of data batch size')
    parser.add_argument('--N0', type=int, default=2, \
            help='initial number of samples')
    parser.add_argument('--Lmin', type=int, default=0, \
            help='minimum refinement level')
    parser.add_argument('--Lmax', type=int, default=8, \
            help='maximum refinement level')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dim', type=int, default=2,
            help='dimension of data')
    args = parser.parse_args()
    
    if args.device=='cpu' or (not torch.cuda.is_available()):
        device='cpu'
    else:
        device = 'cuda:'+str(args.device)
    

    # Target Logistic regression, and synthetic data
    data_X, data_Y = synthetic(args.dim, data_size = args.s0 * args.M**args.Lmax)
    data_X = data_X.to(device=device)
    data_Y = data_Y.to(device=device)
    
    MLMC_CONFIG = {'Lmin':args.Lmin,
            'Lmax':args.Lmax,
            'N0':args.N0,
            'M':args.M,
            'T':5,
            's0':args.s0,
            'n0':100, # initial number of steps at level 0
            'data_X':data_X,
            'data_Y':data_Y,
            'device':device,
            }
    
    # Bayesian log regressor
    bayesian_logregress = Bayesian_logistic(**MLMC_CONFIG)
    
    # 1. Convergence tests
    bayesian_logregress.estimate_alpha_beta_gamma(args.L, args.N, "convergence_test_s.txt")
    bayesian_logregress.save_convergence_test("logistic_level_s_data.txt")

    # 2. get complexities
    Eps = [0.1, 0.01, 0.001,0.0001]#, 0.0005]
    #bayesian_logregress.get_target("convergence_test_s.txt")
    Nl_list, mlmc_cost, std_cost = bayesian_logregress.get_complexities(Eps, "convergence_test_s.txt")

    # 3. plot
    bayesian_logregress.plot(Eps, Nl_list, mlmc_cost, std_cost, "logistic_level_s.pdf")
    
    # 4. save
    bayesian_logregress.save(Eps, Nl_list, mlmc_cost, std_cost, "logistic_level_s_data.txt")
