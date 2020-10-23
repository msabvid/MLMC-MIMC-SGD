import sys
import os
from abc import ABC, abstractmethod
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import copy
import argparse
import numpy as np
import math
from sklearn.datasets import make_blobs
import tqdm
import matplotlib.pyplot as plt

from mlmc_mimc import MLMC 
from lib.data import get_dataset, Dataset_MCMC




class LogisticNets(nn.Module):
    """
    List of logistic regressions
    """
    
    def __init__(self, dim, N):
        """List of logistic networks. 
        We aim to sample from the  posterior of the parameters of the logistic network
        by solving the Langevin sde process
        We will solve N Langevin processes in parallel to estimate moments

        Parameters
        ----------
        dim : int
            Dimension of input data
        N : int
            Number of copies of the logistic regression necessary for Monte Carlo

        """
        
        super().__init__()
        self.params = nn.Parameter(torch.zeros(N, dim+1)) # +1 is for the intercept


    def forward(self, data_X, idx=0):
        y = torch.matmul(data_X, self.params[idx, :].view(-1,1))
        y = torch.sigmoid(y)
        return y
    
    
    def forward_backward_pass(self, data_X, data_Y, U):
        loss_fn = nn.BCELoss(reduction='none')
        x = data_X[U,:] # x has shape (N, subsample_size, (dim+1))
        target = data_Y[U,:]
        
        y = torch.bmm(x,self.params.unsqueeze(2)) # (N, subsample_size, 1)
        y = torch.sigmoid(y)
        
        loss = -loss_fn(y,target).squeeze(2) #!!! I put a minus in front of loss_fn so that we actually compute the log-likelihood! Important for the signs in the Langevin process

        loss = loss.mean(1)

        loss.backward(torch.ones_like(loss))
        return 0 



class SGLD():

    def __init__(self, T, n_steps, data_X, data_Y, device):
        self.data_X = data_X
        self.data_Y = data_Y
        self.dim = data_X.shape[1]-1 # -1 is to subtract column of 1s in the data to account for the intercept
        self.T = T
        self.n_steps = n_steps
        self.device = device
        self.data_size = data_X.shape[0]
        
        # we estimate the MAP, and the log-likelihood of the dataset using the MAP
        self.MAP = self.estimate_MAP(epochs=200,batch_size=100) # object of class LogisticNets
        self.grad_loglik_MAP = self.get_grad_loglik_MAP() # tensor of size (1, self.dim+1)

    def estimate_MAP(self, epochs, batch_size):
        """
        We estimate the MAP using usual SGD with RMSprop
        """
        print("estimating MAP for control variate")
        X_f = LogisticNets(dim=self.dim, N=1).to(device=self.device)
        self.init_weights(X_f)
        optimizer = torch.optim.RMSprop(X_f.parameters(), lr=0.001)
        dataset = Dataset_MCMC(self.data_X.to("cpu"), self.data_Y.to("cpu"))
        dataloader = DataLoader(dataset, batch_size=batch_size,
                shuffle=True, num_workers=1, pin_memory=False)
        loss_fn = nn.BCELoss(reduction='mean')
        pbar = tqdm.tqdm(total=epochs)
        for epoch in range(epochs):
            for idx, (x,y) in enumerate(dataloader):
                optimizer.zero_grad()
                pred = X_f(x.to(self.device))
                loss = loss_fn(pred, y.to(self.device))
                loss.backward()
                optimizer.step()
            pbar.update(1)
        return X_f

    def get_grad_loglik_MAP(self):
        self.MAP.zero_grad()
        loss_fn = nn.BCELoss(reduction='sum')
        pred = self.MAP(self.data_X.to(self.device))
        loss = -loss_fn(pred, self.data_Y.to(self.device)) #!!! I put a minus in front of loss_fn so that we actually compute the log-likelihood! Important for the signs in the Langevin process
        loss.backward()
        
        grad_loglik_MAP = copy.deepcopy(self.MAP.params.grad)
        return grad_loglik_MAP

    @staticmethod
    def init_weights(net, mu=0, std=1):
        """ Init weights with prior

        """
        #net.params.data.copy_(mu + std * torch.randn_like(net.params))
        net.params.data.copy_(torch.zeros_like(net.params))
    
    def grad_logprior(self,x):
        """
        Prior is d-dimensional N(0,1)
        f(x) = 1/sqrt(2pi) * exp(-x^2/2)
        log f(x) = Const - x^2/2
        d/dx log(f(x)) = -x

        TO BE CHANGED if the prior is changed
        """
        return -x

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
    
    def solve(self, N, sf):
        """Solve N stochastic langevin processes with subsample size sf
        It returns E(F(X)) with X sampled from the posterior

        Parameters
        ----------
        N: int
            Number of Langevin Processes to solve
        sf: int
            Subsample size
        """
        hf = self.T/self.n_steps
        sigma = 1/math.sqrt(self.data_size)  
        sum1, sum2 = np.zeros(self.n_steps), np.zeros(self.n_steps) #first and second order moments
        
        pbar = tqdm.tqdm(total=N)
        for N1 in range(0,N,5000):
            N2 = min(5000, N-N1) # we will do batches of paths of size N2
            X = LogisticNets(self.dim, N2).to(device=self.device)
            self.init_weights(X)

            for step in range(self.n_steps):
                dW = math.sqrt(hf) * torch.randn_like(X.params)
                U = np.random.choice(self.data_size, (N2, sf), replace=True)
                X.zero_grad()
                X.forward_backward_pass(self.data_X, self.data_Y, U)
                
                params_updated = X.params.data + hf/2 * (1/self.data_size*self.grad_logprior(X.params.data) +
                        X.params.grad) + sigma*dW
                X.params.data.copy_(params_updated)
                sum1[step] += np.sum(self.Func(X))
                sum2[step] += np.sum(self.Func(X)**2)
            pbar.update(N2)
        
        return sum1/N, sum2/N
    
    
    def solve_with_cv(self, N, sf):
        """Solve N stochastic langevin processes with subsample size sf
        It returns E(F(X)) with X sampled from the posterior

        Parameters
        ----------
        N: int
            Number of Langevin Processes to solve
        sf: int
            Subsample size
        """
        hf = self.T/self.n_steps
        sigma = 1/math.sqrt(self.data_size)  
        sum1, sum2 = np.zeros(self.n_steps), np.zeros(self.n_steps) #first and second order moments
        
        pbar = tqdm.tqdm(total=N)
        for N1 in range(0,N,5000):
            N2 = min(5000, N-N1) # we will do batches of paths of size N2
            X = LogisticNets(self.dim, N2).to(device=self.device)
            # we extend MAP and grad_loklik_MAP to run several processes forward at the same time
            grad_loglik_MAP = self.grad_loglik_MAP.repeat((N2,1))
            MAP = copy.deepcopy(self.MAP)
            MAP.params.data = self.MAP.params.data.repeat((N2,1))

            self.init_weights(X)

            for step in range(self.n_steps):
                dW = math.sqrt(hf) * torch.randn_like(X.params)
                U = np.random.choice(self.data_size, (N2, sf), replace=True)
                X.zero_grad()
                MAP.zero_grad()
                X.forward_backward_pass(self.data_X, self.data_Y, U)
                MAP.forward_backward_pass(self.data_X, self.data_Y, U)
                params_updated = X.params.data + hf/2 * (1/self.data_size*self.grad_logprior(X.params.data) +
                        1/self.data_size*grad_loglik_MAP + (X.params.grad - MAP.params.grad)) + sigma*dW
                X.params.data.copy_(params_updated)

                sum1[step] += np.sum(self.Func(X))
                sum2[step] += np.sum(self.Func(X)**2)
            pbar.update(5000)

        return sum1/N, sum2/N


def make_plots(path,sgld,*args):
    """
    Make plots
    
    Parameters
    ----------
    path: str
        path where plot and data should be saved
    args: List(Dict)
        each element of args is a dictionary with the following (key,value) pairs
        - "moment1":E(F(X))
        - "moment2":E(F(X)**2)
        - "label":str
    """
    if not os.path.exists(path):
        os.path.makedirs(path)
    
    fig, ax = plt.subplots()
    for d in args:
        var = d["moment2"] - d["moment1"]**2
        ax.plot(d["moment1"], label=d["label"])
        ax.fill_between(
        



    



if __name__ == '__main__':
    
    #CONFIGURATION
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=20000, help='number samples from posterior')
    parser.add_argument('--device', default='cpu', help='device')
    parser.add_argument('--dim', type=int, default=2, help='dimension of data')
    parser.add_argument('--data_size', type=int, default=512, help="dataset size")
    parser.add_argument('--subsample_size', type=int, default=32, help="subsample size")
    parser.add_argument('--T', type=int, default=5, help='horizon time')
    parser.add_argument('--n_steps', type=int, default=500, help='inumber of steps in time discretiation')
    args = parser.parse_args()
    
    if args.device=='cpu' or (not torch.cuda.is_available()):
        device='cpu'
    else:
        device = 'cuda:'+str(args.device)
    

    # Target Logistic regression, and synthetic data
    data_X, data_Y = get_dataset(m=args.data_size, d=args.dim,
            type_regression="logistic", data_dir="./data/")

    data_X = data_X.to(device=device)
    data_Y = data_Y.to(device=device)
    
    # path numerical results
    path_results = "./numerical_results/sgld_cv/logistic_d{}_m{}".format(args.dim, args.data_size)
    if not os.path.exists(path_results):
        os.makedirs(path_results)
    
    # SGLD object
    sgld = SGLD(T=args.T, 
            n_steps=args.n_steps,
            data_X=data_X,
            data_Y=data_Y,
            device=device)
    
    # 1. We calculate E(F(X)) and E(F(X)**2) for stochastic Langevin process
    sgld_1, sgld_2 = sgld.solve(N=args.N, sf=args.subsample_size)

    # 2. We calculate E(X) and E(F(X)**2) for stochastic Langevin process with control variate
    sgld_cv_1, sgld_cv_2 = sgld.solve_with_cv(N=args.N, sf=args.subsample_size)

    # Plots


