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
import pickle

from lib.data import get_dataset, Dataset_MCMC
from lib.priors import Gaussian, MixtureGaussians
from lib.models import LogisticNets
from lib.hyperparameters import config_priors


class SGLD():

    def __init__(self, T, n_steps, data_X, data_Y, prior, device, MAP = None):
        self.data_X = data_X
        self.data_Y = data_Y
        self.dim = data_X.shape[1]-1 # -1 is to subtract column of 1s in the data to account for the intercept
        self.T = T
        self.n_steps = n_steps
        self.device = device
        self.data_size = data_X.shape[0]
        self.prior = prior
        
        # we estimate the MAP, and the log-likelihood of the dataset using the MAP
        if MAP:
            X = LogisticNets(self.dim, N=1)#.to(device=self.device)
            X.load_state_dict(MAP)
            X.to(device=self.device)
            self.MAP = X
        else:
            self.MAP = self.estimate_MAP(epochs=args.n_steps,batch_size=self.data_size) # object of class LogisticNets
        self.grad_loglik_MAP = self.get_grad_loglik_MAP() # tensor of size (1, self.dim+1)

    def estimate_MAP(self, epochs, batch_size):
        """
        We estimate the MAP using usual SGD with RMSprop
        """
        hf = 1e-5#min(self.T/self.n_steps, 0.0001)
        epochs = max(epochs, 100000)
        print("estimating MAP for control variate")
        pbar = tqdm.tqdm(total=epochs)
        X = LogisticNets(self.dim, N=1).to(device=self.device)
        self.init_weights(X)
        
        sf=self.data_size
        for step in range(epochs):
            #U = np.random.choice(self.data_size, (1, sf), replace=True)
            U = np.arange(self.data_size).reshape(1,batch_size)
            X.zero_grad()
            drift = self.prior.logprob(X.params) + self.data_size/batch_size * X.loglik(self.data_X, self.data_Y, U)
            drift.backward(torch.ones_like(drift))
            X.params.data.copy_(X.params.data + hf*(X.params.grad))
            pbar.update(1)
            if step % 100 == 0:
                pbar.write("norm grad log prob={}".format(torch.norm(X.params.grad, p="fro").item()))
        return X

    def save_MAP(self, filename):
        torch.save(self.MAP.state_dict(), filename)


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
        hf = 0.1/self.data_size#1e-8#1/self.data_size#self.T/self.n_steps
        #sigma = 1/math.sqrt(self.data_size)  
        sigma = math.sqrt(2)
        sum1, sum2 = np.zeros(self.n_steps), np.zeros(self.n_steps) #first and second order moments
        print("Solving SGLD...")
        n_steps = min(100000, self.n_steps)
        for N1 in range(0,N,5000):
            N2 = min(5000, N-N1) # we will do batches of paths of size N2
            X = LogisticNets(self.dim, N2).to(device=self.device)
            self.init_weights(X)
            # Euler scheme on Stochastic Langevin process
            pbar = tqdm.tqdm(total=n_steps)
            for step in range(n_steps):
                dW = math.sqrt(hf) * torch.randn_like(X.params)
                U = np.random.choice(self.data_size, (N2, sf), replace=True)
                X.zero_grad()
                drift_langevin = self.prior.logprob(X.params) + self.data_size/sf * X.loglik(self.data_X, self.data_Y,U)
                drift_langevin.backward(torch.ones_like(drift_langevin))
                X.params.data.copy_(X.params.data + hf*(X.params.grad) + sigma*dW)
                sum1[step] += np.sum(self.Func(X)) # first moment
                sum2[step] += np.sum(self.Func(X)**2) # second moment
                if step%100==0:
                    pbar.update(100)
        
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
        hf = 0.1/self.data_size#1e-8#1/self.data_size#self.T/self.n_steps
        #sigma = 1/math.sqrt(self.data_size)  
        sigma = math.sqrt(2)
        sum1, sum2 = np.zeros(self.n_steps), np.zeros(self.n_steps) #first and second order moments
        print("Solving SGLD with CV...")
        n_steps = min(self.n_steps, 100000)
        for N1 in range(0,N,5000):
            N2 = min(5000, N-N1) # we will do batches of paths of size N2
            X = LogisticNets(self.dim, N2).to(device=self.device)
            self.init_weights(X)
            # we extend MAP and grad_loklik_MAP to run several processes forward at the same time
            grad_loglik_MAP = self.grad_loglik_MAP.repeat((N2,1))
            MAP = copy.deepcopy(self.MAP)
            MAP.params.data = self.MAP.params.data.repeat((N2,1))
            # Euler scheme on Stochastic Langevin process with cv
            pbar = tqdm.tqdm(total=n_steps)
            for step in range(n_steps):
                dW = math.sqrt(hf) * torch.randn_like(X.params)
                U = np.random.choice(self.data_size, (N2, sf), replace=True)
                X.zero_grad()
                MAP.zero_grad()
                X.forward_backward_pass(self.data_X, self.data_Y, U)
                MAP.forward_backward_pass(self.data_X, self.data_Y, U)
                params_updated = X.params.data + hf * (self.prior.grad_logprob(X.params.data) +
                        grad_loglik_MAP + self.data_size/sf*(X.params.grad - MAP.params.grad)) + sigma*dW
                X.params.data.copy_(params_updated)

                sum1[step] += np.sum(self.Func(X)) # first order moment
                sum2[step] += np.sum(self.Func(X)**2) # second order moment
                if step%100==0:
                    pbar.update(100)

        return sum1/N, sum2/N


def make_plots(path_results,results):
    """
    Make plots and write results
    
    Parameters
    ----------
    path: str
        path where plot and data should be saved
    results: List(Dict)
        each element of args is a dictionary with the following (key,value) pairs
        - "moment1":E(F(X))
        - "moment2":E(F(X)**2)
        - "label":str
    """
    if not os.path.exists(path_results):
        os.path.makedirs(path_results)
    
    fig, ax = plt.subplots()
    for d in results:
        n_steps = range(len(d["moment1"]))
        var = d["moment2"] - d["moment1"]**2
        ax.plot(n_steps, d["moment1"], label=d["label"])
        ax.fill_between(n_steps, d["moment1"]-np.sqrt(var), d["moment1"]+np.sqrt(var), alpha=0.5)
    ax.legend()
    fig.savefig(os.path.join(path_results,"sgld_cv.pdf"))
    

    return 0



def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)



if __name__ == '__main__':
    
    #CONFIGURATION
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=10000, help='number samples from posterior to calculate E(F(X))')
    parser.add_argument('--device', default='cpu', help='device')
    parser.add_argument('--dim',  type=int, default=2, help='dimension of data if data is synthetic')
    parser.add_argument('--data_size', type=int, default=512, help="dataset size is data is synthetic")
    parser.add_argument('--subsample_size', type=int, default=32, help="subsample size")
    parser.add_argument('--T', type=int, default=10, help='horizon time')
    parser.add_argument('--n_steps', type=int, default=10000, help='number of steps in time discretisation')
    parser.add_argument('--seed', type=int, default=1, help='seed')
    parser.add_argument('--type_data', type=str, default="synthetic")
    parser.add_argument('--prior', type=str, default="Gaussian", help="type of prior")
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
    path_results = "./numerical_results/sgld_cv/logistic/{}_d{}_m{}_s{}".format(args.type_data, dim, data_size, args.subsample_size)
    if not os.path.exists(path_results):
        os.makedirs(path_results)
    
    # load the mode
    path_mode = "./numerical_results/sgld_cv/logistic/mode_{}_d{}_m{}".format(args.type_data, dim, data_size)
    if not os.path.exists(path_mode):
        os.makedirs(path_mode)
    try:
        MAP = torch.load(os.path.join(path_mode, "MAP.pth.tar"), map_location="cpu") # state_dict
    except:
        MAP = None
    
    # prior configuration
    PRIORS = {"Gaussian":Gaussian, "MixtureGaussians":MixtureGaussians}
    CONFIG_PRIORS = config_priors(dim, device)
    prior = PRIORS[args.prior](**CONFIG_PRIORS[args.prior])
    
    # SGLD object
    set_seed(args.seed)
    sgld = SGLD(T=args.T, 
            n_steps=args.n_steps,
            data_X=data_X,
            data_Y=data_Y,
            device=device,
            prior=prior,
            MAP=MAP)
    
    # 1. We calculate E(F(X)) and E(F(X)**2) for stochastic Langevin process
    sgld_1, sgld_2 = sgld.solve(N=args.N, sf=args.subsample_size)

    # 2. We calculate E(F(X)) and E(F(X)**2) for stochastic Langevin process with control variate
    sgld_cv_1, sgld_cv_2 = sgld.solve_with_cv(N=args.N, sf=args.subsample_size)

    # Plots and results
    results = [dict(moment1=sgld_1, moment2=sgld_2, label="$E(F(X))$ - sgld"),
         dict(moment1=sgld_cv_1, moment2=sgld_cv_2, label="$E(F(X))$ - sgld_cv")]
    make_plots(path_results, results)
    with open(os.path.join(path_results, "sgld_results.pickle"), "wb") as f:
        pickle.dump(results, f)
    
    sgld.save_MAP(os.path.join(path_mode, "MAP.pth.tar"))

