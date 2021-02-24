import torch
import torch.nn as nn
import numpy as np
import math
from abc import abstractmethod


class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()

    @abtsractmethod
    def loglik(self, **kwargs):
        ...




class LogisticNets(BaseModel):
    """
    List of logistic regressions
    """
    
    def __init__(self, dim, N, **kwargs):
        """List of N logistic networks. 
        We aim to sample from the  posterior of the parameters of the logistic network
        by solving the Langevin sde process
        We will solve *N* Langevin processes in parallel to estimate momentsi

        Parameters
        ----------
        dim : int
            Dimension of input data
        N : int
            Number of copies of the logistic regression necessary for Monte Carlo
        """
        super(BaseModel,self).__init__()
        self.params = nn.Parameter(torch.zeros(N, dim+1)) # +1 is for the intercept


    def forward(self, data_X, idx=0):
        y = torch.matmul(data_X, self.params[idx, :].view(-1,1))
        y = torch.sigmoid(y) # (N, 1)
        return y
    
    
    def loglik(self, data_X, data_Y, U):
        """
        Likelihood of a subsample of the data 
        using the N logistic regressions. This likelihood will be used in the Langevin process

        Parameters
        ----------
        data_X: torch.Tensor
            torch.Tensor of size (N, dim+1) where the +1 is a column of ones to fit the intercept
        data_Y: torch.Tensor
            torch.Tensor of size (N, 1)
        U: np.array
            np.array of size (N, subsample_size) used for subsampling purposes from the data
        """
        self.zero_grad()
        loss_fn = nn.BCELoss(reduction='none')
        x = data_X[U,:] # x has shape (N, subsample_size, (dim+1))
        target = data_Y[U,:]
        
        y = torch.bmm(x,self.params.unsqueeze(2)) # (N, subsample_size, 1)
        y = torch.sigmoid(y)
        
        loss = -loss_fn(y,target).squeeze(2) #!!! I put a minus in front of loss_fn so that we actually compute the log-likelihood! Important for the signs in the Langevin process
        loss = loss.sum(1) #tensor of size (N)
        return loss 
    
    
    def forward_backward_pass(self, data_X, data_Y, U):
        """
        This method directly perform a forward and a backward pass to calculate the likelihood of a subsample of the data 
        using the N logistic regressions. This likelihood will be used in the Langevin process

        Parameters
        ----------
        data_X: torch.Tensor
            torch.Tensor of size (N, dim+1) where the +1 is a column of ones to fit the intercept
        data_Y: torch.Tensor
            torch.Tensor of size (N, 1)
        U: np.array
            np.array of size (N, subsample_size) used for subsampling purposes from the data
        """
        self.zero_grad()
        loss_fn = nn.BCELoss(reduction='none')
        x = data_X[U,:] # x has shape (N, subsample_size, (dim+1))
        target = data_Y[U,:]
        
        y = torch.bmm(x,self.params.unsqueeze(2)) # (N, subsample_size, 1)
        y = torch.sigmoid(y)
        
        loss = -loss_fn(y,target).squeeze(2) #!!! I put a minus in front of loss_fn so that we actually compute the log-likelihood! Important for the signs in the Langevin process
        loss = loss.sum(1)
        loss.backward(torch.ones_like(loss))
        return 0 


class MixtureGaussianNets(BaseModel):
    """
    Model follows a mixture of Gaussians.
    Model taken from Example 5.1 in https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf
    """

    def __init__(self, dim, N, sigma_x, **kwargs):
        """
        Parameters
        ----------
        dim: int
            dimension of the process / quantity of interest
        N: int
            number of processes
        sigma_x: float
            see Example https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf
        """
        super(BaseModel, self).__init__()
        assert dim==2, "dim needs to be 2, see https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf"
        self.params = nn.Parameter(torch.zeros(N, dim))
        self.sigma_x = sigma_x
        self.mixing = [0.5,0.5]

    def loglik(self, data_X, U, **kwargs):
        """
        log-likelihood:
        data_X_i \sim 1/2 N(X_1, \sigma_x^2) + 1/2 N(X_1 + X_2, \sigma_x^2)
        
        Where data_X is the data, and X_1 and X_2 are the parameters of the model. A bit confusing, I know, but
        I want to keep consistency with the rest of the code and our MASGA paper (https://arxiv.org/pdf/2006.06102.pdf) 

        Note
        ----
        The above expression implies a mixture of Normals, not a sum of Normals (which would yield another normal)

        Parameters
        ----------
        data_X: torch.Tensor
            torch.Tensor of shape (N, dim)
        U: np.array
            np.array of shape (N, subsample_size)
        
        """
        
        self.zero_grad()
        x=data_X[U,:] #(N, subsample_size, dim)
        
        mean1 = self.params[:,0].reshape(-1,1,1) #shape (N,1,1)
        mean2 = self.params[:,0].reshape(-1,1,1) + self.params[:,1].reshape(-1,1,1) #shape (N,1,1)
        
        likelihood = 0.5*torch.exp(-1/(2*self.sigma_x**2)*(x-mean1)**2) + 0.5*torch.exp(-1/(2*self.sigma_x**2)*(x-mean2)**2) # (N, subsample_size, dim)
        # we suppose data_X_i \sim 1/2 N(X_1, \sigma_x^2) + 1/2 N(X_1 + X_2, \sigma_x^2) are independent for each covariate
        # Therefore, the join prob is the product of the prob of the marginals
        likelihood = likelihood.prod(2) # (N, subsample_size)
        logL = torch.log(likelihood + 1e-8) # (N, subsample_size)
        # we sum across subsample 
        logL = logL.sum(1) #(N) 
        return logL



# TODO: add other models that can be run forward in the Langevin Process by batches
