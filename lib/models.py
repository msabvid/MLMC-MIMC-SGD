import torch
import torch.nn as nn
import numpy as np


class LogisticNets(nn.Module):
    """
    List of logistic regressions
    """
    
    def __init__(self, dim, N):
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
        super().__init__()
        self.params = nn.Parameter(torch.zeros(N, dim+1)) # +1 is for the intercept


    def forward(self, data_X, idx=0):
        y = torch.matmul(data_X, self.params[idx, :].view(-1,1))
        y = torch.sigmoid(y)
        return y
    
    
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


# TODO: add other models that can be run forward in the Langevin Process by batches
