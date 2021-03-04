import torch
from typing import List
from abc import abstractmethod

class BasePrior():
    

    @abstractmethod
    def grad_logprob(self, x):
        ...


class Gaussian(BasePrior):
    """
    Gaussian Prior with diagonal covariance matrix
    """
    def __init__(self, mu: torch.Tensor, diagSigma: torch.Tensor, **kwargs):
        """
        Parameters
        ----------
        mu: torch.Tensor
            tensor of size (dim)
        Sigma: torch.Tensor
            tensor of size (dim)
        """
        assert diagSigma.dim()==1 , "diagSigma needs to have dimension 1"

        self.mu = mu.view(1,-1)
        self.diagSigma = diagSigma.view(1,-1)

    def logprob(self,x):
        exponent = -0.5 * (x-self.mu)**2/self.diagSigma #(N, dim)
        exponent = exponent.sum(1) # we don't take into account the constant in fron of the exp, since after taking logs, and differentiating, it disappears. 
        return exponent # we return log(exp(exponent)) = exponent 

    def grad_logprob(self, x):
        return -(x-self.mu)/self.diagSigma



class MixtureGaussians(BasePrior):
    """
    Mixture of Gaussians
    """

    def __init__(self, mu: List[torch.Tensor], diagSigma: List[torch.Tensor], mixing: List[float], **kwargs):
        """
        Parameters
        ----------
        mu: list[torch.Tensor]
            list of means of Gaussians of size (dim)
        Sigma: list[torch.Tensor]
            list of covariance matrices of Gaussians of size (dim) <-- I am assuming diagonal covariance matrices
        mixing: list[float]
            mixing coefficients of Gaussian mixture
        """
        assert sum(mixing)==1, "mixing coefficients must sum 1"
        assert len(mu)==len(diagSigma)
        assert len(mu)==len(mixing)

        self.mu = [m.view(1,-1) for m in mu]
        self.diagSigma = [d.view(1,-1) for d in diagSigma]
        self.mixing = mixing

    def logprob(self, x):
        """
        x: torch.Tensor
            Parameters of models. Tensor of shape (N, dim), with N number of models/processes
        """
        prob = 0
        for mu, diagSigma, mixing in zip(self.mu, self.diagSigma, self.mixing):
            exponent = -0.5 * (x-mu)**2/diagSigma # (N, dim)
            exponent = exponent.sum(1) # (N)
            prob += mixing * 1/torch.sqrt(torch.prod(diagSigma))*torch.exp(exponent) # torch.prod(diagSigma) := det(diagSigma)
        return torch.log(prob+1e-8)



