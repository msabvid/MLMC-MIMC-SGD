import torch
from typing import List

class BasePrior():

    def grad_logsample(self, x):
        raise NotImplementedError()


class Gaussian(BasePrior):
    """
    Gaussian Prior with diagonal covariance matrix
    """
    def __init__(self, mu: torch.Tensor, diagSigma: torch.Tensor, **kwargs):
        """
        Parameters
        ----------
        mu: torch.Tensor
            tensor of size (n)
        Sigma: torch.Tensor
            tensor of size (n)
        """
        assert diagSigma.dim()==1 , "diagSigma needs to have dimension 1"

        self.mu = mu.view(1,-1)
        self.diagSigma = diagSigma.view(1,-1)

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
        mu: list[orch.Tensor]
            list of tensor of size (n)
        Sigma: list[torch.Tensor]
            list of tensor of size (n)
        mixing: list[float]
            list of floats
        """
        assert sum(mixing)==1, "mixing coefficients must sum 1"
        assert len(mu)==len(diagSigma)
        assert len(mu)==len(mixing)

        self.mu = [m.view(1,-1) for m in mu]
        self.diagSigma = [d.view(1,-1) for d in diagSigma]
        self.mixing = mixing

    def grad_logprob(self, x):
        grad_logprob = 0
        for mu, diagSigma, mixing in zip(self.mu, self.diagSigma, self.mixing):
            grad_logprob += mixing * (-1) * (x-mu)/diagSigma
        return grad_logprob



