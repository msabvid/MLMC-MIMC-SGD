import math
import torch
import torch.nn as nn
import copy
import numpy as np
import math
import tqdm

from ..mlmc_mimc import MLMC 
from ..lib.data import get_dataset
from ..nn.priors import Gaussian, MixtureGaussians
from ..nn.models import LogisticNets



class Bayesian_MCMC():
    """
    Multi-level Monte Carlo at the subsampling level with antithetic samples 
    """

    def __init__(self, M: int, h0: float, 
            s0: int, n0: int, data_X: torch.Tensor, data_Y: torch.Tensor, prior, model, func, init_func, device):
        """
        Sampler of the posterior distribution using discretisation of Langevin process

        dXt = nabla(logprior(X_t) + loglikelihood(X_t))dt + sqrt(2)dWt

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
        init_func: initialisation function of chains
        device: str
        """
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

    def init_weights(self, net, **kwargs):
        """ Init weights with init function

        """
        net.params.data.copy_(self.init_func(params=net.params.data))


    def euler_step(self, nets, U, sigma, h, dW):
        """Perform a step of Euler scheme in-place on the parameters of the networks

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
        drift_langevin = self.prior.logprob(nets.params) + self.data_size/subsample_size * nets.loglik(data_X = self.data_X, data_y=self.data_Y, U=U)
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
            if isinstance(nets, torch.Tensor):
                F = self.func(nets)
            else:
                F = self.func(nets.params) #torch.norm(nets.params, p=2, dim=1)**2
        return F.cpu().numpy()

    def sample_posterior(self, l, N, subsampling=True, **kwargs):
        """
        Sampling from the posterior using sub-sampling on the dataset and unbiased estimator of the drift of the Langevin process

        Parameters
        ----------
        l: int. Level of subsampling level
        N: int. Number of chains
        
        Returns
        -------
        chain: torch.Tensor of shape [N, self.n0+1, dim]

        """
        dim = self.data_X.shape[1]#-1
        
        hf = self.h0 
        # we re-scale timestep size: h:=h/(2m)
        hf = hf/(2*self.data_size)
        
        sf = self.s0 * self.M ** l
        sigma_f = math.sqrt(2)

        chain = torch.zeros(N, self.n0+1, dim, device=self.device)

        X_f = self.model(dim, N, **kwargs).to(device=self.device)
        self.init_weights(X_f)
        chain[:,0,:] = X_f.params.data.detach()
        dW = torch.zeros_like(X_f.params)
        
        pbar = tqdm.tqdm(total=self.n0)
        for n in range(self.n0):
            dW = math.sqrt(hf) * torch.randn_like(dW)
            if subsampling:
                U = np.random.choice(self.data_size, (N,sf), replace=True)
            else:
                U = np.arange(self.data_size).reshape(1,-1).repeat(N,axis=0)
            self.euler_step(X_f, U, sigma_f, hf, dW)
            chain[:,n+1,:] = X_f.params.data.detach()
            if n%100==0:
                pbar.update(100)

        return chain



