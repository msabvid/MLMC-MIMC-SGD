import torch

def config_priors(dim, device):

    dictGaussian = {"mu": torch.zeros(dim, device=device),
            "diagSigma": torch.ones(dim, device=device)}

    dictMixture = {"mu":[torch.zeros(dim, device=device), 2*torch.ones(dim, device=device)],
            "diagSigma":[torch.ones(dim, device=device)]*2}

    CONFIG = {"Gaussian":dictGaussian, "MixtureGaussian":dictMixture}
    return CONFIG
