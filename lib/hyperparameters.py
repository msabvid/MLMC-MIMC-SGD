import torch

def config_priors(dim, device):

    dictGaussian = {"mu": torch.zeros(dim, device=device),
            "diagSigma": torch.ones(dim, device=device)}

    dictMixture = {"mu":[torch.zeros(dim, device=device), 2*torch.ones(dim, device=device)],
            "diagSigma":[torch.ones(dim, device=device)]*2,
            "mixing":[0.6,0.4]}

    CONFIG = {"Gaussian":dictGaussian, "MixtureGaussians":dictMixture}
    return CONFIG
