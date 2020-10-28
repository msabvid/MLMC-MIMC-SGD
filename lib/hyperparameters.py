import torch

def config_priors(dim, device):

    dictGaussian = {"mu": torch.zeros(dim, device=device),
            "diagSigma": torch.ones(dim, device=device)}

    dictMixture = {"mu":[torch.zeros(dim, device=device), 5*torch.ones(dim, device=device)],
            "diagSigma":[torch.ones(dim, device=device), torch.ones(dim, device=device)],
            "mixing":[0.6,0.4]}

    CONFIG = {"Gaussian":dictGaussian, "MixtureGaussians":dictMixture}
    return CONFIG
