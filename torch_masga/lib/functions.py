import torch
import torch.nn as nn



def norm_sq(x: torch.Tensor):
    """
    f(x) = ||x||_2^2

    Parameters
    ----------
    x: torch.Tensor of size (batch_size, d)

    Returns
    -------
    output: torch.Tensor of shape (batch_size)
    """

    output = torch.norm(x, p=2, dim=1)**2
    return output

def norm1(x: torch.Tensor):
    """
    f(x) = ||x||_1

    Parameters
    ----------
    x: torch.Tensor of size (batch_size, d)

    Returns
    -------
    output: torch.Tesor of size (batch_size)
    """

    output = torch.norm(x, p=1, dim=1)
    return output



def exp(x: torch.Tensor):
    """
    f(x) = exp(||x||_1)

    Parameters
    ----------
    x: torch.Tensor of size (batch_size, d)
    
    Returns
    -------
    output: torch.Tensor of size (batch_size)
    """
    output = torch.exp(norm1(x))
    return output


def bell(x: torch.Tensor):
    """
    f(x) = exp(-||x||^2)

    Parameters
    ----------
    x: torch.Tensor of size (batch_size, d)

    Returns
    -------
    output: torch.Tensor of size (batch_size)
    """
    output = torch.exp(-norm_sq(x))
    return output
