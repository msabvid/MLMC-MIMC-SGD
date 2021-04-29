import torch
import torch.nn as nn




def zeros(params: torch.Tensor):
    return torch.zeros_like(params)

def ones(params: torch.Tensor):
    return torch.ones_like(params)


def init_mode(params: torch.Tensor, filename: str):
    mode = torch.load(filename, map_location=params.device)
    batch_size = params.shape[0]
    return mode['params'].repeat(batch_size, 1)


def normal_(params: torch.Tensor, mean=0.0, std=1.0):
    return std * (mean + torch.randn_like(params))
