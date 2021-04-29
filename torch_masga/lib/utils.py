import torch
import numpy as np



def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
