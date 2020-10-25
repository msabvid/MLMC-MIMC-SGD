import numpy as np
import torch
import os
from torch.utils.data import Dataset



def create_dataset(m: int, d: int, data_dir: str, type_regression="logistic", type_data="synthetic", seed=1):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    np.random.seed(seed)
    torch.manual_seed(seed)

    if type_data=="synthetic" and type_regression=="logistic":
        data_x = 2*torch.randn(m, d)
        data_x = torch.cat([torch.ones(m, 1), data_x],1)
        params = torch.randn(d+1,1)-0.1
        data_y = torch.matmul(data_x,params)
        data_y = torch.sign(torch.clamp(data_y,0))

        data = dict(x=data_x,y=data_y)
        filename = "data_logistic_synthetic_d{}_m{}.pth.tar".format(d,m)
        torch.save(data, os.path.join(data_dir, filename))
    else:
        raise ValueError("Unknown regression {}".format(type_regression))

    return 0



def get_dataset(m: int, d: int, type_regression: str, type_data: str,  data_dir: str):

    filename = os.path.join(data_dir,"data_{}_{}_d{}_m{}.pth.tar".format(type_regression, type_data,d, m))
    if not os.path.exists(filename):
        create_dataset(m=m,d=d,data_dir=data_dir, type_regression=type_regression, type_data=type_data)
    data = torch.load(filename)
    return data["x"], data["y"]


class Dataset_MCMC(Dataset):

    def __init__(self, data_X, data_Y):
        self.data_X = data_X
        self.data_Y = data_Y

    def __len__(self):
        return self.data_X.shape[0]

    def __getitem__(self, idx):
        return self.data_X[idx], self.data_Y[idx]
