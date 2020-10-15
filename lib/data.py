import numpy as np
import torch
import os




def create_dataset(m: int, d: int, data_dir: str, type_regression="logistic", seed=1):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    np.random.seed(seed)
    torch.manual_seed(seed)

    if type_regression=="logistic":
        data_x = 2*torch.randn(m, d)
        data_x = torch.cat([torch.ones(m, 1), data_x],1)
        params = torch.randn(d+1,1)-0.1
        data_y = torch.matmul(data_x,params)
        data_y = torch.sign(torch.clamp(data_y,0))

        data = dict(x=data_x,y=data_y)
        filename = "data_logistic_d{}_m{}.pth.tar".format(d,m)
        torch.save(data, os.path.join(data_dir, filename))
    else:
        raise ValueError("Unknown regression {}".format(type_regression))

    return 0



def get_dataset(m: int, d: int, type_regression: str, data_dir: str):

    filename = os.path.join(data_dir,"data_{}_d{}_m{}.pth.tar".format(type_regression, d, m))
    if not os.path.exists(filename):
        create_dataset(m=m,d=d,data_dir=data_dir,)
    data = torch.load(filename)
    return data["x"], data["y"]

