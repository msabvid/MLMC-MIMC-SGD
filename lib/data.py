import numpy as np
import torch
import os
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


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
    elif type_data=="covtype":
        data = np.loadtxt(os.path.join(data_dir, "covtype.txt"), delimiter=",")
        data_x = data[:,:-1]
        data_y = data[:,-1]
        X_train, _, Y_train, _ = train_test_split(data_x, data_y, train_size=0.2)
        scaler = StandardScaler()
        data_x = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
        Y_train[Y_train>1] = 0
        data_y = torch.tensor(Y_train,dtype=torch.float32).reshape(-1,1)
        data = dict(x=data_x,y=data_y)
        filename = "data_logistic_covtype.pth.tar".format(d)
        torch.save(data, os.path.join(data_dir, filename))
    elif type_regression=='MixtureGaussians':
        theta1, theta2 = 0, 1
        sigma_x = np.sqrt(2)
        u = np.random.rand(m)
        z1 = np.random.randn(m)
        z2 = np.random.randn(m)
        x1, x2 = np.zeros_like(u), np.zeros_like(u)
        x1[np.where(u<=0.5)] = theta1 + sigma_x * z1[np.where(u<=0.5)]
        x2[np.where(u<=0.5)] = theta1 + sigma_x * z2[np.where(u<=0.5)]
        x1[np.where(u>0.5)] = theta1 + theta2 + sigma_x * z1[np.where(u>0.5)]
        x2[np.where(u>0.5)] = theta1 + theta2 + sigma_x * z2[np.where(u>0.5)]
        data_x = np.stack([x1,x2], axis=1)
        data_y = None
        data = dict(x=torch.tensor(data_x, dtype=torch.float32),y=data_y)
        filename = "data_MixtureGaussians_synthetic_d{}_m{}.pth.tar".format(d,m)
        torch.save(data, os.path.join(data_dir, filename))
    else:
        raise ValueError("Unknown regression {}".format(type_regression))

    return 0



def get_dataset(m: int, d: int, type_regression: str, type_data: str,  data_dir: str):

    if type_data == "synthetic":
        filename = os.path.join(data_dir,"data_{}_{}_d{}_m{}.pth.tar".format(type_regression, type_data,d, m))
    elif type_data=="covtype":
        filename = os.path.join(data_dir,"data_{}_{}.pth.tar".format(type_regression, type_data,d, m))
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
