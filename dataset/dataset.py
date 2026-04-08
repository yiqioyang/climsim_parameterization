import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
import xarray as xr




def transform23d(x, mean, std):
    mean = torch.from_numpy(mean.values).float()
    std = torch.from_numpy(std.values).float()
    return (x - mean) / std

def transform_out(x, std):
    std = torch.from_numpy(std.values).float()
    return (x) * std

def transform_q(x, norm_lambda):
    norm_lambda = torch.from_numpy(norm_lambda.values).float()

    return 1 - torch.exp(-norm_lambda * x)


name_mapping = {
    'state_t':'ptend_t',
    'state_q0001':'ptend_q0001', 
    'state_q0002':'ptend_q0002',
    'state_q0003':'ptend_q0003',
    'cam_out_NETSW': 'cam_out_NETSW',
    'cam_out_FLWDS': 'cam_out_FLWDS',
    'cam_out_PRECSC': 'cam_out_PRECSC',
    'cam_out_PRECC': 'cam_out_PRECC',
    'cam_out_SOLS': 'cam_out_SOLS',
    'cam_out_SOLL': 'cam_out_SOLL',
    'cam_out_SOLSD': 'cam_out_SOLSD',
    'cam_out_SOLLD': 'cam_out_SOLLD'
}

def load_norm_mean_std():
    inp_mean = xr.open_dataset("/glade/u/home/qingyuany/ClimSim/preprocessing/normalizations/inputs/input_mean.nc")
    inp_std = xr.open_dataset("/glade/u/home/qingyuany/ClimSim/preprocessing/normalizations/inputs/input_std.nc")
    out_mean = xr.open_dataset("/glade/u/home/qingyuany/ClimSim/preprocessing/normalizations/outputs/output_scale.nc")
    return inp_mean, inp_std, out_mean

def load_lambda():
    lambda1 = pd.read_csv("/glade/u/home/qingyuany/ClimSim/preprocessing/normalizations/inputs/qn_exp_lambda_large.txt", index_col=False, header=None)
    lambda2 = pd.read_csv("/glade/u/home/qingyuany/ClimSim/preprocessing/normalizations/inputs/qc_exp_lambda_large.txt", index_col=False, header=None)
    lambda3 = pd.read_csv("/glade/u/home/qingyuany/ClimSim/preprocessing/normalizations/inputs/qi_exp_lambda_large.txt", index_col=False, header=None)
    lambdas = pd.concat([lambda1.T, lambda2.T, lambda3.T], axis = 1)
    lambdas.columns  = ['state_q0001', 'state_q0002', 'state_q0003']
    return lambdas




class ClimsimBaseDataset(Dataset):
    def __init__(self, input_paths, xname3d, xname2d, yname3d, yname2d, spatial_info = None, n_col = 384, n_lev=60, transform=None):
        self.inp_files = input_paths
        self.out_files = [file.replace("mli", "mlo") for file in self.inp_files]
        self.xname3d = xname3d
        self.xname2d = xname2d
    
        
        self.yname3d = yname3d
        self.yname2d = yname2d
        
        self.n_lev = n_lev
        self.n_col = n_col
        self.transform = transform
        self.spatial_info = spatial_info
        self.inp_mean, self.inp_std, self.out_std = load_norm_mean_std()
        self.lambdas = load_lambda()
    def __len__(self):
        return len(self.inp_files)

    @staticmethod
    def _stack_vars3d(ds, names3d):
        out = torch.from_numpy((ds[names3d].to_array().to_numpy())).float().permute([2,0,1])
        return out
    
    @staticmethod
    def _stack_vars2d(ds, names2d):
        out = torch.from_numpy((ds[names2d].to_array().to_numpy())).float().permute([1,0])
        return out
    
    
    def loadx(self, idx):
        with xr.open_dataset(self.inp_files[idx]) as ds:
            xr_x = ds.load()
        return xr_x

    def loady(self, idx):
        with xr.open_dataset(self.out_files[idx]) as ds:
            xr_y = ds.load()
        return xr_y

class Climsim_Dataset_xy(ClimsimBaseDataset):
    def __getitem__(self, idx):
        xr_x = self.loadx(idx)
        xr_y = self.loady(idx)

        x3d = self._stack_vars3d(xr_x, self.xname3d)
        x2d = self._stack_vars2d(xr_x, self.xname2d)

        y3d = self._stack_vars3d(xr_y, self.yname3d)
        y2d = self._stack_vars2d(xr_y, self.yname2d)

        y3d = y3d - x3d


        for i, name in enumerate(self.xname3d[0:2]):
            x3d[:,i] = transform23d(x3d[:,i], self.inp_mean[name], self.inp_std[name])

        for i, name in enumerate(self.xname3d[2:]):
            x3d[:,(i+2)] = transform_q(x3d[:,(i+2)], self.lambdas[name])

        for i, name in enumerate(self.xname2d):
            x2d[:,i] = transform23d(x2d[:,i], self.inp_mean[name], self.inp_std[name])


        for i, name in enumerate(self.yname2d):
            y2d[:,i] = transform_out(y2d[:,i], self.out_std[name])

        for i, name in enumerate(self.yname3d):
            y3d[:,i] = transform_out(y3d[:,i], self.out_std[name_mapping[name]  ])

        #########

        return x3d, x2d, y3d, y2d


