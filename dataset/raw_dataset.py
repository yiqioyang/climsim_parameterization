import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
import xarray as xr





class ClimsimBaseDataset_raw(Dataset):
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

class Climsim_Dataset_xy_raw(ClimsimBaseDataset_raw):
    def __getitem__(self, idx):
        xr_x = self.loadx(idx)
        xr_y = self.loady(idx)

        x3d = self._stack_vars3d(xr_x, self.xname3d)
        x2d = self._stack_vars2d(xr_x, self.xname2d)

        y3d = self._stack_vars3d(xr_y, self.yname3d)
        y2d = self._stack_vars2d(xr_y, self.yname2d)

        y3d = y3d - x3d



        #########

        return x3d, x2d, y3d, y2d


