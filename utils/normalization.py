import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import xarray as xr
import glob
import matplotlib.pyplot as plt

import glob
import sys

import random


from ..dataset.dataset import ClimsimBaseDataset, Climsim_Dataset_xy



n_samples = 30000
n_workers = 8


inp_files_raw = glob.glob("/glade/work/qingyuany/Climsim/train/*/E3SM*mli*.nc")
out_files_raw = glob.glob("/glade/work/qingyuany/Climsim/train/*/E3SM*mlo*.nc")

random.shuffle(inp_files_raw)
inp_files_raw = inp_files_raw[:n_samples]

inp_files = []
count = 0
for f in inp_files_raw:
    temp = f.replace("mli", "mlo")
    if temp in out_files_raw:
        inp_files.append(f)

    else:
        count += 1



inp_var_3d_nm = ['state_t', 'state_q0001', 'state_q0002', 'state_q0003']
inp_var_2d_nm = ['state_ps', 'pbuf_SOLIN', 'pbuf_LHFLX', 'pbuf_SHFLX']
inp_var_1d_nm = ['ymd', 'tod']
                 

out_var_3d_nm = ['state_t', 'state_q0001', 'state_q0002', 'state_q0003']
out_var_2d_nm = ['cam_out_NETSW',  
                 'cam_out_FLWDS', 'cam_out_PRECSC', 'cam_out_PRECC', 'cam_out_SOLS', 
                 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD']

dataset = Climsim_Dataset_xy(
    input_paths = inp_files,
    xname3d = inp_var_3d_nm,
    xname2d = inp_var_2d_nm,
    xname1d = inp_var_1d_nm,
    yname3d = out_var_3d_nm,
    yname2d = out_var_2d_nm       
)



input_mean_3d = []
input_std_3d = []
input_scale = []

input_mean_2d = []
input_std_2d = []

##########################################
output_mean_3d = []
output_std_3d = []
output_scale = []

output_mean_2d = []
output_std_2d = []





dataloader = DataLoader(dataset, batch_size=32, num_workers=n_workers, pin_memory=True)