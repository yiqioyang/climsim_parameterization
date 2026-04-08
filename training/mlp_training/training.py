
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import xarray as xr
import numpy as np
import sys
import glob
import random
from torch.distributed import get_rank

sys.path.append('/glade/u/home/qingyuany/repos/climsim_diffusion/dataset')
sys.path.append('/glade/u/home/qingyuany/repos/climsim_diffusion/mlp_training')


from dataset import (Climsim_Dataset_xy, transform23d, transform_out, transform_q, 
        name_mapping, load_norm_mean_std, load_lambda, ClimsimBaseDataset, Climsim_Dataset_xy)

from mlp import MLP





def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")

class Trainer:
    def __init__(self, dataloader, model, optimizer, loss_fun, bsz, xynames, save_every, snapshot_path):

        self.dataloader = dataloader
        self.model = model
        self.optimizer = optimizer
        self.loss_fun = loss_fun
        self.bsz = bsz
        self.save_every = save_every
        self.snapshot_path = snapshot_path
        self.epochs_run = 0

        self.x3d_name, self.x2d_name, self.y3d_name, self.y2d_name = xynames

        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.model = DDP(self.model, device_ids=[self.gpu_id])
    
        # if os.path.exists(snapshot_path):
        #     print("Loading snapshot")
        #     self._load_snapshot(snapshot_path)

    
    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")


    def _run_batch(self, x_batch, y_batch):
        self.optimizer.zero_grad()
        y_pred = self.model(x_batch)
        loss = self.loss_fun(y_pred, y_batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def _run_epoch(self, epoch):
        self.dataloader.sampler.set_epoch(epoch)
        for x3d, x2d, y3d, y2d in self.dataloader:
            x3d = x3d.reshape(-1, len(self.x3d_name) * 60)
            x2d =x2d.reshape(-1, len(self.x2d_name))

            y3d = y3d.reshape(-1, len(self.y3d_name) * 60)
            y2d = y2d.reshape(-1, len(self.y2d_name))

            x = torch.cat([x3d, x2d], dim=1).to(self.gpu_id)    
            y = torch.cat([y3d, y2d], dim=1).to(self.gpu_id)
    
            loss = self._run_batch(x, y)


    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            print(f"[GPU{self.gpu_id}]")
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)


def load_train_objs(inp_files, inp_var_3d_nm, inp_var_2d_nm, out_var_3d_nm, out_var_2d_nm):
    dataset = Climsim_Dataset_xy(
        input_paths = inp_files,
        xname3d = inp_var_3d_nm,
        xname2d = inp_var_2d_nm,
        yname3d = out_var_3d_nm,
        yname2d = out_var_2d_nm       
    )   
    model = MLP(input_dim = len(inp_var_3d_nm) * 60 + len(inp_var_2d_nm), output_dim = len(out_var_3d_nm) * 60 + len(out_var_2d_nm))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return dataset, model, optimizer



def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
        sampler=DistributedSampler(dataset)
    )



inp_var_3d_nm = ['state_t', 'state_q0001', 'state_q0002', 'state_q0003']
inp_var_2d_nm = ['state_ps', 'pbuf_SOLIN', 'pbuf_LHFLX', 'pbuf_SHFLX']
                 

out_var_3d_nm = ['state_t', 'state_q0001', 'state_q0002', 'state_q0003']
out_var_2d_nm = ['cam_out_NETSW',  
                 'cam_out_FLWDS', 'cam_out_PRECSC', 'cam_out_PRECC', 'cam_out_SOLS', 
                 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD']

########################################################
xynames = [inp_var_3d_nm, inp_var_2d_nm, out_var_3d_nm, out_var_2d_nm]

inp_files_raw = glob.glob("/glade/work/qingyuany/Climsim/train/*/E3SM*mli*.nc")
out_files_raw = glob.glob("/glade/work/qingyuany/Climsim/train/*/E3SM*mlo*.nc")

inp_files_raw = inp_files_raw[:50000]

inp_files = []
count = 0
for f in inp_files_raw:
    temp = f.replace("mli", "mlo")
    if temp in out_files_raw:
        inp_files.append(f)

    else:
        count += 1







def main(input_files, xynames,save_every: int, total_epochs: int, batch_size: int, snapshot_path: str = "snapshot.pt"):
    x3dname, x2dname, y3dname, y2dname = xynames
    ddp_setup()
    dataset, model, optimizer = load_train_objs(input_files, x3dname, x2dname, y3dname, y2dname)
    train_dataloader = prepare_dataloader(dataset, batch_size)




    loss_fun = torch.nn.MSELoss()

    trainer = Trainer(train_dataloader, model, optimizer, loss_fun, batch_size, xynames, save_every, snapshot_path)
    trainer.train(total_epochs)
    destroy_process_group()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='distributed training with climsim')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    main(inp_files,xynames,args.save_every, args.total_epochs, args.batch_size, '/glade/work/qingyuany/Climsim/ml_results/mlp_training/test.pt')