
import os

import torch
import xarray as xr
import numpy as np

def norm_per_batch(x3d, x2d, x3d_q = None, q_threshold = 10**(-7)):
    
    x3d = x3d.reshape(-1, x3d.shape[-2], x3d.shape[-1])
    x2d = x2d.reshape(-1, x2d.shape[-1])

    batch_size = x3d.shape[0]

    batch3d_sum = x3d.mean(dim = [0,2])
    batch3d_sq_sum = (x3d **2).mean(dim = [0,2])

    batch2d_sum = x2d.mean(dim = [0])
    batch2d_sq_sum = (x2d **2).mean(dim = [0])

    if x3d_q is not None:
        x3d_q = x3d_q.reshape(-1, x3d_q.shape[-2], x3d_q.shape[-1])
        x3d_q_mask = x3d_q > q_threshold
        batch3d_q_sum = (x3d_q *x3d_q_mask).mean(dim = [0,2])
        return batch3d_sum, batch3d_sq_sum, batch2d_sum, batch2d_sq_sum, batch3d_q_sum
    
    else:
        return batch3d_sum, batch3d_sq_sum, batch2d_sum, batch2d_sq_sum


def norm_cumulative(data_loader, x3d_channel, x3d_q_channel, q_threshold=10 ** (-7), device='cpu'):
 

    sample_count = 0
    x3d_mean_sum = 0.0
    x3d_sq_sum = 0.0

    x2d_mean_sum = 0.0
    x2d_sq_sum = 0.0
    
    x3d_q_mean_sum = 0.0

    y3d_mean_sum = 0.0
    y3d_sq_sum = 0.0

    y2d_mean_sum = 0.0
    y2d_sq_sum = 0.0

    loader_i = 0
    for x3d_all, x2d, y3d, y2d in data_loader:
        if loader_i % 50 == 0:
            print(loader_i)
        loader_i += 1

        if loader_i == 3:
            print(x3d_all.shape, x2d.shape, y3d.shape, y2d.shape)


        x3d_all = x3d_all.to(device=device, dtype=torch.float32)
        x2d = x2d.to(device=device, dtype=torch.float32)
        y3d = y3d.to(device=device, dtype=torch.float32)
        y2d = y2d.to(device=device, dtype=torch.float32)

        x3d = x3d_all[:, :,x3d_channel,:]
        x3d_q = x3d_all[:, :,x3d_q_channel,:] 

        batch_size = x3d.shape[0]

        
        xbatch3d_sum, xbatch3d_sq_sum, xbatch2d_sum, xbatch2d_sq_sum, xbatch3d_q_sum = norm_per_batch(x3d, x2d, x3d_q, q_threshold)
        x3d_mean_sum += xbatch3d_sum * batch_size
        x3d_sq_sum += xbatch3d_sq_sum * batch_size
        x2d_mean_sum += xbatch2d_sum * batch_size
        x2d_sq_sum += xbatch2d_sq_sum * batch_size
        x3d_q_mean_sum += xbatch3d_q_sum * batch_size
        
        ybatch3d_sum, ybatch3d_sq_sum, ybatch2d_sum, ybatch2d_sq_sum = norm_per_batch(y3d, y2d)
        y3d_mean_sum += ybatch3d_sum * batch_size
        y3d_sq_sum += ybatch3d_sq_sum * batch_size
        y2d_mean_sum += ybatch2d_sum * batch_size
        y2d_sq_sum += ybatch2d_sq_sum * batch_size
        
        sample_count += batch_size


    x3d_mean_o = x3d_mean_sum/sample_count
    x3d_std_o = torch.sqrt(x3d_sq_sum/sample_count - x3d_mean_o**2)

    x2d_mean_o = x2d_mean_sum/sample_count
    x2d_std_o = torch.sqrt(x2d_sq_sum/sample_count - x2d_mean_o**2)
    
    x3d_q_mean_o = x3d_q_mean_sum/sample_count


    y3d_mean_o = y3d_mean_sum/sample_count
    y3d_std_o = torch.sqrt(y3d_sq_sum/sample_count - y3d_mean_o**2)

    y2d_mean_o = y2d_mean_sum/sample_count
    y2d_std_o = torch.sqrt(y2d_sq_sum/sample_count - y2d_mean_o**2)
    

    out_dict = {
        "x3d_mean": x3d_mean_o,
        "x3d_std": x3d_std_o,
        "x2d_mean": x2d_mean_o,
        "x2d_std": x2d_std_o,
        "x3d_q_mean": x3d_q_mean_o,
        "y3d_mean": y3d_mean_o,
        "y3d_std": y3d_std_o,
        "y2d_mean": y2d_mean_o,
        "y2d_std": y2d_std_o
    }

    return out_dict



def names_values_to_xr(names, values, lev_name="lev"):
    # values can be torch tensor or numpy array
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()

    ds_vars = {}
    for i, name in enumerate(names):
        v = values[i]

        # scalar (0-d) -> store as scalar variable
        if np.ndim(v) == 0:
            ds_vars[name] = float(v)
            print('true')
        else:
            # 1-d profile (e.g. length 60) -> store with lev dimension
            ds_vars[name] = ((lev_name,), np.asarray(v))

    return xr.Dataset(ds_vars)