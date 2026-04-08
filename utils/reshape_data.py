import torch

def reshape_for_mlp(x3d, x2d, y3d, y2d, x3d_name, x2d_name, y3d_name, y2d_name):
    x3d = x3d.reshape(-1, len(x3d_name) * 60)
    x2d = x2d.reshape(-1, len(x2d_name))

    y3d = y3d.reshape(-1, len(y3d_name) * 60)
    y2d = y2d.reshape(-1, len(y2d_name))

    x = torch.cat([x3d, x2d], dim=1)
    y = torch.cat([y3d, y2d], dim=1)

    return x, y
    