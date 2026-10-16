import os
from unicodedata import mirrored
import torch
import trimesh
import numpy as np
from bps_torch.bps import bps_torch


def get_obj_bps(obj_mesh_root):
    np.random.seed(100)
    cur_dir = os.path.dirname(__file__)
    BPS_BASE = torch.from_numpy(np.load(os.path.join(cur_dir, '../mano/assets/bps.npz'))['basis']).to(torch.float32)
    bps = bps_torch(custom_basis = BPS_BASE)
    object_bps = {}
    object_pcds = {}

    for filename in os.listdir(obj_mesh_root):
        obj_name = filename.split('.')[0]
        obj_mesh_path = os.path.join(obj_mesh_root, filename)
        obj_mesh = trimesh.load(obj_mesh_path, process=False)
        obj_sampled_pcd = trimesh.sample.sample_surface(obj_mesh, 1024)[0]
        obj_sampled_pcd = torch.from_numpy(obj_sampled_pcd.astype(np.float32))
        mirror_obj_sampled_pcd = obj_sampled_pcd * torch.tensor([-1., 1., 1.])

        obj_bps = bps.encode(obj_sampled_pcd, feature_type='dists')['dists'].to(torch.float32).cpu().detach().numpy()[0]
        mirror_obj_bps = bps.encode(mirror_obj_sampled_pcd, feature_type='dists')['dists'].to(torch.float32).cpu().detach().numpy()[0]

        object_bps[obj_name] = obj_bps
        object_bps[f'mirror_{obj_name}'] = mirror_obj_bps
        object_pcds[obj_name] = obj_sampled_pcd.cpu().numpy()
        object_pcds[f'mirror_{obj_name}'] = mirror_obj_sampled_pcd.cpu().numpy()

    return object_bps, object_pcds
