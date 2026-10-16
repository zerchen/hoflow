#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File        :oakink.py
#@Date        :2022/07/05 18:51:49
#@Author      :zerui chen
#@Contact     :zerui.chen@inria.fr

import os
from datasets.shards import find_shards
import trimesh
import torch
import numpy as np
from utils.random_utils import np_random


class oakink:
    def __init__(self, cfg, data_split):
        self.name = 'oakink'
        self.cfg = cfg
        self.data_split = data_split
        self.cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.object_mesh_root = os.path.join(self.cur_dir, 'data', "oakink_aligned")

        self.object_pcds = {}
        if os.path.isdir(self.object_mesh_root):
            for mesh_file in os.listdir(self.object_mesh_root):
                if mesh_file.endswith('.ply'):
                    obj_name = mesh_file.split('.')[0]
                    obj_mesh_path = os.path.join(self.object_mesh_root, mesh_file)
                    obj_mesh = trimesh.load(obj_mesh_path, process=False)
                    obj_pcds = trimesh.sample.sample_surface(obj_mesh, 1024)[0].astype(np.float32)
                    self.object_pcds[obj_name] = obj_pcds

        if cfg.task in ["motion_ar", "motion_diffusion", "autoreg_flow", "seq_motion_diffusion"]:
            self.train_shard_path = find_shards(os.path.join(self.cur_dir, "data"), "seq_tars", self.name)
            self.val_shard_path = find_shards(os.path.join(self.cur_dir, "data"), "seq_tars", self.name)
            self.test_shard_path = find_shards(os.path.join(self.cur_dir, "data"), "seq_tars", self.name)
            self.make_sample = self.make_seq_sample
        self.np_random, _ = np_random()

    def make_seq_sample(self, sample, mode='train'):
        meta = sample["meta.json"]
        sample_key = meta["sample_id"]
        obj_name = meta["obj_name"]
        text = meta["text"]

        obj_mesh_path = os.path.join(self.object_mesh_root, f"{obj_name}.ply")
        obj_pcds = torch.from_numpy(self.object_pcds[obj_name])
        window_size = self.cfg.test_window_size
        if "init_obj_rotmat.npy" in sample:
            first_frame_rot = torch.from_numpy(np.asarray(sample["init_obj_rotmat.npy"], dtype=np.float32))
        else:
            first_frame_rot = torch.eye(3, dtype=torch.float32)
        obj_pcds = torch.einsum('ij,nj->ni', first_frame_rot, obj_pcds)

        rh_poses = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(window_size, 16, 3, 3).clone()
        lh_poses = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(window_size, 16, 3, 3).clone()
        rh_trans = torch.zeros((window_size, 3), dtype=torch.float32)
        lh_trans = torch.zeros((window_size, 3), dtype=torch.float32)
        rh_verts = torch.zeros((window_size, 778, 3), dtype=torch.float32)
        lh_verts = torch.zeros((window_size, 778, 3), dtype=torch.float32)
        rh_joints = torch.zeros((window_size, 21, 3), dtype=torch.float32)
        lh_joints = torch.zeros((window_size, 21, 3), dtype=torch.float32)
        rh_transf = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(window_size, 16, 4, 4).clone()
        lh_transf = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(window_size, 16, 4, 4).clone()
        obj_transf = torch.eye(4).unsqueeze(0).expand(window_size, 4, 4).clone()

        input_iter = dict(rh_poses=rh_poses, lh_poses=lh_poses, rh_trans=rh_trans, lh_trans=lh_trans, rh_verts=rh_verts, lh_verts=lh_verts, rh_joints=rh_joints, lh_joints=lh_joints, rh_transf=rh_transf, lh_transf=lh_transf, obj_transf=obj_transf, obj_pcds=obj_pcds, text=text, rh_mask=torch.ones((1), dtype=torch.float32), lh_mask=torch.ones((1), dtype=torch.float32), init_obj_rotmat=first_frame_rot)
        meta_iter = dict(id=sample_key, obj_id=obj_mesh_path, motion_length=window_size)
        return input_iter, meta_iter
