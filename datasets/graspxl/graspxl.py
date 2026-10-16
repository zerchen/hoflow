#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File :graspxl.py
#@Date :2025/08/27 11:18:16
#@Author :Zerui Chen
#@Contact :zeruicasia@gmail.com

import os
from datasets.shards import find_shards
import trimesh
import torch
import numpy as np
from scipy.spatial.transform import Rotation
from utils.pose_utils import pad_with_last
from utils.random_utils import np_random
from kornia.geometry.conversions import axis_angle_to_rotation_matrix, rotation_matrix_to_axis_angle


class graspxl:
    def __init__(self, cfg, data_split):
        self.name = 'graspxl'
        self.cfg = cfg
        self.data_split = data_split
        self.cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.obj_mesh_root = os.path.join(self.cur_dir, 'data', "object_dataset")
        self.rh_rot_center = np.array([0.0957,  0.0064,  0.0062], dtype=np.float32)

        if cfg.task in ["inter_vae"]:
            self.train_shard_path = find_shards(os.path.join(self.cur_dir, "data"), "full_seq_tars", self.name)
            self.val_shard_path = find_shards(os.path.join(self.cur_dir, "data"), "full_seq_tars", self.name)[-1:]
            self.test_shard_path = find_shards(os.path.join(self.cur_dir, "data"), "full_seq_tars", self.name)[-1:]
            self.make_sample = self.make_seq_sample
        elif cfg.task in ["autoreg_flow"]:
            self.train_shard_path = find_shards(os.path.join(self.cur_dir, "data"), "latent_tars", self.name)
            self.val_shard_path = find_shards(os.path.join(self.cur_dir, "data"), "latent_tars", self.name)[-1:]
            self.test_shard_path = find_shards(os.path.join(self.cur_dir, "data"), "full_seq_tars", self.name)[-1:]
            self.make_sample = self.make_latent_sample
        self.np_random, _ = np_random()
    
    def make_seq_sample(self, sample, mode):
        meta = sample["meta.json"]
        sample_key = meta["sample_id"]
        motion_length = meta["seq_len"]
        obj_name = meta["obj_name"]
        obj_size = sample_key.split("-")[1]

        verb = self.np_random.choice(["grasp", "pick up", "relocate", "place", "lift", "move"])
        text = f"{verb} the object"

        obj_mesh_path = os.path.join(self.obj_mesh_root, obj_size, obj_name, f"{obj_name}.obj")
        obj_mesh = trimesh.load(obj_mesh_path, process=False)
        obj_pcds = torch.from_numpy(trimesh.sample.sample_surface(obj_mesh, 1024)[0].astype(np.float32))

        rh_poses = sample["rh_poses.npy"]
        rh_trans = sample["rh_trans.npy"]
        obj_transf = sample["obj_transf.npy"]

        rh_verts = sample["rh_verts.npy"]
        rh_joints = sample["rh_joints.npy"]
        rh_transf = sample["rh_transf.npy"]

        window_size = self.cfg.test_window_size if mode == "test" else self.cfg.window_size
        if self.cfg.task in ["seq_motion_ae", "inter_vae"]:
            if motion_length < window_size:
                rh_poses = torch.from_numpy(pad_with_last(rh_poses, window_size, dim=2))
                rh_trans = torch.from_numpy(pad_with_last(rh_trans, window_size, dim=2))
                obj_transf = torch.from_numpy(pad_with_last(obj_transf, window_size, dim=3))
                rh_verts = torch.from_numpy(pad_with_last(rh_verts, window_size, dim=3))
                rh_joints = torch.from_numpy(pad_with_last(rh_joints, window_size, dim=3))
                rh_transf = torch.from_numpy(pad_with_last(rh_transf, window_size, dim=4))
            else:
                start_idx = 0 if mode == "test" else self.np_random.randint(0, motion_length-window_size + 1)
                rh_poses = torch.from_numpy(rh_poses[start_idx:start_idx+window_size])
                rh_trans = torch.from_numpy(rh_trans[start_idx:start_idx+window_size])
                obj_transf = torch.from_numpy(obj_transf[start_idx:start_idx+window_size])
                rh_verts = torch.from_numpy(rh_verts[start_idx:start_idx+window_size])
                rh_joints = torch.from_numpy(rh_joints[start_idx:start_idx+window_size])
                rh_transf = torch.from_numpy(rh_transf[start_idx:start_idx+window_size])
        elif self.cfg.task in ["motion_diffusion", "motion_ar", "autoreg_flow"]:
            rh_poses = torch.from_numpy(pad_with_last(rh_poses, window_size, dim=2))
            rh_trans = torch.from_numpy(pad_with_last(rh_trans, window_size, dim=2))
            obj_transf = torch.from_numpy(pad_with_last(obj_transf, window_size, dim=3))
            rh_verts = torch.from_numpy(pad_with_last(rh_verts, window_size, dim=3))
            rh_joints = torch.from_numpy(pad_with_last(rh_joints, window_size, dim=3))
            rh_transf = torch.from_numpy(pad_with_last(rh_transf, window_size, dim=4))

        rh_trans = rh_trans - obj_transf[0, :3, 3][None]
        rh_verts = rh_verts - obj_transf[0, :3, 3][None, None]
        rh_joints = rh_joints - obj_transf[0, :3, 3][None, None]
        obj_transf[:, :3, 3] = obj_transf[:, :3, 3] - obj_transf[0, :3, 3][None]

        rh_poses = axis_angle_to_rotation_matrix(rh_poses.reshape(-1, 3)).reshape((window_size, 16, 3, 3))

        if mode == 'train' and self.cfg.use_3d_aug and self.np_random.random() <= 0.5:
            rot_angles = (self.np_random.uniform(-1, 1, 3) * np.array([np.pi/12, np.pi/12, np.pi/6])).astype(np.float32)
            rot_aug_mat = torch.from_numpy(Rotation.from_euler('zyx', rot_angles[::-1]).as_matrix().astype(np.float32))
            rh_poses[:, 0] = rot_aug_mat.expand_as(rh_poses[:, 0]) @ rh_poses[:, 0]
            rh_rot_center = torch.from_numpy(self.rh_rot_center).expand_as(rh_trans)
            rh_trans = (rot_aug_mat @ rh_trans.transpose(1, 0)).transpose(1, 0) + (rot_aug_mat @ rh_rot_center.transpose(1, 0)).transpose(1, 0) - rh_rot_center
            rh_verts = torch.einsum('ij,tnj->tni', rot_aug_mat, rh_verts)
            rh_joints = torch.einsum('ij,tnj->tni', rot_aug_mat, rh_joints)
            homo_rot_aug_mat = torch.eye(4)
            homo_rot_aug_mat[:3, :3] = rot_aug_mat
            obj_transf = homo_rot_aug_mat.expand_as(obj_transf) @ obj_transf
            rh_transf = homo_rot_aug_mat.expand_as(rh_transf) @ rh_transf
            rh_transf[:, :, :3, 3] = rh_joints[:, [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]] - rh_trans.unsqueeze(1)

        first_frame_rot = obj_transf[0, :3, :3].clone()
        obj_transf[:, :3, :3] = torch.einsum('tij,jk->tik', obj_transf[:, :3, :3], first_frame_rot.T)
        obj_pcds = torch.einsum('ij,nj->ni', first_frame_rot, obj_pcds)

        dummy_lh_transf = torch.eye(4).expand_as(rh_transf)
        dummy_lh_poses = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(window_size, 16, 3, 3).clone()
        if mode == 'train':
            input_iter = dict(rh_poses=rh_poses, lh_poses=dummy_lh_poses, rh_trans=rh_trans, lh_trans=torch.zeros_like(rh_trans), rh_verts=rh_verts, lh_verts=torch.zeros_like(rh_verts), rh_joints=rh_joints, lh_joints=torch.zeros_like(rh_joints), rh_transf=rh_transf, lh_transf=dummy_lh_transf, obj_transf=obj_transf, obj_pcds=obj_pcds, text=text, rh_mask=torch.ones((1), dtype=torch.float32), lh_mask=torch.zeros((1), dtype=torch.float32), init_obj_rotmat=first_frame_rot)
            label_iter = dict(rh_mask=torch.ones((1), dtype=torch.float32), lh_mask=torch.zeros((1), dtype=torch.float32))
            meta_iter = dict(id=sample_key, obj_id=obj_mesh_path, motion_length=window_size)
            return input_iter, label_iter, meta_iter
        elif mode == 'test':
            input_iter = dict(rh_poses=rh_poses, lh_poses=dummy_lh_poses, rh_trans=rh_trans, lh_trans=torch.zeros_like(rh_trans), rh_verts=rh_verts, lh_verts=torch.zeros_like(rh_verts), rh_joints=rh_joints, lh_joints=torch.zeros_like(rh_joints), rh_transf=rh_transf, lh_transf=dummy_lh_transf, obj_transf=obj_transf, obj_pcds=obj_pcds, text=text, rh_mask=torch.ones((1), dtype=torch.float32), lh_mask=torch.zeros((1), dtype=torch.float32), init_obj_rotmat=first_frame_rot)
            meta_iter = dict(id=sample_key, obj_id=obj_mesh_path, motion_length=window_size)
            return input_iter, meta_iter

    def make_latent_sample(self, sample, mode='train'):
        meta = sample["meta.json"]
        sample_id = meta["id"]
        text = meta["text"]
        obj_path = meta["obj_id"]

        init_obj_rotmat = torch.from_numpy(sample["init_obj_rotmat.npy"])
        obj_mesh = trimesh.load(obj_path, process=False)
        obj_pcds = torch.from_numpy(trimesh.sample.sample_surface(obj_mesh, 1024)[0].astype(np.float32))
        obj_pcds = (init_obj_rotmat @ obj_pcds.transpose(1, 0)).transpose(1, 0)

        rh_obj_mean = torch.from_numpy(sample["rh_obj_mean.npy"])
        rh_mean = torch.from_numpy(sample["rh_mean.npy"])
        lh_obj_mean = rh_obj_mean
        lh_mean = rh_mean
        rh_mask = torch.ones((1), dtype=torch.float32)
        lh_mask = torch.ones((1), dtype=torch.float32)

        if mode == 'train':
            input_iter = dict(obj_pcds=obj_pcds, text=text)
            label_iter = dict(rh_obj_mean=rh_obj_mean, lh_obj_mean=lh_obj_mean, rh_mean=rh_mean, lh_mean=lh_mean, rh_mask=rh_mask, lh_mask=lh_mask)
            meta_iter = dict(id=sample_id, obj_id=obj_path, motion_length=self.cfg.window_size)
            return input_iter, label_iter, meta_iter
