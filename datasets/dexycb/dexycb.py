#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File        :dexycb.py
#@Date        :2022/07/05 18:51:49
#@Author      :zerui chen
#@Contact     :zerui.chen@inria.fr

import os
from datasets.shards import find_shards
import json
import trimesh
import torch
import numpy as np
from scipy.spatial.transform import Rotation
from utils.pose_utils import pad_with_last
from utils.random_utils import np_random
from kornia.geometry.conversions import axis_angle_to_rotation_matrix, rotation_matrix_to_axis_angle


class dexycb:
    def __init__(self, cfg, data_split):
        self.name = 'dexycb'
        self.cfg = cfg
        self.data_split = data_split
        self.cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.obj_mesh_root = os.path.join(self.cur_dir, 'data', "models")
        self.rh_rot_center = np.array([0.0957,  0.0064,  0.0062], dtype=np.float32)
        self.lh_rot_center = np.array([-0.0957,  0.0064,  0.0062], dtype=np.float32)

        self.object_pcds = {}
        for obj_name in os.listdir(self.obj_mesh_root):
            obj_mesh_path = os.path.join(self.obj_mesh_root, obj_name, "textured_simple.obj")
            obj_mesh = trimesh.load(obj_mesh_path, process=False)
            obj_pcds = trimesh.sample.sample_surface(obj_mesh, 1024)[0].astype(np.float32)
            self.object_pcds[obj_name] = obj_pcds

        if cfg.task in ["inter_vae"]:
            self.train_shard_path = find_shards(os.path.join(self.cur_dir, "data"), "train_seq_tars", self.name)
            self.val_shard_path = find_shards(os.path.join(self.cur_dir, "data"), "test_seq_tars", self.name)
            self.test_shard_path = find_shards(os.path.join(self.cur_dir, "data"), "test_seq_tars", self.name)
            self.make_sample = self.make_seq_sample
        elif cfg.task in ["autoreg_flow"]:
            self.train_shard_path = find_shards(os.path.join(self.cur_dir, "data"), "latent_tars", self.name)
            self.val_shard_path = find_shards(os.path.join(self.cur_dir, "data"), "latent_tars", self.name)
            self.test_shard_path = find_shards(os.path.join(self.cur_dir, "data"), "test_seq_tars", self.name)
            self.make_sample = self.make_latent_sample
        self.np_random, _ = np_random()

    def make_seq_sample(self, sample, mode):
        meta = sample["meta.json"]
        sample_key = meta["sample_id"]
        motion_length = meta["seq_len"]
        obj_name = meta["obj_name"]
        hand_side = meta["hand_side"]
        text = sample["text.txt"] if "text.txt" in sample else meta["text"]

        obj_mesh_path = os.path.join(self.obj_mesh_root, obj_name, "textured_simple.obj")
        obj_pcds = torch.from_numpy(self.object_pcds[obj_name])
        obj_transf = sample["obj_transf.npy"]

        h_poses = sample["h_poses.npy"]
        h_trans = sample["h_trans.npy"]
        h_verts = sample["h_verts.npy"]
        h_joints = sample["h_joints.npy"]
        h_transf = sample["h_transf.npy"]

        window_size = self.cfg.test_window_size if mode == "test" else self.cfg.window_size
        if motion_length < window_size:
            h_poses = torch.from_numpy(pad_with_last(h_poses, window_size, dim=2))
            h_trans = torch.from_numpy(pad_with_last(h_trans, window_size, dim=2))
            obj_transf = torch.from_numpy(pad_with_last(obj_transf, window_size, dim=3))
            h_verts = torch.from_numpy(pad_with_last(h_verts, window_size, dim=3))
            h_joints = torch.from_numpy(pad_with_last(h_joints, window_size, dim=3))
            h_transf = torch.from_numpy(pad_with_last(h_transf, window_size, dim=4))
        else:
            start_idx = 0 if mode == "test" else self.np_random.randint(0, motion_length-window_size + 1)
            h_poses = torch.from_numpy(h_poses[start_idx:start_idx+window_size])
            h_trans = torch.from_numpy(h_trans[start_idx:start_idx+window_size])
            obj_transf = torch.from_numpy(obj_transf[start_idx:start_idx+window_size])
            h_verts = torch.from_numpy(h_verts[start_idx:start_idx+window_size])
            h_joints = torch.from_numpy(h_joints[start_idx:start_idx+window_size])
            h_transf = torch.from_numpy(h_transf[start_idx:start_idx+window_size])

        h_trans = h_trans - obj_transf[0, :3, 3][None]
        h_verts = h_verts - obj_transf[0, :3, 3][None, None]
        h_joints = h_joints - obj_transf[0, :3, 3][None, None]
        obj_transf[:, :3, 3] = obj_transf[:, :3, 3] - obj_transf[0, :3, 3][None]
        h_poses = axis_angle_to_rotation_matrix(h_poses.reshape(-1, 3)).reshape((window_size, 16, 3, 3))

        if mode == 'train' and self.cfg.use_3d_aug and self.np_random.random() <= 0.5:
            rot_angles = (self.np_random.uniform(-1, 1, 3) * np.array([np.pi/12, np.pi/12, np.pi/6])).astype(np.float32)
            rot_aug_mat = torch.from_numpy(Rotation.from_euler('zyx', rot_angles[::-1]).as_matrix().astype(np.float32))
            h_poses[:, 0] = rot_aug_mat.expand_as(h_poses[:, 0]) @ h_poses[:, 0]
            h_rot_center = torch.from_numpy(self.rh_rot_center).expand_as(h_trans) if hand_side == "right" else torch.from_numpy(self.lh_rot_center).expand_as(h_trans)
            h_trans = (rot_aug_mat @ h_trans.transpose(1, 0)).transpose(1, 0) + (rot_aug_mat @ h_rot_center.transpose(1, 0)).transpose(1, 0) - h_rot_center
            h_verts = torch.einsum('ij,tnj->tni', rot_aug_mat, h_verts)
            h_joints = torch.einsum('ij,tnj->tni', rot_aug_mat, h_joints)
            homo_rot_aug_mat = torch.eye(4)
            homo_rot_aug_mat[:3, :3] = rot_aug_mat
            obj_transf = homo_rot_aug_mat.expand_as(obj_transf) @ obj_transf
            h_transf = homo_rot_aug_mat.expand_as(h_transf) @ h_transf
            h_transf[:, :, :3, 3] = h_joints[:, [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]] - h_trans.unsqueeze(1)

        first_frame_rot = obj_transf[0, :3, :3].clone()
        obj_transf[:, :3, :3] = torch.einsum('tij,jk->tik', obj_transf[:, :3, :3], first_frame_rot.T)
        obj_pcds = torch.einsum('ij,nj->ni', first_frame_rot, obj_pcds)

        if hand_side == "right":
            dummy_lh_transf = torch.eye(4).expand_as(h_transf)
            dummy_lh_poses = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(window_size, 16, 3, 3).clone()
            input_iter = dict(rh_poses=h_poses, lh_poses=dummy_lh_poses, rh_trans=h_trans, lh_trans=torch.zeros_like(h_trans), rh_verts=h_verts, lh_verts=torch.zeros_like(h_verts), rh_joints=h_joints, lh_joints=torch.zeros_like(h_joints), rh_transf=h_transf, lh_transf=dummy_lh_transf, obj_transf=obj_transf, obj_pcds=obj_pcds, text=text, rh_mask=torch.ones((1), dtype=torch.float32), lh_mask=torch.zeros((1), dtype=torch.float32), init_obj_rotmat=first_frame_rot)
            label_iter = dict(rh_mask=torch.ones((1), dtype=torch.float32), lh_mask=torch.zeros((1), dtype=torch.float32))
        else:
            dummy_rh_transf = torch.eye(4).expand_as(h_transf)
            dummy_rh_poses = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(window_size, 16, 3, 3).clone()
            input_iter = dict(rh_poses=dummy_rh_poses, lh_poses=h_poses, rh_trans=torch.zeros_like(h_trans), lh_trans=h_trans, rh_verts=torch.zeros_like(h_verts), lh_verts=h_verts, rh_joints=torch.zeros_like(h_joints), lh_joints=h_joints, rh_transf=dummy_rh_transf, lh_transf=h_transf, obj_transf=obj_transf, obj_pcds=obj_pcds, text=text, rh_mask=torch.zeros((1), dtype=torch.float32), lh_mask=torch.ones((1), dtype=torch.float32), init_obj_rotmat=first_frame_rot)
            label_iter = dict(rh_mask=torch.zeros((1), dtype=torch.float32), lh_mask=torch.ones((1), dtype=torch.float32))
        meta_iter = dict(id=sample_key, obj_id=obj_mesh_path, motion_length=window_size)
    
        if mode == 'train':
            return input_iter, label_iter, meta_iter
        elif mode == 'test':
            if self.cfg.task in ["autoreg_flow"]:
                input_iter["rh_mask"] = torch.ones((1), dtype=torch.float32)
                input_iter["lh_mask"] = torch.zeros((1), dtype=torch.float32)
            return input_iter, meta_iter

    def make_latent_sample(self, sample, mode='train'):
        meta = sample["meta.json"]
        sample_id = meta["id"]
        text = meta["text"]
        obj_path = meta["obj_id"]
        obj_name = obj_path.split('/')[-2]

        init_obj_rotmat = torch.from_numpy(np.asarray(sample["init_obj_rotmat.npy"], dtype=np.float32))
        obj_pcds = torch.from_numpy(np.asarray(self.object_pcds[obj_name], dtype=np.float32))
        obj_pcds = (init_obj_rotmat @ obj_pcds.transpose(1, 0)).transpose(1, 0)

        try:
            rh_obj_mean = torch.from_numpy(sample["rh_obj_mean.npy"])
            rh_mean = torch.from_numpy(sample["rh_mean.npy"])
            lh_obj_mean = rh_obj_mean
            lh_mean = rh_mean
            rh_mask = torch.ones((1), dtype=torch.float32)
            lh_mask = torch.ones((1), dtype=torch.float32)
        except:
            lh_obj_mean = torch.from_numpy(sample["lh_obj_mean.npy"])
            lh_mean = torch.from_numpy(sample["lh_mean.npy"])
            rh_obj_mean = lh_obj_mean
            rh_mean = lh_mean
            rh_mask = torch.ones((1), dtype=torch.float32)
            lh_mask = torch.ones((1), dtype=torch.float32)

        if mode == 'train':
            input_iter = dict(obj_pcds=obj_pcds, text=text)
            label_iter = dict(rh_obj_mean=rh_obj_mean, lh_obj_mean=lh_obj_mean, rh_mean=rh_mean, lh_mean=lh_mean, rh_mask=rh_mask, lh_mask=lh_mask)
            meta_iter = dict(id=sample_id, obj_id=obj_path, motion_length=self.cfg.window_size)
            return input_iter, label_iter, meta_iter
