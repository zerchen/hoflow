#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File :net.py
#@Date :2025/08/19 09:36:56
#@Author :Zerui Chen
#@Contact :zeruicasia@gmail.com

import torch
import torch.nn as nn
import numpy as np
import os
import json
import clip
import copy
from bps_torch.bps import bps_torch
from mano.manolayer import ManoLayer
from mano.rot6d import rotation_6d_to_matrix
from networks.ae.seq_pose_vae import ObjectCentricMotionAE
from networks.mar.MARDM import MARDM_models
from kornia.geometry.conversions import rotation_matrix_to_axis_angle, axis_angle_to_rotation_matrix


class model(nn.Module):
    def __init__(self, cfg):
        super(model, self).__init__()
        self.cfg = cfg

        self.mano_right = ManoLayer(flat_hand_mean=True, ncomps=45, side='right', mano_root='../common/mano/assets/', use_pca=False)
        self.bps = bps_torch(custom_basis=torch.from_numpy(np.load('../common/mano/assets/bps.npz')['basis']).to(torch.float32))
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device="cuda")
        self.clip_model.eval()
        for _, param in self.clip_model.named_parameters():
            param.requires_grad = False
        
        self.motion_ae = ObjectCentricMotionAE(cfg)
        self.motion_ae.eval()
        for _, param in self.motion_ae.named_parameters():
            param.requires_grad = False

        self.mardm = MARDM_models[cfg.arch](ae_dim=cfg.mar_output_dim, cond_mode='text', num_layers=cfg.mar_num_layers, dropout=cfg.mar_dropout, cond_drop_prob=cfg.mar_cond_drop_prob)
        
        self.ema_mardm = None
        if hasattr(cfg, 'use_ema') and cfg.use_ema:
            self._create_ema_model()
    
    def _create_ema_model(self):
        self.ema_mardm = copy.deepcopy(self.mardm)
        self.ema_mardm.eval()
        for param in self.ema_mardm.parameters():
            param.requires_grad_(False)
    
    def get_ema_mardm(self):
        return self.ema_mardm
    
    def update_ema(self, ema_decay):
        if self.ema_mardm is not None:
            with torch.no_grad():
                for ema_param, model_param in zip(self.ema_mardm.parameters(), self.mardm.parameters()):
                    ema_param.data.mul_(ema_decay).add_(model_param.data, alpha=(1 - ema_decay))

    def flip_batch(self, inputs):
        device = inputs['obj_transf'].device
        dtype = inputs['obj_transf'].dtype
        mirr_mat = torch.diag(torch.tensor([-1., 1., 1.], device=device, dtype=dtype))
        flip_vec = torch.tensor([-1., 1., 1.], device=device, dtype=dtype)
        
        flip_inputs = {
            'rh_poses': torch.einsum("ij,btnjk,kl->btnil", mirr_mat, inputs['lh_poses'], mirr_mat),
            'rh_trans': inputs['lh_trans'] * flip_vec,
            'rh_verts': inputs['lh_verts'] * flip_vec,
            'rh_joints': inputs['lh_joints'] * flip_vec,
            'obj_pcds': inputs['obj_pcds'] * flip_vec,
            'rh_mask': inputs['lh_mask']
        }
        
        flip_inputs['rh_transf'] = inputs['lh_transf'].clone()
        flip_inputs['rh_transf'][..., :3, :3] = torch.einsum("ij,btnjk,kl->btnil", mirr_mat, inputs['lh_transf'][..., :3, :3], mirr_mat)
        flip_inputs['rh_transf'][..., :3, 3] = inputs['lh_transf'][..., :3, 3] * flip_vec
        
        flip_inputs['obj_transf'] = inputs['obj_transf'].clone()
        flip_inputs['obj_transf'][:, :, :3, :3] = torch.einsum("ij,btjk,kl->btil", mirr_mat, inputs['obj_transf'][:, :, :3, :3], mirr_mat)
        flip_inputs['obj_transf'][:, :, :3, 3] = inputs['obj_transf'][:, :, :3, 3] * flip_vec
        
        return flip_inputs

    def flip_hand_back(self, rec_h_poses, rec_h_trans, rec_h_verts, rec_h_joints, seq_len, device, dtype):
        flip_vec = torch.tensor([-1., 1., 1.], device=device, dtype=dtype)
        batch_size = rec_h_poses.shape[0]

        rec_lh_poses = rec_h_poses.clone().reshape(batch_size, seq_len, 16, 3)
        rec_lh_poses[..., 1] = -rec_lh_poses[..., 1]
        rec_lh_poses[..., 2] = -rec_lh_poses[..., 2]
        rec_lh_poses = rec_lh_poses.reshape(batch_size, seq_len, 48)
        rec_lh_trans = rec_h_trans * flip_vec
        rec_lh_verts = rec_h_verts * flip_vec
        rec_lh_joints = rec_h_joints * flip_vec

        return rec_lh_poses, rec_lh_trans, rec_lh_verts, rec_lh_joints

    def forward(self, inputs, targets=None, metas=None, mode='train'):
        if mode == 'train':
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                text_input = inputs["text"]
                pred_results = {}

                obj_bps = self.bps.encode(inputs['obj_pcds'], feature_type='dists')['dists']

                with torch.no_grad():
                    clip_embedding = clip.tokenize(text_input).cuda()
                    text_feats = self.clip_model.encode_text(clip_embedding)

                rh_obj_mu = targets["rh_obj_mean"]
                rh_mu = targets["rh_mean"]
                lh_obj_mu = targets["lh_obj_mean"]
                lh_mu = targets["lh_mean"]
                seq_len = metas["motion_length"].long()
                
                latent_feats = torch.cat([rh_obj_mu, rh_mu, lh_obj_mu, lh_mu], dim=1)
                pred_results["latent_feats"] = latent_feats

                y = {}
                y["clip"] = text_feats
                y["bps"] = obj_bps

                loss = {}
                loss["token"] = self.mardm.forward_loss(latent_feats, y, seq_len // 4)

                return loss, pred_results
        elif mode == 'test':
            with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
                text_input = inputs["text"]
                seq_len = self.cfg.test_window_size
                token_length = metas["motion_length"].long() // 4

                obj_bps = self.bps.encode(inputs['obj_pcds'], feature_type='dists')['dists']
                clip_embedding = clip.tokenize(text_input).cuda()

                text_feats = self.clip_model.encode_text(clip_embedding)
                y = {}
                y["clip"] = text_feats
                y["bps"] = obj_bps

                mardm_model = self.ema_mardm if self.ema_mardm is not None else self.mardm
                pred_latents = mardm_model.generate(y, token_length, self.cfg.time_steps, self.cfg.cfg_weight, temperature=self.cfg.temperature, hard_pseudo_reorder=self.cfg.hard_pseudo_reorder)

                batch_size = pred_latents.shape[0]
                latent_dim = self.cfg.latent_dim
                
                pred_rh_z_obj = pred_latents[:, :latent_dim, :]
                pred_rh_z_hand = pred_latents[:, latent_dim:2*latent_dim, :]
                pred_lh_z_obj = pred_latents[:, 2*latent_dim:3*latent_dim, :]
                pred_lh_z_hand = pred_latents[:, 3*latent_dim:, :]

                rec_rh_obj, rec_rh_hand = self.motion_ae.decode(pred_rh_z_obj, pred_rh_z_hand)
                rec_lh_obj, rec_lh_hand = self.motion_ae.decode(pred_lh_z_obj, pred_lh_z_hand)

                rec_rh_obj_rot_6d = rec_rh_obj[:, :, :6]
                rec_rh_obj_rotmat = rotation_6d_to_matrix(rec_rh_obj_rot_6d.reshape(-1, 6)).reshape(batch_size, seq_len, 3, 3)
                rec_rh_obj_trans = rec_rh_obj[:, :, 6:9]
                rec_rh_poses_6d = rec_rh_hand[:, :, :96]

                rec_lh_obj_rot_6d = rec_lh_obj[:, :, :6]
                rec_lh_obj_rotmat = rotation_6d_to_matrix(rec_lh_obj_rot_6d.reshape(-1, 6)).reshape(batch_size, seq_len, 3, 3)
                rec_lh_obj_trans = rec_lh_obj[:, :, 6:9]
                rec_lh_poses_6d = rec_lh_hand[:, :, :96]

                flip_vec = torch.tensor([-1., 1., 1.], device=pred_latents.device, dtype=pred_latents.dtype)
                has_rh = bool((inputs["rh_mask"].sum() > 0).item())
                has_lh = bool((inputs["lh_mask"].sum() > 0).item())
                if has_rh and has_lh:
                    rec_obj_trans = rec_rh_obj_trans
                    rec_obj_rotmat = rec_rh_obj_rotmat
                    rec_rh_trans = rec_rh_hand[:, :, 96:99] + rec_obj_trans
                    rec_lh_trans = rec_lh_hand[:, :, 96:99] + rec_obj_trans * flip_vec
                elif has_rh:
                    rec_obj_trans = rec_rh_obj_trans
                    rec_obj_rotmat = rec_rh_obj_rotmat
                    rec_rh_trans = rec_rh_hand[:, :, 96:99] + rec_obj_trans
                elif has_lh:
                    mirr_mat = torch.diag(flip_vec)
                    rec_obj_trans = rec_lh_obj_trans * flip_vec
                    rec_obj_rotmat = torch.einsum("ij,btjk,kl->btil", mirr_mat, rec_lh_obj_rotmat, mirr_mat)
                    rec_lh_trans = rec_lh_hand[:, :, 96:99] + rec_lh_obj_trans

                sample_id = metas['id'][0]
                pred_result = {}
                pred_result['id'] = sample_id
                pred_result['pred_obj_trans'] = rec_obj_trans[0].detach().cpu().numpy().tolist()
                pred_result['pred_obj_rotmat'] = rec_obj_rotmat[0].detach().cpu().numpy().tolist()
                pred_result['init_obj_rotmat'] = inputs['init_obj_rotmat'][0].cpu().numpy().tolist()
                pred_result['obj_mesh_path'] = metas['obj_id'][0]

                if has_rh:
                    rec_rh_poses = rotation_matrix_to_axis_angle(rotation_6d_to_matrix(rec_rh_poses_6d.reshape((-1, 6)))).reshape((batch_size, seq_len, 48))
                    rec_rh_verts, rec_rh_joints, _, _, _ = self.mano_right(rec_rh_poses.reshape(-1, 48), th_trans=rec_rh_trans.reshape(-1, 3))
                    rec_rh_verts = rec_rh_verts.reshape((batch_size, seq_len, 778, 3))
                    rec_rh_joints = rec_rh_joints.reshape((batch_size, seq_len, 21, 3))

                    pred_result['pred_rh_poses'] = rec_rh_poses[0].detach().cpu().numpy().tolist()
                    pred_result['pred_rh_trans'] = rec_rh_trans[0].detach().cpu().numpy().tolist()
                    pred_result['pred_rh_verts'] = rec_rh_verts[0].detach().cpu().numpy().tolist()
                    pred_result['pred_rh_joints'] = rec_rh_joints[0].detach().cpu().numpy().tolist()
                if has_lh:
                    rec_lh_poses = rotation_matrix_to_axis_angle(rotation_6d_to_matrix(rec_lh_poses_6d.reshape((-1, 6)))).reshape((batch_size, seq_len, 48))
                    rec_lh_verts, rec_lh_joints, _, _, _ = self.mano_right(rec_lh_poses.reshape(-1, 48), th_trans=rec_lh_trans.reshape(-1, 3))
                    rec_lh_verts = rec_lh_verts.reshape((batch_size, seq_len, 778, 3))
                    rec_lh_joints = rec_lh_joints.reshape((batch_size, seq_len, 21, 3))
                    rec_lh_poses, rec_lh_trans, rec_lh_verts, rec_lh_joints = self.flip_hand_back(rec_lh_poses, rec_lh_trans, rec_lh_verts, rec_lh_joints, seq_len, rec_rh_obj.device, rec_rh_obj.dtype)

                    pred_result['pred_lh_poses'] = rec_lh_poses[0].detach().cpu().numpy().tolist()
                    pred_result['pred_lh_trans'] = rec_lh_trans[0].detach().cpu().numpy().tolist()
                    pred_result['pred_lh_verts'] = rec_lh_verts[0].detach().cpu().numpy().tolist()
                    pred_result['pred_lh_joints'] = rec_lh_joints[0].detach().cpu().numpy().tolist()

                with open(os.path.join(self.cfg.result_dir, f"{sample_id}.json"), 'w') as f:
                    json.dump(pred_result, f)


def get_model(cfg, is_train):
    ho_model = model(cfg)
    ae_ckpt = torch.load(cfg.ae_ckpt, map_location=torch.device('cpu'))['network']
    new_ae_ckpt = {}
    for k, v in ae_ckpt.items():
        if 'mano' not in k:
            new_k = k.replace('module.motion_ae.', '').replace('module.', '')
            new_ae_ckpt[new_k] = v
    ho_model.motion_ae.load_state_dict(new_ae_ckpt)
    print('Load ObjectCentricMotionAE checkpoint from {}'.format(cfg.ae_ckpt))

    return ho_model
