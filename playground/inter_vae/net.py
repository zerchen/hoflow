#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File :net.py
#@Date :2025/08/19 09:36:56
#@Author :Zerui Chen
#@Contact :zeruicasia@gmail.com

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.random_utils import np_random
from mano.rot6d import rotation_6d_to_matrix
from mano.manolayer import ManoLayer
from networks.ae.seq_pose_vae import ObjectCentricMotionAE
from utils.rotation import rotmat2aa
from kornia.geometry.conversions import rotation_matrix_to_axis_angle


class model(nn.Module):
    """
    Object-Centric Hand-Object Motion Model.
    
    Key improvements over the original architecture:
    1. Object is encoded and decoded independently (not from hand predictions)
    2. Hand decoder is conditioned on object latent
    3. Object pose is predicted ONCE, eliminating the need for averaging
    4. Compatible with both single-hand and bimanual manipulation
    """
    def __init__(self, cfg):
        super(model, self).__init__()
        self.cfg = cfg
        self.motion_ae = ObjectCentricMotionAE(cfg)
        self.mano_right = ManoLayer(flat_hand_mean=True, ncomps=45, side='right', mano_root='../common/mano/assets/', use_pca=False)
        self.l1_loss = torch.nn.L1Loss(reduction='none')
        self.np_random, _ = np_random()

    def flip_batch(self, inputs):
        """Flip left hand to right hand coordinate system"""
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
        """Flip hand predictions back to left hand coordinate system"""
        flip_vec = torch.tensor([-1., 1., 1.], device=device, dtype=dtype)
        batch_size = rec_h_poses.shape[0]
        
        rec_lh_poses = rec_h_poses.clone()
        rec_lh_poses = rec_lh_poses.reshape(batch_size, seq_len, 16, 3)
        rec_lh_poses[..., 1] = -rec_lh_poses[..., 1]  # Negate Y component
        rec_lh_poses[..., 2] = -rec_lh_poses[..., 2]  # Negate Z component
        rec_lh_poses = rec_lh_poses.reshape(batch_size, seq_len, 48)
        rec_lh_trans = rec_h_trans * flip_vec
        rec_lh_verts = rec_h_verts * flip_vec
        rec_lh_joints = rec_h_joints * flip_vec
        
        return rec_lh_poses, rec_lh_trans, rec_lh_verts, rec_lh_joints

    def flip_obj_back(self, rec_obj_rotmat, rec_obj_trans, device, dtype):
        """Flip object predictions back from flipped coordinate system"""
        flip_vec = torch.tensor([-1., 1., 1.], device=device, dtype=dtype)
        mirr_mat = torch.diag(flip_vec)
        
        rec_obj_rotmat_flipped = torch.einsum("ij,btjk,kl->btil", mirr_mat, rec_obj_rotmat, mirr_mat)
        rec_obj_trans_flipped = rec_obj_trans * flip_vec
        
        return rec_obj_rotmat_flipped, rec_obj_trans_flipped

    def precompute(self, inputs):
        with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
            latent = {}
            with torch.no_grad():
                if inputs['rh_mask'].item():
                    _, rh_obj_mu, rh_obj_logvar, _, rh_mu, rh_logvar = self.motion_ae.encode(inputs, sample=True)
                    latent['rh_obj_mean'] = rh_obj_mu[0].detach().cpu()
                    latent['rh_obj_std'] = torch.exp(0.5 * rh_obj_logvar)[0].detach().cpu()
                    latent['rh_mean'] = rh_mu[0].detach().cpu()
                    latent['rh_std'] = torch.exp(0.5 * rh_logvar)[0].detach().cpu()
                if inputs['lh_mask'].item():
                    flipped_inputs = self.flip_batch(inputs)
                    _, lh_obj_mu, lh_obj_logvar, _, lh_mu, lh_logvar = self.motion_ae.encode(flipped_inputs, sample=True)
                    latent['lh_obj_mean'] = lh_obj_mu[0].detach().cpu()
                    latent['lh_obj_std'] = torch.exp(0.5 * lh_obj_logvar)[0].detach().cpu()
                    latent['lh_mean'] = lh_mu[0].detach().cpu()
                    latent['lh_std'] = torch.exp(0.5 * lh_logvar)[0].detach().cpu()
                
            return latent

    def forward(self, inputs, targets=None, metas=None, mode='train'):
        # Prepare inputs: flip left hand to right hand format
        flipped_inputs = self.flip_batch(inputs)
        
        # Collect right hand and flipped-left hand sample masks
        rh_mask = inputs['rh_mask'][:, 0].bool()
        lh_mask = flipped_inputs['rh_mask'][:, 0].bool()  # This is lh_mask after flip
        
        # Keep split inputs for test-time reconstruction
        rh_inputs = {k: v[rh_mask] for k, v in inputs.items() if k in flipped_inputs}
        lh_inputs = {k: v[lh_mask] for k, v in flipped_inputs.items()}
        
        # Get counts
        n_rh = rh_inputs['obj_transf'].shape[0] if rh_mask.any() else 0
        n_lh = lh_inputs['obj_transf'].shape[0] if lh_mask.any() else 0
        seq_len = inputs['obj_transf'].shape[1]
        device = inputs['obj_transf'].device
        dtype = inputs['obj_transf'].dtype

        if mode == 'train':
            # Train on a single right-hand-format batch:
            # [original right-hand samples] + [flipped left-hand samples]
            merged_inputs = {k: torch.cat([rh_inputs[k], lh_inputs[k]], dim=0) for k in lh_inputs}
            train_mask = merged_inputs['rh_mask']
            train_mask = train_mask.any(dim=1) if train_mask.ndim > 1 else train_mask
            train_mask = train_mask.bool()
            train_inputs = {k: v[train_mask] for k, v in merged_inputs.items()}
            n_train = int(train_mask.sum().item())
            return self._forward_train(train_inputs, n_train, seq_len)
        elif mode == 'test':
            return self._forward_test(inputs, rh_inputs, lh_inputs, n_rh, n_lh, seq_len, device, dtype, metas)

    def _forward_train(self, train_inputs, n_train, seq_len):
        """Training forward pass with VAE (includes KL loss)"""
        loss = {}
        motion_results = {}
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            if n_train > 0:
                outputs = self.motion_ae(train_inputs, sample=True)  # VAE mode
                loss, motion_results = self._compute_loss(outputs, train_inputs, n_train, seq_len)
            else:
                raise ValueError(f"No training samples found for batch size {n_train}")
        
        return loss, motion_results

    def _compute_loss(self, outputs, cur_inputs, batch_size, seq_len):
        rec_hand = outputs['rec_hand']  # [B, T, 99]
        rec_obj = outputs['rec_obj']    # [B, T, 9]
        
        # Parse hand predictions
        rec_rh_poses = rotation_6d_to_matrix(rec_hand[:, :, :96].reshape((-1, 6)))
        rec_rh_delta_trans = rec_hand[:, :, 96:99]  # relative to object

        # Parse object predictions (predicted ONCE from object branch)
        rec_obj_rotmat = rotation_6d_to_matrix(rec_obj[:, :, :6].reshape((-1, 6)))
        rec_obj_trans = rec_obj[:, :, 6:9]

        # Recover absolute hand translation for mesh loss
        rec_rh_trans = rec_rh_delta_trans + rec_obj_trans

        # Ground truth
        gt_obj_rot_6d = cur_inputs['obj_transf'][:, :, :2, :3].reshape(batch_size, seq_len, -1)
        gt_obj_trans = cur_inputs['obj_transf'][:, :, :3, 3]
        gt_rh_delta_trans = cur_inputs['rh_trans'] - gt_obj_trans  # relative to object
        gt_rh_verts = cur_inputs['rh_verts']
        gt_rh_poses_6d = cur_inputs['rh_poses'][..., :2, :3].reshape(batch_size, seq_len, -1)

        loss = {}
        loss["pose"] = self.l1_loss(rec_hand[:, :, :96], gt_rh_poses_6d).mean()
        loss["trans"] = self.l1_loss(rec_rh_delta_trans, gt_rh_delta_trans).mean()

        # Mesh loss
        rec_rh_verts, _, _, _, _ = self.mano_right(rotmat2aa(rec_rh_poses.reshape(-1, 1, 16, 9)).reshape(-1, 48), th_trans=rec_rh_trans.reshape(-1, 3))
        rec_rh_verts = rec_rh_verts.reshape((batch_size, seq_len, 778, 3))
        loss["mesh"] = self.l1_loss(rec_rh_verts, gt_rh_verts).mean()

        # Object losses
        loss["obj_rot"] = self.l1_loss(rec_obj[:, :, :6], gt_obj_rot_6d).mean() * 2
        loss["obj_trans"] = self.l1_loss(rec_obj_trans, gt_obj_trans).mean() * 2
        
        # VAE: KL divergence loss
        # KL(q(z|x) || p(z)) where p(z) = N(0, I)
        # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        obj_mu = outputs['obj_mu']  # [B, D, T']
        obj_logvar = outputs['obj_logvar']  # [B, D, T']
        hand_mu = outputs['hand_mu']  # [B, D, T']
        hand_logvar = outputs['hand_logvar']  # [B, D, T']
        
        # Object KL: sum over dimensions, mean over batch and time
        obj_kl = -0.5 * torch.sum(1 + obj_logvar - obj_mu.pow(2) - obj_logvar.exp(), dim=1)  # [B, T']
        loss["obj_kl"] = obj_kl.mean() * self.cfg.kl_weight
        
        # Hand KL: sum over dimensions, mean over batch and time
        hand_kl = -0.5 * torch.sum(1 + hand_logvar - hand_mu.pow(2) - hand_logvar.exp(), dim=1)  # [B, T']
        loss["hand_kl"] = hand_kl.mean() * self.cfg.kl_weight
        
        results = {'rec_rh_poses': rec_rh_poses, 'rec_rh_trans': rec_rh_trans, 'rec_obj_rotmat': rec_obj_rotmat, 'rec_obj_trans': rec_obj_trans}
        
        return loss, results

    def _forward_test(self, inputs, rh_inputs, lh_inputs, n_rh, n_lh, seq_len, device, dtype, metas):
        """Test forward pass - object predicted once, no averaging needed"""
        with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
            # Initialize outputs
            rec_rh_poses = rec_rh_trans = rec_rh_verts = rec_rh_joints = None
            rec_lh_poses = rec_lh_trans = rec_lh_verts = rec_lh_joints = None
            rec_obj_rotmat = rec_obj_trans = None
            flip_vec = torch.tensor([-1., 1., 1.], device=device, dtype=dtype)
            
            # Process right hand if present
            if n_rh > 0:
                rh_outputs = self.motion_ae(rh_inputs, sample=False)  # Use mean for inference
                rec_rh_hand = rh_outputs['rec_hand'].float()
                rec_rh_obj = rh_outputs['rec_obj'].float()

                # Parse object first (needed to recover absolute hand trans)
                rec_obj_rotmat = rotation_6d_to_matrix(rec_rh_obj[:, :, :6].reshape((-1, 6))).reshape((n_rh, seq_len, 3, 3))
                rec_obj_trans = rec_rh_obj[:, :, 6:9]

                # Parse hand (network predicts relative trans, recover absolute)
                rec_rh_poses = rotation_matrix_to_axis_angle(rotation_6d_to_matrix(rec_rh_hand[:, :, :96].reshape((-1, 6)))).reshape((n_rh, seq_len, 48))
                rec_rh_trans = rec_rh_hand[:, :, 96:99] + rec_obj_trans

                rec_rh_verts, rec_rh_joints, _, _, _ = self.mano_right(rec_rh_poses.reshape(-1, 48), th_trans=rec_rh_trans.reshape(-1, 3))
                rec_rh_verts = rec_rh_verts.reshape((n_rh, seq_len, 778, 3))
                rec_rh_joints = rec_rh_joints.reshape((n_rh, seq_len, 21, 3))
            
            # Process left hand if present
            if n_lh > 0:
                lh_outputs = self.motion_ae(lh_inputs, sample=False)  # Use mean for inference
                rec_lh_hand = lh_outputs['rec_hand'].float()
                rec_lh_obj = lh_outputs['rec_obj'].float()

                # Parse object in flipped coordinates
                rec_obj_rotmat_lh = rotation_6d_to_matrix(rec_lh_obj[:, :, :6].reshape((-1, 6))).reshape((n_lh, seq_len, 3, 3))
                rec_obj_trans_lh = rec_lh_obj[:, :, 6:9]

                # Parse hand (in flipped coordinates, network predicts relative trans)
                rec_h_poses = rotation_matrix_to_axis_angle(rotation_6d_to_matrix(rec_lh_hand[:, :, :96].reshape((-1, 6)))).reshape((n_lh, seq_len, 48))
                # Keep left-hand translation consistent with exported object trajectory.
                if rec_obj_trans is not None:
                    shared_obj_trans_flipped = rec_obj_trans * flip_vec
                else:
                    shared_obj_trans_flipped = rec_obj_trans_lh
                rec_h_trans = rec_lh_hand[:, :, 96:99] + shared_obj_trans_flipped

                rec_h_verts, rec_h_joints, _, _, _ = self.mano_right(rec_h_poses.reshape(-1, 48), th_trans=rec_h_trans.reshape(-1, 3))
                rec_h_verts = rec_h_verts.reshape((n_lh, seq_len, 778, 3))
                rec_h_joints = rec_h_joints.reshape((n_lh, seq_len, 21, 3))

                # Flip hand back to left hand coordinates
                rec_lh_poses, rec_lh_trans, rec_lh_verts, rec_lh_joints = self.flip_hand_back(rec_h_poses, rec_h_trans, rec_h_verts, rec_h_joints, seq_len, device, dtype)

                # If no right hand, use object from left hand branch (flipped back)
                if rec_obj_rotmat is None:
                    rec_obj_rotmat, rec_obj_trans = self.flip_obj_back(rec_obj_rotmat_lh, rec_obj_trans_lh, device, dtype)
            
            sample_id = metas['id'][0]
            pred_result = {}
            pred_result['id'] = sample_id
            pred_result['obj_mesh_path'] = metas['obj_id'][0]
            pred_result['pred_obj_rotmat'] = rec_obj_rotmat[0].detach().cpu().numpy().tolist()
            pred_result['pred_obj_trans'] = rec_obj_trans[0].detach().cpu().numpy().tolist()
            pred_result['gt_obj_rotmat'] = inputs['obj_transf'][:, :, :3, :3][0].cpu().numpy().tolist()
            pred_result['gt_obj_trans'] = inputs['obj_transf'][:, :, :3, 3][0].cpu().numpy().tolist()
            pred_result['init_obj_rotmat'] = inputs['init_obj_rotmat'][0].cpu().numpy().tolist()
            if rec_rh_poses is not None:
                pred_result.update(
                    {
                        'pred_rh_poses': rec_rh_poses[0].detach().cpu().numpy().tolist(),
                        'pred_rh_trans': rec_rh_trans[0].detach().cpu().numpy().tolist(),
                        'pred_rh_verts': rec_rh_verts[0].detach().cpu().numpy().tolist(),
                        'pred_rh_joints': rec_rh_joints[0].detach().cpu().numpy().tolist(),
                        'gt_rh_verts': inputs['rh_verts'][0].cpu().numpy().tolist(),
                        'gt_rh_joints': inputs['rh_joints'][0].cpu().numpy().tolist(),
                    }
                )
            
            if rec_lh_poses is not None:
                pred_result.update(
                    {
                        'pred_lh_poses': rec_lh_poses[0].detach().cpu().numpy().tolist(),
                        'pred_lh_trans': rec_lh_trans[0].detach().cpu().numpy().tolist(),
                        'pred_lh_verts': rec_lh_verts[0].detach().cpu().numpy().tolist(),
                        'pred_lh_joints': rec_lh_joints[0].detach().cpu().numpy().tolist(),
                        'gt_lh_verts': inputs['lh_verts'][0].cpu().numpy().tolist(),
                        'gt_lh_joints': inputs['lh_joints'][0].cpu().numpy().tolist(),
                    }
                )
            
            with open(os.path.join(self.cfg.result_dir, f"{sample_id}.json"), 'w') as f:
                json.dump(pred_result, f)


def get_model(cfg, is_train):
    ho_model = model(cfg)
    return ho_model
