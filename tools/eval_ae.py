#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File        :eval_ae.py
#@Date        :2022/07/19 10:32:37
#@Author      :zerui chen
#@Contact     :zerui.chen@inria.fr

import os
import argparse
import json
import numpy as np
import yaml
import torch
from tqdm import tqdm
from loguru import logger
from glob import glob


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-e', required=True, type=str, help='Directory containing prediction JSON files')
    args = parser.parse_args()

    return args


def eval_seq_ae(json_path):
    import trimesh
    from pytorch3d.loss import chamfer_distance

    with open(json_path, 'r') as f:
        json_data = json.load(f)
    sample_id = json_data["id"]

    try:
        pred_rh_verts = np.array(json_data["pred_rh_verts"], dtype=np.float32)
        pred_rh_joints = np.array(json_data["pred_rh_joints"], dtype=np.float32)
        gt_rh_verts = np.array(json_data["gt_rh_verts"], dtype=np.float32)
        gt_rh_joints = np.array(json_data["gt_rh_joints"], dtype=np.float32)
        rh_verts_err = np.mean(np.linalg.norm(pred_rh_verts - gt_rh_verts, axis=-1)) * 1000.
        rh_joints_err = np.mean(np.linalg.norm(pred_rh_joints - gt_rh_joints, axis=-1)) * 1000.
    except:
        rh_verts_err = None
        rh_joints_err = None

    try:
        pred_lh_verts = np.array(json_data["pred_lh_verts"], dtype=np.float32)
        pred_lh_joints = np.array(json_data["pred_lh_joints"], dtype=np.float32)
        gt_lh_verts = np.array(json_data["gt_lh_verts"], dtype=np.float32)
        gt_lh_joints = np.array(json_data["gt_lh_joints"], dtype=np.float32)
        lh_verts_err = np.mean(np.linalg.norm(pred_lh_verts - gt_lh_verts, axis=-1)) * 1000.
        lh_joints_err = np.mean(np.linalg.norm(pred_lh_joints - gt_lh_joints, axis=-1)) * 1000.
    except:
        lh_verts_err = None
        lh_joints_err = None

    obj_mesh = trimesh.load(json_data["obj_mesh_path"], process=False)
    obj_verts = trimesh.sample.sample_surface_even(obj_mesh, 1024)[0]
    pred_obj_trans = np.array(json_data["pred_obj_trans"], dtype=np.float32)
    pred_obj_rotmat = np.array(json_data["pred_obj_rotmat"], dtype=np.float32)
    pred_obj_verts = np.einsum('tij,nj->tni', pred_obj_rotmat, obj_verts) + pred_obj_trans[:, None]

    gt_obj_trans = np.array(json_data["gt_obj_trans"], dtype=np.float32)
    gt_obj_rotmat = np.array(json_data["gt_obj_rotmat"], dtype=np.float32)
    gt_obj_verts = np.einsum('tij,nj->tni', gt_obj_rotmat, obj_verts) + gt_obj_trans[:, None]

    obj_trans_err = np.mean(np.linalg.norm(pred_obj_trans - gt_obj_trans, axis=-1)) * 1000
    obj_chamfer_err = chamfer_distance(torch.from_numpy(pred_obj_verts).float().cuda(), torch.from_numpy(gt_obj_verts).float().cuda())[0].cpu().numpy().item() * 1e4

    error_dict = {'id': sample_id, 'rh_MPVPE': rh_verts_err, 'rh_MPJPE': rh_joints_err, 'lh_MPVPE': lh_verts_err, 'lh_MPJPE': lh_joints_err, 'obj_trans_err': obj_trans_err, 'obj_chamfer_err': obj_chamfer_err}

    return error_dict


def main():
    args = parse_args()
    with open(os.path.join(args.dir, '../exp.yaml'), 'r') as f:
        cfg = yaml.safe_load(f)
    if cfg.get('task') != 'inter_vae':
        raise ValueError('eval_ae.py only supports inter_vae; use eval_motion.py for generated motions.')

    output_dir = args.dir
    json_files = sorted(glob(os.path.join(output_dir, '*.json')))
    if len(json_files) == 0:
        logger.error(f"No JSON files found in {output_dir}")
        return
    logger.info(f"Found {len(json_files)} samples to evaluate")

    import pandas as pd

    summary = []
    for json_path in tqdm(json_files, desc="Evaluating"):
        error_dict = eval_seq_ae(json_path)
        eval_keys = tuple(error_dict.keys())
        eval_vals = tuple(error_dict.values())
        summary.append(eval_vals)

    summary = sorted(summary, reverse=False, key=lambda result: result[0])
    summary_filename = "eval_result.txt"

    with open(os.path.join(output_dir, '..', summary_filename), "w") as f:
        eval_result = [[] for _ in range(len(eval_keys) - 1)]
        name_list = list(eval_keys)
        data_list = []

        for result in summary:
            data_sample = [result[0]]
            for i in range(len(eval_keys) - 1):
                if result[i + 1] is not None and not np.isnan(result[i + 1]):
                    eval_result[i].append(result[i + 1])
                    data_sample.append(round(result[i + 1], 3))
                else:
                    data_sample.append(None)
            data_list.append(data_sample)

        f.write(pd.DataFrame(data_list, columns=name_list, index=[''] * len(summary), dtype=str).to_string())
        f.write('\n')

        # Filter out NaN values
        for idx in range(len(eval_result)):
            eval_result[idx] = [x for x in eval_result[idx] if not np.isnan(x)]

        for i in range(len(eval_keys) - 1):
            cur_metric = "{}: {:.4f}\n".format(eval_keys[i + 1], np.mean(eval_result[i]) if eval_result[i] else 0)
            print(cur_metric, end=''); f.write(cur_metric)


if __name__ == "__main__":
    main()
