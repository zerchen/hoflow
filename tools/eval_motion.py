#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File        :eval.py
#@Date        :2022/07/19 10:32:37
#@Author      :zerui chen
#@Contact     :zerui.chen@inria.fr

import os
import sys
import argparse
import yaml
import json
import numpy as np
from tqdm import tqdm
import torch
from loguru import logger
import _init_paths
from _init_paths import add_path, this_dir
import threading
import concurrent.futures
from glob import glob
from torch.nn.functional import pdist
from math import sqrt, isnan
import pandas as pd
import trimesh
import shutil
from motion_metrics.physic_metrics import PhysicEvaluation


def pad_with_last(feat, max_frame, dim=2):
        nf = feat.shape[0]
        if nf >= max_frame:
            feat = feat[:max_frame, ...]
        else:
            if dim == 2:
                feat = np.pad(feat, ((0, max_frame-nf), (0, 0)), 'constant', constant_values=0)
            elif dim == 3 :
                feat = np.pad(feat, ((0, max_frame-nf), (0, 0), (0, 0)), 'constant', constant_values=0)
            feat[nf:, ...] = feat[nf-1, ...]
        return feat    


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-e', required=True, type=str)
    parser.add_argument('--num_proc', default=10, type=int)
    parser.add_argument('--gt', action='store_true')
    args = parser.parse_args()

    return args


_THREAD_LOCAL = threading.local()


def _get_thread_context():
    if not hasattr(_THREAD_LOCAL, 'evaluator'):
        _THREAD_LOCAL.evaluator = PhysicEvaluation()
        _THREAD_LOCAL.rh_faces = np.load(os.path.join('..', 'common', 'mano', 'assets', 'rhand_faces.npy'))
        _THREAD_LOCAL.lh_faces = np.load(os.path.join('..', 'common', 'mano', 'assets', 'lhand_faces.npy'))
        _THREAD_LOCAL.obj_mesh_cache = {}
    return _THREAD_LOCAL.evaluator, _THREAD_LOCAL.rh_faces, _THREAD_LOCAL.lh_faces, _THREAD_LOCAL.obj_mesh_cache


def _get_object_mesh_template(obj_mesh_path, obj_mesh_cache):
    if obj_mesh_path in obj_mesh_cache:
        return obj_mesh_cache[obj_mesh_path]

    obj_mesh = trimesh.load(obj_mesh_path, process=False)
    num_faces = len(obj_mesh.faces)
    if num_faces > 1000:
        target_face_count = 1000
    elif num_faces > 4:
        target_face_count = max(4, int(num_faces * 0.5))
    else:
        target_face_count = None

    if target_face_count is not None and target_face_count < num_faces:
        obj_mesh = obj_mesh.simplify_quadric_decimation(face_count=target_face_count)
    trimesh.repair.fix_normals(obj_mesh)
    obj_can_verts = np.array(obj_mesh.vertices, dtype=np.float32)
    obj_faces = np.array(obj_mesh.faces, dtype=np.int32)
    obj_mesh_cache[obj_mesh_path] = (obj_can_verts, obj_faces)
    return obj_can_verts, obj_faces


def _to_optional_array(json_data, key):
    value = json_data.get(key, None)
    if value is None:
        return None
    return np.asarray(value, dtype=np.float32)


def _build_diversity_feature(json_data):
    pred_rh_verts = _to_optional_array(json_data, "pred_rh_verts")
    pred_lh_verts = _to_optional_array(json_data, "pred_lh_verts")
    active_hands = int(pred_rh_verts is not None) + int(pred_lh_verts is not None)
    if active_hands == 0:
        raise ValueError("Missing both pred_rh_verts and pred_lh_verts in prediction json.")

    pred_rh_verts = torch.from_numpy(pred_rh_verts)[::10] if pred_rh_verts is not None else None
    pred_lh_verts = torch.from_numpy(pred_lh_verts)[::10] if pred_lh_verts is not None else None

    if pred_rh_verts is None:
        pred_rh_verts = torch.zeros_like(pred_lh_verts)
    if pred_lh_verts is None:
        pred_lh_verts = torch.zeros_like(pred_rh_verts)

    T = min(pred_rh_verts.shape[0], pred_lh_verts.shape[0])
    pred_rh_verts = pred_rh_verts[:T]
    pred_lh_verts = pred_lh_verts[:T]
    feat = torch.cat([pred_rh_verts.reshape(T, -1), pred_lh_verts.reshape(T, -1)], dim=-1)
    return feat, T, active_hands


def _safe_mean(values):
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))


def _evaluate_motion(pred_motion_path):
    evaluator, rh_faces, lh_faces, obj_mesh_cache = _get_thread_context()

    with open(pred_motion_path, 'r') as f:
        json_data = json.load(f)

    sample_idx = os.path.splitext(os.path.basename(pred_motion_path))[0]
    rh_verts = _to_optional_array(json_data, "pred_rh_verts")
    lh_verts = _to_optional_array(json_data, "pred_lh_verts")
    if rh_verts is None:
        raise ValueError(f"Missing pred_rh_verts in prediction json: {pred_motion_path}")

    obj_mesh_path = json_data['obj_mesh_path']
    obj_can_verts, obj_faces = _get_object_mesh_template(obj_mesh_path, obj_mesh_cache)
    init_obj_rotmat = np.array(json_data['init_obj_rotmat'], dtype=np.float32)
    obj_can_verts = np.einsum('ij,nj->ni', init_obj_rotmat, obj_can_verts)

    obj_rot = np.array(json_data['pred_obj_rotmat'], dtype=np.float32)
    obj_trans = np.array(json_data['pred_obj_trans'], dtype=np.float32)
    obj_verts = np.einsum('tij,vj->tvi', obj_rot, obj_can_verts) + obj_trans[:, None, :]

    result = evaluator(
        obj_verts,
        obj_faces,
        rh_verts,
        rh_faces,
        lh_verts,
        lh_faces if lh_verts is not None else None,
    )
    error_dict = {}
    error_dict['id'] = sample_idx
    
    metric_keys = [
        ('rhand', 'inter_volume_contact', 1e6),
        ('lhand', 'inter_volume_contact', 1e6),
        ('rhand', 'inter_depth_contact', 1e2),
        ('lhand', 'inter_depth_contact', 1e2),
        ('rhand', 'contact_ratio_contact', 1e2),
        ('lhand', 'contact_ratio_contact', 1e2),
    ]
    
    for hand_key, metric_key, multiplier in metric_keys:
        error_dict_key = f"{hand_key}_{metric_key}"
        if hand_key in result and metric_key in result[hand_key]:
            error_dict[error_dict_key] = result[hand_key][metric_key] * multiplier
        else:
            error_dict[error_dict_key] = None
    ivu_list = []
    for hand_key in ["rhand", "lhand"]:
        hand_result = result.get(hand_key, {})
        hand_surface_area = hand_result.get("hand_surface_area_mean", None)
        hand_contact_ratio = hand_result.get("contact_ratio_mean", None)
        hand_inter_volume = hand_result.get("inter_volume_mean", None)
        if hand_surface_area is None or hand_contact_ratio is None or hand_inter_volume is None:
            continue
        hand_contact_region = hand_surface_area * hand_contact_ratio
        hand_ivu = hand_inter_volume / hand_contact_region if hand_contact_region > 0 else 0
        ivu_list.append(hand_ivu)
    error_dict['IVU'] = (sum(ivu_list) / len(ivu_list) * 1e2) if len(ivu_list) > 0 else None
    error_dict['Phy'] = result["off_ground_contact_rate"] * 1e2

    return error_dict


def main():
    # argument parse and create log
    args = parse_args()

    if not args.dir.endswith(".json"):
        testset = args.dir.strip('/').split('/')[-1].split('_')[1]
        exec(f'from datasets.{testset}.{testset} import {testset}')
        with open(os.path.join(args.dir, '../exp.yaml'), 'r') as f:
            cfg = yaml.safe_load(f)

    txt2motion = {}
    all_preds_filenames = glob(os.path.join(args.dir, '*.json'))
    for filename in all_preds_filenames:
        if testset == 'grab':
            obj_name = filename.split('/')[-1].split('.')[0].split("-")[1].split('_')[0]
            verb = filename.split('/')[-1].split('.')[0].split("-")[1].split('_')[1]
            text = f"{verb} the {obj_name}"
            if text not in txt2motion:
                txt2motion[text] = []
            txt2motion[text].append(filename)
        if testset == 'oakink':
            obj_name = filename.split('/')[-1].split('.')[0].split('_')[0]
            verb = filename.split('/')[-1].split('.')[0].split('-')[1]
            text = f"{verb} the {obj_name}"
            if text not in txt2motion:
                txt2motion[text] = []
            txt2motion[text].append(filename)
        if testset == 'dexycb':
            with open(filename, 'r') as f:
                data = json.load(f)
            obj_name = data['obj_mesh_path'].split('/')[-2]
            verb = "grasp"
            text = f"{verb} the {obj_name}"
            if text not in txt2motion:
                txt2motion[text] = []
            txt2motion[text].append(filename)

    diversity_per_txt = {}
    all_feat = []
    all_active_hands = []
    for text, file_list in txt2motion.items():
        feat = []
        text_active_hands = []
        for filename in file_list:
            with open(filename, 'r') as f:
                data = json.load(f)
            cur_feat, T, active_hands = _build_diversity_feature(data)
            feat.append(cur_feat)
            text_active_hands.append(active_hands)
            all_active_hands.append(active_hands)
        
        feat = torch.stack(feat, dim=0)
        repeat = feat.shape[0]
        feat = feat.reshape(repeat, -1)
        all_feat.append(feat)

        num_hands_for_norm = max(text_active_hands) if len(text_active_hands) > 0 else 1
        diversity_per_txt[text] = torch.mean(pdist(feat, p=2)) / sqrt(T * 778 * num_hands_for_norm)

    all_feat = torch.cat(all_feat, dim=0)
    overall_num_hands_for_norm = max(all_active_hands) if len(all_active_hands) > 0 else 1
    over_all_div = torch.mean(pdist(all_feat, p=2)).item() / sqrt(T * 778 * overall_num_hands_for_norm)
    sd_list = list(diversity_per_txt.values())
    sample_diversity = float(sum(sd_list) / len(sd_list))

    summary = []
    num_proc = max(1, int(args.num_proc))
    if num_proc == 1:
        for sample in tqdm(all_preds_filenames):
            error_dict = _evaluate_motion(sample)
            summary.append(tuple(error_dict.values()))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_proc) as executor:
            for error_dict in tqdm(executor.map(_evaluate_motion, all_preds_filenames), total=len(all_preds_filenames)):
                summary.append(tuple(error_dict.values()))

    summary = sorted(summary, reverse=False, key=lambda result: result[0])
    summary_filename = f"eval_result_{testset}_{cfg['testset_split']}.txt"

    output_summary_filename = os.path.join(args.dir, '..', summary_filename)
    with open(output_summary_filename, "w") as f:
        eval_result = [[] for i in range(8)]
        name_list = ['sample_id', 'rhand_IV', 'lhand_IV', 'rhand_ID', 'lhand_ID', 'rhand_CR', 'lhand_CR', 'IVU', 'Phy']
        data_list = []
        for idx, result in enumerate(summary):
            data_sample = [result[0]]
            for i in range(8):
                if result[i + 1] is not None:
                    eval_result[i].append(result[i + 1])
                    data_sample.append(round(result[i + 1], 3))
                else:
                    data_sample.append(result[i + 1])
            data_list.append(data_sample)
        f.write(pd.DataFrame(data_list, columns=name_list, index=[''] * len(summary), dtype=str).to_string())
        f.write('\n')

        for idx, _ in enumerate(eval_result):
            new_array = []
            for number in eval_result[idx]:
                if not np.isnan(number):
                    new_array.append(number)
            eval_result[idx] = new_array

        mean_rhand_IV = "mean rhand IV: {}\n".format(_safe_mean(eval_result[0]))
        mean_lhand_IV = "mean lhand IV: {}\n".format(_safe_mean(eval_result[1]))
        mean_rhand_ID = "mean rhand ID: {}\n".format(_safe_mean(eval_result[2]))
        mean_lhand_ID = "mean lhand ID: {}\n".format(_safe_mean(eval_result[3]))
        mean_rhand_CR = "mean rhand CR: {}\n".format(_safe_mean(eval_result[4]))
        mean_lhand_CR = "mean lhand CR: {}\n".format(_safe_mean(eval_result[5]))
        mean_IVU = "mean IVU: {}\n".format(_safe_mean(eval_result[6]))
        mean_Phy = "mean Phy: {}\n".format(_safe_mean(eval_result[7]))
        mean_SD = "mean SD: {}\n".format(sample_diversity)
        mean_OD = "mean OD: {}\n".format(over_all_div)

        print(mean_rhand_IV); f.write(mean_rhand_IV)
        print(mean_lhand_IV); f.write(mean_lhand_IV)
        print(mean_rhand_ID); f.write(mean_rhand_ID)
        print(mean_lhand_ID); f.write(mean_lhand_ID)
        print(mean_rhand_CR); f.write(mean_rhand_CR)
        print(mean_lhand_CR); f.write(mean_lhand_CR)
        print(mean_IVU); f.write(mean_IVU)
        print(mean_Phy); f.write(mean_Phy)
        print(mean_SD); f.write(mean_SD)
        print(mean_OD); f.write(mean_OD)


if __name__ == "__main__":
    main()
