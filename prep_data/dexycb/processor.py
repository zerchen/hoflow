"""Read DexYCB sequence poses and convert them to HO-Flow motion arrays.

Raw pose format: https://github.com/NVlabs/dex-ycb-toolkit
Object-held-out split: LatentHOI's dexycb_preprocessing_all_seq.py.
"""

from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from datasets.dexycb.toolkit.dex_ycb import DexYCBDataset

TEST_OBJECTS = frozenset((
    '002_master_chef_can', '011_banana', '025_mug', '061_foam_brick',
))


def read_sequence(data_root, pose_path, mode):
    """Read one world-coordinate pose.npz; skip the opposite object split."""
    data_root, pose_path = Path(data_root), Path(pose_path)
    with (pose_path.parent / 'meta.yml').open() as stream:
        meta = yaml.safe_load(stream)
    object_index = meta['ycb_grasp_ind']
    object_name = DexYCBDataset.ycb_classes[meta['ycb_ids'][object_index]]
    if (object_name in TEST_OBJECTS) != (mode == 'test'):
        return None
    side = meta['mano_sides'][0]
    if side not in ('right', 'left'):
        raise ValueError(f'Unsupported hand side: {side}')
    mesh_path = data_root / 'models' / object_name / 'textured_simple.obj'
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    calib_path = data_root / 'calibration' / f"mano_{meta['mano_calib'][0]}" / 'mano.yml'
    with calib_path.open() as stream:
        betas = np.asarray(yaml.safe_load(stream)['betas'], dtype=np.float32)
    with np.load(pose_path) as poses:
        hand = np.asarray(poses['pose_m'][:, 0], dtype=np.float32)
        obj = np.asarray(poses['pose_y'][:, object_index], dtype=np.float32)
    if hand.shape != (len(obj), 51) or obj.shape[1:] != (7,):
        raise ValueError(f'Invalid pose shapes in {pose_path}: {hand.shape}, {obj.shape}')
    valid = np.any(hand != 0, axis=1)
    valid &= np.isfinite(hand).all(axis=1) & np.isfinite(obj).all(axis=1)
    valid &= np.linalg.norm(obj[:, :4], axis=1) > 0
    valid &= np.any(obj != [0., 0., 0., 1., 0., 0., 0.], axis=1)
    if not valid.any():
        return None
    hand, obj = hand[valid], obj[valid]
    obj_transf = np.broadcast_to(np.eye(4, dtype=np.float32), (len(obj), 4, 4)).copy()
    # DexYCB pose_y stores quaternions in (x, y, z, w) order, followed by translation.
    obj_transf[:, :3, :3] = Rotation.from_quat(obj[:, :4]).as_matrix()
    obj_transf[:, :3, 3] = obj[:, 4:]
    return {
        'id': '-'.join(pose_path.parent.relative_to(data_root).parts),
        'obj_name': object_name,
        'hand_side': side,
        'hand': hand,
        'betas': betas,
        'obj_transf': obj_transf,
    }


def process_sequence(data):
    """Refit subject-specific PCA hands to the flat, zero-shape MANO model."""
    from prep_data.grab.processor import build_hand_transf, solve_hand

    hand = data['hand']
    poses, trans, verts, joints, rotations = solve_hand(
        45, len(hand), hand[:, 3:48], data['betas'], hand[:, 48:], hand[:, :3],
        data['hand_side'], flat_hand_mean=False,
    )
    obj_transf = data['obj_transf'].copy()
    origin = obj_transf[0, :3, 3].copy()
    trans -= origin
    verts -= origin
    joints -= origin
    obj_transf[:, :3, 3] -= origin
    transforms = build_hand_transf(rotations, joints, trans)
    object_label = ' '.join(data['obj_name'].split('_')[1:])
    return {
        'id': data['id'],
        'obj_name': data['obj_name'],
        'hand_side': data['hand_side'],
        'seq_len': len(hand),
        'text': f'grasp the {object_label}',
        'nested_tar_files': {
            'h_poses': poses,
            'h_trans': trans,
            'h_verts': verts,
            'h_joints': joints,
            'h_transf': transforms,
            'obj_transf': obj_transf,
        },
    }
