"""Process GRAB sequences into per-sequence samples (seq tars only)."""

import os
import sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot

ROOT = os.path.dirname(os.path.abspath(__file__))
MANO_ROOT = os.path.join(ROOT, "..", "..", "common", "mano", "assets")
sys.path.insert(0, os.path.join(ROOT, "..", "..", "common"))

NON_TIP = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
RH_CTR = np.float32([0.0957, 0.0064, 0.0062])
LH_CTR = np.float32([-0.0957, 0.0064, 0.0062])
TEST_OBJ = ["apple", "mug", "train", "elephant"]
AXIS = np.diag(np.float32([-1, -1, 1]))
AXIS4 = np.eye(4, dtype=np.float32)
AXIS4[:3, :3] = AXIS


def prepare_params(params, mask):
    return {k: v[mask].astype(np.float32) for k, v in params.items()}


def filter_contact_frames(seq_data):
    mask = (seq_data["contact"].item()["object"] > 0).any(axis=1)
    s, e = 0, len(mask) - 1
    while s < len(mask) and not mask[s]:
        s += 1
    while e > 0 and not mask[e]:
        e -= 1
    mask[s:e] = True
    return mask


def solve_hand(n_pca, T, pca, beta, trans, rot, side, flat_hand_mean=True):
    from mano.manolayer import ManoLayer

    with torch.no_grad():
        pca_layer = ManoLayer(
            flat_hand_mean=flat_hand_mean,
            ncomps=n_pca,
            side=side,
            mano_root=MANO_ROOT,
            use_pca=True,
        ).cuda()
        gt_verts, _, gt_poses, _, _ = pca_layer(
            torch.from_numpy(np.concatenate([rot, pca], 1)).cuda(),
            torch.from_numpy(beta).cuda().unsqueeze(0).expand(T, -1),
            torch.from_numpy(trans).cuda(),
        )
    gt_poses = gt_poses.cpu()

    layer = ManoLayer(
        flat_hand_mean=True, side=side, mano_root=MANO_ROOT, use_pca=False
    ).cuda()
    g_rot = torch.tensor(gt_poses[:, :3].numpy(), requires_grad=True, device="cuda")
    f_rot = torch.tensor(gt_poses[:, 3:].numpy(), requires_grad=True, device="cuda")
    g_trans = torch.tensor(trans, requires_grad=True, device="cuda")
    opt = torch.optim.Adam(
        [
            {"params": g_trans, "lr": 0.01},
            {"params": f_rot, "lr": 0.01},
            {"params": g_rot, "lr": 0.01},
        ]
    )
    sched = torch.optim.lr_scheduler.StepLR(opt, 200, 0.5)
    for _ in range(400):
        v, _, _, _, _ = layer(
            th_pose_coeffs=torch.cat([g_rot, f_rot], 1), th_trans=g_trans
        )
        loss = ((v - gt_verts) ** 2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
        sched.step()

    with torch.no_grad():
        g_rot, f_rot, g_trans = g_rot.detach(), f_rot.detach(), g_trans.detach()
        v, j, p, tf, _ = layer(
            th_pose_coeffs=torch.cat([g_rot, f_rot], 1), th_trans=g_trans
        )
    return (
        p.cpu().numpy(),
        g_trans.cpu().numpy(),
        v.cpu().numpy(),
        j.cpu().numpy(),
        tf[:, :, :3, :3].cpu().numpy(),
    )


def build_hand_transf(global_rot, joints_21, trans):
    """Build (T, 16, 4, 4) per-joint transforms from global rotations and joints."""
    T = global_rot.shape[0]
    tf = np.zeros((T, 16, 4, 4), dtype=np.float32)
    tf[:, :, :3, :3] = global_rot
    tf[:, :, :3, 3] = joints_21[:, NON_TIP] - trans[:, None]
    tf[:, :, 3, 3] = 1.0
    return tf


def sequence_file_meta(data_root, sbj, fn):
    """
    Lightweight check: load only meta + contact, no MANO. Use to skip files
    before calling process_sequence_file.

    Returns:
        (obj_name, intent, T) if file has at least 1 contact frame and is valid, else None.
    """
    seq_path = os.path.join(data_root, "data", sbj, fn)
    try:
        seq = np.load(seq_path, allow_pickle=True)
    except Exception:
        return None
    obj_name = seq["obj_name"].item()
    intent = seq["motion_intent"].item()
    mask = filter_contact_frames(seq)
    T = int(mask.sum())
    if T < 1:
        return None
    return (obj_name, intent, T)


def process_sequence_file(data_root, sbj, fn, mode):
    """
    Load one GRAB sequence file and return a list of sequence samples (one per stride).

    Each sample is a dict suitable for ShardManager.add_sample_from_dict with
    nested_tar_files (seq-level arrays). Returns [] if file should be skipped.

    Call sequence_file_meta() first to avoid running this when the file would be skipped.
    """
    seq_path = os.path.join(data_root, "data", sbj, fn)
    seq = np.load(seq_path, allow_pickle=True)
    obj_name = seq["obj_name"].item()
    intent = seq["motion_intent"].item()

    if (mode == "train" and obj_name in TEST_OBJ) or (
        mode == "test" and obj_name not in TEST_OBJ
    ):
        return []

    mask = filter_contact_frames(seq)
    T = int(mask.sum())
    if T < 1:
        return []

    rh_p = prepare_params(seq["rhand"].item()["params"], mask)
    lh_p = prepare_params(seq["lhand"].item()["params"], mask)
    ob_p = prepare_params(seq["object"].item()["params"], mask)

    obj_trans = ob_p["transl"]
    obj_rot = Rot.from_rotvec(ob_p["global_orient"]).as_matrix().transpose(0, 2, 1)

    n_pca = seq["n_comps"].item()
    gender = seq["gender"].item()
    rh_beta = np.load(
        os.path.join(data_root, "data", gender, f"{sbj}_rhand_betas.npy")
    ).astype(np.float32)
    lh_beta = np.load(
        os.path.join(data_root, "data", gender, f"{sbj}_lhand_betas.npy")
    ).astype(np.float32)
    sid = f"{sbj}-{fn.split('.')[0]}"

    # Solve both hands
    rh_po, rh_tr, rh_vt, rh_jt, rh_gr = solve_hand(
        n_pca, T, rh_p["hand_pose"], rh_beta,
        rh_p["transl"], rh_p["global_orient"], "right",
    )
    lh_po, lh_tr, lh_vt, lh_jt, lh_gr = solve_hand(
        n_pca, T, lh_p["hand_pose"], lh_beta,
        lh_p["transl"], lh_p["global_orient"], "left",
    )

    # Axis-flip global rotation
    rh_po[:, :3] = Rot.from_matrix(
        AXIS[None] @ Rot.from_rotvec(rh_po[:, :3]).as_matrix()
    ).as_rotvec()
    lh_po[:, :3] = Rot.from_matrix(
        AXIS[None] @ Rot.from_rotvec(lh_po[:, :3]).as_matrix()
    ).as_rotvec()

    # Center + flip translation
    rh_tr = (AXIS @ (rh_tr - obj_trans[[0]]).T).T + AXIS @ RH_CTR - RH_CTR
    lh_tr = (AXIS @ (lh_tr - obj_trans[[0]]).T).T + AXIS @ LH_CTR - LH_CTR

    # Center + flip joints/verts
    for arr in [rh_jt, rh_vt, lh_jt, lh_vt]:
        arr -= obj_trans[[0]][:, None]
        arr[:, :, :2] *= -1.0

    # Per-joint global rotation after axis flip + build transforms
    rh_gr = AXIS[None, None] @ rh_gr
    lh_gr = AXIS[None, None] @ lh_gr
    rh_tf = build_hand_transf(rh_gr, rh_jt, rh_tr)
    lh_tf = build_hand_transf(lh_gr, lh_jt, lh_tr)

    # Object transform
    obj_tc = obj_trans - obj_trans[[0]]
    obj_tf = np.zeros((T, 4, 4), dtype=np.float32)
    obj_tf[:, :3, :3] = obj_rot
    obj_tf[:, :3, 3] = obj_tc
    obj_tf[:, 3, 3] = 1.0
    obj_tf = AXIS4[None] @ obj_tf

    # Stride-6 subsampling → one sequence sample per stride
    samples = []
    for t in range(6):
        s = slice(t, None, 6)
        sl = int(rh_po[s].shape[0])
        if sl < 1:
            continue
        key = f"{sid}-{t}"
        samples.append({
            "id": key,
            "obj_name": obj_name,
            "seq_len": sl,
            "motion_intent": intent,
            "nested_tar_files": {
                "rh_poses": rh_po[s],
                "lh_poses": lh_po[s],
                "rh_trans": rh_tr[s],
                "lh_trans": lh_tr[s],
                "obj_transf": obj_tf[s],
                "rh_verts": rh_vt[s],
                "lh_verts": lh_vt[s],
                "rh_joints": rh_jt[s],
                "lh_joints": lh_jt[s],
                "rh_transf": rh_tf[s],
                "lh_transf": lh_tf[s],
            },
        })
    return samples
