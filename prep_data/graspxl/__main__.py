import argparse
import json
import os
import sys
import torch
import trimesh
import tarfile
import pickle
import numpy as np
from glob import glob
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
sys.path.insert(0, 'common')
from mano.manolayer import ManoLayer
from ..converter import SceneConverterBase
from .processor import load_motion
from kornia.geometry.conversions import rotation_matrix_to_axis_angle


# ============================================================================
# Data Validation Functions (integrated from diagnose_graspxl.py)
# ============================================================================

class DataValidationConfig:
    """Configuration for data validation thresholds"""
    # Rotation checks
    MAX_ROTATION_ANGLE = 3.5  # radians (~200 degrees)
    ROTATION_ORTHO_TOLERANCE = 1e-2
    GIMBAL_LOCK_THRESHOLD = 0.999
    
    # Translation checks
    MAX_HAND_TRANSLATION = 5.0  # meters
    MAX_OBJECT_TRANSLATION = 10.0  # meters
    
    # Trajectory checks
    MAX_TRAJECTORY_JUMP = 0.5  # meters per frame
    
    # Value checks
    MAX_EXTREME_VALUE = 100.0


def check_rotation_matrix_validity(rotmat, tolerance=1e-2):
    """Check if rotation matrix is valid (orthogonal, det=1)"""
    # Check orthogonality: R @ R.T should be identity
    identity_check = rotmat @ rotmat.T
    ortho_error = np.abs(identity_check - np.eye(3)).max()
    if ortho_error > tolerance:
        return False, f"non_orthogonal(err={ortho_error:.4f})"
    
    # Check determinant (should be 1 for proper rotation)
    det = np.linalg.det(rotmat)
    if abs(det - 1.0) > tolerance:
        return False, f"bad_det({det:.4f})"
    
    return True, None


def check_rotation_near_gimbal_lock(rotmat, threshold=0.999):
    """Check if rotation is near gimbal lock"""
    if abs(rotmat[2, 0]) > threshold:
        return False, f"gimbal_lock(R[2,0]={rotmat[2, 0]:.4f})"
    return True, None


def validate_sample_data(rh_poses, rh_trans, rh_verts, rh_joints, rh_transf, obj_transf, 
                         config=None, verbose=False):
    """
    Comprehensive validation of sample data.
    
    Returns:
        is_valid: bool - True if sample passes all validation checks
        issues: list - List of issue descriptions if invalid
    """
    if config is None:
        config = DataValidationConfig()
    
    issues = []
    
    # 1. Check for NaN/Inf values
    arrays_to_check = {
        "rh_poses": rh_poses,
        "rh_trans": rh_trans,
        "rh_verts": rh_verts,
        "rh_joints": rh_joints,
        "rh_transf": rh_transf,
        "obj_transf": obj_transf,
    }
    
    for name, arr in arrays_to_check.items():
        if arr is None:
            issues.append(f"missing:{name}")
            continue
        if np.isnan(arr).any():
            issues.append(f"nan:{name}")
        if np.isinf(arr).any():
            issues.append(f"inf:{name}")
        if np.abs(arr).max() > config.MAX_EXTREME_VALUE:
            issues.append(f"extreme_value:{name}(max={np.abs(arr).max():.2f})")
    
    if issues:
        return False, issues
    
    # 2. Check rotation angles (axis-angle representation)
    if rh_poses is not None:
        # rh_poses contains axis-angle rotations
        # Check global rotation (first 3 values) and local rotations
        pose_norms = np.linalg.norm(rh_poses.reshape(-1, 3), axis=-1)
        if pose_norms.max() > config.MAX_ROTATION_ANGLE:
            issues.append(f"extreme_rotation:rh_poses(max={pose_norms.max():.4f}rad)")
    
    # 3. Check translation magnitudes
    if rh_trans is not None:
        trans_mag = np.linalg.norm(rh_trans, axis=-1)
        if trans_mag.max() > config.MAX_HAND_TRANSLATION:
            issues.append(f"large_translation:rh_trans(max={trans_mag.max():.2f}m)")
    
    if obj_transf is not None:
        obj_trans_mag = np.linalg.norm(obj_transf[:, :3, 3], axis=-1)
        if obj_trans_mag.max() > config.MAX_OBJECT_TRANSLATION:
            issues.append(f"large_translation:obj_transf(max={obj_trans_mag.max():.2f}m)")
    
    # 4. Check trajectory jumps (frame-to-frame displacement)
    if rh_trans is not None and len(rh_trans) > 1:
        trans_diff = np.diff(rh_trans, axis=0)
        trans_jump = np.linalg.norm(trans_diff, axis=-1)
        if trans_jump.max() > config.MAX_TRAJECTORY_JUMP:
            issues.append(f"trajectory_jump:rh_trans(max={trans_jump.max():.4f}m)")
    
    if obj_transf is not None and len(obj_transf) > 1:
        obj_trans_diff = np.diff(obj_transf[:, :3, 3], axis=0)
        obj_jump = np.linalg.norm(obj_trans_diff, axis=-1)
        if obj_jump.max() > config.MAX_TRAJECTORY_JUMP:
            issues.append(f"trajectory_jump:obj_transf(max={obj_jump.max():.4f}m)")
    
    # 5. Check rotation matrix validity (for transformation matrices)
    if obj_transf is not None:
        # Check first and last frames of object transform
        for t in [0, len(obj_transf) - 1]:
            rotmat = obj_transf[t, :3, :3]
            valid, msg = check_rotation_matrix_validity(rotmat, config.ROTATION_ORTHO_TOLERANCE)
            if not valid:
                issues.append(f"obj_rotation:frame{t}:{msg}")
            
            valid, msg = check_rotation_near_gimbal_lock(rotmat, config.GIMBAL_LOCK_THRESHOLD)
            if not valid:
                issues.append(f"gimbal_lock:obj_frame{t}:{msg}")
    
    if rh_transf is not None:
        # Check first frame of hand transforms (checking all would be too slow)
        for j in range(min(16, rh_transf.shape[1])):
            rotmat = rh_transf[0, j, :3, :3]
            valid, msg = check_rotation_matrix_validity(rotmat, config.ROTATION_ORTHO_TOLERANCE)
            if not valid:
                issues.append(f"hand_rotation:joint{j}:{msg}")
                break  # One bad joint is enough to flag the sample
    
    is_valid = len(issues) == 0
    
    if verbose and not is_valid:
        print(f"  Validation issues: {issues}")
    
    return is_valid, issues


# ============================================================================
# Canonical Transform Computation
# ============================================================================

def compute_canonical_transform_graspxl(rh_verts, rh_joints):
    """
    Compute z-axis rotation to place hand in natural right-hand manipulation pose.
    
    Coordinate system:
    - Origin: center of table
    - Person stands at y < 0 (behind table, looking toward +y)
    - Right hand is at x > 0
    
    Target canonical pose for right hand manipulation:
    - Wrist at y <= 0 (on person's side of table)
    - Hand center at x >= 0 (right side)
    - Palm faces -z (down for top grasp) or -x (toward body midline for side grasp)
    - Thumb position consistent with right hand (not flipped to look like left hand)
    
    ONLY uses z-axis rotation (around vertical axis) because:
    - Z-rotation = person walking around the table, completely natural
    - X/Y rotation = unnatural arm twisting that would be impossible
    
    Args:
        rh_verts: (T, N, 3) MANO hand mesh vertices (778 vertices per frame), numpy array
        rh_joints: (T, 21, 3) MANO hand joint positions, numpy array
    
    Returns:
        rotation_matrix: (3, 3) z-axis rotation to apply to all vertices, numpy array
    """
    init_hand = rh_verts[0]  # (778, 3) for MANO mesh
    hand_center = init_hand.mean(axis=0)
    
    # Get joint positions from first frame
    # MANO joint indices: 0=wrist, 4=thumb_tip, 8=index_tip, 12=middle_tip, 16=ring_tip, 20=pinky_tip
    wrist = rh_joints[0, 0]
    thumb_tip = rh_joints[0, 4]
    pinky_tip = rh_joints[0, 20]
    
    # Compute palm normal using PCA (shortest axis = palm thickness direction)
    centered = init_hand - hand_center
    cov = centered.T @ centered
    _, eigenvectors = np.linalg.eigh(cov)
    palm_normal = eigenvectors[:, 0]
    
    # Determine correct sign of palm normal (should point from dorsal to palm)
    projections = centered @ palm_normal
    pos_count = (projections > projections.std()).sum()
    neg_count = (projections < -projections.std()).sum()
    if pos_count > neg_count:
        palm_normal = -palm_normal
    
    # Target directions for palm facing
    target_palm_down = np.array([0.0, 0.0, -1.0])  # Palm facing down (-z)
    target_palm_inward = np.array([-1.0, 0.0, 0.0])  # Palm facing toward body midline (-x)
    
    best_rotation = np.eye(3)
    best_score = float('-inf')
    
    # Only z-axis rotations (0°, 90°, 180°, 270°)
    # This is equivalent to the person standing at different positions around the table
    for angle_z in [0, np.pi/2, np.pi, 3*np.pi/2]:
        cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
        rot_z = np.array([
            [cos_z, -sin_z, 0],
            [sin_z, cos_z, 0],
            [0, 0, 1]
        ])
        
        # Apply rotation
        rotated_wrist = rot_z @ wrist
        rotated_center = rot_z @ hand_center
        rotated_palm_normal = rot_z @ palm_normal
        rotated_thumb_tip = rot_z @ thumb_tip
        rotated_pinky_tip = rot_z @ pinky_tip
        
        # Score this configuration
        score = 0.0
        
        # 1. Wrist position: y <= 0 (person's side of table)
        if rotated_wrist[1] <= 0:
            score += 20.0
        else:
            score -= 40.0  # Heavy penalty - must be on person's side
        
        # 2. Hand center: x >= 0 (right side for right hand)
        if rotated_center[0] >= 0:
            score += 15.0
        else:
            score -= 20.0  # Penalty for left side
        
        # 3. Palm facing: -z (down) or -x (toward body midline)
        palm_down_align = np.dot(rotated_palm_normal, target_palm_down)
        palm_inward_align = np.dot(rotated_palm_normal, target_palm_inward)
        best_palm_align = max(palm_down_align, palm_inward_align * 0.8)
        score += best_palm_align * 10.0
        
        # 4. Right hand constraint: ensure thumb is on correct side
        # For right hand with palm down (-z): thumb should be toward +y (away from person)
        # For right hand with palm inward (-x): thumb should be toward -z (down)
        if palm_down_align > 0.5:
            # Palm facing down: thumb should be at higher y than pinky
            if rotated_thumb_tip[1] > rotated_pinky_tip[1]:
                score += 15.0  # Correct right hand orientation
            else:
                score -= 30.0  # Would look like left hand
        elif palm_inward_align > 0.5:
            # Palm facing inward: thumb should be at lower z than pinky
            if rotated_thumb_tip[2] < rotated_pinky_tip[2]:
                score += 15.0  # Correct right hand orientation
            else:
                score -= 30.0  # Would look like left hand
        
        if score > best_score:
            best_score = score
            best_rotation = rot_z.copy()
    
    return best_rotation


class GraspXLConverter(SceneConverterBase):
    def __init__(self, input_dir: str, output_dir: str, max_sample_per_shard: int = 100, group_seq: bool = False, skip_validation: bool = False, verbose: bool = False, resume: bool = False):
        super().__init__(dataset_name="graspxl", input_dir=input_dir, output_dir=output_dir, max_sample_per_shard=max_sample_per_shard)
        self.group_seq = group_seq
        self.rh_rot_center = np.array([0.0957,  0.0064,  0.0062], dtype=np.float32)
        self.mano_right = ManoLayer(flat_hand_mean=False, side="right", mano_root='common/mano/assets/', use_pca=False).cuda()
        
        # Validation settings
        self.skip_validation = skip_validation
        self.verbose = verbose
        self.validation_config = DataValidationConfig()
        
        # Resume settings
        self.resume = resume
        self.processed_ids = set()
        self.checkpoint_path = os.path.join(output_dir, ".processed_ids.pkl")
        
        if self.resume:
            self._load_processed_ids()
        
        # Statistics tracking
        self.stats = {
            "total_processed": 0,
            "valid_samples": 0,
            "invalid_samples": 0,
            "skipped_samples": 0,  # Due to missing files, mesh issues, etc.
            "resumed_skipped": 0,  # Skipped due to already processed
            "issue_counts": defaultdict(int),
            "invalid_sample_ids": [],
        }
    
    def _load_processed_ids(self):
        """Load processed IDs from checkpoint file or scan existing shards"""
        # First try to load from checkpoint file (faster)
        if os.path.exists(self.checkpoint_path):
            print(f"Loading checkpoint from {self.checkpoint_path}...")
            with open(self.checkpoint_path, 'rb') as f:
                self.processed_ids = pickle.load(f)
            print(f"  Loaded {len(self.processed_ids)} processed IDs from checkpoint")
            self._setup_shard_continuation()
            return
        
        # Fall back to scanning existing shards
        print("Scanning existing shards for processed IDs...")
        shard_pattern = os.path.join(self.output_dir, "*.tar")
        shard_files = sorted(glob(shard_pattern))
        
        if not shard_files:
            print("  No existing shards found")
            return
        
        for shard_file in tqdm(shard_files, desc="Scanning shards"):
            try:
                with tarfile.open(shard_file, 'r') as tar:
                    for member in tar.getmembers():
                        # Extract sample ID from metadata json files
                        if member.name.endswith('.meta.json'):
                            f = tar.extractfile(member)
                            if f:
                                try:
                                    meta = json.loads(f.read().decode('utf-8'))
                                    sample_id = meta.get('sample_id')
                                    if sample_id:
                                        self.processed_ids.add(sample_id)
                                except:
                                    pass
            except Exception as e:
                print(f"  Warning: Could not read shard {shard_file}: {e}")
        
        print(f"  Found {len(self.processed_ids)} already processed sequence IDs")
        
        # Save checkpoint for faster future resumption
        self._save_checkpoint()
        
        # Setup shard manager to continue from last shard
        self._setup_shard_continuation()
    
    def _setup_shard_continuation(self):
        """Configure shard manager to continue from last existing shard"""
        shard_pattern = os.path.join(self.output_dir, "*.tar")
        shard_files = sorted(glob(shard_pattern))
        
        if not shard_files:
            return
        
        # Find the last shard index
        last_shard = shard_files[-1]
        # Extract shard index from filename like "graspxl-002832.tar"
        shard_name = os.path.basename(last_shard)
        try:
            shard_idx = int(shard_name.split('-')[-1].replace('.tar', ''))
            # Start from next shard (the last one might be incomplete, but we'll skip processed IDs anyway)
            self.shard_manager.shard_index = shard_idx + 1
            print(f"  Will continue from shard index {self.shard_manager.shard_index}")
        except Exception as e:
            print(f"  Warning: Could not parse shard index from {shard_name}: {e}")
    
    def _save_checkpoint(self):
        """Save processed IDs to checkpoint file"""
        with open(self.checkpoint_path, 'wb') as f:
            pickle.dump(self.processed_ids, f)
    
    def _update_checkpoint(self, sample_id: str):
        """Add a sample ID to processed set and periodically save checkpoint"""
        self.processed_ids.add(sample_id)
        # Save checkpoint every 1000 samples
        if len(self.processed_ids) % 1000 == 0:
            self._save_checkpoint()

    def _process_seq_folder(self, seq_path: str, seq_id: str):
        if self.verbose:
            print(f"\nProcessing scene {seq_id}")
        result = self._process_sample(seq_path, seq_id)
        if result == "valid":
            if self.verbose:
                print(f"✓ Successfully processed scene {seq_id}")
        elif result == "invalid":
            if self.verbose:
                print(f"✗ Skipped invalid scene {seq_id}")
        else:
            if self.verbose:
                print(f"- Skipped scene {seq_id} (missing data)")

    def _process_sample(self, seq: Path, seq_id: str):
        # Skip if already processed (resume mode)
        if self.resume and seq_id in self.processed_ids:
            self.stats["resumed_skipped"] += 1
            return "resumed"
        
        self.stats["total_processed"] += 1
        
        rh_init_poses, rh_trans, obj_transf, obj_name = load_motion(seq)
        
        obj_size = seq_id.split("-")[1]
        obj_mesh_path = self.input_dir / "object_dataset" / obj_size / obj_name / f"{obj_name}.obj"
        try:
            obj_mesh = trimesh.load(obj_mesh_path, process=False)
            points, _ = trimesh.sample.sample_surface(obj_mesh, 1024)
        except:
            self.stats["skipped_samples"] += 1
            return "skipped"

        if points is None or len(obj_mesh.vertices) < 50 or len(points) < 1024:
            self.stats["skipped_samples"] += 1
            return "skipped"
        
        # Validate input data from processor
        if rh_init_poses is None or rh_trans is None or obj_transf is None or obj_name is None:
            self.stats["skipped_samples"] += 1
            return "skipped"
        
        if len(rh_init_poses) == 0 or len(rh_trans) == 0 or len(obj_transf) == 0:
            self.stats["skipped_samples"] += 1
            return "skipped"
        
        if np.isnan(rh_init_poses).any() or np.isnan(rh_trans).any() or np.isnan(obj_transf).any():
            self.stats["invalid_samples"] += 1
            self.stats["issue_counts"]["nan_in_raw_data"] += 1
            self.stats["invalid_sample_ids"].append(seq_id)
            return "invalid"
        
        if np.isinf(rh_init_poses).any() or np.isinf(rh_trans).any() or np.isinf(obj_transf).any():
            self.stats["invalid_samples"] += 1
            self.stats["issue_counts"]["inf_in_raw_data"] += 1
            self.stats["invalid_sample_ids"].append(seq_id)
            return "invalid"

        rh_verts, rh_joints, rh_poses, rh_transf, _ = self.mano_right(torch.from_numpy(rh_init_poses).cuda(), th_trans=torch.from_numpy(rh_trans).cuda())
        rh_verts = rh_verts.detach().cpu().numpy()
        rh_joints = rh_joints.detach().cpu().numpy()
        rh_poses = rh_poses.detach().cpu().numpy()
        rh_transf = rh_transf.detach().cpu().numpy()

        # Basic MANO output validation
        mano_outputs = {
            "rh_verts": rh_verts,
            "rh_joints": rh_joints,
            "rh_poses": rh_poses,
            "rh_transf": rh_transf,
        }
        for name, arr in mano_outputs.items():
            if arr is None or len(arr) == 0:
                self.stats["skipped_samples"] += 1
                return "skipped"
            if np.isnan(arr).any() or np.isinf(arr).any():
                self.stats["invalid_samples"] += 1
                self.stats["issue_counts"][f"nan_inf_in_{name}"] += 1
                self.stats["invalid_sample_ids"].append(seq_id)
                return "invalid"

        # Validate sequence length consistency
        seq_len = len(rh_verts)
        if not all(len(arr) == seq_len for arr in [rh_joints, rh_poses, rh_trans, rh_transf, obj_transf]):
            self.stats["skipped_samples"] += 1
            return "skipped"
        
        # Comprehensive data validation (rotation angles, trajectory jumps, etc.)
        if not self.skip_validation:
            is_valid, issues = validate_sample_data(
                rh_poses=rh_poses,
                rh_trans=rh_trans,
                rh_verts=rh_verts,
                rh_joints=rh_joints,
                rh_transf=rh_transf,
                obj_transf=obj_transf,
                config=self.validation_config,
                verbose=self.verbose
            )
            
            if not is_valid:
                self.stats["invalid_samples"] += 1
                self.stats["invalid_sample_ids"].append(seq_id)
                for issue in issues:
                    # Extract issue type (before the colon)
                    issue_type = issue.split(":")[0] if ":" in issue else issue
                    self.stats["issue_counts"][issue_type] += 1
                return "invalid"

        canonical_rot = compute_canonical_transform_graspxl(rh_verts, rh_joints)
        canonical_rot = canonical_rot.astype(np.float32)
        
        rh_verts = np.einsum('tni,ji->tnj', rh_verts, canonical_rot)
        rh_joints = np.einsum('tji,ki->tjk', rh_joints, canonical_rot)
        rh_trans = rh_joints[:, 0] - self.rh_rot_center
        
        rh_transf_rot = rh_transf[:, :, :3, :3]
        rh_transf_rot = np.einsum('ij,tpjk->tpik', canonical_rot, rh_transf_rot)
        rh_transf[:, :, :3, :3] = rh_transf_rot
        rh_transf[:, :, :3, 3] = rh_joints[:, [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]] - rh_trans[:, None]
        rh_poses[:, :3] = rotation_matrix_to_axis_angle(torch.from_numpy(rh_transf[:, 0, :3, :3])).numpy()
        
        obj_transf_rot = obj_transf[:, :3, :3]
        obj_transf_trans = obj_transf[:, :3, 3]
        obj_transf_rot = np.einsum('ij,tjk->tik', canonical_rot, obj_transf_rot)
        obj_transf_trans = np.einsum('ij,tj->ti', canonical_rot, obj_transf_trans)
        obj_transf[:, :3, :3] = obj_transf_rot
        obj_transf[:, :3, 3] = obj_transf_trans

        # Sample passed all validation - add to dataset
        self.stats["valid_samples"] += 1
        self._update_checkpoint(seq_id)
        
        array_key = "nested_tar_files" if self.group_seq else "flattened_tar_files"
        data_dict = {
            "id": seq_id,
            "obj_name": obj_name,
            "seq_len": len(rh_verts),
            array_key: {
                "rh_verts": rh_verts,
                "rh_joints": rh_joints,
                "rh_poses": rh_poses,
                "rh_trans": rh_trans,
                "rh_transf": rh_transf,
                "obj_transf": obj_transf,
            },
        }
        self.shard_manager.add_sample_from_dict(data_dict)
        return "valid"

    def convert_dataset(self, dataset_dir: Path):
        for mano_folder in ["mano_dataset_1", "mano_dataset_2", "mano_dataset_3"]:
            for size_folder in ["small", "medium", "large"]:
                folder_path = os.path.join(dataset_dir, mano_folder, size_folder)
                if not os.path.exists(folder_path):
                    continue
                for video_folder in os.listdir(folder_path):
                    seq_file_list = glob(os.path.join(dataset_dir, mano_folder, size_folder, video_folder, "*.npy"))
                    for seq_file in seq_file_list:
                        filename = Path(seq_file).name.split(".")[0]
                        seq_id = f"{mano_folder}-{size_folder}-{video_folder}-{filename}"
                        try:
                            self._process_seq_folder(seq_file, seq_id)
                        except Exception as e:
                            if self.verbose:
                                print(f"Error processing {seq_id}: {e}")
                            self.stats["skipped_samples"] += 1
    
    def print_validation_summary(self):
        """Print a summary of validation statistics"""
        print("\n" + "=" * 60)
        print("DATA VALIDATION SUMMARY")
        print("=" * 60)
        
        if self.resume and self.stats['resumed_skipped'] > 0:
            print(f"\nResumed processing:")
            print(f"  Already processed (skipped): {self.stats['resumed_skipped']}")
        
        print(f"\nTotal samples processed this run: {self.stats['total_processed']}")
        print(f"  Valid samples:   {self.stats['valid_samples']} ({100*self.stats['valid_samples']/max(1, self.stats['total_processed']):.1f}%)")
        print(f"  Invalid samples: {self.stats['invalid_samples']} ({100*self.stats['invalid_samples']/max(1, self.stats['total_processed']):.1f}%)")
        print(f"  Skipped samples: {self.stats['skipped_samples']} ({100*self.stats['skipped_samples']/max(1, self.stats['total_processed']):.1f}%)")
        
        if self.resume:
            total_valid = len(self.processed_ids)
            print(f"\nTotal valid samples (including previous runs): {total_valid}")
        
        if self.stats['issue_counts']:
            print("\nIssue breakdown:")
            for issue_type, count in sorted(self.stats['issue_counts'].items(), key=lambda x: -x[1]):
                print(f"  {issue_type}: {count}")
        
        print("=" * 60)
    
    def save_validation_report(self, output_path: str = None):
        """Save detailed validation report to JSON file"""
        if output_path is None:
            output_path = os.path.join(self.output_dir, "validation_report.json")
        
        report = {
            "summary": {
                "total_processed": self.stats["total_processed"],
                "valid_samples": self.stats["valid_samples"],
                "invalid_samples": self.stats["invalid_samples"],
                "skipped_samples": self.stats["skipped_samples"],
                "valid_percentage": 100 * self.stats["valid_samples"] / max(1, self.stats["total_processed"]),
            },
            "issue_counts": dict(self.stats["issue_counts"]),
            "invalid_sample_ids": self.stats["invalid_sample_ids"][:1000],  # Limit to first 1000
            "validation_config": {
                "max_rotation_angle": self.validation_config.MAX_ROTATION_ANGLE,
                "max_hand_translation": self.validation_config.MAX_HAND_TRANSLATION,
                "max_object_translation": self.validation_config.MAX_OBJECT_TRANSLATION,
                "max_trajectory_jump": self.validation_config.MAX_TRAJECTORY_JUMP,
                "max_extreme_value": self.validation_config.MAX_EXTREME_VALUE,
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nValidation report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="GraspXL Dataset to WebDataset Converter")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True, help="Directory for WebDataset output")
    parser.add_argument("--max_sample_per_shard", type=int,default=2000, help="Maximum number of samples per shard")
    parser.add_argument("--group_seq", action="store_true", help="Group sequences into a single sample")
    parser.add_argument("--skip_validation", action="store_true", 
                        help="Skip comprehensive data validation (faster but may include bad samples)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress")
    parser.add_argument("--save_report", action="store_true", help="Save validation report to JSON file")
    parser.add_argument("--resume", action="store_true", 
                        help="Resume from previous run, skipping already processed samples")
    
    parser.add_argument("--max_rotation_angle", type=float, default=3.5,
                        help="Maximum rotation angle in radians (default: 3.5)")
    parser.add_argument("--max_trajectory_jump", type=float, default=0.5,
                        help="Maximum frame-to-frame displacement in meters (default: 0.5)")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"ERROR: Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    converter = GraspXLConverter(
        input_dir=args.input_dir, 
        output_dir=args.output_dir, 
        max_sample_per_shard=args.max_sample_per_shard, 
        group_seq=args.group_seq,
        skip_validation=args.skip_validation,
        verbose=args.verbose,
        resume=args.resume
    )
    
    # Apply custom validation thresholds if specified
    converter.validation_config.MAX_ROTATION_ANGLE = args.max_rotation_angle
    converter.validation_config.MAX_TRAJECTORY_JUMP = args.max_trajectory_jump
    
    print(f"Starting GraspXL conversion...")
    print(f"  Input:  {args.input_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Resume: {'ENABLED' if args.resume else 'DISABLED'}")
    print(f"  Validation: {'DISABLED' if args.skip_validation else 'ENABLED'}")
    if not args.skip_validation:
        print(f"    - Max rotation angle: {args.max_rotation_angle} rad")
        print(f"    - Max trajectory jump: {args.max_trajectory_jump} m")
    
    converter.convert_dataset(input_path)
    converter.finalize()
    
    # Save final checkpoint
    if args.resume or converter.stats['valid_samples'] > 0:
        converter._save_checkpoint()
        print(f"\nCheckpoint saved: {converter.checkpoint_path}")
    
    converter.print_validation_summary()
    
    if args.save_report:
        converter.save_validation_report()

if __name__ == "__main__":
    main()
