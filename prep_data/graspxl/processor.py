"""Load GraspXL motion arrays from a sequence file."""

import numpy as np
from scipy.spatial.transform import Rotation as Rot


def load_motion(path):
    """Return hand poses, translations, object transforms, and object name."""
    motion_data = np.load(path, allow_pickle=True).item()

    rh_trans = motion_data["right_hand"]['trans']
    rh_poses = np.concatenate([motion_data["right_hand"]['rot'], motion_data["right_hand"]['pose']], axis=1)

    obj_name = next(key for key in motion_data if key != "right_hand")
    obj_rot = motion_data[obj_name]["rot"]
    obj_trans = motion_data[obj_name]["trans"]
    obj_transf = np.zeros((len(obj_rot), 4, 4), dtype=np.float32)
    obj_transf[:, :3, :3] = Rot.from_rotvec(obj_rot).as_matrix()
    obj_transf[:, :3, 3] = obj_trans
    obj_transf[:, 3, 3] = 1

    return rh_poses, rh_trans, obj_transf, obj_name
