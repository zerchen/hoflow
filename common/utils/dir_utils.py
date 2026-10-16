import os
import sys
import cv2
import json
import trimesh
import numpy as np

def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)


def export_pose_results(path, pose_result, metas):
    if isinstance(pose_result, list):
        num_frames = len(pose_result)
        for i in range(num_frames):
            sample_id = metas['id'][i][0]
            with open(os.path.join(path, sample_id + '.json'), 'w') as f:
                output = dict()
                output['cam_extr'] = metas['cam_extr'][i][0][:3, :3].cpu().numpy().tolist()
                output['cam_intr'] = metas['cam_intr'][i][0][:3, :3].cpu().numpy().tolist()
                if pose_result[i] is not None:
                    for key in pose_result[i].keys():
                        if pose_result[i][key] is not None:
                            output[key] = pose_result[i][key][0].cpu().numpy().tolist()
                        else:
                            continue
                json.dump(output, f)
    else:
        sample_id = metas['id'][0]
        with open(os.path.join(path, sample_id + '.json'), 'w') as f:
            output = dict()
            output['cam_extr'] = metas['cam_extr'][0][:3, :3].cpu().numpy().tolist()
            output['cam_intr'] = metas['cam_intr'][0][:3, :3].cpu().numpy().tolist()
            if pose_result is not None:
                for key in pose_result.keys():
                    if pose_result[key] is not None:
                        output[key] = pose_result[key][0].cpu().numpy().tolist()
                    else:
                        continue
            json.dump(output, f)


def export_point_cloud_results(path, recon_scale, point_cloud_result, metas):
    sample_id = metas['id'][0]
    img = metas['img'][0].cpu().numpy()[:, :, ::-1].astype(np.uint8)
    hand_verts_3d = metas['right_hand_verts_3d'][0].cpu().numpy()
    cur_dir = os.path.dirname(__file__)
    mano_faces = np.load(os.path.join(cur_dir, '..', 'mano', 'assets', 'closed_fmano.npy'))
    hand_mesh = trimesh.Trimesh(vertices=hand_verts_3d, faces=mano_faces)

    # try:
        # pc = point_cloud_result["pointclouds_up"][0].cpu().numpy() * recon_scale
    # except:
        # pc = point_cloud_result["pointclouds"][0].cpu().numpy() * recon_scale
    # objtrans = point_cloud_result['objtrans'][0].cpu().numpy()
    # handpalm = point_cloud_result['handpalm'][0].cpu().numpy()
    # pc_cam = pc + objtrans + handpalm
    # cam_intr = metas['cam_intr'][0, :3, :3].cpu().numpy()
    # pc_2d = (cam_intr @ pc_cam.transpose(1, 0)).transpose(1, 0)
    # pc_2d = (pc_2d[:, :2] / pc_2d[:, [2]]).astype(np.int32)
    # for i in range(len(pc_2d)):
        # if pc_2d[i, 0] >=0 and pc_2d[i, 1] >=0 and pc_2d[i, 0] < 224 and pc_2d[i, 1] < 224:
            # cv2.circle(img, pc_2d[i], 1, (255, 0, 0), 4)

    cv2.imwrite(os.path.join(path, sample_id + '.jpg'), img)
    hand_mesh.export(os.path.join(path, sample_id + '.obj'))

    with open(os.path.join(path, sample_id + '.json'), 'w') as f:
        output = dict()
        output['cam_extr'] = metas['cam_extr'][0][:3, :3].cpu().numpy().tolist()
        output['cam_intr'] = metas['cam_intr'][0][:3, :3].cpu().numpy().tolist()
        if point_cloud_result is not None:
            for key in point_cloud_result.keys():
                if key == "pointclouds":
                    output_pointcloud = point_cloud_result[key][0].cpu().numpy() * recon_scale
                    output[key] = output_pointcloud.tolist()
                elif key == "objtrans":
                    output_obj_trans = point_cloud_result[key][0].cpu().numpy()
                    output[key] = output_obj_trans.tolist()
                elif key == "handpalm":
                    handpalm = point_cloud_result[key][0].cpu().numpy()
                    output[key] = handpalm.tolist()
                if key == "pointclouds_up":
                    output_pointcloud = point_cloud_result[key][0].cpu().numpy() * recon_scale
                    output[key] = output_pointcloud.tolist()
                else:
                    continue
        json.dump(output, f)


def export_diffusion_results(path, diffusion_result, metas):
    sample_id = metas['id'][0]
    cur_dir = os.path.dirname(__file__)

    with open(os.path.join(path, sample_id + '.json'), 'w') as f:
        output = dict()
        if diffusion_result is not None:
            for key in diffusion_result.keys():
                output[key] = diffusion_result[key][0].cpu().numpy().tolist()
        json.dump(output, f)