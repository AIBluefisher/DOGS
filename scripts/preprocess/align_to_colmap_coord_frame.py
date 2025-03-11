# pylint: disable=[E1101]

import os
import argparse
import copy
from collections import OrderedDict
from typing import List

import torch
import numpy as np

from conerf.datasets.utils import fetch_ply, store_ply
from conerf.geometry.align_poses import align_ate_c2b_use_a2b
from conerf.geometry.pose_util import rotation_distance
from conerf.pycolmap.pycolmap.scene_manager import SceneManager


def evaluate_camera_alignment(aligned_pred_poses, poses_gt):
    """
    measure errors in rotation and translation
    """
    R_aligned, t_aligned = aligned_pred_poses.split([3, 1], dim=-1)
    R_gt, t_gt = poses_gt.split([3, 1], dim=-1)

    R_error = rotation_distance(R_aligned[..., :3, :3], R_gt[..., :3, :3])
    t_error = (t_aligned - t_gt)[..., 0].norm(dim=-1)

    mean_rotation_error = np.rad2deg(R_error.mean().cpu()).item()
    mean_position_error = t_error.mean().item()
    med_rotation_error = np.rad2deg(R_error.median().cpu()).item()
    med_position_error = t_error.median().item()

    return {'R_error_mean': mean_rotation_error, "t_error_mean": mean_position_error,
            'R_error_med': med_rotation_error, 't_error_med': med_position_error}


def load_colmap_format_pose(colmap_dir: str, image_names_db=None):
    manager = SceneManager(colmap_dir, load_points=False)
    manager.load()

    colmap_image_data = manager.images
    w2c_mats = []
    image_names = []
    image_name_to_pose = {}

    # Extract extrinsic & intrinsic matrices.
    bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
    for k in colmap_image_data:
        im_data = colmap_image_data[k]

        if image_names_db is not None and (im_data.name not in image_names_db):
            # print(f'image name: {im_data.name}')
            continue

        w2c = np.concatenate([
            np.concatenate(
                [im_data.R(), im_data.tvec.reshape(3, 1)], 1), bottom
        ], axis=0)
        image_names.append(im_data.name)
        image_name_to_pose[im_data.name] = w2c

    image_names = sorted(image_names)
    for i, image_name in enumerate(image_names):
        w2c_mats.append(image_name_to_pose[image_name])

    w2c_mats = np.stack(w2c_mats, axis=0)
    # Convert extrinsics to camera-to-world.
    camtoworlds = np.linalg.inv(w2c_mats)

    return camtoworlds, image_name_to_pose.keys()


def transform_and_write_model(input_dir: str, output_dir: str, sim3: List):
    from conerf.pycolmap.pycolmap.rotation import Quaternion  # pylint: disable=C0415

    scale, rotation, translation = sim3[0], sim3[1], sim3[2]
    rotation = rotation.squeeze(0)  # [3,3]
    translation = translation.squeeze(0).T  # [1,3]

    manager = SceneManager(input_dir, load_points=True)
    manager.load()

    image_data = manager.images
    points3D = manager.points3D

    trans_points3D = scale * (rotation @ points3D.T).T + translation
    trans_image_data = OrderedDict()
    bottom = np.array([0, 0, 0, 1]).reshape(1, 4)

    for k in image_data:
        im_data = image_data[k]
        trans_im_data = copy.deepcopy(im_data)

        world_to_cam = np.concatenate([
            np.concatenate(
                [im_data.R(), im_data.tvec.reshape(3, 1)], 1), bottom
        ], axis=0)

        cam_to_world = np.linalg.inv(world_to_cam)
        cam_to_world[:3, :3] = rotation @ cam_to_world[:3, :3]
        cam_to_world[:3, 3:] = scale * \
            (rotation @ cam_to_world[:3, 3:]) + translation.T
        world_to_cam = np.linalg.inv(cam_to_world)

        trans_im_data.q = Quaternion.FromR(world_to_cam[:3, :3])
        trans_im_data.tvec = world_to_cam[:3, 3]
        trans_image_data[k] = trans_im_data

    manager.images = trans_image_data
    manager.points3D = trans_points3D
    manager.save(output_dir, binary=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root_dir",
                        type=str,
                        default="",
                        help="absolute path of the dataset")
    parser.add_argument("--scene",
                        type=str,
                        default="",
                        help="scene name of the dataset")
    parser.add_argument("--method",
                        type=str,
                        default="zero_gs",
                        help="scene name of the dataset")
    args = parser.parse_args()

    data_root_dir = args.data_root_dir
    scene = args.scene
    method = args.method

    colmap_model_path = os.path.join(
        # data_root_dir, scene, "sparse/manhattan_world")
        data_root_dir, scene, "sparse/0")
    target_model_path = os.path.join(data_root_dir, scene, f"{method}/0")
    print(f'target_model_path: {target_model_path}')

    output_target_model_path = os.path.join(
        data_root_dir, scene, f"{method}/manhattan_world")
    os.makedirs(output_target_model_path, exist_ok=True)

    colmap_poses_c2w, colmap_image_names = load_colmap_format_pose(
        colmap_model_path)
    target_poses_c2w, target_image_names = load_colmap_format_pose(
        target_model_path)

    if len(colmap_image_names) > len(target_image_names):
        colmap_poses_c2w, _ = load_colmap_format_pose(
            colmap_model_path, target_image_names
        )
    else:
        target_poses_c2w, _ = load_colmap_format_pose(
            target_model_path, colmap_image_names
        )

    assert len(colmap_poses_c2w) == len(target_poses_c2w)
    print(f'num poses: {len(colmap_poses_c2w)}')

    target_poses_c2w, s, rotation, t = align_ate_c2b_use_a2b(
        torch.from_numpy(target_poses_c2w),  # pylint: disable=E1101
        torch.from_numpy(colmap_poses_c2w)   # pylint: disable=E1101
    )
    sim3 = [s, rotation, t]

    pose_error = evaluate_camera_alignment(
        target_poses_c2w,
        torch.from_numpy(colmap_poses_c2w).float()
    )
    print(f'pose_error: {pose_error}')

    # TODO(chenyu): for ace0, transform the .ply file.
    transform_and_write_model(
        target_model_path, output_target_model_path, sim3)

    ply_path = os.path.join(data_root_dir, scene, f"{method}/0/points3D.ply")
    if os.path.exists(ply_path):
        print(f't shape: {t.shape}, rotation shape: {rotation.shape}')
        point_cloud = fetch_ply(ply_path)
        points3d = point_cloud.points
        colors = point_cloud.colors * 255.0
        points3d = s * (rotation.squeeze(0) @ points3d.T).T + t.squeeze(-1)
        points3d = points3d.reshape(-1, 3)

        new_ply_path = os.path.join(
            data_root_dir, scene, f"{method}/manhattan_world/points3D.ply")
        store_ply(new_ply_path, points3d, colors)
