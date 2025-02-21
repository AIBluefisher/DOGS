# pylint: disable=[E1101,C0206,E0401,C0103]

import os
import math
import random
from typing import List, Dict

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import trimesh

from conerf.datasets.utils import (
    minify, points_in_bbox2D, get_block_info_dir,
    store_ply, fetch_ply, save_bounding_boxes, save_colmap_ply, save_colmap_images,
    compute_rainbow_color, compute_bounding_box3D
)
from conerf.geometry.cluster import Grid2DClustering
from conerf.pycolmap.pycolmap.scene_manager import SceneManager
from scripts.preprocess.colmap_to_nerf import (
    get_val_images,
    MEGA_NERF_PREPROCESSED_SCENE,
    MEGA_NERF_PREPROCESSED_SCENE_WITH_MAPPINGS,
)


def find_all_images_with_subdir(image_dir: str) -> List:
    image_names = []
    for root, dirs, files in os.walk(image_dir):  # pylint: disable=W0612
        for file in files:
            full_image_name = os.path.join(root, file)
            image_name_start_index = len(image_dir)
            if any(full_image_name.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']):
                image_names.append(full_image_name[image_name_start_index+1:])
    image_names.sort()

    return image_names


def image_name_to_indices(image_paths: List, image_names: List) -> List:
    image_name_to_index = {}
    for i, image_path in enumerate(image_paths):
        image_name_to_index[image_path] = i

    image_indices = []
    for image_name in image_names:
        assert image_name in image_paths
        index = image_name_to_index[image_name]
        image_indices.append(index)

    return image_indices


def load_images(image_indices: np.array, image_paths: List, image_split: str = "train"):
    images = []
    if len(image_paths) == 0:
        return images

    pbar = tqdm.trange(len(image_indices),
                       desc=f"Loading {image_split} images")
    for image_index, image_path in enumerate(image_paths):
        if image_index in image_indices:
            image = torch.from_numpy(
                imageio.imread(image_path)).to(torch.uint8)
            image = (image / 255.0).clamp(0.0, 1.0)
            images.append(image)
            pbar.update(1)

    return images


def scratch_data(
    indices: np.array,
    image_paths: List,
    normal_paths: List,
    poses: np.array,
    intrinsics: np.array,
    split: str = "train"
) -> Dict:
    images = load_images(indices, image_paths, split) \
        if split != "train" else None
    normals = load_images(indices, normal_paths, split+"/normal") \
        if split != "train" else None
    camtoworlds = torch.from_numpy(poses[indices]).float()
    intrinsics = torch.from_numpy(intrinsics[indices]).float()

    return {
        "rgbs": images,
        "normals": normals,
        "poses": camtoworlds,
        "intrinsics": intrinsics,
        "image_paths": [image_paths[i] for i in indices],
    }


def cluster_image_in_grid(
    camtoworlds: np.ndarray,
    save_dir: str,
    all_image_indices,
    bbox_scale_factor: List,
    image_index_to_image_id: Dict,
    num_blocks: int = 1,
    mx: int = 1,
    my: int = 1,
):
    points3d = camtoworlds[..., :3, -1]
    labels, bboxes, exp_bboxes, transform_world_to_obb = Grid2DClustering(
        points=points3d,
        num_blocks=num_blocks,
        scale_factor=bbox_scale_factor[:2],
        p0=0, p1=1,
        mx=mx, my=my,
    )

    split_result_file = open(
        os.path.join(save_dir, "cluster.txt"),
        "w",
        encoding="utf-8",
    )
    for image_index, label in enumerate(labels):
        image_id = image_index_to_image_id[image_index]
        print(f'{image_id} {label}', file=split_result_file)
    split_result_file.close()

    block_image_ids = dict()
    for block_id, bbox in enumerate(exp_bboxes):
        image_ids = points_in_bbox2D(points3d[:, :2], bbox, transform_world_to_obb)

        if block_id not in block_image_ids:
            block_image_ids[block_id] = list()
        block_image_ids[block_id].append(all_image_indices[image_ids])

    bboxes = np.stack(bboxes, axis=0)
    exp_bboxes = np.stack(exp_bboxes, axis=0)

    return block_image_ids, bboxes, exp_bboxes, transform_world_to_obb


def cluster_points_in_grid(
    points3d: np.ndarray,
    colors: np.ndarray,
    save_dir: str,
    bbox_scale_factor: List,
    num_blocks: int = 1,
    mx: int = 1,
    my: int = 1,
    use_prior_center: bool = False,
    transform_world_to_obb: np.ndarray = None,
):
    _, bboxes, exp_bboxes, transform_world_to_obb = Grid2DClustering(
        points=points3d,
        num_blocks=num_blocks,
        scale_factor=bbox_scale_factor[:2],
        p0=0.00001, p1=0.99999,
        mx=mx, my=my,
        use_prior_center=use_prior_center,
        transform_world_to_obb=transform_world_to_obb,
    )

    for block_id, bbox in enumerate(exp_bboxes):
        points_in_bbox_indices = points_in_bbox2D(
            points3d[:, :2], bbox, transform_world_to_obb)
        local_points3d = points3d[points_in_bbox_indices]
        local_colors = colors[points_in_bbox_indices]

        local_ply_path = os.path.join(save_dir, f"points3D_{block_id}.ply")
        if not os.path.exists(local_ply_path):
            print(f"Converting point3d.bin to point3d_{block_id}.ply " +
                  "will happen only the first time you open the scene.")
            store_ply(local_ply_path, local_points3d, local_colors)

    bboxes = np.stack(bboxes, axis=0)
    exp_bboxes = np.stack(exp_bboxes, axis=0)

    return bboxes, exp_bboxes, transform_world_to_obb


def load_colmap(
    root_fp: str,
    subject_id: str,
    split: str,
    factor: int = 1,
    val_interval: int = 0,
    multi_blocks: bool = False,
    num_blocks: int = 1,
    bbox_scale_factor: List = [1.0, 1.0, 1.0],
    scale: bool = True,
    rotate: bool = True,
    use_manhattan_world: bool = False,
    model_folder: str = 'sparse',
    load_specified_images: bool = False,
    load_normal: bool = False,
    mx: int = None,
    my: int = None,
):
    assert factor in [1, 2, 4, 8]

    data_dir = os.path.join(root_fp, subject_id)
    colmap_dir = os.path.join(
        data_dir, model_folder,
        "manhattan_world" if use_manhattan_world else "0"
    )
    ori_image_dir = os.path.join(data_dir, "images")
    if factor > 1:
        ori_image_dir = ori_image_dir + f"_{factor}"

    train_image_name = []
    if load_specified_images:
        images_to_train_path = os.path.join(data_dir, "images_to_train.txt")
        if os.path.exists(images_to_train_path):
            file = open(images_to_train_path, 'r', encoding='utf-8')
            line = file.readline()
            while line:
                data = line.split(' ')
                image_id, image_path = int(data[0]), str(data[1])
                image_name = image_path[len(ori_image_dir)+1:].strip()
                train_image_name.append(image_name)
                line = file.readline()

    if mx is not None and my is not None:
        assert num_blocks == mx * my

    if factor != 1:
        minify(basedir=data_dir, factors=[factor])

    ply_path = os.path.join(colmap_dir, "points3D.ply")

    manager = SceneManager(colmap_dir, load_points=True)
    manager.load()

    points3d, colors = manager.points3D, manager.point3D_colors
    colmap_image_data = manager.images
    colmap_camera_data = manager.cameras
    image_name_to_image_id = manager.name_to_image_id
    image_names, w2c_mats, intrinsics = [], [], []
    image_index_to_image_id = {}

    # Extract extrinsic & intrinsic matrices.
    bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
    for k in colmap_image_data:
        im_data = colmap_image_data[k]
        if load_specified_images and len(train_image_name) > 0 \
                and im_data.name not in train_image_name:
            continue
        image_names.append(im_data.name)

        camera_id = im_data.camera_id
        camera = colmap_camera_data[camera_id]

        intrinsics.append(np.array([
            [camera.fx / factor, 0, camera.cx / factor],
            [0, camera.fy / factor, camera.cy / factor],
            [0, 0, 1]]
        ))

        w2c = np.concatenate([
            np.concatenate(
                [im_data.R(), im_data.tvec.reshape(3, 1)], 1), bottom
        ], axis=0)
        w2c_mats.append(w2c)
    w2c_mats = np.stack(w2c_mats, axis=0)
    intrinsics = np.stack(intrinsics, axis=0)

    # Convert extrinsics to camera-to-world.
    camtoworlds = np.linalg.inv(w2c_mats)

    # Previous Nerf results were generated with images sorted by filename,
    # ensure metrics are reported on the same test set.
    inds = np.argsort(image_names)
    image_names = [image_names[i] for i in inds]
    camtoworlds = camtoworlds[inds]
    intrinsics = intrinsics[inds]
    for image_index, image_name in enumerate(image_names):
        image_id = image_name_to_image_id[image_name]
        image_index_to_image_id[image_index] = image_id

    if use_manhattan_world:
        # We use COLMAP's `model_orientation_aligner` to align the model axis such that
        # the y-axis points towards downward. Therefore, we transform the coordinate axis
        # to let the z-axis points towards the ground plane.
        # NOTE: for the `building` scene, the z-axis point towards the ground using COLMAP's
        # aligner. Therefore, we do not make such coordinate transformation for it.
        if subject_id.lower() != "building":
            T = np.array([
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, -1, 0, 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)
            camtoworlds = np.einsum("nij, ki -> nkj", camtoworlds, T)
            points3d = (T[:3, :3] @ points3d.T).T  # [Np, 3]

        bbox_path = os.path.join(colmap_dir, "bounding_box.txt")
        bounding_box = compute_bounding_box3D(torch.from_numpy(points3d))
        save_bounding_boxes([bounding_box], bbox_path)

    if scale:
        # normalize the scene
        T, scale = similarity_from_cameras(
            camtoworlds, strict_scaling=False
        )
        camtoworlds = np.einsum("nij, ki -> nkj", camtoworlds, T)
        camtoworlds[:, :3, 3:4] *= scale

        points3d = scale * (T[:3, :3] @ points3d.T +
                            T[:3, 3][..., None]).T  # [Np, 3]

        if rotate:
            # Rotate the scene to align with ground plane.
            camtoworlds, points3d, _, _ = normalize_poses(
                torch.from_numpy(camtoworlds).float(),
                torch.from_numpy(points3d).float(),
                up_est_method="ground",
                center_est_method="lookat",
            )
            camtoworlds = camtoworlds.numpy()

    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply " +
              "will happen only the first time you open the scene.")
        store_ply(ply_path, points3d, colors)
    else:  # TODO(chenyu): Only be executed when point clouds are not generated by COLMAP.
        pcd = fetch_ply(ply_path)
        points3d = pcd.points
        colors = pcd.colors * 255.0  # colors are converted in the range of [0,1] when store ply.

    image_dir_suffix = f"_{factor}" if factor > 1 else ""
    image_folder = "images" if root_fp.find('MatrixCity') < 0 else ""
    colmap_image_dir = os.path.join(data_dir, image_folder).rstrip('/')
    image_dir = os.path.join(data_dir, image_folder +
                             image_dir_suffix).rstrip('/')
    for d in [image_dir, colmap_image_dir]:
        if not os.path.exists(d):
            raise ValueError(f"Image folder {d} does not exist.")

    # Downsampled images may have different names vs images used for COLMAP,
    # so we need to map between the two sorted lists of files.
    colmap_files = find_all_images_with_subdir(colmap_image_dir)
    image_files = find_all_images_with_subdir(image_dir)
    colmap_to_image = dict(zip(colmap_files, image_files))
    image_paths = [
        os.path.join(image_dir, colmap_to_image[f]) for f in image_names
    ]

    normal_paths = []
    if load_normal:
        normal_dir = os.path.join(data_dir, "normals" + image_dir_suffix)
        for image_path in image_paths:
            ext = os.path.splitext(image_path)[1]
            normal_path = image_path.replace(
                image_dir, normal_dir).replace(ext, '.png')
            normal_paths.append(normal_path)

    ################################## For test set #################################
    if split == 'test':
        n_test_traj_steps = 60
        camtoworlds = create_spheric_poses(
            torch.from_numpy(camtoworlds)[:, :3, 3], n_steps=n_test_traj_steps
        ).float()
        dummy_image = imageio.imread(image_paths[0])
        test_images = torch.zeros(
            (n_test_traj_steps, *(dummy_image.shape)), dtype=torch.uint8)

        # visualize_poses(camtoworlds)
        return {
            "rgbs": test_images,
            "poses": camtoworlds,
            "intrinsics": torch.from_numpy(intrinsics).float(),
            "image_paths": [None] * n_test_traj_steps,
        }

    num_images = len(image_paths)
    all_indices = np.arange(num_images)
    all_val_indices = all_indices[all_indices % val_interval == 0] \
        if val_interval > 0 else np.array([])

    if subject_id in MEGA_NERF_PREPROCESSED_SCENE or \
       subject_id in MEGA_NERF_PREPROCESSED_SCENE_WITH_MAPPINGS:
        val_image_names = get_val_images(
            scene_dir=data_dir,
            colmap_image_folder=os.path.join(image_dir),
            scene_name=subject_id,
        )
        all_val_indices = image_name_to_indices(image_paths, val_image_names)
        all_val_indices = np.array(all_val_indices, dtype=np.int64)

    if root_fp.find('MatrixCity') >= 0:
        all_val_indices = []
        for i, image_name in enumerate(image_paths):
            if image_name.find('test') >= 0:
                all_val_indices.append(i)
        all_val_indices = np.array(all_val_indices, dtype=np.int64)

    ################################# For validation set ###############################
    if split == "val":
        return scratch_data(
            all_val_indices, image_paths, normal_paths, camtoworlds, intrinsics, split)

    ############################ For train set (Non-block mode) ########################
    if not multi_blocks:
        indices = set(all_indices) - set(all_val_indices)
        indices = np.array(list(indices), dtype=np.int64)
        return scratch_data(indices, image_paths, normal_paths, camtoworlds, intrinsics, split)

    ############################### For train set (Block mode) ##########################
    # Compute camera bounding box for each block.
    block_save_dir = get_block_info_dir(data_dir, num_blocks, mx, my)
    os.makedirs(block_save_dir, exist_ok=True)
    bboxes_path = os.path.join(block_save_dir, "bounding_boxes.txt")
    origin_bboxes_path = os.path.join(
        block_save_dir, "bounding_boxes_origin.txt")
    obb_transform_path = os.path.join(
        block_save_dir, "world_to_obb_transform.npy")

    block_image_ids, camera_bboxes, exp_camera_bboxes, image_world_to_obb = cluster_image_in_grid(
        camtoworlds, block_save_dir, all_indices, bbox_scale_factor,
        image_index_to_image_id, num_blocks, mx, my,
    )
    
    use_prior_center = False
    if subject_id.lower() == "sci-art" or subject_id.lower() == "artsquad" or \
       subject_id.lower() == "building":
        use_prior_center = True

    point_bboxes, exp_point_bboxes, _ = cluster_points_in_grid(
        points3d, colors, block_save_dir, bbox_scale_factor,
        num_blocks, mx, my, use_prior_center, image_world_to_obb,
    )
    save_bounding_boxes(exp_camera_bboxes.tolist() +
                        exp_point_bboxes.tolist(), bboxes_path, 'w')
    save_bounding_boxes(camera_bboxes.tolist() +
                        point_bboxes.tolist(), origin_bboxes_path, 'w')
    with open(obb_transform_path, "wb") as npy_file:
        np.save(npy_file, image_world_to_obb)

    all_xyz, all_rgb = [], []
    for block_id, bbox in enumerate(point_bboxes):
        points_in_bbox_indices = points_in_bbox2D(points3d[:, :2], bbox, image_world_to_obb)
        local_points3d = points3d[points_in_bbox_indices]

        obb_local_points3d = np.zeros_like(local_points3d)
        obb_local_points3d[:, :2] = trimesh.transform_points(
            local_points3d[:, :2], image_world_to_obb)
        obb_local_points3d[:, 2] = local_points3d[:, 2]

        all_xyz.append(torch.from_numpy(obb_local_points3d))
        rgb = compute_rainbow_color(block_id).reshape(1, -1)
        rgb = rgb.expand(local_points3d.shape[0], -1)
        all_rgb.append(rgb)
    all_xyz = torch.concat(all_xyz, dim=0)
    all_rgb = torch.concat(all_rgb, dim=0)
    ply_path = os.path.join(block_save_dir, "cluster_points3D.txt")
    save_colmap_ply(all_xyz, all_rgb, ply_path)

    block_camtoworlds = [None] * num_blocks
    block_normals = [None] * num_blocks
    block_intrinsics = [None] * num_blocks
    block_image_paths = [None] * num_blocks

    pbar = tqdm.trange(
        num_blocks, desc=f"Loading {split} images for {num_blocks} Blocks")
    for block_id in block_image_ids:
        image_ids = sorted(block_image_ids[block_id])
        image_ids = np.array(image_ids)
        print(f'block#{block_id} images: {image_ids.shape}')

        # Select the split.
        all_block_indices = list(range(0, image_ids.shape[0]))
        indices = image_ids[np.array(
            [i for i in all_block_indices])]  # pylint: disable=R1721

        image_list, local_camtoworlds, local_intrinsics = [], [], []
        normal_list = []
        for image_index, image_path in enumerate(image_paths):
            if image_index in indices and image_index not in all_val_indices:
                image_list.append(image_path)
                local_camtoworlds.append(camtoworlds[image_index])
                local_intrinsics.append(intrinsics[image_index])
                if load_normal:
                    normal_list.append(torch.from_numpy(
                        imageio.imread(normal_paths[image_index])).to(torch.uint8)
                    )

        block_image_paths[block_id] = image_list
        block_normals[block_id] = normal_list
        block_camtoworlds[block_id] = torch.from_numpy(
            np.stack(local_camtoworlds, axis=0)).float()
        block_intrinsics[block_id] = torch.from_numpy(
            np.stack(local_intrinsics, axis=0)).float()

        pbar.update(1)

    save_colmap_images(block_camtoworlds, num_images, block_save_dir)

    return {
        "rgbs": None,  # block_images,
        "normals": None,  # block_normals,
        "poses": block_camtoworlds,
        "intrinsics": block_intrinsics,
        "image_paths": block_image_paths,
    }


def similarity_from_cameras(c2w, strict_scaling):
    """
    reference: nerf-factory
    Get a similarity transform to normalize dataset
    from c2w (OpenCV convention) cameras
    :param c2w: (N, 4)
    :return T (4,4) , scale (float)
    """
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]

    # (1) Rotate the world so that z+ is the up axis
    # we estimate the up axis by averaging the camera up axes
    ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)

    up_camspace = np.array([0.0, -1.0, 0.0])
    c = (up_camspace * world_up).sum()
    cross = np.cross(world_up, up_camspace)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    # if c > -1:
    #     R_align = np.eye(3) + skew + (skew @ skew) * 1 / \
    #         (1 + c)  # pylint: disable=C0103
    # else:
    #     # In the unlikely case the original data has y+ up axis,
    #     # rotate 180-deg about x axis
    #     R_align = np.array(  # pylint: disable=C0103
    #         [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    R_align = np.array(  # pylint: disable=C0103
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    #  R_align = np.eye(3) # DEBUG
    R = R_align @ R
    fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)
    t = (R_align @ t[..., None])[..., 0]

    # (2) Recenter the scene using camera center rays
    # find the closest point to the origin for each camera's center ray
    nearest = t + (fwds * -t).sum(-1)[:, None] * fwds

    # median for more robustness
    translate = -np.median(nearest, axis=0)

    #  translate = -np.mean(t, axis=0)  # DEBUG

    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align

    # (3) Rescale the scene using camera distances
    scale_fn = np.max if strict_scaling else np.median
    scale = 1.0 / scale_fn(np.linalg.norm(t + translate, axis=-1))

    return transform, scale


def get_center(pts):
    center = pts.mean(0)
    dis = (pts - center[None, :]).norm(p=2, dim=-1)
    mean, std = dis.mean(), dis.std()
    q25, q75 = torch.quantile(dis, 0.25), torch.quantile(dis, 0.75)
    valid = (dis > mean - 1.5 * std) & (dis < mean + 1.5 * std) & \
            (dis > mean - (q75 - q25) * 1.5) & (dis < mean + (q75 - q25) * 1.5)
    center = pts[valid].mean(0)
    return center


def normalize_poses(
    poses,
    pts,
    up_est_method: str = "ground",
    center_est_method: str = "lookat"
):
    if center_est_method == 'camera':
        # estimation scene center as the average of all camera positions
        center = poses[..., :3, 3].mean(0)
    elif center_est_method == 'lookat':
        # estimation scene center as the average of the intersection of
        # selected pairs of camera rays
        cams_ori = poses[..., :3, 3]
        cams_dir = poses[:, :3, :3] @ torch.as_tensor([0., 0., -1.])
        cams_dir = F.normalize(cams_dir, dim=-1)
        A = torch.stack([cams_dir, -cams_dir.roll(1, 0)], dim=-1)
        b = -cams_ori + cams_ori.roll(1, 0)
        t = torch.linalg.lstsq(A, b).solution
        center = (torch.stack([cams_dir, cams_dir.roll(1, 0)], dim=-1) *
                  t[:, None, :] + torch.stack([
                      cams_ori, cams_ori.roll(1, 0)
                  ], dim=-1)).mean((0, 2))
    elif center_est_method == 'point':
        # first estimation scene center as the average of all camera positions
        # later we'll use the center of all points bounded by the cameras as the
        # final scene center
        center = poses[..., :3, 3].mean(0)
    else:
        raise NotImplementedError(
            f'Unknown center estimation method: {center_est_method}')

    if up_est_method == 'ground':
        # estimate up direction as the normal of the estimated ground plane
        # use RANSAC to estimate the ground plane in the point cloud
        import pyransac3d as pyrsc  # pylint: disable=C0415

        # Fix the seed every time we the load the dataset.
        random.seed(0)
        ground = pyrsc.Plane()

        plane_eq = ground.fit(pts.numpy(), thresh=0.01)[0]
        # A, B, C, D in Ax + By + Cz + D = 0
        plane_eq = torch.as_tensor(plane_eq)
        # plane normal as up direction
        z = F.normalize(plane_eq[:3], dim=-1)
        signed_distance = (
            torch.cat([pts, torch.ones_like(pts[..., 0:1])], dim=-1) * plane_eq
        ).sum(-1)
        if signed_distance.mean() < 0:
            z = -z  # flip the direction if points lie under the plane
    elif up_est_method == 'camera':
        # estimate up direction as the average of all camera up directions
        z = F.normalize((poses[..., 3] - center).mean(0), dim=0)
    else:
        raise NotImplementedError(
            f'Unknown up estimation method: {up_est_method}')

    # new axis
    y = torch.as_tensor([z[1], -z[0], 0.])
    x = F.normalize(y.cross(z), dim=0)
    y = z.cross(x)

    if center_est_method == 'point':
        # rotation
        Rc = torch.stack([x, y, z], dim=1)
        R = Rc.T
        inv_trans = torch.cat([
            torch.cat([R, torch.as_tensor([[0., 0., 0.]]).T], dim=1),
            torch.as_tensor([[0., 0., 0., 1.]])
        ], dim=0)
        poses_norm = (inv_trans @ poses)[:, :3]
        pts = (inv_trans @ torch.cat(
            [pts, torch.ones_like(pts[:, 0:1])], dim=-1
        )[..., None])[:, :3, 0]

        # translation and scaling
        poses_min, poses_max = poses_norm[..., 3].min(
            0)[0], poses_norm[..., 3].max(0)[0]
        pts_fg = pts[
            (poses_min[0] < pts[:, 0]) & (pts[:, 0] < poses_max[0]) &
            (poses_min[1] < pts[:, 1]) & (pts[:, 1] < poses_max[1])]
        center = get_center(pts_fg)
        tc = center.reshape(3, 1)
        t = -tc
        inv_trans = torch.cat([
            torch.cat([torch.eye(3), t], dim=1),
            torch.as_tensor([[0., 0., 0., 1.]])
        ], dim=0)
        poses_norm = (inv_trans @ poses)
    else:
        # rotation and translation
        Rc = torch.stack([x, y, z], dim=1)
        tc = center.reshape(3, 1)
        R, t = Rc.T, -Rc.T @ tc
        inv_trans = torch.cat([
            torch.cat([R, t], dim=1), torch.as_tensor([[0., 0., 0., 1.]])
        ], dim=0)

        poses_norm = (inv_trans @ poses)  # (N_images, 4, 4)
        pts = (R @ pts.T + t).T

    return poses_norm, pts, R, t


def create_spheric_poses(cameras, n_steps=120):
    center = torch.as_tensor([0., 0., 0.], dtype=cameras.dtype)
    mean_d = (cameras - center[None, :]).norm(p=2, dim=-1).mean()
    mean_h = cameras[:, 2].mean()
    r = (mean_d ** 2 - mean_h ** 2).sqrt()
    up = torch.as_tensor([0., 0., -1.], dtype=center.dtype)

    camtoworlds = []
    for theta in torch.linspace(0, 2 * math.pi, n_steps):
        cam_pos = torch.stack([r * theta.cos(), r * theta.sin(), mean_h])
        l = F.normalize(center - cam_pos, p=2, dim=0)
        s = F.normalize(l.cross(up), p=2, dim=0)
        u = F.normalize(s.cross(l), p=2, dim=0)
        c2w = torch.cat([torch.stack([s, u, l], dim=1),
                        cam_pos[:, None]], axis=1)
        c2w = torch.cat(
            [c2w, torch.tensor([[0, 0, 0, 1]], dtype=c2w.dtype)], axis=0)

        camtoworlds.append(c2w)

    camtoworlds = torch.stack(camtoworlds, dim=0)

    return camtoworlds
