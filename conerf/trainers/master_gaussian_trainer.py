# pylint: disable=[E1101,W0201,W0621]

import os
import copy
import warnings

from typing import List, Dict, Tuple
from omegaconf import OmegaConf
import tqdm
import omegaconf

import torch
import torch.nn.functional as F
import torch.distributed.rpc as rpc
import numpy as np
import trimesh

from torch.distributed.rpc import remote, rpc_async

from conerf.datasets.dataset_base import MiniDataset
from conerf.datasets.utils import (
    create_dataset, save_colmap_ply, compute_rainbow_color, get_block_info_dir,
    points_in_bbox2D, compute_bounding_box2D
)
from conerf.model.gaussian_fields.gaussian_splat_model import GaussianSplatModel
from conerf.model.gaussian_fields.prune import calculate_v_imp_score, prune_list
from conerf.trainers.implicit_recon_trainer import ImplicitReconTrainer
from conerf.trainers.slave_gaussian_trainer import SlaveGaussianSplatTrainer
from conerf.utils.config import config_parser, load_config
from conerf.utils.utils import setup_seed

MASTER_NAME = 'MasterGaussianSplatTrainer'
WORKER_NAME = 'trainer'
warnings.filterwarnings("ignore", category=UserWarning)


def fuse_block_gaussians(
    block_gaussians: Dict,
    point_bboxes: List = None,
    world_to_obb_transform: np.ndarray = None,
    test_dir: str = ""
) -> Tuple:
    xyz, opacity = [], []
    features_dc, features_rest = [], []
    scaling, quaternion = [], []
    densify_point_bboxes = [None] * len(point_bboxes)

    all_xyz = []
    all_rgb = []

    # Concatenate all 3D Gaussians.
    print('Fusing 3D Gaussians...')
    for block_id, model in block_gaussians.items():
        if point_bboxes is not None:
            point_bbox = point_bboxes[block_id].reshape(2, 3).numpy()
            print(
                f'[BLOCK#{block_id}] Before removing points: {model.get_xyz.shape[0]}')
            points = model.get_xyz.cpu().numpy()
            obb_points2d = trimesh.transform_points(
                points[:, :2], world_to_obb_transform)

            # Re-estimate the bounding box after densification for each block.
            densify_point_bbox = compute_bounding_box2D(
                torch.from_numpy(obb_points2d), [1.0, 1.0], -1.0, 1.0, 0.001, 0.999)
            densify_point_bboxes[block_id] = densify_point_bbox

            # Remove points that out of the initial grid bounding box.
            valid_gaussian_indices = points_in_bbox2D(obb_points2d, point_bbox)
            model.extract_sub_gaussians(valid_gaussian_indices)
            print(
                f'[BLOCK#{block_id}] After removing points: {model.get_xyz.shape[0]}\n')

        xyz.append(model.get_xyz)
        features_dc.append(model.get_features_dc)
        features_rest.append(model.get_features_rest)
        scaling.append(model.get_raw_scaling)
        quaternion.append(model.get_raw_quaternion)
        opacity.append(model.get_raw_opacity)

        points_coors = model.get_xyz.detach().cpu()
        all_xyz.append(points_coors)
        rgb = compute_rainbow_color(block_id).reshape(1, -1)
        rgb = rgb.expand(points_coors.shape[0], -1)
        all_rgb.append(rgb)
        ply_path = os.path.join(test_dir, f"fuse_points3D_{block_id}.ply")
        model.save_ply(ply_path)

    all_xyz = torch.concat(all_xyz, dim=0)
    all_rgb = torch.concat(all_rgb, dim=0)
    ply_path = os.path.join(test_dir, "non_overlap_points3D.txt")
    save_colmap_ply(all_xyz, all_rgb, ply_path)

    xyz = torch.concat(xyz, dim=0)
    features_dc = torch.concat(features_dc, dim=0)
    features_rest = torch.concat(features_rest, dim=0)
    scaling = torch.concat(scaling, dim=0)
    quaternion = torch.concat(quaternion, dim=0)
    opacity = torch.concat(opacity, dim=0)

    return xyz, features_dc, features_rest, scaling, quaternion, opacity, densify_point_bboxes


def prune_gaussians_after_merge(
    config: OmegaConf,
    gaussians: GaussianSplatModel,
    camera_blocks: List,
    device: str
) -> int:
    cameras = []
    for camera_block in camera_blocks:
        cameras += camera_block
    num_total_images = len(cameras)

    color_bkgd = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    gaussian_list, imp_list = prune_list(  # pylint: disable=W0612
        gaussians, cameras, config.pipeline, color_bkgd
    )
    v_list = calculate_v_imp_score(gaussians, imp_list, config.prune.v_pow)
    gaussians.prune_gaussians(0.4 * config.prune.prune_percent, v_list)

    return num_total_images


def select_gaussians_in_each_block(
    bboxes: List,
    gaussians: GaussianSplatModel,
    device: str,
    test_dir: str,
    world_to_obb_transform: np.ndarray = None,
):
    points = gaussians.get_xyz.detach().cpu()
    global_indices = []

    # Regroup points inside in the same bounding box into the same block.
    for block_id, bbox in enumerate(bboxes):
        bbox = bbox.reshape(2, 3).numpy()
        point_indices = points_in_bbox2D(
            points.numpy()[:, :2], bbox, world_to_obb_transform)
        global_indices.append(torch.from_numpy(point_indices))

    # Count the frequency of each gaussian point and remove gaussians that
    # are not in all bounding boxes.
    all_point_indices = torch.cat(global_indices)
    visibility_count = torch.bincount(all_point_indices)
    # Some points may not inside the precomputed bounding-boxes and therefore
    # are not visible in all views.
    valid_gaussian_indices = torch.argwhere(visibility_count).squeeze(-1)
    gaussians.extract_sub_gaussians(valid_gaussian_indices)

    points = gaussians.get_xyz.detach().cpu()
    local_gaussians, global_indices = [], []

    # Regroup points inside in the same bounding box into the same block.
    for block_id, bbox in enumerate(bboxes):
        bbox = bbox.reshape(2, 3).numpy()
        point_indices = points_in_bbox2D(
            points.numpy()[:, :2], bbox, world_to_obb_transform)
        global_indices.append(torch.from_numpy(point_indices))
        print(f'num gaussians in block#{block_id}: {point_indices.shape}')
        sub_gaussians = gaussians.get_sub_gaussians(point_indices)
        local_gaussians.append(sub_gaussians)

        ply_path = os.path.join(test_dir, f"points3D_{block_id}.ply")
        sub_gaussians.save_ply(ply_path)

    # Count the frequency of each gaussian point.
    all_point_indices = torch.cat(global_indices)
    visibility_count = torch.bincount(all_point_indices).to(device)
    assert (visibility_count != 0).all(
    ), "visibility count has zero elements!"

    return visibility_count, global_indices, local_gaussians


def read_bounding_boxes(colmap_dir: str, suffix: str = ''):
    bbox_path = os.path.join(colmap_dir, f"bounding_boxes{suffix}.txt")
    file = open(bbox_path, "r", encoding="utf-8")
    line = file.readline()

    # The first K/2 bboxes are for cameras, the last K/2 boxes are for points.
    num_bboxes = int(line.split(' ')[0]) // 2
    bboxes = []
    for _ in range(num_bboxes):
        file.readline()

    line = file.readline()
    while line:
        data = line.split(' ')
        bbox = torch.zeros(6)
        bbox[0], bbox[1], bbox[2] = float(
            data[0]), float(data[1]), float(data[2])
        bbox[3], bbox[4], bbox[5] = float(
            data[3]), float(data[4]), float(data[5])
        bboxes.append(bbox)
        line = file.readline()

    file.close()
    return bboxes


class MasterGaussianSplatTrainer(ImplicitReconTrainer):
    def __init__(self, config: OmegaConf) -> None:
        self.config = config
        self.device = "cuda"

        self.load_val_dataset()

        self.remote_workers = {}

        # Initialize local trainers remotely on other workers.
        if self.config.dataset.get("mx", None) is not None and \
           self.config.dataset.get("my", None) is not None:
            self.num_blocks = self.config.dataset.mx * self.config.dataset.my
        else:
            self.num_blocks = config.dataset.num_blocks
        print(f'[MasterGaussianSplatTrainer] num blocks: {self.num_blocks}')
        self.init_block_trainers(sub_gaussians=None)

        self.gaussians = GaussianSplatModel(
            max_sh_degree=self.config.texture.max_sh_degree,
            percent_dense=self.config.geometry.percent_dense,
        )
        self.gaussians.active_sh_degree = self.gaussians.max_sh_degree

        super().__init__(
            config=self.config,
            prefetch_dataset=False,
            trainset=None,
            valset=self.val_dataset,
            model=self.gaussians
        )
        self.model = self.gaussians

        data_dir = os.path.join(config.dataset.root_dir, config.dataset.scene)
        bbox_dir = get_block_info_dir(
            data_dir,
            self.config.dataset.num_blocks,
            self.config.dataset.get("mx", None),
            self.config.dataset.get("my", None),
        )
        self.ori_point_bboxes = read_bounding_boxes(bbox_dir, '_origin')
        self.exp_point_bboxes = read_bounding_boxes(bbox_dir)

        obb_transform_path = os.path.join(
            bbox_dir, "world_to_obb_transform.npy")
        npy_file = open(obb_transform_path, "rb")
        self.world_to_obb_transform = np.load(npy_file)
        npy_file.close()
        print(
            f'[MasterGaussianSplatTrainer] world_to_obb_transform: {self.world_to_obb_transform}')

    def init_block_trainers(self, sub_gaussians: List = None, sub_masks: Dict = None):
        assert sub_gaussians is None or len(sub_gaussians) == self.num_blocks
        self.remote_workers = {}

        for block_id in range(self.num_blocks):
            local_config = copy.deepcopy(self.config)
            local_config.expname = local_config.expname + f'/block_{block_id}'
            local_config.trainer.enable_tensorboard = False

            val_dataset = None
            model = None if sub_gaussians is None else sub_gaussians[block_id]
            mask = None if sub_masks is None else sub_masks[block_id]

            # worker_name = WORKER_NAME.format(block_id + 1)
            worker_name = f"worker{block_id+1}"
            device_id = (block_id + 1) % 4  # local_world_size
            self.remote_workers[block_id] = remote(
                worker_name,
                SlaveGaussianSplatTrainer,
                args=(local_config, False, None, val_dataset,
                      model, mask, block_id, device_id)
            )

    def setup_training_params(self):
        pass

    def build_networks(self):
        pass

    def setup_optimizer(self):
        pass

    def compose_state_dicts(self) -> None:
        self.state_dicts = {
            "models": dict(),
            "optimizers": dict(),
            "schedulers": dict(),  # No scheduler needs to be stored.
            "meta_data": dict(),
        }

        # meta data for construction models.
        self.state_dicts["meta_data"]["active_sh_degree"] = self.gaussians.active_sh_degree
        self.state_dicts["meta_data"]["xyz"] = self.gaussians.get_xyz
        self.state_dicts["meta_data"]["features_dc"] = self.gaussians.get_features_dc
        self.state_dicts["meta_data"]["features_rest"] = self.gaussians.get_features_rest
        self.state_dicts["meta_data"]["scaling"] = self.gaussians.get_raw_scaling
        self.state_dicts["meta_data"]["quaternion"] = self.gaussians.get_raw_quaternion
        self.state_dicts["meta_data"]["opacity"] = self.gaussians.get_raw_opacity
        self.state_dicts["meta_data"]["max_radii2D"] = self.gaussians.max_radii2D
        self.state_dicts["meta_data"]["xyz_gradient_accum"] = self.gaussians.xyz_gradient_accum
        self.state_dicts["meta_data"]["denom"] = self.gaussians.denom
        self.state_dicts["meta_data"]["spatial_lr_scale"] = 0

        if self.iteration >= self.config.geometry.densify_end_iter and \
           self.config.trainer.admm.enable:
            self.state_dicts["meta_data"]["rho_xyz"] = self.rho_xyz
            self.state_dicts["meta_data"]["rho_fdc"] = self.rho_fdc
            self.state_dicts["meta_data"]["rho_fr"] = self.rho_fr
            self.state_dicts["meta_data"]["rho_s"] = self.rho_s
            self.state_dicts["meta_data"]["rho_q"] = self.rho_q
            self.state_dicts["meta_data"]["rho_o"] = self.rho_o

    def load_val_dataset(self):
        val_config = copy.deepcopy(self.config)
        val_config.dataset.multi_blocks = False
        val_config.dataset.num_blocks = 1
        self.val_dataset = create_dataset(
            config=val_config,
            split=val_config.dataset.val_split,
            num_rays=None,
            apply_mask=val_config.dataset.apply_mask,
            device=self.device
        )

    def setup_penalty_parameters(self, num_gaussians: int):
        scale_factor = 1.0 / num_gaussians

        self.rho_xyz = scale_factor * self.config.trainer.admm.alpha_xyz
        self.rho_fdc = scale_factor * self.config.trainer.admm.alpha_fdc
        self.rho_fr = scale_factor * self.config.trainer.admm.alpha_fr
        self.rho_s = scale_factor * self.config.trainer.admm.alpha_s
        self.rho_q = scale_factor * self.config.trainer.admm.alpha_q
        self.rho_o = scale_factor * self.config.trainer.admm.alpha_o

    @torch.no_grad()
    def adapt_penalty_parameters(
        self,
        primal_res_dict: Dict,
        dual_res_dict: Dict,
    ):
        mu = self.config.trainer.admm.mu  # pylint: disable=C0103
        tau_inc = self.config.trainer.admm.tau_inc
        tau_dec = self.config.trainer.admm.tau_dec

        # Large values of 'rho' place a large penalty on violations of primal
        # feasibility and so tend to produce small primal residuals.

        if primal_res_dict["xyz"] > mu * dual_res_dict["xyz"]:
            self.rho_xyz *= tau_inc
        elif dual_res_dict["xyz"] > mu * primal_res_dict["xyz"]:
            self.rho_xyz /= tau_dec

        if primal_res_dict["feat_dc"] > mu * dual_res_dict["feat_dc"]:
            self.rho_fdc *= tau_inc
        elif dual_res_dict["feat_dc"] > mu * primal_res_dict["feat_dc"]:
            self.rho_fdc /= tau_dec

        if primal_res_dict["feat_rest"] > mu * dual_res_dict["feat_rest"]:
            self.rho_fr *= tau_inc
        elif dual_res_dict["feat_rest"] > mu * primal_res_dict["feat_rest"]:
            self.rho_fr /= tau_dec

        if primal_res_dict["scale"] > mu * dual_res_dict["scale"]:
            self.rho_s *= tau_inc
        elif dual_res_dict["scale"] > mu * primal_res_dict["scale"]:
            self.rho_s /= tau_dec

        if primal_res_dict["quaternion"] > mu * dual_res_dict["quaternion"]:
            self.rho_q *= tau_inc
        elif dual_res_dict["quaternion"] > mu * primal_res_dict["quaternion"]:
            self.rho_q /= tau_dec

        if primal_res_dict["opacity"] > mu * dual_res_dict["opacity"]:
            self.rho_o *= tau_inc
        elif dual_res_dict["opacity"] > mu * primal_res_dict["opacity"]:
            self.rho_o /= tau_dec

    @torch.no_grad()
    def set_block_penalty_parameters(self):
        futures = []
        for worker in self.remote_workers.values():
            futures.append(worker.rpc_async().set_penalty_parameters(
                self.rho_xyz,
                self.rho_fdc,
                self.rho_fr,
                self.rho_s,
                self.rho_q,
                self.rho_o,
            ))

        for future in futures:
            future.wait()

    @torch.no_grad()
    def compute_primal_residual(self) -> Dict:
        xyz_primal_res = 0
        feat_dc_primal_res = 0
        feat_rest_primal_res = 0
        scale_primal_res = 0
        quaternion_primal_res = 0
        opacity_primal_res = 0

        for k, gaussians in self.block_gaussians.items():
            indices = self.global_indices[k]
            xyz_primal_res += F.mse_loss(
                self.gaussians.get_xyz[indices], gaussians.get_xyz
            )
            feat_dc_primal_res += F.mse_loss(
                self.gaussians.get_features_dc[indices], gaussians.get_features_dc
            )
            feat_rest_primal_res += F.mse_loss(
                self.gaussians.get_features_rest[indices], gaussians.get_features_rest
            )
            scale_primal_res += F.mse_loss(
                self.gaussians.get_raw_scaling[indices], gaussians.get_raw_scaling
            )
            quaternion_primal_res += F.mse_loss(
                self.gaussians.get_raw_quaternion[indices], gaussians.get_raw_quaternion
            )
            opacity_primal_res += F.mse_loss(
                self.gaussians.get_raw_opacity[indices], gaussians.get_raw_opacity
            )

        return {
            "xyz": xyz_primal_res,
            "feat_dc": feat_dc_primal_res,
            "feat_rest": feat_rest_primal_res,
            "scale": scale_primal_res,
            "quaternion": quaternion_primal_res,
            "opacity": opacity_primal_res,
        }

    @torch.no_grad()
    def compute_dual_residual(self) -> Dict:
        xyz_dual_res = self.rho_xyz * \
            F.mse_loss(self.prev_xyz, self.gaussians.get_xyz)
        feat_dc_dual_res = self.rho_fdc * F.mse_loss(
            self.prev_features_dc, self.gaussians.get_features_dc)
        feat_rest_dual_res = self.rho_fr * F.mse_loss(
            self.prev_features_rest, self.gaussians.get_features_rest)
        scale_dual_res = self.rho_s * \
            F.mse_loss(self.prev_scaling, self.gaussians.get_raw_scaling)
        quat_dual_res = self.rho_q * F.mse_loss(
            self.prev_quaternion, self.gaussians.get_raw_quaternion)
        opacity_dual_res = self.rho_o * F.mse_loss(
            self.prev_opacity, self.gaussians.get_raw_opacity)

        return {
            "xyz": xyz_dual_res,
            "feat_dc": feat_dc_dual_res,
            "feat_rest": feat_rest_dual_res,
            "scale": scale_dual_res,
            "quaternion": quat_dual_res,
            "opacity": opacity_dual_res,
        }

    @torch.no_grad()
    def update_block_dual_variables(self):
        futures = []
        for worker in self.remote_workers.values():
            futures.append(worker.rpc_async().update_dual_variables())

        for future in futures:
            future.wait()

    def collect_block_gaussians(self) -> Dict:
        xyz, opacity = [], []
        features_dc, features_rest = [], []
        scaling, quaternion = [], []

        block_gaussians, futures = {}, {}
        for block_id, worker in self.remote_workers.items():
            futures[block_id] = worker.rpc_async().send_local_model()

        for block_id, future in futures.items():
            xyz, features_dc, features_rest, scaling, quaternion, opacity = future.wait()
            gaussians = GaussianSplatModel(
                max_sh_degree=self.config.texture.max_sh_degree,
                percent_dense=self.config.geometry.percent_dense,
            )
            gaussians.init_from_external_properties(
                xyz, features_dc, features_rest, scaling, quaternion, opacity
            )
            block_gaussians[block_id] = gaussians

        return block_gaussians

    def collect_block_masks(self) -> Dict:
        block_masks, futures = {}, {}
        for block_id, worker in self.remote_workers.items():
            futures[block_id] = worker.rpc_async().send_local_mask()

        for block_id, future in futures.items():
            block_masks[block_id] = future.wait()

        return block_masks

    def collect_block_losses(self):
        futures = {}
        for block_id, worker in self.remote_workers.items():
            futures[block_id] = worker.rpc_async().send_local_loss()

        for k, future in futures.items():
            loss = future.wait()
            self.scalars_to_log[f"block{k}/psnr"] = loss["train/psnr"]
            self.scalars_to_log[f"block{k}/loss"] = loss["train/loss"]
            self.scalars_to_log[f"block{k}/l1_loss"] = loss["train/l1_loss"]
            self.scalars_to_log[f"block{k}/ema_loss"] = loss["train/ema_loss"]
            self.scalars_to_log[f"block{k}/points"] = loss["train/points"]

            if (self.iteration > self.config.geometry.densify_end_iter) and \
                (self.iteration % self.config.trainer.admm.consensus_interval == 0) and \
                    self.config.trainer.admm.enable:
                self.scalars_to_log[f"block{k}/xyz_penalty"] = loss["train/xyz_penalty"]
                self.scalars_to_log[f"block{k}/feat_dc_penalty"] = loss["train/feat_dc_penalty"]
                self.scalars_to_log[f"block{k}/feat_rest_penalty"] = loss["train/feat_rest_penalty"]
                self.scalars_to_log[f"block{k}/scaling_penalty"] = loss["train/scaling_penalty"]
                self.scalars_to_log[f"block{k}/quat_penalty"] = loss["train/quaternion_penalty"]
                self.scalars_to_log[f"block{k}/opacity_penalty"] = loss["train/opacity_penalty"]

    @torch.no_grad()
    def broadcast_global_gaussian_splat(self):
        futures = []
        for block_id, worker in self.remote_workers.items():
            indices = self.global_indices[block_id]
            xyz, features_dc, features_rest, scaling, quaternion, opacity = \
                self.gaussians.get_all_properties(indices=indices)
            futures.append(
                worker.rpc_async().set_global_gaussians(
                    xyz, features_dc, features_rest, scaling, quaternion, opacity
                ))

        for future in futures:
            future.wait()

    @torch.no_grad()
    def gaussian_splat_consensus(self):
        self.prev_xyz = self.gaussians.get_xyz
        self.prev_features_dc = self.gaussians.get_features_dc
        self.prev_features_rest = self.gaussians.get_features_rest
        self.prev_scaling = self.gaussians.get_raw_scaling
        self.prev_quaternion = self.gaussians.get_raw_quaternion
        self.prev_opacity = self.gaussians.get_raw_opacity

        # Set all properties of global 3D GS to zeros.
        self.gaussians.reinitialize()

        self.block_gaussians = self.collect_block_gaussians()
        for k, gaussians in self.block_gaussians.items():
            indices = self.global_indices[k]
            self.gaussians.plus_gaussians(gaussians, indices=indices)

        self.gaussians.average_gaussians(
            count=self.visibility_count.unsqueeze(-1))

    @torch.no_grad()
    def fuse_local_gaussians(self):
        block_gaussians = self.collect_block_gaussians()
        block_masks = None # self.collect_block_masks()

        xyz, features_dc, features_rest, scaling, quaternion, opacity, _ = \
            fuse_block_gaussians(
                block_gaussians,
                self.ori_point_bboxes,
                self.world_to_obb_transform,
                self.evaluator.output_dir
            )
        self.gaussians.init_from_external_properties(
            xyz, features_dc, features_rest, scaling, quaternion, opacity, optimizable=False
        )
        num_gaussians = self.gaussians.get_xyz.shape[0]
        print(f'[fuse_local_gaussians] num gaussians: {xyz.shape}')

        data_dir = os.path.join(self.config.dataset.root_dir, self.config.dataset.scene)
        camera_blocks = []
        for block_id in range(self.num_blocks):
            block_dir = get_block_info_dir(
                data_dir,
                self.config.dataset.num_blocks,
                self.config.dataset.get("mx", None),
                self.config.dataset.get("my", None),
            )
            block_dir = os.path.join(block_dir, f'block_{block_id}')
            dataset = MiniDataset(cameras=[], camtoworlds=None, block_id=block_id)
            dataset.read(path=block_dir, read_image=False, block_id=block_id, device='cpu')
            camera_blocks.append(dataset.cameras)

        prune_gaussians_after_merge(self.config, self.gaussians, camera_blocks, self.device)
        num_gaussians = self.gaussians.get_xyz.shape[0]
        print(f'[AFTER PRUNING] xyz shape: {self.gaussians.get_xyz.shape}')

        self.evaluator.models = [self.gaussians]

        if not self.config.trainer.admm.enable:
            return

        self.visibility_count, self.global_indices, local_gaussians = \
            select_gaussians_in_each_block(
                self.exp_point_bboxes, self.gaussians,
                self.device, self.evaluator.output_dir, self.world_to_obb_transform
            )

        # Empty remote worker resources.
        for worker in self.remote_workers.values():
            worker.rpc_async().clean().wait()

        self.init_block_trainers(
            sub_gaussians=local_gaussians, sub_masks=block_masks)
        self.setup_penalty_parameters(num_gaussians=num_gaussians)
        self.set_block_penalty_parameters()

        for block_id, worker in self.remote_workers.items():
            worker.rpc_async().update_iteration(self.iteration).wait()
            worker.rpc_async().setup_dual_variables().wait()
            worker.rpc_async().enable_admm_training().wait()

        torch.cuda.empty_cache()

    def train(self):
        desc = f"Training {self.config.expname}"
        pbar = tqdm.trange(self.config.trainer.max_iterations, desc=desc)

        iter_start = self.load_checkpoint(
            load_optimizer=not self.config.trainer.no_load_opt,
            load_scheduler=not self.config.trainer.no_load_scheduler,
        )
        if iter_start >= self.config.trainer.max_iterations:
            return

        score = 0
        self.iteration = iter_start
        pbar.update(self.iteration)

        while self.iteration < self.config.trainer.max_iterations:
            data_batch = ""

            self.train_iteration(data_batch=data_batch)

            # log to tensorboard.
            if self.iteration % self.config.trainer.n_tensorboard == 0:
                self.log_info()

            if self.iteration % self.config.trainer.n_checkpoint == 0:
                self.compose_state_dicts()
                self.save_checkpoint(score=score)

            if self.iteration % self.config.trainer.n_validation == 0 and \
               self.iteration != 0:
                score = self.validate()

            pbar.update(self.config.trainer.admm.consensus_interval)

            if self.iteration > self.config.trainer.max_iterations:
                break

        if self.config.trainer.n_checkpoint % self.config.trainer.n_validation != 0 and \
           self.config.trainer.n_validation % self.config.trainer.n_checkpoint != 0:
            score = self.validate()
            self.compose_state_dicts()
            self.save_checkpoint(score=score)

        self.train_done = True

    def train_iteration(self, data_batch) -> None:  # pylint: disable=[W0613]
        # (1) Make async RPC to optimize local gaussian splats on all trainers.
        futures = []
        for worker in self.remote_workers.values():
            futures.append(rpc_async(
                worker.owner(),
                worker.rpc_sync().train_every_x_interval,
                args=("", self.config.trainer.admm.consensus_interval,)
            ))

        # Wait until all observers have finished this iteration.
        for future in futures:
            future.wait()

        self.iteration += self.config.trainer.admm.consensus_interval

        if self.iteration % self.config.trainer.n_tensorboard == 0:
            self.collect_block_losses()

        # The consensus and sharing of gaussian splats only activated when the
        # densification step finished.
        if self.iteration < self.config.geometry.densify_end_iter or \
           not self.config.trainer.admm.enable:
            return

        if self.iteration >= self.config.geometry.densify_end_iter and \
           self.gaussians.get_xyz.shape[0] == 0:
            self.fuse_local_gaussians()
            self.validate()

        # if self.iteration % self.config.trainer.admm.consensus_interval == 0:

        # (2) Global gaussian splats consensus by averaging all local gaussian splats.
        self.gaussian_splat_consensus()

        # (3) Broadcast the global gaussian splats to all local trainers.
        self.broadcast_global_gaussian_splat()

        # (4) Update the dual variables in all local trainers.
        self.update_block_dual_variables()

        # (5) Compute the primal residuals and dual residuals.
        primal_res_dict = self.compute_primal_residual()
        dual_res_dict = self.compute_dual_residual()

        # (6) Self-adaptation of penalty parameters and update the penalty
        # parameters in each block.
        if self.iteration <= self.config.trainer.admm.stop_adapt_iter:
            self.adapt_penalty_parameters(
                primal_res_dict=primal_res_dict,
                dual_res_dict=dual_res_dict
            )
            self.set_block_penalty_parameters()

        self.scalars_to_log["penalty/rho_xyz"] = self.rho_xyz
        self.scalars_to_log["penalty/rho_fdc"] = self.rho_fdc
        self.scalars_to_log["penalty/rho_fr"] = self.rho_fr
        self.scalars_to_log["penalty/rho_s"] = self.rho_s
        self.scalars_to_log["penalty/rho_q"] = self.rho_q
        self.scalars_to_log["penalty/rho_o"] = self.rho_o
        self.scalars_to_log["train/primal_residual"] = \
            sum(primal_res_dict.values()).detach().item()
        self.scalars_to_log["train/dual_residual"] = \
            sum(dual_res_dict.values()).detach().item()

    def validate(self) -> float:
        futures = []
        for worker in self.remote_workers.values():
            futures.append(worker.rpc_async().validate())

        if not self.config.trainer.admm.enable:
            self.fuse_local_gaussians()

        if self.iteration >= self.config.geometry.densify_end_iter:
            super().validate()

        for future in futures:
            future.wait()

        return 0

    def log_info(self) -> None:
        super().log_info()

    def save_checkpoint(self, score: float = 0.0):
        # futures = {}
        # for block_id, worker in self.remote_workers.items():
        #     futures[block_id] = worker.rpc_async().save_checkpoint(0)

        super().save_checkpoint(score=score)

        # for _, future in futures.items():
        #     future.wait()

    def load_checkpoint(
        self,
        load_model=True,     # pylint: disable=W0613
        load_optimizer=True,  # pylint: disable=W0613
        load_scheduler=True,  # pylint: disable=W0613
        load_meta_data=False  # pylint: disable=W0613
    ) -> int:
        iter_start = super().load_checkpoint(
            False, load_optimizer, False, load_meta_data=True
        )

        meta_data = self.state_dicts["meta_data"]
        self.gaussians.init_from_external_properties(
            meta_data["xyz"], meta_data["features_dc"], meta_data["features_rest"],
            meta_data["scaling"], meta_data["quaternion"], meta_data["opacity"],
        )

        # penalty parameters
        if iter_start >= self.config.geometry.densify_end_iter and \
           self.config.trainer.admm.enable:
            self.rho_xyz = self.state_dicts["meta_data"]["rho_xyz"]
            self.rho_fdc = self.state_dicts["meta_data"]["rho_fdc"]
            self.rho_fr = self.state_dicts["meta_data"]["rho_fr"]
            self.rho_s = self.state_dicts["meta_data"]["rho_s"]
            self.rho_q = self.state_dicts["meta_data"]["rho_q"]
            self.rho_o = self.state_dicts["meta_data"]["rho_o"]

        return iter_start


def run(config: OmegaConf):
    """ Distributed function to be implemented later. """
    rank = int(
        os.environ['RANK'])  # Defines the ID of a worker in the world (all nodes combined)
    # The rank of the worker group (0 - max_nnodes)
    group_rank = int(os.environ["GROUP_RANK"])
    # Defines the ID of a worker within a node.
    local_rank = int(os.environ["LOCAL_RANK"])
    # Defines the total number of workers.
    world_size = int(os.environ['WORLD_SIZE'])
    local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])

    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=64, rpc_timeout=800)
    print(f'rank: {rank}; local_rank: {local_rank}, group_rank: {group_rank} ' +
          f'world size: {world_size}, local_world_size: {local_world_size}')

    if group_rank == 0 and local_rank == 0:
        print('master set device map')
        # Set device mapping to enable transfer data across different GPUs.
        # https://h-huang.github.io/tutorials/recipes/cuda_rpc.html
        for i in range(1, world_size):
            worker_device_id = i % 4  # local_world_size
            # options.set_device_map(f"worker{i}", {0: i-1})
            options.set_device_map(f"worker{i}", {0: worker_device_id})

        # https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.init_rpc
        rpc.init_rpc(
            name=MASTER_NAME,
            rank=0,
            world_size=world_size,
            rpc_backend_options=options,
        )

        gs_master = MasterGaussianSplatTrainer(config=config)
        gs_master.train()

    else:
        worker_name = f"worker{rank}"
        rpc.init_rpc(
            worker_name,
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options,
        )

    # Block until all rpcs finish, and shutdown the RPC instance.
    rpc.shutdown()


if __name__ == "__main__":
    args = config_parser()

    # parse YAML config to OmegaConf
    config = load_config(args.config)
    config["config_file_path"] = args.config

    assert config.dataset.scene != ""

    setup_seed(config.seed)

    torch.distributed.init_process_group(backend='nccl')

    scenes = []
    if (
        type(config.dataset.scene) == omegaconf.listconfig.ListConfig  # pylint: disable=C0123
    ):
        scene_list = list(config.dataset.scene)
        for sc in config.dataset.scene:
            scenes.append(sc)
    else:
        scenes.append(config.dataset.scene)

    for scene in scenes:
        data_dir = os.path.join(config.dataset.root_dir, scene)
        assert os.path.exists(data_dir), f"Dataset does not exist: {data_dir}!"

        local_config = copy.deepcopy(config)
        local_config.expname = (
            f"{config.neural_field_type}_{config.task}_{config.dataset.name}_{scene}"
        )
        local_config.expname = local_config.expname + "_" + args.suffix
        local_config.dataset.scene = scene

        run(local_config)
