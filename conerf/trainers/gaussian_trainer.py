# pylint: disable=[E1101,R1719,W0201]

import os
import random
import copy
from random import randint
from typing import List
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
import numpy as np

from conerf.base.model_base import ModelBase
from conerf.datasets.dataset_base import MiniDataset
from conerf.datasets.utils import (
    fetch_ply, compute_nerf_plus_plus_norm, create_dataset, get_block_info_dir
)
from conerf.loss.ssim_torch import ssim
from conerf.model.gaussian_fields.gaussian_splat_model import GaussianSplatModel
from conerf.model.gaussian_fields.masks import AppearanceEmbedding
from conerf.render.gaussian_render import render
from conerf.model.gaussian_fields.prune import calculate_v_imp_score, prune_list
from conerf.trainers.implicit_recon_trainer import ImplicitReconTrainer
from conerf.visualization.pose_visualizer import visualize_cameras


# NOTE: We define a callable class instead of a closure since a closure cannot
# be serializable by torch's rpc.
class ExponentialLR: # pylint: disable=[R0903]
    def __init__(
        self,
        lr_init: float,
        lr_final: float,
        lr_delay_steps: int = 0,
        lr_delay_mult: float = 1.0,
        max_steps: int = 1000000
    ):
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.lr_delay_steps = lr_delay_steps
        self.lr_delay_mult = lr_delay_mult
        self.max_steps = max_steps

    def __call__(self, step):
        if step < 0 or (self.lr_init == 0.0 and self.lr_final == 0.0):
            return 0.0

        if self.lr_delay_steps > 0:
            delay_rate = self.lr_delay_mult + (1 - self.lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / self.lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0

        t = np.clip(step / self.max_steps, 0, 1)
        log_lerp = np.exp(np.log(self.lr_init) * (1 - t) + np.log(self.lr_final) * t)

        return delay_rate * log_lerp


def extract_pose_tensor_from_cameras(
    cameras: List,
    noises: torch.Tensor = None,
    corrections: torch.Tensor = None,
) -> torch.Tensor:
    num_cameras = len(cameras)
    poses = torch.zeros((num_cameras, 4, 4), dtype=torch.float32).cuda()

    for i in range(num_cameras):
        poses[i, :3, :3] = cameras[i].R
        poses[i, :3, 3] = cameras[i].t
        poses[i, 3, 3] = 1.0

        # Add noise.
        if noises is not None:
            poses[i] = poses[i] @ noises[i]

        # Correct noise.
        if corrections is not None:
            poses[i] = poses[i] @ corrections[i]

    return poses


def load_val_dataset(config: OmegaConf, device: str = 'cuda'):
    val_config = copy.deepcopy(config)
    val_config.dataset.multi_blocks = False
    val_config.dataset.num_blocks = 1
    val_dataset = create_dataset(
        config=val_config,
        split=val_config.dataset.val_split,
        num_rays=None,
        apply_mask=val_config.dataset.apply_mask,
        device=device
    )
    return val_dataset


class GaussianSplatTrainer(ImplicitReconTrainer):
    """
    Trainer for 3D Gaussian Splatting model.
    """

    def __init__(
        self,
        config: OmegaConf,
        prefetch_dataset: bool = True,
        trainset=None,
        valset=None,
        model: ModelBase = None,
        block_id: int = None,
        device_id: int = 0,
    ) -> None:
        self.gaussians = None

        if trainset is None and block_id is not None:
            mx = config.dataset.get("mx", None) # pylint: disable=C0103
            my = config.dataset.get("my", None) # pylint: disable=C0103
            data_dir = os.path.join(config.dataset.root_dir, config.dataset.scene)
            data_dir = get_block_info_dir(data_dir, config.dataset.num_blocks, mx, my)
            block_dir = os.path.join(data_dir, f'block_{block_id}')
            trainset = MiniDataset(cameras=[], camtoworlds=None, block_id=block_id)
            trainset.read(path=block_dir, block_id=block_id, device='cpu')

        if valset is None:
            valset = load_val_dataset(config, 'cpu')

        self.admm_enabled = False

        super().__init__(config, prefetch_dataset, trainset, valset, model, block_id, device_id)

    def init_gaussians(self):
        # Using semantic alias to better understand the code.
        self.gaussians = self.model
        # Initialize 3D Gaussians from COLMAP point clouds.
        data_dir = os.path.join(
            self.config.dataset.root_dir, self.config.dataset.scene)
        pcl_name = "points3D"
        if self.config.dataset.multi_blocks:
            pcl_name += f"_{self.train_dataset.current_block}"
            mx = self.config.dataset.get("mx", None) # pylint: disable=C0103
            my = self.config.dataset.get("my", None) # pylint: disable=C0103
            data_dir = get_block_info_dir(data_dir, self.config.dataset.num_blocks, mx, my)
            colmap_ply_path = os.path.join(data_dir, f"{pcl_name}.ply")
        else:
            dense = '' if self.config.dataset.init_ply_type == "sparse" else "_dense"
            colmap_ply_path = os.path.join(
                data_dir, self.config.dataset.model_folder, "0",
                f"{pcl_name}{dense}.ply"
            )
            print(f'Initialize 3DGS using {colmap_ply_path}')

        point_cloud = fetch_ply(colmap_ply_path)
        self.gaussians.init_from_colmap_pcd(point_cloud)

        # Precompute 3D filter for anti-aliasing.
        if self.config.texture.anti_aliasing:
            self.gaussians.compute_3D_filter(cameras=self.train_cameras)

    def build_networks(self):
        self.model = GaussianSplatModel(
            max_sh_degree=self.config.texture.max_sh_degree,
            percent_dense=self.config.geometry.percent_dense,
            anti_aliasing=self.config.texture.anti_aliasing,
            device=self.device,
        )

        if self.config.geometry.get("mask", False):
            self.mask = AppearanceEmbedding(len(self.train_dataset.cameras)).to(self.device)

        self.init_gaussians()

    def setup_training_params(self):
        self.color_bkgd = torch.tensor(
            [0, 0, 0], dtype=torch.float32, device=self.device)
        self.ema_loss = 0.0
        self.use_white_bkgd = \
            True if self.config.dataset.apply_mask else False
        self.viewpoint_stack = None
        self.depth_threshold = self.config.geometry.get("depth_threshold", 0.0)

        random.shuffle(self.train_dataset.cameras)
        self.train_cameras = self.train_dataset.cameras.copy()

        spatial_lr_scale = self.config.geometry.get("spatial_lr_scale", -1)
        if spatial_lr_scale < 0:
            self.spatial_lr_scale = compute_nerf_plus_plus_norm(self.train_cameras)
        else:
            self.spatial_lr_scale = spatial_lr_scale

        # High resolution index.
        self.high_resolution_index = []
        for index, camera in enumerate(self.train_cameras):
            if camera.width >= 800:
                self.high_resolution_index.append(index)

    def setup_optimizer(self):
        # Trivial hack when model is passed to the constructor.
        if self.gaussians is None:
            self.gaussians = self.model

        lr_config = self.config.optimizer.lr
        lr_params = [
            {
                'params': [self.gaussians.get_xyz],
                'lr': lr_config.position_init * self.spatial_lr_scale, "name": "xyz",
            },
            {
                'params': [self.gaussians.get_features_dc],
                'lr': lr_config.feature, "name": "f_dc",
            },
            {
                'params': [self.gaussians.get_features_rest],
                'lr': lr_config.feature / 20.0, "name": "f_rest",
            },
            {
                'params': [self.gaussians.get_raw_opacity],
                'lr': lr_config.opacity, "name": "opacity",
            },
            {
                'params': [self.gaussians.get_raw_scaling],
                'lr': lr_config.scaling, "name": "scaling",
            },
            {
                'params': [self.gaussians.get_raw_quaternion],
                'lr': lr_config.quaternion, "name": "quaternion",
            },
        ]

        self.mask_optimizer = None
        if 'mask' in self.config.geometry and self.config.geometry.mask:
            self.mask_optimizer = torch.optim.Adam(self.mask.parameters(), lr=lr_config.mask)

        self.optimizer = torch.optim.Adam(lr_params, lr=0.0, eps=1e-15)
        self.xyz_scheduler = ExponentialLR(
            lr_init=lr_config.position_init * self.spatial_lr_scale,
            lr_final=lr_config.position_final * self.spatial_lr_scale,
            lr_delay_mult=lr_config.position_delay_mult,
            max_steps=lr_config.position_max_iterations
        )

        self.setup_pose_optimizer()

    def setup_pose_optimizer(self):
        if not self.optimize_camera_poses:
            return

        self.gt_poses = extract_pose_tensor_from_cameras(self.train_dataset.cameras)

        pose_params = []
        for camera in self.train_dataset.cameras:
            if camera.image_index == 0:
                print('skip image 0!')
                continue

            pose_params.append({
                "params": [camera.delta_rot],
                "lr": self.config.optimizer.lr_pose.rot,
                "name": f"rot_{camera.image_index}",
            })
            pose_params.append({
                "params": [camera.delta_trans],
                "lr": self.config.optimizer.lr_pose.trans,
                "name": f"trans_{camera.image_index}",
            })

        self.pose_optimizer = torch.optim.Adam(pose_params)

    def update_learning_rate(self):
        """Learning rate scheduling per step."""
        if self.xyz_scheduler is None:
            return

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler(self.iteration)
                param_group['lr'] = lr
                return lr

    def training_resolution(self) -> int:
        if not self.config.geometry.get('coarse-to-fine', False):
            return 1

        n_interval = 3
        iteration_threshold = min(20000, self.config.geometry.densify_end_iter) // n_interval
        resolution = 2 ** max(n_interval - self.iteration // iteration_threshold - 1, 0)

        return resolution

    def update_iteration(self, iteration: int):
        self.iteration = iteration

    @torch.no_grad()
    def send_local_model(self):
        xyz, features_dc, features_rest, scaling, quaternion, opacity = \
            self.gaussians.get_all_properties()

        return xyz, features_dc, features_rest, scaling, quaternion, opacity

    @torch.no_grad()
    def send_local_loss(self):
        return self.scalars_to_log

    def setup_dual_variables(self):
        # Set dual variables for all the properties of 3D Gaussians:
        # xyz, features_dc, features_rest, scaling, quaternion, opacity.
        # We follow DBACC (Distributed Bundle Adjustment based on Camera Consensus) to
        # initialize all dual variables to zeros.
        self.u_xyz = torch.zeros_like(self.gaussians.get_xyz, requires_grad=False)
        self.u_fdc = torch.zeros_like(self.gaussians.get_features_dc, requires_grad=False)
        self.u_fr = torch.zeros_like(self.gaussians.get_features_rest, requires_grad=False)
        self.u_s = torch.zeros_like(self.gaussians.get_raw_scaling, requires_grad=False)
        self.u_q = torch.zeros_like(self.gaussians.get_raw_quaternion, requires_grad=False)
        self.u_o = torch.zeros_like(self.gaussians.get_raw_opacity, requires_grad=False)

    @torch.no_grad()
    def update_dual_variables(self):
        # Apply over-relaxation by:
        #   u^{k+1} = u^k + (1 + alpha^k) * (x^{k+1} - z^{k+1}).
        over_relaxation_factor = 1 + self.config.trainer.admm.over_relaxation_coeff
        self.u_xyz += over_relaxation_factor * (
            self.gaussians.get_xyz - self.global_xyz
        )
        self.u_fdc += over_relaxation_factor * (
            self.gaussians.get_features_dc - self.global_features_dc
        )
        self.u_fr += over_relaxation_factor * (
            self.gaussians.get_features_rest - self.global_features_rest
        )
        self.u_s += over_relaxation_factor * (
            self.gaussians.get_raw_scaling - self.global_scaling
        )
        self.u_q += over_relaxation_factor * (
            self.gaussians.get_raw_quaternion - self.global_quaternion
        )
        self.u_o += over_relaxation_factor * (
            self.gaussians.get_raw_opacity - self.global_opacity
        )

    def set_penalty_parameters(
        self,
        rho_xyz: float,
        rho_fdc: float,
        rho_fr: float,
        rho_s: float,
        rho_q: float,
        rho_o: float,
    ):
        self.rho_xyz = rho_xyz
        self.rho_fdc = rho_fdc
        self.rho_fr = rho_fr
        self.rho_s = rho_s
        self.rho_q = rho_q
        self.rho_o = rho_o

    def set_global_indices(self, global_indices: torch.Tensor):
        self.global_indices = global_indices

    def set_global_gaussians(
        self,
        global_xyz,
        global_features_dc,
        global_features_rest,
        global_scaling,
        global_quaternion,
        global_opacity
    ):
        self.global_xyz = global_xyz
        self.global_features_dc = global_features_dc
        self.global_features_rest = global_features_rest
        self.global_scaling = global_scaling
        self.global_quaternion = global_quaternion
        self.global_opacity = global_opacity

    def enable_admm_training(self):
        self.admm_enabled = True

    def add_admm_penalties(self, loss):
        xyz = self.gaussians.get_xyz
        features_dc = self.gaussians.get_features_dc
        features_rest = self.gaussians.get_features_rest
        scaling = self.gaussians.get_raw_scaling
        quaternion = self.gaussians.get_raw_quaternion
        opacity = self.gaussians.get_raw_opacity

        xyz_penalty = 0.5 * self.rho_xyz * F.mse_loss(
            xyz + self.u_xyz, self.global_xyz
        )
        feat_dc_penalty = 0.5 * self.rho_fdc * F.mse_loss(
            features_dc + self.u_fdc, self.global_features_dc
        )
        feat_rest_penalty = 0.5 * self.rho_fr * F.mse_loss(
            features_rest + self.u_fr, self.global_features_rest
        )
        scaling_penalty = 0.5 * self.rho_s * F.mse_loss(
            scaling + self.u_s, self.global_scaling
        )
        quaternion_penalty = 0.5 * self.rho_q * F.mse_loss(
            quaternion + self.u_q, self.global_quaternion
        )
        opacity_penalty = 0.5 * self.rho_o * F.mse_loss(
            opacity + self.u_o, self.global_opacity
        )

        loss += xyz_penalty
        loss += feat_dc_penalty
        loss += feat_rest_penalty
        loss += scaling_penalty
        loss += quaternion_penalty
        loss += opacity_penalty

        self.scalars_to_log["train/xyz_penalty"] = xyz_penalty.detach().item()
        self.scalars_to_log["train/feat_dc_penalty"] = feat_dc_penalty.detach().item()
        self.scalars_to_log["train/feat_rest_penalty"] = feat_rest_penalty.detach().item()
        self.scalars_to_log["train/scaling_penalty"] = scaling_penalty.detach().item()
        self.scalars_to_log["train/quaternion_penalty"] = quaternion_penalty.detach().item()
        self.scalars_to_log["train/opacity_penalty"] = opacity_penalty.detach().item()

        return loss

    def train_every_x_interval(self, data_batch, interval: int = 100):
        for i in range(interval): # pylint: disable=W0612
            self.increment_iteration()
            self.train_iteration(data_batch=data_batch)

    def train_iteration(self, data_batch) -> None: # pylint: disable=W0613
        self.gaussians.train()
        self.update_learning_rate()

        # Increase the levels of SH up to a maximum degree.
        if self.iteration % 1000 == 0:
            self.gaussians.increase_SH_degree()

        # Pick a random camera.
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.train_dataset.cameras.copy() # pylint: disable=W0201
        camera = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack) - 1))
        resolution = self.training_resolution()
        camera_origin = camera.copy_to_device(self.device)
        camera = camera.downsample(resolution).copy_to_device(self.device)
        self.scalars_to_log['train/resolution'] = resolution

        # Pick a random high resolution camera.
        if self.config.texture.anti_aliasing and \
           (random.random() < 0.3 and self.config.geometry.sample_more_high_resolution):
            camera = self.train_cameras[
                self.high_resolution_index[randint(0, len(self.high_resolution_index) - 1)]
            ]

        # Since we only update on the copy of cameras, the update won't
        # be accumulated continuously on the same cameras.
        if self.optimize_camera_poses and (camera.image_index != 0) and \
           (self.iteration > self.config.geometry.opt_pose_start_iter):
            image_index = camera.image_index

        render_results = render(
            gaussian_splat_model=self.gaussians,
            viewpoint_camera=camera,
            pipeline_config=self.config.pipeline,
            bkgd_color=self.color_bkgd,
            kernel_size=self.config.geometry.kernel_size,
            anti_aliasing=self.config.texture.anti_aliasing,
            subpixel_offset=None,
            depth_threshold=self.depth_threshold * self.spatial_lr_scale,
            device=self.device,
        )
        colors, screen_space_points, visibility_filter, radii = (
            render_results["rendered_image"],
            render_results["screen_space_points"],
            render_results["visibility_filter"],
            render_results["radii"],
        )

        # Compute loss.
        if self.config.geometry.get("mask", False):
            pixels = camera.image.permute(2, 0, 1)
            loss_ssim = ssim(pixels, colors)
            image_size = camera.image.shape[:-1]
            camera = camera_origin.downsample(32).copy_to_device(self.device)
            mask = self.mask(camera.image.permute(2, 0, 1), camera.image_index, image_size)
            # print(f"[Debug] {mask}")
            loss_rgb_l1 = F.l1_loss(colors * mask, pixels)
            lambda_dssim = self.config.loss.lambda_dssim
            loss = (1.0 - lambda_dssim) * loss_rgb_l1 + \
                lambda_dssim * (1.0 - loss_ssim) + \
                0.5 * torch.mean((mask - 1) ** 2.) # Regularization for mask
        else:
            pixels = camera.image.permute(2, 0, 1)  # [RGB, height, width]
            loss_rgb_l1 = F.l1_loss(colors, pixels)
            loss_ssim = ssim(pixels, colors)
            lambda_dssim = self.config.loss.lambda_dssim
            loss = (1.0 - lambda_dssim) * loss_rgb_l1 + \
                lambda_dssim * (1.0 - loss_ssim)

        if self.admm_enabled:
            loss = self.add_admm_penalties(loss)

        loss.backward()

        self.ema_loss = 0.4 * loss.detach().item() + 0.6 * self.ema_loss  # pylint: disable=W0201

        mse = F.mse_loss(colors, pixels)
        psnr = -10.0 * torch.log(mse) / np.log(10.0)

        # training statistics.
        self.scalars_to_log["train/psnr"] = psnr.detach().item()
        self.scalars_to_log["train/loss"] = loss.detach().item()
        self.scalars_to_log["train/l1_loss"] = loss_rgb_l1.detach().item()
        self.scalars_to_log["train/ema_loss"] = self.ema_loss
        self.scalars_to_log["train/points"] = self.gaussians.get_xyz.shape[0]

        with torch.no_grad():
            # Densification.
            if self.iteration < self.config.geometry.densify_end_iter:
                # Keep track of max radii in image-space for pruning.
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter]
                )
                self.gaussians.add_densification_stats(
                    screen_space_points, visibility_filter, pixels=render_results["pixels"])

                if self.iteration > self.config.geometry.densify_start_iter and \
                        self.iteration % self.config.geometry.densification_interval == 0:
                    size_threshold = 20 \
                        if self.iteration > self.config.geometry.opacity_reset_interval else None
                    self.gaussians.densify_and_prune(
                        max_grad=self.config.geometry.densify_grad_threshold,
                        min_opacity=0.005,
                        extent=self.spatial_lr_scale,
                        max_screen_size=size_threshold,
                        optimizer=self.optimizer,
                    )

                    if self.config.texture.anti_aliasing:
                        self.gaussians.compute_3D_filter(cameras=self.train_cameras)

                if self.iteration % self.config.geometry.opacity_reset_interval == 0 or \
                (self.use_white_bkgd and self.iteration == self.config.geometry.densify_start_iter):
                    if self.config.texture.anti_aliasing:
                        self.gaussians.reset_opacity_with_3D_filter(self.optimizer)
                    else:
                        self.gaussians.reset_opacity(self.optimizer)

            if self.config.texture.anti_aliasing and \
            (self.iteration % 100 == 0 and self.iteration > self.config.geometry.densify_end_iter):
                if self.iteration < self.config.trainer.max_iterations - 100:
                    # Do not update in the end of training.
                    self.gaussians.compute_3D_filter(cameras=self.train_cameras)

            if self.iteration in list(self.config.prune.iterations):
                gaussian_list, imp_list = prune_list( # pylint: disable=W0612
                    self.gaussians, self.train_dataset.cameras.copy(),
                    self.config.pipeline, self.color_bkgd
                )
                v_list = calculate_v_imp_score(self.gaussians, imp_list, self.config.prune.v_pow)
                i = self.config.prune.iterations.index(self.iteration)
                self.gaussians.prune_gaussians_with_opt(
                    (self.config.prune.prune_decay**i) * self.config.prune.prune_percent,
                    v_list, self.optimizer
                )

        # Optimizer step.
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        if self.mask_optimizer is not None:
            self.mask_optimizer.step()
            self.mask_optimizer.zero_grad(set_to_none=True)

        if self.pose_optimizer is not None:
            self.pose_optimizer.step()
            self.pose_optimizer.zero_grad(set_to_none=True)

        if self.pose_scheduler is not None:
            self.pose_scheduler.step()

        if self.iteration % self.config.trainer.n_checkpoint == 0:
            self.compose_state_dicts()

        # Update camera pose.
        if self.optimize_camera_poses and (camera.image_index != 0) and \
           (self.iteration > self.config.geometry.opt_pose_start_iter):
            self.train_dataset.cameras[image_index].update_camera_pose()

        # Visualize camera pose.
        if self.pose_optimizer is not None and \
           self.iteration % self.config.trainer.n_tensorboard == 0:
            with torch.no_grad():
                pred_poses = extract_pose_tensor_from_cameras(self.train_dataset.cameras)
                visualize_cameras(
                    self.visdom,
                    self.iteration,
                    poses=[torch.inverse(self.gt_poses), torch.inverse(pred_poses)],
                    cam_depth=0.1
                )

    def compose_state_dicts(self) -> None:
        self.state_dicts = {
            "models": dict(),
            "optimizers": dict(),
            "schedulers": dict(),  # No scheduler needs to be stored.
            "meta_data": dict(),
        }

        # self.state_dicts["models"]["model"] = None
        self.state_dicts["optimizers"]["optimizer"] = self.optimizer

        # Pose related items.
        if self.delta_pose is not None:
            self.state_dicts["models"]["delta_pose"] = self.delta_pose
        if self.config.get("pose_optimizer", None) is not None:
            self.state_dicts["optimizers"]["pose_optimizer"] = self.pose_optimizer
        if self.config.get("pose_scheduler", None) is not None:
            self.state_dicts["schedulers"]["pose_scheduler"] = self.pose_scheduler

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
        self.state_dicts["meta_data"]["spatial_lr_scale"] = self.spatial_lr_scale

        if self.config.dataset.multi_blocks:
            self.state_dicts["meta_data"]["block_id"] = self.train_dataset.current_block
        self.state_dicts["meta_data"]["camera_poses"] = self.train_dataset.camtoworlds

    def load_checkpoint(
        self,
        load_model=True,     # pylint: disable=W0613
        load_optimizer=True, # pylint: disable=W0613
        load_scheduler=True, # pylint: disable=W0613
        load_meta_data=False # pylint: disable=W0613
    ) -> int:
        iter_start = super().load_checkpoint(
            False, load_optimizer, False, load_meta_data=True
        )

        meta_data = self.state_dicts["meta_data"]
        self.gaussians.active_sh_degree = meta_data["active_sh_degree"]
        self.gaussians.set_xyz(meta_data["xyz"])
        self.gaussians.set_features_dc(meta_data["features_dc"])
        self.gaussians.set_features_rest(meta_data["features_rest"])
        self.gaussians.set_raw_scaling(meta_data["scaling"])
        self.gaussians.set_raw_quaternion(meta_data["quaternion"])
        self.gaussians.set_raw_opacity(meta_data["opacity"])
        self.gaussians.max_radii2D = meta_data["max_radii2D"]
        self.gaussians.xyz_gradient_accum = meta_data["xyz_gradient_accum"]
        self.gaussians.denom = meta_data["denom"]
        self.spatial_lr_scale = meta_data["spatial_lr_scale"] # pylint: disable=W0201

        return iter_start