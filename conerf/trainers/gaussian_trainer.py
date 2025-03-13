# pylint: disable=[E1101,R1719,W0201]

import os
import random
import copy
from typing import List
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
import numpy as np

from conerf.base.model_base import ModelBase
from conerf.base.task_queue import ImageReader
from conerf.datasets.utils import (
    fetch_ply, compute_nerf_plus_plus_norm, create_dataset, get_block_info_dir
)
# from conerf.loss.ssim_torch import ssim
from conerf.model.gaussian_fields.gaussian_splat_model import GaussianSplatModel
from conerf.model.gaussian_fields.masks import DecoupledAppearanceEmbedding
from conerf.model.gaussian_fields.pose_embed import CameraOptModule
from conerf.model.gaussian_fields.app_embed import AppearanceOptModule
from conerf.render.gaussian_render import render, render_gsplat
from conerf.model.gaussian_fields.prune import calculate_v_imp_score, prune_list
from conerf.trainers.implicit_recon_trainer import ImplicitReconTrainer
from conerf.visualization.pose_visualizer import visualize_cameras

from diff_gaussian_rasterization import SparseGaussianAdam
from fused_ssim import fused_ssim


# NOTE: We define a callable class instead of a closure since a closure cannot
# be serialized by torch's rpc.
class ExponentialLR:  # pylint: disable=[R0903]
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
        log_lerp = np.exp(np.log(self.lr_init) * (1 - t) +
                          np.log(self.lr_final) * t)

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
        appear_embedding: torch.nn.Module = None,
        block_id: int = None,
        device_id: int = 0,
    ) -> None:
        self.gaussians = None

        if valset is None:
            valset = load_val_dataset(config, 'cpu')

        self.admm_enabled = False

        super().__init__(config, prefetch_dataset,
                         trainset, valset, model, appear_embedding, block_id, device_id)

    def init_gaussians(self):
        # Using semantic alias to better understand the code.
        self.gaussians = self.model
        # Initialize 3D Gaussians from COLMAP point clouds.
        data_dir = os.path.join(
            self.config.dataset.root_dir, self.config.dataset.scene)
        colmap_dir = os.path.join(
            data_dir, self.config.dataset.get(
                "model_folder", "sparse"),  # model_folder,
            "manhattan_world" if self.config.dataset.get(
                "use_manhattan_world", False) else "0"
        )
        pcl_name = "points3D"
        if self.config.dataset.multi_blocks:
            pcl_name += f"_{self.train_dataset.current_block}"
            mx = self.config.dataset.get("mx", None)  # pylint: disable=C0103
            my = self.config.dataset.get("my", None)  # pylint: disable=C0103
            data_dir = get_block_info_dir(
                data_dir, self.config.dataset.num_blocks, mx, my)
            colmap_ply_path = os.path.join(data_dir, f"{pcl_name}.ply")
        else:
            dense = '' if self.config.dataset.init_ply_type == "sparse" else "_dense"
            colmap_ply_path = os.path.join(
                colmap_dir, f"{pcl_name}{dense}.ply")
        print(f'Initialize 3DGS using {colmap_ply_path}')

        point_cloud = fetch_ply(colmap_ply_path)
        self.gaussians.init_from_colmap_pcd(
            point_cloud,
            # self.config.appearance.get("app_feat_dim", None),
            app_feat_dim=None,
            image_idxs=self.train_camera_idxs
            if self.config.appearance.use_trained_exposure else None
        )
        bounding_box = ImplicitReconTrainer.read_bounding_box(colmap_dir)
        self.bounding_box = torch.tensor(bounding_box, dtype=torch.float32) \
            if bounding_box is not None else None

    def build_networks(self):
        self.model = GaussianSplatModel(
            max_sh_degree=self.config.texture.max_sh_degree,
            percent_dense=self.config.geometry.percent_dense,
            device=self.device,
        )

        # Decoupled appearance embedding.
        self.dec_app_embedding = None
        if self.config.geometry.get("mask", False):
            self.dec_app_embedding = DecoupledAppearanceEmbedding(
                len(self.train_dataset.cameras)).to(self.device)

        # Appearance embedding.
        self.app_embedding = None
        self.app_module = None
        if self.config.appearance.use_app_embed:
            print('Use App Embedding!')
            feature_dim = self.config.appearance.app_feat_dim
            embed_dim = self.config.appearance.app_embed_dim
            self.app_module = AppearanceOptModule(
                len(self.train_dataset.cameras),
                feature_dim,
                embed_dim,
                self.config.texture.max_sh_degree,
            ).to(self.device)
            # Initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)

            features = torch.rand(
                len(self.gaussians.get_xyz), feature_dim
            ).to(self.device)
            self.app_embedding = torch.nn.Parameter(features)

        self.init_gaussians()

    def setup_training_params(self):
        self.color_bkgd = torch.tensor(
            [0, 0, 0], dtype=torch.float32, device=self.device)
        self.ema_loss = 0.0
        self.use_white_bkgd = \
            True if self.config.dataset.apply_mask else False

        self.image_reader = None

        random.shuffle(self.train_dataset.cameras)
        self.train_cameras = self.train_dataset.cameras.copy()
        self.train_camera_idxs = [
            camera.image_index for camera in self.train_cameras]
        self.image_idx_to_global_index = {
            idx: i for i, idx in enumerate(self.train_camera_idxs)
        }

        spatial_lr_scale = self.config.geometry.get("spatial_lr_scale", -1)
        if spatial_lr_scale < 0:
            self.spatial_lr_scale = compute_nerf_plus_plus_norm(
                self.train_cameras)
        else:
            self.spatial_lr_scale = spatial_lr_scale

    def setup_optimizer(self):
        # Trivial hack when model is passed to the constructor.
        if self.gaussians is None:
            self.gaussians = self.model

        self.setup_gaussian_optimizer()
        self.setup_appearance_mask_optimizer()
        self.setup_exposure_optimizer()
        self.setup_pose_optimizer()
        self.setup_app_optimizer()

        lr_config = self.config.optimizer.lr
        self.xyz_scheduler = ExponentialLR(
            lr_init=lr_config.position_init * self.spatial_lr_scale,
            lr_final=lr_config.position_final * self.spatial_lr_scale,
            lr_delay_mult=lr_config.position_delay_mult,
            max_steps=lr_config.position_max_iterations
        )

    def setup_gaussian_optimizer(self):
        lr_config = self.config.optimizer.lr
        lr_params = [
            {
                'params': [self.gaussians.get_xyz],
                'lr': lr_config.position_init * self.spatial_lr_scale, "name": "xyz",
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

        if self.app_embedding is None:
            lr_params.append({
                'params': [self.gaussians.get_features_dc],
                'lr': lr_config.feature, "name": "f_dc",
            })
            lr_params.append({
                'params': [self.gaussians.get_features_rest],
                'lr': lr_config.feature / 20.0, "name": "f_rest",
            })
        else:
            lr_params.append({
                'params': [self.gaussians.get_colors],
                'lr': self.config.optimizer.lr_app.colors, "name": "colors",
            })
            lr_params.append({
                'params': [self.app_embedding],
                'lr': self.config.optimizer.lr_app.app_embed, "name": "app_embed",
            })

        self.optimizer = torch.optim.Adam(lr_params, lr=0.0, eps=1e-15)
        # self.optimizer = SparseGaussianAdam(lr_params, lr=0.0, eps=1e-15)

    def setup_appearance_mask_optimizer(self):
        lr_config = self.config.optimizer.lr

        self.mask_optimizer = None
        if self.mask is not None:
            self.mask_optimizer = torch.optim.Adam(
                self.mask.parameters(), lr=lr_config.mask
            )

    def setup_exposure_optimizer(self):
        lr_config = self.config.optimizer.lr

        self.exposure_optimizer = None
        self.exposure_scheduler = None
        if self.config.appearance.use_trained_exposure:
            self.exposure_optimizer = torch.optim.Adam(
                [self.gaussians.get_exposure])
            self.exposure_scheduler = ExponentialLR(
                lr_init=lr_config.exposure_lr_init,
                lr_final=lr_config.exposure_lr_final,
                lr_delay_steps=lr_config.exposure_lr_delay_steps,
                lr_delay_mult=lr_config.exposure_lr_delay_mult,
                max_steps=lr_config.exposure_max_iterations,
            )

    def setup_pose_optimizer(self):
        if not self.optimize_camera_poses:
            return

        self.gt_poses = extract_pose_tensor_from_cameras(
            self.train_dataset.cameras)

        # pose_params = []
        # for camera in self.train_dataset.cameras:
        #     if camera.image_index == 0:
        #         print('skip image 0!')
        #         continue

        #     pose_params.append({
        #         "params": [camera.delta_rot],
        #         "lr": self.config.optimizer.lr_pose.rot,
        #         "name": f"rot_{camera.image_index}",
        #     })
        #     pose_params.append({
        #         "params": [camera.delta_trans],
        #         "lr": self.config.optimizer.lr_pose.trans,
        #         "name": f"trans_{camera.image_index}",
        #     })

        # self.pose_optimizer = torch.optim.Adam(pose_params)
        self.pose_adjust = CameraOptModule(
            len(self.train_dataset.cameras)).to(self.device)
        self.pose_adjust.zero_init()
        self.pose_optimizer = torch.optim.Adam(
            self.pose_adjust.parameters(),
            lr=self.config.optimizer.lr_pose,
            weight_decay=1e-6,
        )
        self.pose_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.pose_optimizer, gamma=0.01 ** (1.0 /
                                                self.config.trainer.max_iterations)
        )

    def setup_app_optimizer(self):
        self.app_optimizers = []
        if self.config.appearance.use_app_embed:
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=self.config.optimizer.lr_app.app_module * 10.0,
                    weight_decay=1e-6,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=self.config.optimizer.lr_app.app_module,
                )
            ]

    def update_learning_rate(self):
        """Learning rate scheduling per step."""
        self._update_gaussian_params_lr()
        self._update_exposure_params_lr()

    def _update_gaussian_params_lr(self):
        if self.xyz_scheduler is None:
            return

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler(self.iteration)
                param_group['lr'] = lr
                return lr

    def _update_exposure_params_lr(self):
        if self.exposure_scheduler is None:
            return

        for param_group in self.exposure_optimizer.param_groups:
            param_group['lr'] = self.exposure_scheduler(self.iteration)

    def training_resolution(self) -> int:
        if not self.config.geometry.get('coarse-to-fine', False):
            return 1

        n_interval = 3
        iteration_threshold = min(
            20000, self.config.geometry.densify_end_iter) // n_interval
        resolution = 2 ** max(n_interval - self.iteration //
                              iteration_threshold - 1, 0)

        return resolution

    def add_admm_penalties(self, loss):
        raise NotImplementedError

    def train_iteration(self, data_batch) -> None:  # pylint: disable=W0613
        self.gaussians.train()
        self.update_learning_rate()

        # Increase the levels of SH up to a maximum degree.
        if self.iteration % 1000 == 0:
            self.gaussians.increase_SH_degree()

        # Training finished and safely exit.
        if (self.iteration - 1) >= self.config.trainer.max_iterations:
            self.image_reader.safe_exit()
            return

        # Pick a random camera.
        if (self.iteration - 1) % len(self.train_cameras) == 0 or self.image_reader is None:
            random.shuffle(self.train_cameras)
            image_list = [camera.image_path for camera in self.train_cameras]

            # Ensure all items in the queue have been gotten and processed
            # in the last epoch.
            if self.image_reader is not None:
                self.image_reader.safe_exit()

            self.image_reader = ImageReader(
                num_channels=self.config.dataset.get('num_channels', 3),
                image_list=image_list
            )
            self.image_reader.add_task(None)

        image_index, image = self.image_reader.get_image()
        camera = copy.deepcopy(self.train_cameras[image_index])
        camera.image = copy.deepcopy(image)
        resolution = self.training_resolution()
        camera_origin = camera.copy_to_device(self.device) \
            if self.mask is not None else None
        camera = camera.downsample(resolution).copy_to_device(self.device)
        self.scalars_to_log['train/resolution'] = resolution

        # Since we only update on the copy of cameras, the update won't
        # be accumulated continuously on the same cameras.
        if self.optimize_camera_poses and (camera.image_index != 0) and \
           (self.iteration > self.config.geometry.opt_pose_start_iter):
            image_index = camera.image_index

        global_index = self.image_idx_to_global_index[camera.image_index]
        render_results = render_gsplat(
            gaussian_splat_model=self.gaussians,
            viewpoint_camera=camera,
            pipeline_config=self.config.pipeline,
            bkgd_color=self.color_bkgd,
            anti_aliasing=self.config.texture.anti_aliasing,
            separate_sh=False,  # True,
            use_trained_exposure=self.config.appearance.use_trained_exposure,
            depth_threshold=self.config.geometry.depth_threshold,
            app_embedding=self.app_embedding,
            app_module=self.app_module,
            global_index=global_index,
            device=self.device,
        )
        colors, screen_space_points, visibility_filter, radii = (
            render_results["rendered_image"],
            render_results["screen_space_points"],
            render_results["visibility_filter"],
            render_results["radii"],
        )

        # Compute loss.
        lambda_dssim = self.config.loss.lambda_dssim
        lambda_mask = self.config.loss.lambda_mask
        pixels = camera.image.permute(2, 0, 1)  # [RGB, height, width]
        # loss_ssim = ssim(pixels, colors)
        loss_ssim = fused_ssim(colors.unsqueeze(0), pixels.unsqueeze(0))
        if self.mask is not None:
            image_size = camera.image.shape[:-1]
            camera = camera_origin.downsample(32).copy_to_device(self.device)
            mask = self.mask(camera.image.permute(2, 0, 1),
                             camera.image_index, image_size)
            loss_rgb_l1 = F.l1_loss(colors * mask, pixels)
            loss = (1.0 - lambda_dssim) * loss_rgb_l1 + \
                lambda_dssim * (1.0 - loss_ssim) + \
                lambda_mask * torch.mean((mask - 1) **
                                         2.)  # Regularization for mask
        else:
            loss_rgb_l1 = F.l1_loss(colors, pixels)
            loss = (1.0 - lambda_dssim) * loss_rgb_l1 + \
                lambda_dssim * (1.0 - loss_ssim)

        loss_scaling = render_results["scaling"].prod(dim=1).mean()
        loss += self.config.loss.lambda_scale * loss_scaling

        if self.admm_enabled:
            loss = self.add_admm_penalties(loss)

        loss.backward()

        self.ema_loss = 0.4 * loss.detach().item() + 0.6 * \
            self.ema_loss  # pylint: disable=W0201

        mse = F.mse_loss(colors, pixels)
        psnr = -10.0 * torch.log(mse) / np.log(10.0)

        # training statistics.
        self.scalars_to_log["train/psnr"] = psnr.detach().item()
        self.scalars_to_log["train/loss"] = loss.detach().item()
        self.scalars_to_log["train/l1_loss"] = loss_rgb_l1.detach().item()
        self.scalars_to_log["train/scale_loss"] = loss_scaling.detach().item()
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
                    screen_space_points, visibility_filter, pixels.shape[2], pixels.shape[1])

                if self.iteration > self.config.geometry.densify_start_iter and \
                        self.iteration % self.config.geometry.densification_interval == 0:
                    size_threshold = 20 \
                        if self.iteration > self.config.geometry.opacity_reset_interval else None
                    self.gaussians.densify_and_prune(
                        max_grad=self.config.geometry.densify_grad_threshold,
                        min_opacity=self.config.geometry.min_opacity,
                        extent=self.spatial_lr_scale,
                        max_screen_size=size_threshold,
                        optimizer=self.optimizer,
                        bounding_box=self.bounding_box,
                    )

                if self.iteration % self.config.geometry.opacity_reset_interval == 0 or \
                        (self.use_white_bkgd and self.iteration == self.config.geometry.densify_start_iter):
                    self.gaussians.reset_opacity(self.optimizer)

            if self.iteration in list(self.config.prune.iterations):
                gaussian_list, imp_list = prune_list(  # pylint: disable=W0612
                    self.gaussians, self.train_dataset.cameras.copy(),
                    self.config.pipeline, self.color_bkgd
                )
                v_list = calculate_v_imp_score(
                    self.gaussians, imp_list, self.config.prune.v_pow)
                i = self.config.prune.iterations.index(self.iteration)
                self.gaussians.prune_gaussians_with_opt(
                    (self.config.prune.prune_decay**i) *
                    self.config.prune.prune_percent,
                    v_list, self.optimizer
                )

        # Optimizer step.

        self.optimizer.step()
        # visible = radii > 0
        # self.optimizer.step(visible, radii.shape[0])
        self.optimizer.zero_grad(set_to_none=True)

        if self.exposure_optimizer is not None:
            self.exposure_optimizer.step()
            self.exposure_optimizer.zero_grad(set_to_none=True)

        if self.mask_optimizer is not None:
            self.mask_optimizer.step()
            self.mask_optimizer.zero_grad(set_to_none=True)

        if self.pose_optimizer is not None:
            self.pose_optimizer.step()
            self.pose_optimizer.zero_grad(set_to_none=True)

        if self.pose_scheduler is not None:
            self.pose_scheduler.step()

        for optimizer in self.app_optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

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
                pred_poses = extract_pose_tensor_from_cameras(
                    self.train_dataset.cameras)
                visualize_cameras(
                    self.visdom,
                    self.iteration,
                    poses=[torch.inverse(self.gt_poses),
                           torch.inverse(pred_poses)],
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

        self.state_dicts["meta_data"]["camera_poses"] = self.train_dataset.camtoworlds

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
        self.spatial_lr_scale = meta_data["spatial_lr_scale"]  # pylint: disable=W0201

        return iter_start
