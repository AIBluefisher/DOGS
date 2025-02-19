# pylint: disable=E1101

import copy
import random

import torch
import torch.nn.functional as F
import numpy as np

from conerf.loss.ssim_torch import ssim
from conerf.base.task_queue import ImageReader
from conerf.render.gaussian_render import render
from conerf.model.deblur.varnet import VarNet
from conerf.trainers.gaussian_trainer import GaussianSplatTrainer

from fused_ssim import fused_ssim


class DeblurGaussianSplatTrainer(GaussianSplatTrainer):
    """
    Reimplementation of the ECCV 2024 paper: "Deblurring 3D Gaussian Splatting".
    """

    def build_networks(self):
        super().build_networks()

        self.deblur_net = None
        if self.config.geometry.get("deblur", False):
            self.deblur_net = VarNet(
                self.config.deblur.position_latent_dim,
                self.config.deblur.view_latent_dim,
                self.config.deblur.net_depth,
                self.config.deblur.net_width,
                self.config.deblur.use_position_offset,
                self.config.deblur.num_gaussian_sets,
                self.device,
            ).to(self.device)

        point_max = self.gaussians.get_xyz.amax(0)
        point_min = self.gaussians.get_xyz.amin(0)
        self.prune_bbox = point_max - point_min

    def setup_optimizer(self):
        super().setup_optimizer()

        lr_config = self.config.optimizer.lr
        self.deblur_optimizer = None

        # self.deblur_optimizer = torch.optim.Adam(
        #     self.deblur_net.parameters(), lr=lr_config.deblur, eps=1e-15
        # )
        self.optimizer.add_param_group({
            'params': self.deblur_net.parameters(),
            'lr': lr_config.deblur,
            'name': 'mlp',
        })

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

        render_results = render(
            gaussian_splat_model=self.gaussians,
            viewpoint_camera=camera,
            pipeline_config=self.config.pipeline,
            bkgd_color=self.color_bkgd,
            anti_aliasing=self.config.texture.anti_aliasing,
            separate_sh=False,  # True,
            use_trained_exposure=self.config.appearance.use_trained_exposure,
            depth_threshold=self.config.geometry.depth_threshold,
            deblur_net=self.deblur_net,
            lambda_scale=self.config.deblur.lambda_scale,
            lambda_position=self.config.deblur.lambda_position,
            use_position_offset=self.config.deblur.use_position_offset,
            max_clamp=self.config.deblur.max_clamp,
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
        loss_ssim = fused_ssim(colors.unsqueeze(0), pixels.unsqueeze(0))
        # loss_ssim = ssim(pixels, colors)
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

        # loss_scaling = render_results["scaling"].prod(dim=1).mean()
        # loss += self.config.loss.lambda_scale * loss_scaling

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
        # self.scalars_to_log["train/scale_loss"] = loss_scaling.detach().item()
        self.scalars_to_log["train/ema_loss"] = self.ema_loss
        self.scalars_to_log["train/points"] = self.gaussians.get_xyz.shape[0]

        with torch.no_grad():
            # Densification.
            if self.iteration < self.config.geometry.densify_end_iter:
                # Keep track of max radii in image-space for pruning.
                if type(visibility_filter) == list:
                    self.gaussians.max_radii2D[visibility_filter[-1]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter[-1]],
                        radii[-1][visibility_filter[-1]]
                    )
                else:
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter],
                        radii[visibility_filter]
                    )
                self.gaussians.add_densification_stats(
                    screen_space_points, visibility_filter)

                if self.iteration > self.config.geometry.densify_start_iter and \
                        self.iteration % self.config.geometry.densification_interval == 0:
                    # size_threshold = 20 \
                    #     if self.iteration > self.config.geometry.opacity_reset_interval else None
                    size_threshold = None
                    self.gaussians.densify_and_prune(
                        max_grad=self.config.geometry.densify_grad_threshold,
                        min_opacity=self.config.geometry.min_opacity,
                        extent=self.spatial_lr_scale,
                        max_screen_size=size_threshold,
                        optimizer=self.optimizer,
                        bounding_box=self.bounding_box,
                        prune_depth=True,
                    )

                # Add extra points.
                if self.iteration == self.config.deblur.extra_points_iteration:
                    volume = self.prune_bbox[0] * \
                        self.prune_bbox[1] * self.prune_bbox[2]
                    if self.config.deblur.points_rate > 0.0:
                        num_points = int(
                            min(volume / (self.config.deblur.points_rate ** 3), 200000))
                    else:
                        num_points = self.config.deblur.num_extra_points
                    print(f'Allocated {num_points} points')

                    self.gaussians.allocate_extra_points(
                        distance=self.config.deblur.distance,
                        num_nearest_neighbor=self.config.deblur.num_nearest_neighbor,
                        num_points=self.config.deblur.num_extra_points,
                        bound=self.config.deblur.points_bound,
                    )
                    # After adding extra points to existing 3DGS, we need to update the
                    # parameters in original optimizer.
                    self.setup_gaussian_optimizer()
                    self.optimizer.add_param_group({
                        'params': self.deblur_net.parameters(),
                        'lr': self.config.optimizer.lr.deblur,
                        'name': 'mlp',
                    })

                # if self.iteration % self.config.geometry.opacity_reset_interval == 0 or \
                #    (self.use_white_bkgd and self.iteration == self.config.geometry.densify_start_iter):
                #     self.gaussians.reset_opacity(self.optimizer)

        # Optimizer step.

        # if type(visibility_filter) == list:
        #     self.optimizer.step(visibility_filter[-1], radii[-1].shape[0])
        # else:
        #     self.optimizer.step(visibility_filter, radii.shape[0])
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        # if self.deblur_optimizer is not None:
        #     self.deblur_optimizer.step()
        #     self.deblur_optimizer.zero_grad(set_to_none=True)

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

        if self.iteration % self.config.trainer.n_checkpoint == 0:
            self.compose_state_dicts()

        # Update camera pose.
        if self.optimize_camera_poses and (camera.image_index != 0) and \
           (self.iteration > self.config.geometry.opt_pose_start_iter):
            self.train_dataset.cameras[image_index].update_camera_pose()
