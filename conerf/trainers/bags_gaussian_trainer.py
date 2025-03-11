# pylint: disable=E1101

import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf

from conerf.base.task_queue import ImageReader
from conerf.render.gaussian_render import render
from conerf.model.deblur.bags_net import BagsSingleResKernelNet, BagsMultiResKernelNet
from conerf.trainers.gaussian_trainer import GaussianSplatTrainer

from fused_ssim import fused_ssim


def tv_loss(grids):
    """
    https://github.com/apchenstu/TensoRF/blob/4ec894dc1341a2201fe13ae428631b58458f105d/utils.py#L139

    Args:
        grids: stacks of explicit feature grids (stacked at dim 0)
    Returns:
        average total variation loss for neighbor rows and columns.
    """
    number_of_grids = grids.shape[0]
    h_tv_count = grids[:, :, 1:, :].shape[1] * grids[:,
                                                     :, 1:, :].shape[2] * grids[:, :, 1:, :].shape[3]
    w_tv_count = grids[:, :, :, 1:].shape[1] * grids[:,
                                                     :, :, 1:].shape[2] * grids[:, :, :, 1:].shape[3]
    h_tv = torch.pow((grids[:, :, 1:, :] - grids[:, :, :-1, :]), 2).sum()
    w_tv = torch.pow((grids[:, :, :, 1:] - grids[:, :, :, :-1]), 2).sum()
    return 2 * (h_tv / h_tv_count + w_tv / w_tv_count) / number_of_grids


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)

    return torch.flatten(emb, -2, -1)


def get_2d_emb(batch_size, x, y, out_ch, device):
    out_ch = int(np.ceil(out_ch / 4) * 2)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, out_ch, 2).float() / out_ch))

    pos_x = torch.arange(x, device=device).type(
        inv_freq.type()) * 2 * np.pi / x
    pos_y = torch.arange(y, device=device).type(
        inv_freq.type()) * 2 * np.pi / y

    sin_inp_x = torch.einsum("i,j->ij", pos_x, inv_freq)
    sin_inp_y = torch.einsum("i,j->ij", pos_y, inv_freq)
    emb_x = get_emb(sin_inp_x).unsqueeze(1)
    emb_y = get_emb(sin_inp_y)
    emb = torch.zeros((x, y, out_ch * 2), device=device)
    emb[:, :, : out_ch] = emb_x
    emb[:, :, out_ch: 2 * out_ch] = emb_y

    return emb[None, :, :, :].repeat(batch_size, 1, 1, 1)


class BagsTrainer(GaussianSplatTrainer):
    """
    Reimplementation of the ECCV 2024 paper:
        "BAGS: Blur Agnostic Gaussian Splatting through Multi-Scale Kernel Modeling"
    """

    def __init__(
        self,
        config: OmegaConf,
        prefetch_dataset=True,
        trainset=None,
        valset=None,
        model=None,
        appear_embedding=None,
        block_id=None,
        device_id=0
    ):
        super().__init__(
            config, prefetch_dataset, trainset, valset, model, appear_embedding, block_id, device_id
        )

        # Hard-coded for reset the 'xyz' learning rate at final scale.
        self.final_scale_iteration = 9000

        self.ks1 = self.deblur_net.kernel_size1
        self.ks2 = self.deblur_net.kernel_size2
        self.ks3 = self.deblur_net.kernel_size3
        self.unfold1 = nn.Unfold(kernel_size=(
            self.ks1, self.ks1), padding=self.ks1 // 2).to(self.device)
        self.unfold2 = nn.Unfold(kernel_size=(
            self.ks2, self.ks2), padding=self.ks2 // 2).to(self.device)
        self.unfold3 = nn.Unfold(kernel_size=(
            self.ks3, self.ks3), padding=self.ks3 // 2).to(self.device)

        # TODO(chenyu): for single-resolution deblur net.

    def build_networks(self):
        super().build_networks()

        self.deblur_net = None
        if self.config.geometry.get("deblur", False):
            self.deblur_net = BagsMultiResKernelNet(
                num_images=len(self.train_dataset),
            ).to(self.device)

    def setup_optimizer(self):
        super().setup_optimizer()

        lr_config = self.config.optimizer.lr
        self.deblur_optimizer = None
        if self.deblur_net is not None:
            self.deblur_optimizer = torch.optim.Adam(
                self.deblur_net.parameters(), lr=lr_config.deblur
            )

    def training_resolution(self):
        # Hard-coded according to the official code of BAGS.
        upsample_iters = [3000, 6000]

        if self.iteration < upsample_iters[0]:
            resolution = 4
        elif upsample_iters[0] <= self.iteration < upsample_iters[1]:
            resolution = 2
        else:
            resolution = 1

        return resolution

    def _training_miscs(self):
        # Hard-coded according to the official code of BAGS.
        upsample_iters = [3000, 6000]

        if self.iteration < upsample_iters[0]:
            kernel_size, unfold = self.ks1, self.unfold1

        elif upsample_iters[0] <= self.iteration < upsample_iters[1]:
            kernel_size, unfold = self.ks2, self.unfold2

        else:
            kernel_size, unfold = self.ks3, self.unfold3

        return kernel_size, unfold

    def _reset_xyz_learning_rate(self, iteration: int):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler(iteration)
                param_group['lr'] = lr
                return lr

    def train_iteration(self, data_batch):
        self.gaussians.train()

        reset_iteration = self.iteration if self.iteration <= self.final_scale_iteration \
            else self.iteration - self.final_scale_iteration
        if self.iteration <= self.final_scale_iteration:
            self.update_learning_rate()
        else:
            self._reset_xyz_learning_rate(reset_iteration)

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

        render_results = render(
            gaussian_splat_model=self.gaussians,
            viewpoint_camera=camera,
            pipeline_config=self.config.pipeline,
            bkgd_color=self.color_bkgd,
            anti_aliasing=self.config.texture.anti_aliasing,
            separate_sh=False, # True,
            use_trained_exposure=self.config.appearance.use_trained_exposure,
            depth_threshold=self.config.geometry.depth_threshold,
            device=self.device,
        )
        colors, screen_space_points, visibility_filter, radii, depth = (
            render_results["rendered_image"],
            render_results["screen_space_points"],
            render_results["visibility_filter"],
            render_results["radii"],
            render_results["depth"],
        )

        if self.iteration > 250:  # TODO(chenyu): use `reset_iteration` or `self.iteration`?
            shuffle_rgb = colors.unsqueeze(0)
            shuffle_depth = depth.unsqueeze(0) - depth.min()
            shuffle_depth = shuffle_depth / shuffle_depth.max()

            pos_enc = get_2d_emb(
                1, shuffle_rgb.shape[-2], shuffle_rgb.shape[-1], 16, self.device)
            kernel_size, unfold = self._training_miscs()

            kernel_weights, mask = self.deblur_net(
                image_index, pos_enc,
                torch.cat([shuffle_rgb, shuffle_depth], dim=1).detach(),
                self.iteration
            )
            patches = unfold(shuffle_rgb)
            patches = patches.view(1, 3, kernel_size ** 2, shuffle_rgb.shape[-2],
                                   shuffle_rgb.shape[-1])
            kernel_weights = kernel_weights.unsqueeze(1)
            rgb = torch.sum(patches * kernel_weights, 2)[0]
            mask = mask[0]

            colors = mask * rgb + (1 - mask) * colors

            if self.iteration in [251, 3000, 6000]:
                print(f'colors shape: {colors.shape}')
                print(f'shuffle_rgb shape: {shuffle_rgb.shape}')
                print(f'shuffle_depth shape: {shuffle_depth.shape}')
                print(f'pos_enc shape: {pos_enc.shape}')
                print(f'kernel size: {kernel_size}')
                print(f'kernel weights shape: {kernel_weights.shape}, mask shape: {mask.shape}')
                print(f'patches shape: {patches.shape}')
                print(f'rgb shape: {rgb.shape}')
                print(f'mask shape: {mask.shape}; colors shape: {colors.shape}')

        # Compute loss.
        lambda_dssim = self.config.loss.lambda_dssim
        lambda_mask = self.config.loss.lambda_mask
        pixels = camera.image.permute(2, 0, 1)  # [RGB, height, width]
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

        if self.iteration > 250:  # TODO(chenyu): use `reset_iteration` or `self.iteration`?
            depth_loss = self.config.loss.lambda_depth * tv_loss(shuffle_depth)
            mask_loss = self.config.loss.lambda_deblur_mask * mask.mean()
            total_variance_loss = self.config.loss.lambda_rgbtv * \
                tv_loss(shuffle_rgb)
            loss = loss + depth_loss + mask_loss + total_variance_loss

            self.scalars_to_log["train/depth_loss"] = depth_loss.detach().item()
            self.scalars_to_log["train/mask_loss"] = mask_loss.detach().item()
            self.scalars_to_log["train/tv_loss"] = total_variance_loss.detach().item()

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
            if reset_iteration < self.config.geometry.densify_end_iter:
                # Keep track of max radii in image-space for pruning.
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter]
                )
                self.gaussians.add_densification_stats(
                    screen_space_points, visibility_filter)

                if reset_iteration > self.config.geometry.densify_start_iter and \
                        reset_iteration % self.config.geometry.densification_interval == 0:
                    size_threshold = 20 \
                        if reset_iteration > self.config.geometry.opacity_reset_interval else None
                    
                    # Use different threshold for final scale training.
                    if self.iteration <= self.final_scale_iteration:
                        densify_grad_threshold = self.config.deblur.init_densify_grad_threshold
                        min_opacity = self.config.deblur.init_min_opacity
                    else:
                        densify_grad_threshold = self.config.geometry.densify_grad_threshold
                        min_opacity = self.config.geometry.min_opacity
                    
                    self.gaussians.densify_and_prune(
                        max_grad=densify_grad_threshold,
                        min_opacity=min_opacity,
                        extent=self.spatial_lr_scale,
                        max_screen_size=size_threshold,
                        optimizer=self.optimizer,
                        bounding_box=self.bounding_box,
                    )

                if reset_iteration % self.config.geometry.opacity_reset_interval == 0 or \
                        (self.use_white_bkgd and reset_iteration == self.config.geometry.densify_start_iter):
                    self.gaussians.reset_opacity(self.optimizer)

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

        if self.iteration % self.config.trainer.n_checkpoint == 0:
            self.compose_state_dicts()
