# pylint: disable=[E1101,R1719,W0201]

import os
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F

from conerf.base.model_base import ModelBase
from conerf.datasets.dataset_base import MiniDataset
from conerf.datasets.utils import get_block_info_dir
from conerf.trainers.gaussian_trainer import GaussianSplatTrainer


class SlaveGaussianSplatTrainer(GaussianSplatTrainer):
    """
    Trainer for 3D Gaussian Splatting model on slave nodes.
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

        if trainset is None and block_id is not None:
            mx = config.dataset.get("mx", None)  # pylint: disable=C0103
            my = config.dataset.get("my", None)  # pylint: disable=C0103
            data_dir = os.path.join(
                config.dataset.root_dir, config.dataset.scene)
            data_dir = get_block_info_dir(
                data_dir, config.dataset.num_blocks, mx, my)
            block_dir = os.path.join(data_dir, f'block_{block_id}')
            trainset = MiniDataset(
                cameras=[], camtoworlds=None, block_id=block_id)
            trainset.read(path=block_dir, block_id=block_id, device='cpu')

        super().__init__(config, prefetch_dataset,
                         trainset, valset, model, appear_embedding, block_id, device_id)

    def clean(self):
        self.gaussians = None
        self.model = None
        self.mask = None
        self.ckpt_manager = None
        self.train_dataset = None
        self.val_dataset = None
        self.optimizer = None
        self.xyz_scheduler = None
        self.mask_optimizer = None
        if self.mask is not None:
            self.mask_optimizer = None
        self.evaluator = None

        torch.cuda.empty_cache()

    def update_iteration(self, iteration: int):
        self.iteration = iteration

    @torch.no_grad()
    def send_local_model(self):
        xyz, features_dc, features_rest, scaling, quaternion, opacity = \
            self.gaussians.get_all_properties()

        return xyz, features_dc, features_rest, scaling, quaternion, opacity

    @torch.no_grad()
    def send_local_mask(self):
        return self.mask

    @torch.no_grad()
    def send_local_loss(self):
        return self.scalars_to_log

    def setup_dual_variables(self):
        # Set dual variables for all the properties of 3D Gaussians:
        # xyz, features_dc, features_rest, scaling, quaternion, opacity.
        # We follow DBACC (Distributed Bundle Adjustment based on Camera Consensus) to
        # initialize all dual variables to zeros.
        self.u_xyz = torch.zeros_like(
            self.gaussians.get_xyz, requires_grad=False)
        self.u_fdc = torch.zeros_like(
            self.gaussians.get_features_dc, requires_grad=False)
        self.u_fr = torch.zeros_like(
            self.gaussians.get_features_rest, requires_grad=False)
        self.u_s = torch.zeros_like(
            self.gaussians.get_raw_scaling, requires_grad=False)
        self.u_q = torch.zeros_like(
            self.gaussians.get_raw_quaternion, requires_grad=False)
        self.u_o = torch.zeros_like(
            self.gaussians.get_raw_opacity, requires_grad=False)

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
        for i in range(interval):  # pylint: disable=W0612
            self.increment_iteration()
            self.train_iteration(data_batch=data_batch)

    def compose_state_dicts(self) -> None:
        super().compose_state_dicts()

        if self.config.dataset.multi_blocks:
            self.state_dicts["meta_data"]["block_id"] = self.train_dataset.current_block
            self.state_dicts["meta_data"]["admm_enabled"] = self.admm_enabled

            if self.admm_enabled:
                # dual variables
                self.state_dicts["meta_data"]["u_xyz"] = self.u_xyz
                self.state_dicts["meta_data"]["u_fdc"] = self.u_fdc
                self.state_dicts["meta_data"]["u_fr"] = self.u_fr
                self.state_dicts["meta_data"]["u_s"] = self.u_s
                self.state_dicts["meta_data"]["u_q"] = self.u_q
                self.state_dicts["meta_data"]["u_o"] = self.u_o

                # penalty parameters
                self.state_dicts["meta_data"]["rho_xyz"] = self.rho_xyz
                self.state_dicts["meta_data"]["rho_fdc"] = self.rho_fdc
                self.state_dicts["meta_data"]["rho_fr"] = self.rho_fr
                self.state_dicts["meta_data"]["rho_s"] = self.rho_s
                self.state_dicts["meta_data"]["rho_q"] = self.rho_q
                self.state_dicts["meta_data"]["rho_o"] = self.rho_o

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

        if self.config.dataset.multi_blocks:
            self.admm_enabled = self.state_dicts["meta_data"]["admm_enabled"]
            if self.admm_enabled:
                # dual variables
                self.u_xyz = self.state_dicts["meta_data"]["u_xyz"]
                self.u_fdc = self.state_dicts["meta_data"]["u_fdc"]
                self.u_fr = self.state_dicts["meta_data"]["u_fr"]
                self.u_s = self.state_dicts["meta_data"]["u_s"]
                self.u_q = self.state_dicts["meta_data"]["u_q"]
                self.u_o = self.state_dicts["meta_data"]["u_o"]

                # penalty parameters
                self.rho_xyz = self.state_dicts["meta_data"]["rho_xyz"]
                self.rho_fdc = self.state_dicts["meta_data"]["rho_fdc"]
                self.rho_fr = self.state_dicts["meta_data"]["rho_fr"]
                self.rho_s = self.state_dicts["meta_data"]["rho_s"]
                self.rho_q = self.state_dicts["meta_data"]["rho_q"]
                self.rho_o = self.state_dicts["meta_data"]["rho_o"]

        return iter_start
