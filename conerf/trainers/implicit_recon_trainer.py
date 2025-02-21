import math
import os

from omegaconf import OmegaConf

import torch
import tqdm

from conerf.datasets.utils import create_dataset
from conerf.utils.utils import parse_scheduler, parse_optimizer
from conerf.trainers.trainer import BaseTrainer
from conerf.base.checkpoint_manager import CheckPointManager
from conerf.base.model_base import ModelBase
from eval import create_evaluator


class ImplicitReconTrainer(BaseTrainer):
    """
    Base class for training implicit 3D reconstruction models.
    """

    def __init__(  # pylint: disable=W0231
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
        self.trainer_name = "ImplicitReconTrainer"
        self.device = f"cuda:{device_id}"
        self.config = config
        self.block_id = block_id

        self.output_path = os.path.join(
            config.dataset.root_dir, "out", config.expname)
        if self.config.trainer.local_rank == 0:
            os.makedirs(self.output_path, exist_ok=True)

        self.log_learning_rate = True

        self.model = model.to(self.device) if model is not None else None
        self.mask = appear_embedding.to(self.device) \
            if appear_embedding is not None else None
        self.delta_pose = None
        self.optimize_camera_poses = False

        self.pose_optimizer = None
        self.pose_scheduler = None

        self.focal_optimizer = None
        self.focal_scheduler = None

        self.epoch = 0
        self.iteration = 0
        self.scalars_to_log = dict()
        self.images_to_log = dict()
        self.ckpt_manager = CheckPointManager(
            save_path=self.output_path,
            max_to_keep=1000,
            keep_checkpoint_every_n_hours=0.5,
            verbose=config.get('verbose', False),
        )

        self.train_done = False
        self.num_rays = self.config.dataset.get("num_rays", 256)

        if prefetch_dataset:
            self.load_dataset()
        else:
            self.train_dataset = trainset
            self.val_dataset = valset

        self._setup_visualizer()
        self.setup_training_params()
        self.build_pose_refiner()

        # Functions need to be overwritten.
        if model is None:
            self.build_networks()
        self.setup_optimizer()
        self.setup_loss_functions()
        self.compose_state_dicts()

        self.evaluator = create_evaluator(
            config,
            load_val_data=False,
            valset=self.val_dataset,
            models=[self.model],
            meta_data=[self.state_dicts["meta_data"]],
            device=self.device,
        ) if self.val_dataset is not None else None

    @classmethod
    def read_bounding_box(cls, colmap_dir: str, suffix: str = ''):
        bbox_path = os.path.join(colmap_dir, f"bounding_box{suffix}.txt")
        if not os.path.exists(bbox_path):
            return None

        file = open(bbox_path, "r", encoding="utf-8")
        file.readline()  # Omit the first line.

        bbox = [None] * 6

        line = file.readline()
        while line:
            data = line.split(' ')
            bbox[0], bbox[1], bbox[2] = float(
                data[0]), float(data[1]), float(data[2])
            bbox[3], bbox[4], bbox[5] = float(
                data[3]), float(data[4]), float(data[5])
            line = file.readline()

        file.close()
        return bbox

    def reset_checkpoint_manager(self):
        """Reset checkpoint manager when output path changed."""
        self.ckpt_manager = CheckPointManager(
            save_path=self.output_path,
            max_to_keep=1000,
            keep_checkpoint_every_n_hours=0.5,
        )

    def setup_training_params(self):
        """Set training parameters from configuration file."""
        self.max_num_rays = self.config.dataset.max_num_rays
        enable_background = self.config.get('enable_background', False)
        num_samples_per_ray_bg = 0
        if enable_background:
            num_samples_per_ray_bg = self.config.geometry_bg.num_samples_per_ray
        num_samples_per_ray = self.config.dataset.num_samples_per_ray + num_samples_per_ray_bg
        self.target_sample_batch_size = self.num_rays * num_samples_per_ray

        aabb = self.config.dataset.aabb
        if self.config.dataset.auto_aabb and self.train_dataset.BBOX is None:
            camera_locs = torch.cat(
                [self.train_dataset.camtoworlds, self.val_dataset.camtoworlds]
            )[:, :3, -1]
            aabb = torch.cat(
                [camera_locs.min(dim=0).values, camera_locs.max(dim=0).values]
            ).tolist()
        elif self.train_dataset.BBOX is not None:
            aabb = self.train_dataset.BBOX

        self.contraction_type = None

        self.grid_resolution = self.config.dataset.grid_resolution
        self.scene_aabb = (
            torch.tensor(aabb)
            if self.config.dataset.unbounded
            else torch.tensor(self.config.dataset.aabb, dtype=torch.float32)
        )
        self.near_plane = self.config.dataset.near_plane
        self.far_plane = self.config.dataset.far_plane
        self.grid_levels = self.config.dataset.grid_levels
        self.cone_angle = self.config.dataset.cone_angle
        self.alpha_thre = self.config.dataset.alpha_thre

        # setup the scene bounding box.
        if self.config.dataset.unbounded:
            self.near_plane = (
                self.train_dataset.NEAR if self.train_dataset.NEAR > 0 else 0.2  # 0.01
            )
            self.far_plane = (
                self.train_dataset.FAR if self.train_dataset.FAR > 0 else 1.0e3  # 1.0e3
            )
            self.render_step_size = 1e-2  # 1e-3
            self.cone_angle = 10 ** (math.log10(self.far_plane) /
                                     num_samples_per_ray) - 1
        else:
            self.render_step_size = (
                (self.scene_aabb[3:] - self.scene_aabb[:3]).max()
                * math.sqrt(3)
                / self.config.dataset.num_samples_per_ray
            ).item()  # 5e-3
            # self.render_step_size = 0.001

        # print(f'render step size: {self.render_step_size}')
        # print("Using aabb", self.scene_aabb)
        # print(f'Cone angle: {self.cone_angle}')

    def load_dataset(self):
        self.train_dataset = create_dataset(
            config=self.config,
            # train=True,
            split=self.config.dataset.train_split,
            num_rays=self.num_rays,
            apply_mask=self.config.dataset.apply_mask,
            device=self.device
        )
        self.val_dataset = create_dataset(
            config=self.config,
            # train=False,
            split=self.config.dataset.val_split,
            num_rays=None,
            apply_mask=self.config.dataset.apply_mask,
            device=self.device
        )

    def build_networks(self):
        raise NotImplementedError

    def build_pose_refiner(self):
        if self.config.optimizer.get("lr_pose", None) is None:
            return

        self.optimize_camera_poses = True

    def setup_optimizer(self):
        self.optimizer = parse_optimizer(self.config.optimizer, self.model)
        self.scheduler = parse_scheduler(self.config.scheduler, self.optimizer)
        self.grad_scaler = torch.cuda.amp.GradScaler(2**10)

        self.setup_pose_optimizer()

    def setup_pose_optimizer(self):
        if self.config.get("pose_optimizer", None) is not None:
            self.pose_optimizer = parse_optimizer(  # pylint: disable=[W0201]
                self.config.pose_optimizer, self.delta_pose
            )
        if self.config.get("pose_scheduler", None) is not None:
            scheduler = getattr(torch.optim.lr_scheduler,
                                self.config.pose_scheduler.name)

            if self.config.pose_scheduler.lr_end:
                assert self.config.pose_scheduler.name == "ExponentialLR"
                self.config.pose_scheduler.gamma = (
                    self.config.pose_scheduler.lr_end / self.config.pose_optimizer.args.lr
                ) ** (1. / self.config.trainer.max_iterations)

            self.pose_scheduler = scheduler(  # pylint: disable=[W0201]
                self.pose_optimizer,
                gamma=self.config.pose_scheduler.gamma
            )

    def setup_loss_functions(self):
        pass

    def record_memory_stats(self):
        stats = torch.cuda.memory_stats(self.device)
        # print(stats['allocated_bytes.all.current'])
        self.scalars_to_log["memory/current"] = stats['allocated_bytes.all.current'] / (
            1024 ** 3)
        self.scalars_to_log["memory/peak"] = stats['allocated_bytes.all.peak'] / \
            (1024 ** 3)
        self.scalars_to_log["memory/allocated"] = \
            stats['allocated_bytes.all.allocated'] / (1024 ** 3)
        self.scalars_to_log["memory/freed"] = stats['allocated_bytes.all.freed'] / (
            1024 ** 3)

    def update_meta_data(self):
        """Metadata that stored into checkpoint."""
        if self.config.dataset.multi_blocks:
            self.state_dicts["meta_data"]["block_id"] = self.train_dataset.current_block
        self.state_dicts["meta_data"]["camera_poses"] = self.train_dataset.camtoworlds

    def compose_state_dicts(self) -> None:
        self.state_dicts = {
            "models": dict(),
            "optimizers": dict(),
            "schedulers": dict(),
            "meta_data": dict(),
        }

        self.state_dicts["models"]["model"] = self.model
        self.state_dicts["optimizers"]["optimizer"] = self.optimizer
        self.state_dicts["schedulers"]["scheduler"] = self.scheduler

        # Pose related items.
        if self.delta_pose is not None:
            self.state_dicts["models"]["delta_pose"] = self.delta_pose
        if self.config.get("pose_optimizer", None) is not None:
            self.state_dicts["optimizers"]["pose_optimizer"] = self.pose_optimizer
        if self.config.get("pose_scheduler", None) is not None:
            self.state_dicts["schedulers"]["pose_scheduler"] = self.pose_scheduler

        # meta data for construction models.
        self.state_dicts["meta_data"]["aabb"] = self.scene_aabb
        self.state_dicts["meta_data"]["unbounded"] = self.config.dataset.unbounded
        self.state_dicts["meta_data"]["grid_resolution"] = self.grid_resolution
        self.state_dicts["meta_data"]["contraction_type"] = self.contraction_type
        self.state_dicts["meta_data"]["near_plane"] = self.near_plane
        self.state_dicts["meta_data"]["far_plane"] = self.far_plane
        self.state_dicts["meta_data"]["render_step_size"] = self.render_step_size
        self.state_dicts["meta_data"]["alpha_thre"] = self.alpha_thre
        self.state_dicts["meta_data"]["cone_angle"] = self.cone_angle
        self.state_dicts["meta_data"]["grid_levels"] = self.grid_levels

    def increment_iteration(self):
        self.iteration += 1

    def train(self):
        desc = (
            f"Training {self.config.expname}"
            if not self.config.dataset.multi_blocks
            else f"Training {self.config.expname} block_{self.train_dataset.current_block}"
        )
        desc += f" ({len(self.train_dataset)} images)"
        pbar = tqdm.trange(self.config.trainer.max_iterations,
                           desc=desc, leave=False)

        iter_start = self.load_checkpoint(
            load_optimizer=not self.config.trainer.no_load_opt,
            load_scheduler=not self.config.trainer.no_load_scheduler,
        )
        # iter_start -= 1
        if iter_start >= self.config.trainer.max_iterations:
            return

        score = 0
        while self.iteration < iter_start:
            pbar.update(1)
            self.increment_iteration()

        while self.iteration < self.config.trainer.max_iterations:
            for i in range(len(self.train_dataset)):  # pylint: disable=C0200
                data_batch = self.train_dataset[i] \
                    if self.config.neural_field_type.find("gs") < 0 else None
                self.increment_iteration()

                # log to tensorboard.
                if self.iteration % self.config.trainer.n_tensorboard == 0:
                    self.log_info()

                if self.iteration % (self.config.trainer.n_tensorboard * 100) == 0:
                    self.log_images()

                if self.iteration % self.config.trainer.n_checkpoint == 0:
                    self.save_checkpoint(score=score)

                if self.iteration % self.config.trainer.n_validation == 0 and \
                   self.iteration != 0:
                    score = self.validate()

                self.train_iteration(data_batch=data_batch)

                pbar.update(1)

                if self.iteration > self.config.trainer.max_iterations:
                    break

            self.epoch += 1

        if self.config.trainer.n_checkpoint % self.config.trainer.n_validation != 0 and \
           self.config.trainer.n_validation % self.config.trainer.n_checkpoint != 0:
            score = self.validate()
            self.save_checkpoint(score=score)

        self.train_done = True

    def train_iteration(self, data_batch) -> None:
        raise NotImplementedError

    @torch.no_grad()
    def validate(self) -> float:
        if self.evaluator is not None:
            metrics = self.evaluator.eval(iteration=self.iteration)
            psnr_avg = metrics[0]['psnr']

            if self.writer is not None:
                self.writer.add_scalar(
                    "val/psnr", psnr_avg, global_step=self.iteration
                )

            if self.train_dataset is not None:
                self.train_dataset.training = True

            torch.cuda.empty_cache()

            return psnr_avg
        return 0.0
