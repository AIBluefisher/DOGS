# pylint: disable=[E1101,W0621]

import copy
import os
import time
import json
from typing import List, Literal
import tqdm

import torch

from omegaconf import OmegaConf

from conerf.base.checkpoint_manager import CheckPointManager
from conerf.evaluators.evaluator import (
    Evaluator, compute_psnr, compute_lpips, compute_ssim, color_correct
)
from conerf.model.gaussian_fields.gaussian_splat_model import GaussianSplatModel
from conerf.utils.utils import save_images, get_subdirs, colorize


class GaussianSplatEvaluator(Evaluator):
    """Class for evaluating NeRF models."""

    def __init__(
        self,
        config: OmegaConf,
        load_train_data: bool = False,
        trainset=None,
        load_val_data: bool = True,
        valset=None,
        load_test_data: bool = False,
        testset=None,
        models: List = None,
        meta_data: List = None,
        verbose: bool = False,
        device: str = "cuda",
    ) -> None:
        super().__init__(
            config,
            load_train_data,
            trainset,
            load_val_data,
            valset,
            load_test_data,
            testset,
            models,
            meta_data,
            verbose,
            device,
        )

        self.color_bkgd = torch.tensor(
            [0, 0, 0], dtype=torch.float32, device=self.device)

    def _prepare_model_init_params(self, model_type: Literal["global", "local"] = "local"):
        pass

    def _build_networks(self, *args, **kwargs):  # pylint: disable=[W0613]
        model = GaussianSplatModel(
            max_sh_degree=self.config.texture.max_sh_degree,
            percent_dense=self.config.geometry.percent_dense,
        )
        return model

    def setup_metadata(self):
        """Set up meta data that are required to initialize/evaluate a model."""
        # meta data for construction models.
        meta_data = {
            "active_sh_degree": None,
            "xyz": None,
            "features_dc": None,
            "features_rest": None,
            "scaling": None,
            "quaternion": None,
            "opacity": None,
            "max_radii2D": None,
            "xyz_gradient_accum": None,
            "denom": None,
            "spatial_lr_scale": None,
        }
        if self.config.dataset.multi_blocks:
            meta_data["block_id"] = None

        return meta_data

    def load_model(self):
        self.meta_data, self.models = [], []  # pylint: disable=W0201
        ckpt_manager = CheckPointManager(verbose=False)

        input_model_dir = os.path.join(
            self.config.dataset.root_dir, 'out', self.config.expname)

        assert os.path.exists(input_model_dir), \
            f"input model directory does not exist: {input_model_dir}"
        if self.config.dataset.multi_blocks:
            model_dirs = get_subdirs(input_model_dir, "block_")
        else:
            model_dirs = [input_model_dir]

        pbar = tqdm.trange(len(model_dirs), desc="Loading Models", leave=True)
        for model_dir in model_dirs:
            local_config = copy.deepcopy(self.config.trainer)
            local_config.ckpt_path = os.path.join(model_dir, 'model.pth')
            assert os.path.exists(local_config.ckpt_path), \
                f"checkpoint does not exist: {local_config.ckpt_path}"

            # Load meta data at first.
            meta_data = self.setup_metadata()
            iteration = ckpt_manager.load(local_config, meta_data=meta_data)
            self.meta_data.append(meta_data)
            self.model_iterations.append(iteration)

            model = self._build_networks()
            ckpt_manager.load(
                local_config,
                models=None,
                optimizers=None,
                schedulers=None,
                meta_data=meta_data
            )

            model.active_sh_degree = meta_data["active_sh_degree"]
            model.set_xyz(meta_data["xyz"])
            model.set_features_dc(meta_data["features_dc"])
            model.set_features_rest(meta_data["features_rest"])
            model.set_raw_scaling(meta_data["scaling"])
            model.set_raw_quaternion(meta_data["quaternion"])
            model.set_raw_opacity(meta_data["opacity"])
            # model.max_radii2D = meta_data["max_radii2D"]
            # model.xyz_gradient_accum = meta_data["xyz_gradient_accum"]
            # model.denom = meta_data["denom"]

            self.models.append(model)

            pbar.update(1)

    def eval(
        self,
        iteration: int = None,
        split: Literal["val", "test"] = "val",
    ) -> dict:
        """
        Main logic for evaluation.
        """
        metrics = dict()
        if split == "val":
            assert self.val_dataset is not None
            dataset = self.val_dataset
        elif split == "test":
            assert self.test_dataset is not None
            dataset = self.test_dataset
        else:
            if self.verbose:
                print(f'[WARNING] {split} set does not exist!')
            return

        eval_dir = os.path.join(self.eval_dir, split)
        os.makedirs(eval_dir, exist_ok=True)

        num_blocks = len(self.models)

        meta_data = self.meta_data[0]
        meta_data["split"] = split
        for k, model in enumerate(self.models):
            if model.get_xyz.device == torch.device('cpu'):
                model = model.to("cuda")
            else:
                model.eval()
            
            # meta_data = self.meta_data[k]
            iteration = self.model_iterations[k] if iteration is None else iteration

            val_dir = eval_dir
            if self.config.dataset.multi_blocks and len(self.models) > 1:
                val_dir = os.path.join(eval_dir, f"block_{k}")
            os.makedirs(val_dir, exist_ok=True)

            if self.verbose:
                print(f'Results are saving to: {val_dir}')

            if split == "val" and iteration >= self.config.trainer.max_iterations:
                splat_dir = os.path.join(val_dir, "splats")
                os.makedirs(splat_dir, exist_ok=True)
                splat_path = os.path.join(splat_dir, f"iter_{iteration}.splat")
                model.save_splat(splat_path)

                ply_dir = os.path.join(val_dir, "ply")
                os.makedirs(ply_dir, exist_ok=True)
                ply_path = os.path.join(ply_dir, f"iter_{iteration}.ply")
                model.save_ply(ply_path)
                colmap_ply_path = os.path.join(
                    ply_dir, f"iter_{iteration}_points3D.txt")
                model.save_colmap_ply(colmap_ply_path)

            image_dir = os.path.join(val_dir, "images")
            if iteration is not None:
                image_dir = os.path.join(image_dir, f'iter_{iteration}')
            os.makedirs(image_dir, exist_ok=True)

            cameras = dataset.cameras
            pbar = tqdm.trange(
                len(cameras), desc=f"Validating {self.config.expname}", leave=False
            )
            psnrs, ssims, lpips, render_times, render_mems = {}, {}, {}, {}, {}

            for i in range(len(cameras)):  # pylint: disable=C0200
                camera = cameras[i]
                camera = camera.copy_to_device(self.device)

                psnrs[i], ssims[i], lpips[i], render_times[i], render_mems[i] = self._eval(
                    camera, model, meta_data, image_dir, i
                )

                pbar.update(1)

            avg_psnr = sum(psnrs.values()) / len(psnrs)
            avg_ssim = sum(ssims.values()) / len(ssims)
            avg_lpips = sum(lpips.values()) / len(lpips)
            avg_time = sum(render_times.values()) / len(render_times)
            avg_mem = sum(render_mems.values()) / len(render_mems)

            metric_key = k if k < num_blocks else "global"
            metrics[metric_key] = {
                'iteration': iteration,
                'all_psnr': psnrs,
                'all_ssim': ssims,
                'all_lpips': lpips,
                'all_times': render_times,
                'all_mems': render_mems,
                'psnr': avg_psnr,
                'ssim': avg_ssim,
                'lpips': avg_lpips,
                'time': avg_time,
                'memory': avg_mem,
                "points": model.get_xyz.shape[0],
            }

            if split == "test":
                video_name = os.path.join(eval_dir, 'render.mp4')
                rendered_image_dir = os.path.join(image_dir, "rgb_test")
                os.system(
                    f"ffmpeg -framerate 10 -i {rendered_image_dir}/%3d.png " +
                    "-vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' " +
                    f"-c:v libx264 -pix_fmt yuv420p {video_name}")

        metric_file = os.path.join(eval_dir, 'metrics.json')
        json_obj = json.dumps(metrics, indent=4)
        if self.verbose:
            print(f'Saving metrics to {metric_file}')
        with open(metric_file, 'a', encoding='utf-8') as json_file:
            json_file.write(json_obj)

        return metrics

    @torch.no_grad()
    def _eval(self, data, model, meta_data, eval_dir, image_index):  # pylint: disable=W0613
        pixels = data.image  # [height, width, RGB]

        if self.config.neural_field_type == "gs":
            from conerf.render.gaussian_render import render  # pylint: disable=C0415
        elif self.config.neural_field_type == "scaffold_gs":
            from conerf.render.scaffold_gs_render import render  # pylint: disable=C0415
        else:
            raise NotImplementedError

        # rendering
        torch.cuda.reset_peak_memory_stats()
        time_start = time.time()

        render_results = render(
            gaussian_splat_model=model,
            viewpoint_camera=data,
            pipeline_config=self.config.pipeline,
            bkgd_color=self.color_bkgd,
            anti_aliasing=self.config.texture.anti_aliasing,
            separate_sh=False, # True,
        )

        render_time = time.time() - time_start
        render_max_mem = torch.cuda.max_memory_allocated() / (1024.0 ** 2)

        colors, screen_space_points, visibility_filter, radii = (  # pylint: disable=W0612
            render_results["rendered_image"],
            render_results["screen_space_points"],
            render_results["visibility_filter"],
            render_results["radii"],
        )
        colors, depth = render_results["rendered_image"], render_results["depth"]

        pixels, colors = pixels.cpu(), colors.cpu()
        colors = torch.clamp(colors, 0, 1)
        depth = colorize(depth.cpu().squeeze(0), cmap_name="jet")

        if meta_data["split"] == "val":
            colors_cc = color_correct(colors.permute(
                1, 2, 0).numpy(), pixels.numpy())
            colors_cc = torch.from_numpy(colors_cc).permute(2, 0, 1)
        else:
            colors_cc = colors

        image_dict = {}
        image_dict["rgb_gt"] = pixels
        image_dict["rgb_test"] = colors_cc.permute(1, 2, 0)
        image_dict["depth"] = depth

        save_images(save_dir=eval_dir,
                    image_dict=image_dict, index=image_index)

        if meta_data["split"] == "val":
            pixels = pixels[None, ...].to(self.device).permute(0, 3, 1, 2)
            colors_cc = colors_cc[None, ...].to(self.device)
            psnr = compute_psnr(pixels, colors_cc).item()
            ssim = compute_ssim(pixels, colors_cc)
            lpips = compute_lpips(self.lpips_loss, pixels, colors_cc)
        else:
            psnr, ssim, lpips = 0, 0, 0

        return psnr, ssim, lpips, render_time, render_max_mem

    def _export_mesh(self, model, iteration, mesh_dir):
        pass
