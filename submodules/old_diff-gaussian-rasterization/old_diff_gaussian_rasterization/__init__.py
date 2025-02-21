#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C
# modification by VITA, Kevin
# added the f_count, a count flag to count the number of time a guassian is activated.


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(
        item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)


def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    if raster_settings.f_count:
        return _RasterizeGaussians.forward_count(
            means3D,
            means2D,
            sh,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3Ds_precomp,
            raster_settings,
        )

    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
        )
        gaussians_count, important_score, num_rendered, color, depth, alpha, radii, geomBuffer, binningBuffer, imgBuffer = \
            None, None, None, None, None, None, None, None, None, None
        
        # Invoke C++/CUDA rasterizer.
        if raster_settings.f_count:
            args = args + (raster_settings.f_count,)
            if raster_settings.debug:
                # Copy them before they can be corrupted
                cpu_args = cpu_deep_copy_tuple(args)
                try:
                    gaussians_count, important_score, num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = \
                        _C.count_gaussians(*args)
                except Exception as ex:
                    torch.save(cpu_args, "snapshot_fw.dump")
                    print(
                        "\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                    raise ex
            else:
                gaussians_count, important_score, num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = \
                    _C.count_gaussians(*args)
        else:
            if raster_settings.debug:
                # Copy them before they can be corrupted
                cpu_args = cpu_deep_copy_tuple(args)
                try:
                    num_rendered, color, depth, alpha, radii, pixels, geomBuffer, binningBuffer, imgBuffer = \
                        _C.rasterize_gaussians(*args)
                except Exception as ex:
                    torch.save(cpu_args, "snapshot_fw.dump")
                    print(
                        "\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                    raise ex
            else:
                num_rendered, color, depth, alpha, radii, pixels, geomBuffer, binningBuffer, imgBuffer = \
                    _C.rasterize_gaussians(*args)
        
        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations,
                              cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, alpha)
        # ctx.count = gaussians_count
        # ctx.important_score = important_score

        if raster_settings.f_count:
            return gaussians_count, important_score, color, radii
        
        return color, radii, depth, alpha, pixels

    @staticmethod
    def forward_count(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):
        assert (raster_settings.f_count)
        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
            raster_settings.f_count
        )
        
        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            # Copy them before they can be corrupted
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                gaussians_count, important_score, num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = \
                    _C.count_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print(
                    "\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            gaussians_count, important_score, num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = \
                _C.count_gaussians(*args)

        return gaussians_count, important_score, color, radii

    @staticmethod
    # def backward(ctx, grad_out_color, _):
    def backward(ctx, grad_color, grad_radii, grad_depth, grad_alpha, _):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, alpha = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D,
                radii,
                colors_precomp,
                scales,
                rotations,
                raster_settings.scale_modifier,
                cov3Ds_precomp,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                # grad_out_color,
                grad_color,
                grad_depth,
                grad_alpha,
                sh,
                raster_settings.sh_degree,
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                alpha,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            # Copy them before they can be corrupted
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, depth = \
                    _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print(
                    "\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, depth = \
                _C.rasterize_gaussians_backward(*args)

        if raster_settings.depth_threshold > 0:
            # Calculate the scaling factor.
            scaling_factor = torch.minimum(torch.ones_like(depth), (depth / raster_settings.depth_threshold) ** 2)
            
            def scale_tensor(tensor, scaling_factor):
                num_dims = len(tensor.shape)
                for _ in range(num_dims - 2):
                    scaling_factor = scaling_factor.unsqueeze(-1)
                scaling_factor_expanded = scaling_factor.expand_as(tensor)
                return tensor * scaling_factor_expanded
            
            grad_means2D = scale_tensor(grad_means2D, scaling_factor)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    depth_threshold: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool
    f_count: bool


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)

        return visible

    def forward(self, means3D, means2D, opacities, shs=None, colors_precomp=None, scales=None, rotations=None, cov3D_precomp=None):

        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception(
                'Please provide excatly one of either SHs or precomputed colors!')

        if ((scales is None or rotations is None) and cov3D_precomp is None) or \
            ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception(
                'Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
        )
    # TODO(Kevin add counter version of forward)

    def forward_count(self, means3D, means2D, opacities, shs=None, colors_precomp=None, scales=None, rotations=None, cov3D_precomp=None):

        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception(
                'Please provide excatly one of either SHs or precomputed colors!')

        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception(
                'Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
        )
