# pylint: disable=[E1101]

import math
import torch

from omegaconf import OmegaConf

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer
)

from conerf.geometry.camera import Camera
from conerf.model.gaussian_fields.gaussian_splat_model import GaussianSplatModel
from conerf.model.gaussian_fields.sh_utils import eval_sh


def render(
    gaussian_splat_model: GaussianSplatModel,
    viewpoint_camera: Camera,
    pipeline_config: OmegaConf,
    bkgd_color: torch.Tensor,
    scaling_modifier: float = 1.0,
    anti_aliasing: bool = False,
    override_color: torch.Tensor = None,
    separate_sh: bool = False,
    use_trained_exposure: bool = False,
    depth_threshold: float = 0.0,
    device="cuda:0",
):
    # Create zero tensor. We will use it to make pytorch return
    # gradients of the 2D (screen-space) means.
    screen_space_points = torch.zeros_like(
        gaussian_splat_model.get_xyz,
        dtype=gaussian_splat_model.get_xyz.dtype,
        requires_grad=True,
        device=device
    ) + 0
    try:
        screen_space_points.retain_grad()
    except:  # pylint: disable=W0702
        pass

    # Set up rasterization configuration
    tan_fov_x = math.tan(viewpoint_camera.fov_x * 0.5)
    tan_fov_y = math.tan(viewpoint_camera.fov_y * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.height),
        image_width=int(viewpoint_camera.width),
        tanfovx=tan_fov_x,
        tanfovy=tan_fov_y,
        bg=bkgd_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_to_camera,
        projmatrix=viewpoint_camera.projective_matrix,
        sh_degree=gaussian_splat_model.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipeline_config.debug,
        antialiasing=anti_aliasing,
        depth_threshold=depth_threshold,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = gaussian_splat_model.get_xyz
    means2D = screen_space_points
    opacity = gaussian_splat_model.get_opacity

    # If the precomputed 3D covariance is not provided, we compute it from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipeline_config.compute_cov3D_python:
        cov3D_precomp = gaussian_splat_model.get_covariance(scaling_modifier)
    else:
        rotations = gaussian_splat_model.get_quaternion
        scales = gaussian_splat_model.get_scaling

    # Pre-compute colors from SHs in Python if they are not provided.
    # If not, then SH -> RGB conversion will be done by rasterizer.
    spherical_harmonics = None
    precomputed_colors = None
    if override_color is None:
        if pipeline_config.convert_SHs_python:
            shs_view = (
                gaussian_splat_model.get_features.transpose(1, 2)
                .view(-1, 3, (gaussian_splat_model.max_sh_degree + 1) ** 2)
            )
            dir_pp = (
                gaussian_splat_model.get_xyz - viewpoint_camera.camera_center.repeat(
                    gaussian_splat_model.get_features.shape[0], 1)
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(
                gaussian_splat_model.active_sh_degree,
                shs_view,
                dir_pp_normalized
            )
            precomputed_colors = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            # spherical_harmonics = gaussian_splat_model.get_features
            if separate_sh:
                dc = gaussian_splat_model.get_features_dc
                spherical_harmonics = gaussian_splat_model.get_features_rest
            else:
                spherical_harmonics = gaussian_splat_model.get_features
    else:
        precomputed_colors = override_color

    # Rasterize visible Gaussians to image to obtain their radii (on screen).
    if separate_sh:
        rendered_image, radii, depth = rasterizer(
            means3D=means3D,
            means2D=means2D,
            dc=dc,
            shs=spherical_harmonics,
            colors_precomp=precomputed_colors,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
    else:
        rendered_image, radii, depth = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=spherical_harmonics,
            colors_precomp=precomputed_colors,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

    # Apply exposure to rendered image (training only)
    if use_trained_exposure:
        exposure = gaussian_splat_model.get_exposure_from_id(
            viewpoint_camera.image_index)
        rendered_image = torch.matmul(
            rendered_image.permute(1, 2, 0), exposure[:3, :3]
        ).permute(2, 0, 1) + exposure[:3, 3, None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    results = {
        "rendered_image": rendered_image,  # [RGB, height, width]
        "screen_space_points": screen_space_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "scaling": scales,
        "depth": depth,
    }

    return results


def count_render(
    gaussian_splat_model: GaussianSplatModel,
    viewpoint_camera: Camera,
    pipeline_config: OmegaConf,
    bkgd_color: torch.Tensor,
    scaling_modifier: float = 1.0,
    anti_aliasing: bool = False,
    override_color=None,
    subpixel_offset=None,
    device="cuda:0",
):
    # Create zero tensor. We will use it to make pytorch return
    # gradients of the 2D (screen-space) means.
    screen_space_points = torch.zeros_like(
        gaussian_splat_model.get_xyz,
        dtype=gaussian_splat_model.get_xyz.dtype,
        requires_grad=True,
        device=device
    ) + 0
    try:
        screen_space_points.retain_grad()
    except:  # pylint: disable=W0702
        pass

    # Set up rasterization configuration
    tan_fov_x = math.tan(viewpoint_camera.fov_x * 0.5)
    tan_fov_y = math.tan(viewpoint_camera.fov_y * 0.5)

    if subpixel_offset is None:
        subpixel_offset = torch.zeros((
            int(viewpoint_camera.height),
            int(viewpoint_camera.width),
            2),
            dtype=torch.float32,
            device=device
        )

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.height),
        image_width=int(viewpoint_camera.width),
        tanfovx=tan_fov_x,
        tanfovy=tan_fov_y,
        bg=bkgd_color,
        scale_modifier=scaling_modifier,
        depth_threshold=0.0,
        viewmatrix=viewpoint_camera.world_to_camera,
        projmatrix=viewpoint_camera.projective_matrix,
        sh_degree=gaussian_splat_model.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipeline_config.debug,
        f_count=True,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D = gaussian_splat_model.get_xyz
    means2D = screen_space_points
    opacity = gaussian_splat_model.get_opacity

    # If the precomputed 3D covariance is not provided, we compute it from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipeline_config.compute_cov3D_python:
        cov3D_precomp = gaussian_splat_model.get_covariance(scaling_modifier)
    else:
        rotations = gaussian_splat_model.get_quaternion
        scales = gaussian_splat_model.get_scaling

    # Pre-compute colors from SHs in Python if they are not provided.
    # If not, then SH -> RGB conversion will be done by rasterizer.
    spherical_harmonics = None
    precomputed_colors = None
    if override_color is None:
        if pipeline_config.convert_SHs_python:
            shs_view = (
                gaussian_splat_model.get_features.transpose(1, 2)
                .view(-1, 3, (gaussian_splat_model.max_sh_degree + 1) ** 2)
            )
            dir_pp = (
                gaussian_splat_model.get_xyz - viewpoint_camera.camera_center.repeat(
                    gaussian_splat_model.get_features.shape[0], 1)
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(
                gaussian_splat_model.active_sh_degree,
                shs_view,
                dir_pp_normalized
            )
            precomputed_colors = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            spherical_harmonics = gaussian_splat_model.get_features
    else:
        precomputed_colors = override_color

    # Rasterize visible Gaussians to image to obtain their radii (on screen).
    gaussians_count, important_score, rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=spherical_harmonics,
        colors_precomp=precomputed_colors,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "rendered_image": rendered_image,  # [RGB, height, width]
        "screen_space_points": screen_space_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "gaussians_count": gaussians_count,
        "important_score": important_score,
    }
