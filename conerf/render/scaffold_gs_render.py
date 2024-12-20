# pylint: disable=E1101

import math

import torch
from omegaconf import OmegaConf

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer
)

from conerf.geometry.camera import Camera
from conerf.model.gaussian_fields.scaffold_gs import ScaffoldGS


def render(
    gaussian_splat_model: ScaffoldGS,
    viewpoint_camera: Camera,
    pipeline_config: OmegaConf,
    bkgd_color: torch.Tensor,
    visible_mask: torch.Tensor = None,
    scaling_modifier: float = 1.0,
    anti_aliasing: bool = False,
    override_color: torch.Tensor = None,
    separate_sh: bool = False,
    use_trained_exposure: bool = False,
    depth_threshold: float = 0.0,
    device="cuda:0",
):
    # Get neural gaussian attributes.
    xyz, opacity, color, scaling, quaternion, neural_opacity, combined_mask = \
        gaussian_splat_model.generate_neural_gaussians(
            viewpoint_camera, visible_mask)

    # Create zero tensor. We will use it to make pytorch return
    # gradients of the 2D (screen-space) means.
    screen_space_points = torch.zeros_like(
        xyz, dtype=xyz.dtype, requires_grad=True, device=device
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
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipeline_config.debug,
        antialiasing=anti_aliasing,
        depth_threshold=depth_threshold,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D, means3D = screen_space_points, xyz
    scales, rotations = scaling, quaternion

    # Rasterize visible Gaussians to image to obtain their radii (on screen).
    rendered_image, radii, depth = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=color,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )

    # Apply exposure to rendered image (training only)
    if use_trained_exposure:
        exposure = gaussian_splat_model.get_exposure_from_id(viewpoint_camera.image_index)
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
        "combined_mask": combined_mask,
        "neural_opacity": neural_opacity,
        "scaling": scaling,
        "depth": depth,
    }

    return results


def prefilter_voxel(
    gaussian_splat_model: ScaffoldGS,
    viewpoint_camera: Camera,
    pipeline_config: OmegaConf,
    bkgd_color: torch.Tensor,
    scaling_modifier: float = 1.0,
    device="cuda:0",
):
    # Create zero tensor. We will use it to make pytorch return
    # gradients of the 2D (screen-space) means.
    screen_space_points = torch.zeros_like(
        gaussian_splat_model.get_anchor,
        dtype=gaussian_splat_model.get_anchor.dtype,
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
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipeline_config.debug,
        antialiasing=False,
        depth_threshold=0.0,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D = gaussian_splat_model.get_anchor

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

    radii_pure = rasterizer.visible_filter(
        means3D=means3D,
        scales=scales[:, :3],
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    return radii_pure > 0
