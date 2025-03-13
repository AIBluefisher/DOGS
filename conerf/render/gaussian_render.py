# pylint: disable=[E1101]

import math
import torch
import torch.nn.functional as F

from omegaconf import OmegaConf

# from diff_gaussian_rasterization import (
from old_diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer
)
from gsplat import rasterization

from conerf.geometry.camera import Camera
from conerf.model.gaussian_fields.app_embed import AppearanceOptModule
from conerf.model.gaussian_fields.gaussian_splat_model import GaussianSplatModel
from conerf.model.gaussian_fields.sh_utils import eval_sh


def render_gsplat(
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
    # per-image appearance embedding.
    app_embedding: torch.Tensor = None,
    app_module: AppearanceOptModule = None,
    global_index: int = None,
    # Deblur related parameters.
    deblur_net: torch.nn.Module = None,
    lambda_scale: float = 0.01,
    lambda_position: float = 0.01,
    use_position_offset: bool = False,
    max_clamp: float = 1.1,
    device="cuda:0",
):
    means3D = gaussian_splat_model.get_xyz
    opacity = gaussian_splat_model.get_opacity
    K = viewpoint_camera.K
    scales = gaussian_splat_model.get_scaling * scaling_modifier
    rotations = gaussian_splat_model.get_quaternion

    if override_color is not None:
        colors = override_color  # [N,3]
        sh_degree = None
    else:
        colors = gaussian_splat_model.get_features  # [N,K,3]
        sh_degree = gaussian_splat_model.active_sh_degree

    viewmat = viewpoint_camera.world_to_camera.transpose(0, 1)

    if deblur_net is not None:
        _positions = means3D.detach()
        _scales = scales.detach()
        _rotations = rotations.detach()
        _viewdirs = viewpoint_camera.camera_center.repeat(means3D.shape[0], 1)

        delta_scales, delta_rotations, delta_positions = deblur_net(
            _positions, _scales, _rotations, _viewdirs)
        delta_scales = torch.clamp(
            lambda_scale * delta_scales + (1 - lambda_scale), min=1.0, max=max_clamp)
        delta_rotations = torch.clamp(
            lambda_scale * delta_rotations + (1 - lambda_scale), min=1.0, max=max_clamp)

        if not use_position_offset:  # Defocus blur.
            scales = scales * delta_scales
            rotations = rotations * delta_rotations
        else:  # Motion blur.
            delta_positions = lambda_position * delta_positions
            # Reshape to M 3D Gaussian sets.
            delta_positions = delta_positions.view(
                -1, 3, deblur_net.num_gaussian_sets - 1)
            delta_positions = torch.cat([
                delta_positions,
                torch.zeros((means3D.shape[0], 3, 1), dtype=means3D.dtype, device=means3D.device)
            ], dim=-1)
            delta_scales = delta_scales.view(-1,
                                             3, deblur_net.num_gaussian_sets)
            delta_rotations = delta_rotations.view(
                -1, 4, deblur_net.num_gaussian_sets)

            renders, radiis, depths = [], [], []
            screen_space_points_set, visibility_filters = [], []
            for i in range(deblur_net.num_gaussian_sets):
                positions = means3D + delta_positions[..., i]
                trans_scales = scales * delta_scales[..., i]
                trans_rotations = rotations * delta_rotations[..., i]

                render_colors, _, info = rasterization(
                    means=positions,
                    quats=trans_rotations,
                    scales=trans_scales,
                    opacities=opacity.squeeze(-1),
                    colors=colors,
                    viewmats=viewmat[None],
                    Ks=K,
                    backgrounds=bkgd_color[None],
                    width=int(viewpoint_camera.width),
                    height=int(viewpoint_camera.height),
                    packed=False,
                    sh_degree=sh_degree,
                    render_mode="RGB+ED",
                )

                rendered_image = render_colors[..., :3][0].permute(2, 0, 1)
                depth = render_colors[..., 3:][0].permute(2, 0, 1)
                radii = info["radii"].squeeze(0)  # [N,]
                try:
                    info["means2d"].retain_grad()  # [1,N,2]
                except:  # pylint: disable=W0702
                    pass
                
                renders.append(rendered_image)
                radiis.append(radii)
                depths.append(depth)
                screen_space_points_set.append(info["means2d"])
                visibility_filters.append(radii > 0)

            rendered_image = sum(renders) / len(renders)
            depth = sum(depths) / len(depths)
            return {
                "rendered_image": rendered_image,
                "screen_space_points": screen_space_points_set,
                "visibility_filter": visibility_filters,
                "radii": radiis,
                "scaling": scales,
                "depth": depths,
            }

    render_colors, _, info = rasterization(
        means=means3D,    # [N,3]
        quats=rotations,  # [N,4]
        scales=scales,    # [N,3]
        opacities=opacity.squeeze(-1),  # [N,]
        colors=colors,
        viewmats=viewmat[None],  # [1,4,4]
        Ks=K,  # [1,3,3]
        backgrounds=bkgd_color[None],
        width=int(viewpoint_camera.width),
        height=int(viewpoint_camera.height),
        packed=False,
        sh_degree=sh_degree,
        render_mode="RGB+ED",
    )

    rendered_image = render_colors[..., :3][0].permute(2, 0, 1)
    depth = render_colors[..., 3:][0].permute(2, 0, 1)
    radii = info["radii"].squeeze(0)  # [N,]
    try:
        info["means2d"].retain_grad()  # [1,N,2]
    except:  # pylint: disable=W0702
        pass

    return {
        "rendered_image": rendered_image,  # [3,H,W]
        "screen_space_points": info["means2d"],
        "visibility_filter": radii > 0,
        "radii": radii,
        "scaling": scales,
        "depth": depth,
    }


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
    # per-image appearance embedding.
    app_embedding: torch.Tensor = None,
    app_module: AppearanceOptModule = None,
    global_index: int = None,
    # Deblur related parameters.
    deblur_net: torch.nn.Module = None,
    lambda_scale: float = 0.01,
    lambda_position: float = 0.01,
    use_position_offset: bool = False,
    max_clamp: float = 1.1,
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
        # antialiasing=anti_aliasing,
        depth_threshold=depth_threshold,
        f_count=False,
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

    if app_embedding is not None:
        embed_ids = torch.tensor([global_index], dtype=torch.int32).to(device)
        dirs = means3D[None, :, :] - viewpoint_camera.camera_center[None, :]
        colors = app_module(
            features=app_embedding,
            embed_ids=embed_ids,
            dirs=dirs,
            sh_degree=3,
        )
        colors = colors + gaussian_splat_model.get_colors
        override_color = torch.sigmoid(colors)

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

    if deblur_net is not None:
        _positions = means3D.detach()
        _scales = scales.detach()
        _rotations = rotations.detach()
        _viewdirs = viewpoint_camera.camera_center.repeat(means3D.shape[0], 1)

        delta_scales, delta_rotations, delta_positions = deblur_net(
            _positions, _scales, _rotations, _viewdirs)
        delta_scales = torch.clamp(
            lambda_scale * delta_scales + (1 - lambda_scale), min=1.0, max=max_clamp)
        delta_rotations = torch.clamp(
            lambda_scale * delta_rotations + (1 - lambda_scale), min=1.0, max=max_clamp)

        if not use_position_offset:  # Defocus blur.
            scales = scales * delta_scales
            rotations = rotations * delta_rotations
        else:  # Motion blur.
            delta_positions = lambda_position * delta_positions
            # Reshape to M 3D Gaussian sets.
            delta_positions = delta_positions.view(
                -1, 3, deblur_net.num_gaussian_sets - 1)
            delta_positions = torch.cat([
                delta_positions,
                torch.zeros((means3D.shape[0], 3, 1), dtype=means3D.dtype, device=means3D.device)
            ], dim=-1)
            delta_scales = delta_scales.view(-1,
                                             3, deblur_net.num_gaussian_sets)
            delta_rotations = delta_rotations.view(
                -1, 4, deblur_net.num_gaussian_sets)

            renders, radiis, depths = [], [], []
            screen_space_points_set, visibility_filters = [], []
            for i in range(deblur_net.num_gaussian_sets):
                _screen_space_points = torch.zeros_like(
                    gaussian_splat_model.get_xyz,
                    dtype=gaussian_splat_model.get_xyz.dtype,
                    requires_grad=True,
                    device=device
                ) + 0
                try:
                    _screen_space_points.retain_grad()
                except:  # pylint: disable=W0702
                    pass

                _means2D = _screen_space_points
                positions = means3D + delta_positions[..., i]
                trans_scales = scales * delta_scales[..., i]
                trans_rotations = rotations * delta_rotations[..., i]

                if separate_sh:
                    rendered_image, radii, depth = rasterizer(
                        means3D=positions,
                        means2D=_means2D,
                        dc=dc,
                        shs=spherical_harmonics,
                        colors_precomp=precomputed_colors,
                        opacities=opacity,
                        scales=trans_scales,
                        rotations=trans_rotations,
                        cov3D_precomp=cov3D_precomp,
                    )
                else:
                    # rendered_image, radii, depth = rasterizer(
                    rendered_image, radii, depth, _, __ = rasterizer(
                        means3D=positions,
                        means2D=_means2D,
                        shs=spherical_harmonics,
                        colors_precomp=precomputed_colors,
                        opacities=opacity,
                        scales=trans_scales,
                        rotations=trans_rotations,
                        cov3D_precomp=cov3D_precomp,
                    )
                renders.append(rendered_image)
                radiis.append(radii)
                depths.append(depth)
                screen_space_points_set.append(_screen_space_points)
                visibility_filters.append(radii > 0)

            rendered_image = sum(renders) / len(renders)
            depth = sum(depths) / len(depths)
            return {
                "rendered_image": rendered_image,
                "screen_space_points": screen_space_points_set,
                "visibility_filter": visibility_filters,
                "radii": radiis,
                "scaling": scales,
                "depth": depths,
            }

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
        # rendered_image, radii, depth = rasterizer(
        rendered_image, radii, depth, _, __ = rasterizer(
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
        exposure = gaussian_splat_model.get_exposure_from_index(global_index)
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
