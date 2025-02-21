# pylint: disable=[E1101,E1102,C0103]

import os
from typing import List
from dataclasses import dataclass

import tqdm
import torch
import torch.nn.functional as F
import numpy as np

from conerf.datasets.utils import Rays
from conerf.geometry.camera import Camera


def inverse_pose(pose: torch.Tensor):
    inv_pose = torch.zeros_like(pose)
    inv_pose[..., :3, :3] = pose[..., :3, :3].transpose(-2, -1)
    inv_pose[..., :3,  3] = -inv_pose[..., :3, :3] @ pose[..., :3, 3]
    inv_pose[..., -1, -1] = 1
    return inv_pose


def compose_camera(
    image_index: int,
    image_path: str,
    image: torch.Tensor,
    camtoworld: torch.Tensor,
    intrinsics: torch.Tensor,
    normal: torch.Tensor = None,
    channels: int = 3,
    device: str = "cuda:0",
) -> Camera:
    # channels = image.shape[-1]
    if channels == 4:
        c2w = camtoworld
        # Change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP
        # (Y down, Z forward).
        c2w[:3, 1:3] *= -1
        world_to_cam = inverse_pose(c2w)
    else:
        world_to_cam = inverse_pose(camtoworld)

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    height, width = int(cy * 2), int(cx * 2)
    camera = Camera(
        image_index=image_index,
        world_to_camera=world_to_cam,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        image_path=image_path,
        image=image,
        width=width,
        height=height,
        normal=normal,
        device=device
    )
    return camera


def compose_cameras(
    image_paths: List[str],
    images: torch.Tensor,
    camtoworlds: torch.Tensor,
    intrinsics: torch.Tensor,
    normals: torch.Tensor = None,
    channels: int = 3,
    device: str = "cuda:0",
) -> List:
    """
    Compose raw data into the 'Camera' class.
    """
    cameras = []
    num_images = len(camtoworlds)
    pbar = tqdm.trange(num_images, desc="Composing cameras")
    for i in range(num_images):
        camera = compose_camera(
            image_index=i,
            image_path=image_paths[i],
            image=images[i] if images is not None else None,
            camtoworld=camtoworlds[i],
            intrinsics=intrinsics[i],
            normal=normals[i] if normals is not None else None,
            channels=channels,
            device=device,
        )
        cameras.append(camera)
        pbar.update(1)

    return cameras


@dataclass
class MiniDataset:
    def __init__(
        self,
        cameras: List[Camera] = [],
        camtoworlds: torch.Tensor = None,
        block_id: int = -1,
    ) -> None:
        self.cameras = cameras
        self.camtoworlds = camtoworlds
        self.current_block = block_id

    def __len__(self):
        return len(self.cameras)

    def write(self, path: str):
        cameras_path = os.path.join(path, "cameras")
        os.makedirs(cameras_path, exist_ok=True)
        pbar = tqdm.trange(
            len(self.cameras),
            desc=f"Storing cameras for block#{self.current_block}", leave=False
        )
        for i, camera in enumerate(self.cameras):
            camera.write(cameras_path, i)
            pbar.update(1)

        cameratoworlds_path = os.path.join(path, "cameratoworlds.pt")
        if not os.path.exists(cameratoworlds_path):
            torch.save(self.camtoworlds, cameratoworlds_path)

    def read(
        self,
        path: str,
        block_id: int = 0,
        read_image: bool = True,
        read_normal: bool = False,
        device='cuda'
    ):
        cameras_path = os.path.join(path, "cameras")
        camera_files = [
            filename for filename in os.listdir(cameras_path) \
                if os.path.isfile(os.path.join(cameras_path, filename))
        ]
        pbar = tqdm.trange(
            len(camera_files), desc=f"Loading cameras for block#{block_id}", leave=False)
        for camera_file in camera_files:
            abs_camera_file = os.path.join(cameras_path, camera_file)
            self.cameras.append(
                Camera.read(abs_camera_file, read_image, read_normal, device=device)
            )
            pbar.update(1)

        cameratoworlds_path = os.path.join(path, "cameratoworlds.pt")
        self.camtoworlds = torch.load(cameratoworlds_path, map_location=torch.device(device))
        self.current_block = block_id


class DatasetBase(torch.utils.data.Dataset):
    """
    Single subject data loader for training and evaluation.
    """
    SPLITS = ["train", "test", "val"]
    BBOX = None
    # WIDTH, HEIGHT = 0, 0
    NEAR, FAR = 0., 0.
    OPENGL_CAMERA = None
    DATA_TYPE = None  # ["SYNTHETIC", "REAL_WORLD"]

    def __init__(
        self,
        subject_id: str,
        root_fp: str,
        split: str,
        data_split_json: str = "",
        color_bkgd_aug: str = "white",
        num_rays: int = None,
        near: float = None,
        far: float = None,
        batch_over_images: bool = True,
        factor: int = 1,
        val_interval: int = 0,
        multi_blocks: bool = False,
        num_blocks: int = 1,
        **kwargs
    ) -> None:
        super().__init__()

        assert split in self.SPLITS, "%s" % split
        assert color_bkgd_aug in ["white", "black", "random"]

        self.apply_mask = kwargs["apply_mask"]
        self.split = split
        self.val_interval = val_interval
        self.num_rays = num_rays
        self.data_split_json = data_split_json
        if near is not None:
            self.NEAR = near
        if far is not None:
            self.FAR = far
        self.training = (num_rays is not None) and (
            split in ["train", "trainval"]
        )
        self.color_bkgd_aug = color_bkgd_aug
        self.batch_over_images = batch_over_images
        self.multi_blocks = multi_blocks
        self.masks = None
        self.bbox_scale_factor = kwargs["bbox_scale_factor"] if "bbox_scale_factor" \
            in kwargs.keys() else [1.0, 1.0, 1.0]
        self.scale = kwargs["scale"]
        self.rotate = kwargs["rotate"]
        self.use_manhattan_world = kwargs["use_manhattan_world"]
        self.model_folder = kwargs["model_folder"]
        self.load_specified_images = kwargs["load_specified_images"]
        self.load_normal = False if "load_normal" not in kwargs else kwargs["load_normal"]
        self.device = kwargs["device"]
        self.mx = kwargs["mx"]
        self.my = kwargs["my"]

        if self.mx is not None and self.my is not None:
            num_blocks = self.mx * self.my

        # Logic to load specific dataset.
        data = self.load_data(root_fp, subject_id, split, factor, num_blocks)
        if self.multi_blocks and split == "train":
            self._block_images, self._block_camtoworlds, self._block_intrinsics = \
                data["rgbs"], data["poses"], data["intrinsics"]
            block_image_paths = data['image_paths']
            block_normals = data["normals"]

            self._block_cameras = [None] * num_blocks
            self.num_blocks = num_blocks
            self._current_block_id = 0

            for k in range(self.num_blocks):
                normals = block_normals[k] if self.load_normal else None
                self._block_cameras[k] = compose_cameras(
                    block_image_paths[k], None, # self._block_images[k],
                    self._block_camtoworlds[k], self._block_intrinsics[k],
                    normals=normals,
                    channels=kwargs["num_channels"],
                    device='cpu',
                )
        else:
            images, camtoworlds, intrinsics, image_paths = \
                data["rgbs"], data["poses"], data["intrinsics"], data["image_paths"]
            normals = data["normals"] if self.load_normal else None
            if not self.batch_over_images:
                self._cameras = compose_cameras(
                    image_paths, images, camtoworlds, intrinsics,
                    normals=normals,
                    channels=kwargs["num_channels"],
                    device='cpu',
                )
            else:
                self._images = images
            self._camtoworlds = camtoworlds
            self._intrinsics = intrinsics

        # Check for parameter validity.
        self._check()

    def _check(self) -> None:
        # if self.DATA_TYPE == "SYNTHETIC":
        #     assert self.WIDTH > 0 and self.HEIGHT > 0
        # assert self.NEAR > 0 and self.FAR > 0
        assert self.OPENGL_CAMERA is not None

    @property
    def current_block(self):
        """Access current block identifier."""
        assert self.multi_blocks is True
        return self._current_block_id

    @property
    def images(self):
        """Access images (in current block)."""
        if self.multi_blocks is False:
            return self._images
        return self._block_images[self._current_block_id]

    @property
    def camtoworlds(self):
        """Access camera poses (in current block)."""
        if self.multi_blocks is False:
            return self._camtoworlds
        return self._block_camtoworlds[self._current_block_id]

    @property
    def intrinsics(self):
        """Access intrinsics (in current block)."""
        if self.multi_blocks is False:
            return self._intrinsics
        return self._block_intrinsics[self._current_block_id]

    @property
    def cameras(self):
        """Access cameras (in current block)."""
        if self.multi_blocks is False or self.split != 'train':
            return self._cameras
        return self._block_cameras[self._current_block_id]

    def to_device(self):
        """Move related data (images, camera poses) to device."""
        if self.training:
            print('[WARNING] Data are required to cached on CPU, ' +
                  'cannot be moved to GPU currently.')
            return

        if self.multi_blocks and self.split == "train":
            self._block_camtoworlds[self.current_block] = \
                self._block_camtoworlds[self.current_block].to(self.device)
            self._block_intrinsics[self.current_block] = \
                self._block_intrinsics[self.current_block].to(self.device)
        else:
            # self._images = self._images.to(self.device)
            self._camtoworlds = self._camtoworlds.to(self.device)
            self._intrinsics = self._intrinsics.to(self.device)
            # if self.masks is not None:
            #     self.masks = self.masks.to(self.device)

    def move_to_next_block(self):
        """Move data to next block."""
        if self._current_block_id == self.num_blocks - 1:
            return
        self._current_block_id += 1

    def move_to_block(self, block_id):
        """Move data to the specified block."""
        assert block_id < self.num_blocks, f"Invalid block id: {block_id}"
        self._current_block_id = block_id

    def load_data(
        self,
        root_fp: str,
        subject_id: str,
        split: str,
        factor: int = 1,
        num_blocks: int = 1
    ):
        """Virtual function for loading data. Should implemented in child class."""
        raise NotImplementedError

    def __len__(self):
        if self.batch_over_images:
            return len(self.images)
        return len(self.cameras)

    @torch.no_grad()
    def __getitem__(self, index):
        data = self.fetch_data(index)
        data = self.preprocess(data)
        return data

    def get_background_color(self):
        """Generate background color."""
        if self.training:
            if self.color_bkgd_aug == "random":
                color_bkgd = torch.rand(3, device=self.device)
            elif self.color_bkgd_aug == "white":
                color_bkgd = torch.ones(3, device=self.device)
            elif self.color_bkgd_aug == "black":
                color_bkgd = torch.zeros(3, device=self.device)
        else:
            # just use white during inference
            color_bkgd = torch.ones(3, device=self.device)

        return color_bkgd

    def preprocess(self, data):
        """
        Process the fetched / cached data with randomness.
        """
        pixels, rays, color_bkgd = data["rgb"], data["rays"], data["color_bkgd"]

        if self.DATA_TYPE == "SYNTHETIC":
            pixels, alpha = torch.split(pixels, [3, 1], dim=-1)
            pixels = pixels * alpha + color_bkgd * (1.0 - alpha)

        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "rays": rays,  # [n_rays,] or [h, w]
            "color_bkgd": color_bkgd,  # [3,]
            **{k: v for k, v in data.items() if k not in ["rgb", "rays"]},
        }

    def update_num_rays(self, num_rays):
        """Dynamically update the number of rays during training."""
        self.num_rays = num_rays

    def fetch_data(self, index):
        """
        Fetch the data (it maybe cached for multiple batches).
        """
        num_rays = self.num_rays

        color_bkgd = self.get_background_color()
        camera = self.cameras[index]

        if self.training:
            if self.batch_over_images:
                image_id = torch.randint(0, len(self.images), size=(num_rays,), device=self.device)
            else:
                image_id = torch.from_numpy(np.array([index] * num_rays)).to(self.device)
            x = torch.randint(0, camera.width, size=(num_rays,), device=self.device)
            y = torch.randint(0, camera.height, size=(num_rays,), device=self.device)
        else:
            image_id = [index]
            x, y = torch.meshgrid(
                torch.arange(camera.width, device=self.device),
                torch.arange(camera.height, device=self.device),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()

        # generate rays
        rgb = self.images[image_id][y, x].to(self.device) / 255.0  # (num_rays, 3/4)

        if self.apply_mask:
            mask = self.masks[image_id, y, x].to(self.device) / 255.0
            rgb = rgb * mask[..., None] + color_bkgd * (1 - mask[..., None])

        image_channels = 4 if self.DATA_TYPE == "SYNTHETIC" else 3
        c2w = self.camtoworlds[image_id]  # (num_rays, 3, 4)
        fx, fy = camera.fx, camera.fy
        cx, cy = camera.cx, camera.cy
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - cx + 0.5) / fx,
                    (y - cy + 0.5) / fy
                    * (-1.0 if self.OPENGL_CAMERA else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if self.OPENGL_CAMERA else 1.0),
        )  # [num_rays, 3]

        # [n_cams, height, width, 3]
        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        viewdirs = directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True
        )

        if self.training:
            origins = torch.reshape(origins, (num_rays, 3))
            viewdirs = torch.reshape(viewdirs, (num_rays, 3))
            rgb = torch.reshape(rgb, (num_rays, image_channels))
        else:
            origins = torch.reshape(origins, (camera.height, camera.width, 3))
            viewdirs = torch.reshape(
                viewdirs, (camera.height, camera.width, 3))
            rgb = torch.reshape(
                rgb, (camera.height, camera.width, image_channels))

        rays = Rays(origins=origins, viewdirs=viewdirs)

        return {
            "rgb": rgb,  # [h, w, 3] or [num_rays, 3]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
            "color_bkgd": color_bkgd,
            "image_index": image_id,
        }
