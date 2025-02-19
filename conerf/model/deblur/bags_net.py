# pylint: disable=E1101

import torch
import torch.nn as nn


class BagsSingleResKernelNet(nn.Module):
    """
    Single-resolution kernel for defocus blur.
    """
    def __init__(
        self,
        num_images: int,
        image_embed_dim: int = 32,
        kernel_size: int = 17,
        use_rgbd: bool = True,
    ):
        super().__init__()

        self.num_images = num_images
        self.use_rgbd = use_rgbd

        self.image_embedding = nn.Embedding(self.num_images, image_embed_dim)

        rgbd_dim = 32 if self.use_rgbd else 0
        posi_embed_dim = 16

        self.mlp_base = nn.Sequential(
            nn.Conv2d(32 + posi_embed_dim + rgbd_dim,
                      64, 1, bias=False), nn.ReLU(),
            nn.Conv2d(64, 64, 1, bias=False), nn.ReLU(),
        )
        self.mlp_head = nn.Conv2d(64, kernel_size ** 2, 1, bias=False)
        self.mlp_mask = nn.Conv2d(64, 1, 1, bias=False)

        self.conv_rgbd = nn.Sequential(
            nn.Conv2d(4, 64, 5, padding=2), nn.ReLU(), nn.InstanceNorm2d(64),
            nn.Conv2d(64, 64, 5, padding=2), nn.ReLU(), nn.InstanceNorm2d(64),
            nn.Conv2d(64, 32, 3, padding=1),
        )

    def forward(self, image_index: int, pos_enc: torch.Tensor, image: torch.Tensor):
        latent = self.image_embedding(torch.LongTensor(
            [image_index]).to(image.device))[None, None]
        latent = latent.expand(
            pos_enc.shape[0], pos_enc.shape[1], pos_enc.shape[2], pos_enc.shape[-1]
        )

        inp = torch.cat([latent, pos_enc], dim=-1).permute(0, 3, 1, 2)

        if self.use_rgbd:
            feat = self.conv_rgbd(image)
            feat = self.mlp_base(torch.cat([inp, feat], dim=1))
        else:
            feat = self.mlp_base(inp)

        weight = torch.softmax(self.mlp_head(feat), dim=1)
        mask = torch.sigmoid(self.mlp_mask(feat))

        return weight, mask


class BagsMultiResKernelNet(nn.Module):
    """
    Multi-resolution kernel for motion blur.
    """
    def __init__(
        self,
        num_images: int,
        image_embed_dim: int = 32,
        kernel_size1: int = 5,
        kernel_size2: int = 9,
        kernel_size3: int = 17,
        use_rgbd: bool = True,
    ):
        super().__init__()

        self.num_images = num_images
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2
        self.kernel_size3 = kernel_size3
        self.use_rgbd = use_rgbd

        self.image_embedding = nn.Embedding(self.num_images, image_embed_dim)

        rgbd_dim = 32 if self.use_rgbd else 0
        posi_embed_dim = 16

        self.mlp_base1 = nn.Sequential(
            nn.Conv2d(32 + posi_embed_dim + rgbd_dim,
                      64, 1, bias=False), nn.ReLU(),
        )
        self.mlp_head1 = nn.Conv2d(64, kernel_size1 ** 2, 1, bias=False)
        self.mlp_mask1 = nn.Conv2d(64, 1, 1, bias=False)

        self.mlp_base2 = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False), nn.ReLU(),
        )
        self.mlp_head2 = nn.Conv2d(64, kernel_size2 ** 2, 1, bias=False)
        self.mlp_mask2 = nn.Conv2d(64, 1, 1, bias=False)

        self.mlp_base3 = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False), nn.ReLU(),
        )
        self.mlp_head3 = nn.Conv2d(64, kernel_size3 ** 2, 1, bias=False)
        self.mlp_mask3 = nn.Conv2d(64, 1, 1, bias=False)

        self.conv_rgbd = nn.Sequential(
            nn.Conv2d(4, 64, 5, padding=2), nn.ReLU(), nn.InstanceNorm2d(64),
            nn.Conv2d(64, 64, 5, padding=2), nn.ReLU(), nn.InstanceNorm2d(64),
            nn.Conv2d(64, 32, 3, padding=1),
        )

    def forward(self, image_index: int, pos_enc: torch.Tensor, image: torch.Tensor, step: int):
        latent = self.image_embedding(torch.LongTensor(
            [image_index]).to(image.device))[None, None]
        latent = latent.expand(
            pos_enc.shape[0], pos_enc.shape[1], pos_enc.shape[2], latent.shape[-1]
        )

        inp = torch.cat([latent, pos_enc], dim=-1).permute(0, 3, 1, 2)

        if self.use_rgbd:
            feat = self.conv_rgbd(image)
            feat = self.mlp_base1(torch.cat([inp, feat], dim=1))
        else:
            feat = self.mlp_base1(inp)

        #
        if 250 < step < 3000:
            weight = torch.softmax(self.mlp_head1(feat), dim=1)
            mask = torch.sigmoid(self.mlp_mask1(feat))
        elif 3000 <= step < 6000:
            feat = self.mlp_base2(feat)

            weight = torch.softmax(self.mlp_head2(feat), dim=1)
            mask = torch.sigmoid(self.mlp_mask2(feat))
        else:
            feat = self.mlp_base2(feat)
            feat = self.mlp_base3(feat)

            weight = torch.softmax(self.mlp_head3(feat), dim=1)
            mask = torch.sigmoid(self.mlp_mask3(feat))

        return weight, mask
