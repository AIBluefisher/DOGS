# pylint: disable=E1101

import torch
from torch import nn
import torch.nn.functional as F


class DecoupledAppearanceEmbedding(nn.Module):
    """
    Implementation of appearance embedding in the paper 
    VastGaussian: https://arxiv.org/abs/2402.17427
    downsampling factor = 32
    """
    def __init__(self, num_views: int, embedding_dim: int = 64) -> None:
        super().__init__()
        self.appearance_embedding = nn.Parameter(
            torch.zeros(num_views, embedding_dim),
        )
        self.fusion = nn.Conv2d(embedding_dim + 3, 256, kernel_size=3, padding=1)
        in_channels = [256, 128, 64, 32]
        self.upsample = nn.Sequential(
            *[nn.Sequential(
                nn.PixelShuffle(2),
                nn.Conv2d(i // 4, i // 2, kernel_size=3, padding=1),
                nn.ReLU(),
            ) for i in in_channels]
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 3, kernel_size=3, padding=1),
            # nn.Sigmoid() # FIXME: necessary?
        )

    def forward(self, image: torch.Tensor, index: int, image_size: tuple[int]):
        """
        Args:
            image: 3 x H x W
            index: index of the image
            image_size: original size of the image
        
        Return:
            out: 3 x H x W
        """
        embedding = self.appearance_embedding[index]
        C, H, W = image.shape  # pylint: disable=C0103

        out = torch.cat([image, embedding[:, None, None].repeat(1, H, W)], dim=0)
        out = self.fusion(out)
        out = self.upsample(out)
        out = F.interpolate(out.unsqueeze(0), size=image_size, mode='bilinear')[0]
        out = self.out_conv(out)

        return out
