"""Super-Resolution branch for SRE-YOLO."""
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class SRBranch(nn.Module):
    """Upscales tapped feature maps from the backbone to HR space.

    Args:
        in_channels: Number of input channels from the tapped feature map.
        upscale: Spatial upscaling factor (default 2 → 640→1280).
        n_resblocks: Number of residual blocks in the decoder.
        decoder: 'shallow' or 'deep' decoder variant.
    """

    def __init__(
        self,
        in_channels: int,
        upscale: int = 2,
        n_resblocks: int = 16,
        decoder: str = "deep",
    ):
        super().__init__()
        mid = 64 if decoder == "shallow" else 128
        self.head = nn.Conv2d(in_channels, mid, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(mid) for _ in range(n_resblocks)])
        self.upsample = nn.Sequential(
            nn.Conv2d(mid, mid * upscale ** 2, 3, padding=1),
            nn.PixelShuffle(upscale),
        )
        self.tail = nn.Conv2d(mid, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        x = x + self.body(x)
        x = self.upsample(x)
        return self.tail(x)
