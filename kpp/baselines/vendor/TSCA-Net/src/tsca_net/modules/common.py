
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEModule(nn.Module):
   

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class DoubleConv(nn.Module):
    """(Conv2d -> GroupNorm -> Mish) x 2 -- basic U-Net block."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.Mish(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.Mish(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetEncoder(nn.Module):
    

    def __init__(self, in_channels: int = 3, out_channels: int = 2) -> None:
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(128, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        self.head = nn.Conv2d(32, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            seg: [B, out_channels, H, W] probabilities
        """
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.sigmoid(self.head(d1))


class CNNEncoder(nn.Module):
    """Strided CNN encoder for goal heatmap generation.

    Args:
        channels: list of channel sizes [in, h1, h2, ..., out]
    """

    def __init__(self, channels: List[int]) -> None:
        super().__init__()
        layers = []
        for i in range(len(channels) - 1):
            layers += [
                nn.Conv2d(channels[i], channels[i + 1], 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(min(8, channels[i + 1]), channels[i + 1]),
                nn.Mish(inplace=True),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNDecoder(nn.Module):
    """Transposed CNN decoder symmetric to :class:`CNNEncoder`.

    Args:
        channels: list of channel sizes [in, h1, ..., out]
    """

    def __init__(self, channels: List[int]) -> None:
        super().__init__()
        layers = []
        for i in range(len(channels) - 1):
            out_ch = channels[i + 1]
            layers += [
                nn.ConvTranspose2d(channels[i], out_ch, 4, stride=2, padding=1, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.Mish(inplace=True),
            ]
        layers.append(nn.Conv2d(channels[-1], 1, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InputModule(nn.Module):


    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.scene_encoder = UNetEncoder(in_channels=3, out_channels=cfg.scene_channels)
        self.seg_up = nn.Upsample(size=(cfg.map_h, cfg.map_w), mode="bilinear", align_corners=False)

    def forward(
        self,
        obs_traj: torch.Tensor,
        traj_heatmap: torch.Tensor,
        scene_img: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
  
        B, N, T, _ = obs_traj.shape
        H, W = self.cfg.map_h, self.cfg.map_w

        if scene_img is not None:
            seg = self.scene_encoder(scene_img)
            seg = self.seg_up(seg)
        else:
            seg = torch.zeros(B, self.cfg.scene_channels, H, W, device=obs_traj.device)
            seg[:, 0] = 1.0

        seg_exp = seg.unsqueeze(1).expand(-1, N, -1, -1, -1)
        I = torch.cat([seg_exp, traj_heatmap], dim=2)
        return I, obs_traj


class CliqueEncoder(nn.Module):


    def __init__(self, channels: List[int], out_ch: int = 64, map_h: int = 256, map_w: int = 256) -> None:
        super().__init__()
        self.map_h, self.map_w = map_h, map_w
        blocks = []
        in_ch = 2  # (dx, dy)
        for c in channels:
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, c, 3, padding=1, bias=False),
                    nn.GroupNorm(min(8, c), c),
                    nn.Mish(inplace=True),
                    nn.Conv2d(c, c, 1, bias=False),
                    nn.GroupNorm(min(8, c), c),
                    nn.Mish(inplace=True),
                )
            )
            in_ch = c
        self.blocks = nn.ModuleList(blocks)
        self.se = SEModule(channels[-1], reduction=8)
        self.proj = nn.Conv2d(channels[-1], out_ch, 1)

    def forward(self, clique_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clique_map: [B, N, 2, K, T] or [B*N, 2, K, T]
        Returns:
            feat: [B, N, out_ch, H, W]
        """
        if clique_map.dim() == 5:
            B, N = clique_map.shape[:2]
            x = clique_map.reshape(B * N, *clique_map.shape[2:])
        else:
            x = clique_map
            B, N = x.shape[0], 1

        for blk in self.blocks:
            res = x
            x = blk(x)
            if x.shape == res.shape:
                x = x + res

        x = self.se(x)
        x = self.proj(x)
        x = F.interpolate(x, size=(self.map_h, self.map_w), mode="bilinear", align_corners=False)
        return x.view(B, N, x.shape[1], self.map_h, self.map_w)
