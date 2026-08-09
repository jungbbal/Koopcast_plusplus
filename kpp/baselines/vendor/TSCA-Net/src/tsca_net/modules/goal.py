"""TSCA-enhanced two-tiered goal module (paper Sec. II-C)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..functional import sample_from_heatmap
from .common import CliqueEncoder, CNNDecoder, CNNEncoder
from .tsca import TSCACliqueEncoder, TSCAModule


class FirstGoalNetwork(nn.Module):
    """CNN encoder-decoder for the initial goal heatmap (paper Eq. 2)."""

    def __init__(self, cfg) -> None:
        super().__init__()
        in_ch = cfg.scene_channels + cfg.heatmap_channels
        
        enc_ch = [in_ch] + cfg.goal_enc_ch[1:]
        self.encoder = CNNEncoder(enc_ch)
        self.decoder = CNNDecoder(cfg.goal_dec_ch)

    def forward(self, I: torch.Tensor) -> torch.Tensor:
        """
        Args:
            I: [B*N, fusion_ch, H, W]
        Returns:
            M1: [B*N, 1, H, W]  probability heatmap
        """
        return self.decoder(self.encoder(I))


class SecondGoalNetwork(nn.Module):
    """TSCA-conditioned second-tier goal network (paper Eq. 7)."""

    def __init__(self, cfg) -> None:
        super().__init__()
        in_ch = cfg.scene_channels + cfg.heatmap_channels + 64
        enc_ch = [in_ch, 32, 64, 64, 64]
        self.encoder = CNNEncoder(enc_ch)
        self.decoder = CNNDecoder([64, 64, 64, 32, 32])

    def forward(self, I: torch.Tensor, I_C: torch.Tensor) -> torch.Tensor:
        """
        Args:
            I:   [B*N, fusion_ch, H, W]
            I_C: [B*N, 64, H, W]        TSCA clique heatmap
        Returns:
            M2: [B*N, 1, H, W]
        """
        x = torch.cat([I, I_C], dim=1)
        return self.decoder(self.encoder(x))


class TSCAGoalModule(nn.Module):


    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.first_goal = FirstGoalNetwork(cfg)
        self.tsca = TSCAModule(cfg) if cfg.use_tsca else None
        self.tsca_enc = TSCACliqueEncoder(cfg)
        self.second_goal = SecondGoalNetwork(cfg)

        if not cfg.use_tsca:
            self.clique_enc_fallback = CliqueEncoder(cfg.clique_enc_ch, out_ch=64, map_h=cfg.map_h, map_w=cfg.map_w)

    def _build_static_clique(self, goal_pts: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        
        g = goal_pts.unsqueeze(3)
        h = history.unsqueeze(2)
        return g - h

    def forward(self, I: torch.Tensor, p_h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        
        B, N, fc, H, W = I.shape
        I_flat = I.view(B * N, fc, H, W)
        M1_flat = self.first_goal(I_flat)
        M1 = M1_flat.view(B, N, 1, H, W)

        goal_pts = sample_from_heatmap(M1, K=self.cfg.tsca_sample_K)

        time_gates = None
        if self.cfg.use_tsca and self.tsca is not None:
            tsca_mat, time_gates = self.tsca(goal_pts, p_h)
            I_C = self.tsca_enc(tsca_mat)
        else:
            static_mat = self._build_static_clique(goal_pts, p_h)
            perm = static_mat.permute(0, 1, 4, 2, 3)
            I_C = self.clique_enc_fallback(perm)

        I_C_flat = I_C.view(B * N, 64, H, W)
        M2_flat = self.second_goal(I_flat, I_C_flat)
        M2 = M2_flat.view(B, N, 1, H, W)

        return M1, M2, time_gates
