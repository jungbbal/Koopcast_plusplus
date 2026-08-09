

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .common import CliqueEncoder


class TemporalEmbedding(nn.Module):


    def __init__(self, max_len: int, embed_dim: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(max_len, embed_dim)
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:

        return self.emb(timesteps)


class TSCAModule(nn.Module):


    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        embed = cfg.temporal_embed_dim

        self.temporal_emb = TemporalEmbedding(cfg.obs_len, embed)

        
        self.gate_net = nn.Sequential(
            nn.Linear(embed + 1, 32, bias=False),
            nn.LayerNorm(32),
            nn.Mish(inplace=True),
            nn.Linear(32, 16),
            nn.Mish(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        
        self.spatial_attn = nn.Linear(2, 2, bias=True)
        nn.init.eye_(self.spatial_attn.weight)

    def forward(
        self,
        goal_pts: torch.Tensor,
        history: torch.Tensor,
        t_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        B, N, K, _ = goal_pts.shape
        T = history.shape[2]

        if t_idx is None:
            t_idx = torch.arange(T, device=goal_pts.device)

        t_emb = self.temporal_emb(t_idx)  # [T, embed_dim]
        t_emb = t_emb.view(1, 1, 1, T, -1).expand(B, N, K, -1, -1)

        g = goal_pts.unsqueeze(3)  # [B, N, K, 1, 2]
        h = history.unsqueeze(2)  # [B, N, 1, T, 2]
        delta = g - h  # [B, N, K, T, 2]

        a_delta = self.spatial_attn(delta)  # [B, N, K, T, 2]

        dist = delta.norm(dim=-1, keepdim=True)  # [B, N, K, T, 1]
        gate_in = torch.cat([t_emb, dist], dim=-1)
        gates = self.gate_net(gate_in)  # [B, N, K, T, 1]

        tsca_mat = a_delta * gates
        return tsca_mat, gates.squeeze(-1)


class TSCACliqueEncoder(nn.Module):
    

    def __init__(self, cfg) -> None:
        super().__init__()
        self.encoder = CliqueEncoder(channels=cfg.clique_enc_ch, out_ch=64, map_h=cfg.map_h, map_w=cfg.map_w)

    def forward(self, tsca_matrix: torch.Tensor) -> torch.Tensor:
        
        x = tsca_matrix.permute(0, 1, 4, 2, 3)  # [B, N, 2, K, T]
        return self.encoder(x)
