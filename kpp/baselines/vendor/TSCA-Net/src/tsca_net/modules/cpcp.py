

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class SocialGraphBuilder(nn.Module):


    def __init__(self, cfg) -> None:
        super().__init__()
        self.radius = cfg.social_radius

    def forward(
        self,
        positions: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        radius: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        r = radius or self.radius
        diff = positions.unsqueeze(2) - positions.unsqueeze(1)  # [B,N,N,2]
        dist = diff.norm(dim=-1)
        mask = (dist < r) & (dist > 0)
        if valid_mask is not None:
            pair_valid = valid_mask.unsqueeze(1).bool() & valid_mask.unsqueeze(2).bool()
            mask = mask & pair_valid
        adj = torch.exp(-dist) * mask.float()
        adj = adj / (adj.sum(dim=-1, keepdim=True) + 1e-8)
        return adj, mask


class CPCPModule(nn.Module):


    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        self.nbr_enc = nn.LSTM(
            input_size=2,
            hidden_size=cfg.social_embed_dim,
            num_layers=1,
            batch_first=True,
        )

        self.potential_mlp = nn.Sequential(
            nn.Linear(cfg.social_embed_dim + 2, 128),
            nn.Mish(inplace=True),
            nn.Linear(128, cfg.lstm_hidden_dim),
        )

    def forward(
        self,
        pred_pos: torch.Tensor,
        all_histories: torch.Tensor,
        valid_mask: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
       
        B, N, T, _ = all_histories.shape

        # Encode every agent's history exactly once, reused as both the
        # "self" query context and the neighbour context for all other
        # agents (O(N), not O(N x neighbours)).
        hist_flat = all_histories.reshape(B * N, T, 2)
        _, (h_n, _) = self.nbr_enc(hist_flat)
        enc = h_n.squeeze(0).view(B, N, self.cfg.social_embed_dim)  # [B, N, social_dim]

        weights = adj * valid_mask.unsqueeze(1)  # zero out padded neighbours
        social_ctx = torch.einsum("bij,bjd->bid", weights, enc)  # [B, N, social_dim]

        self_diff = pred_pos.unsqueeze(2) - all_histories  # [B, N, T, 2]
        D = self_diff.norm(dim=-1).sum(dim=-1, keepdim=True)  # [B, N, 1]
        U_self = torch.exp(-D)

        combined = social_ctx + U_self * social_ctx
        cpcp_in = torch.cat([combined, pred_pos], dim=-1)
        return self.potential_mlp(cpcp_in)
