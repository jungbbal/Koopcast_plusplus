
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .akgr import AdaptiveKANLayer, KANLayer
from .cpcp import CPCPModule, SocialGraphBuilder


class CPCPAKGRPredictor(nn.Module):


    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        self.history_proj = nn.Linear(2, cfg.lstm_hidden_dim)
        self.lstm_cell = nn.LSTMCell(cfg.lstm_hidden_dim, cfg.lstm_hidden_dim)

        self.cpcp = CPCPModule(cfg) if cfg.use_cpcp else None
        self.social_graph = SocialGraphBuilder(cfg)

        if cfg.use_akgr:
            self.kan = AdaptiveKANLayer(cfg)
        else:
            self.kan_fixed = KANLayer(cfg.kan_hidden_dim, cfg.kan_output_dim, cfg.kan_base_grid, cfg.kan_spline_order)

        motion_in = cfg.kan_output_dim + cfg.lstm_hidden_dim + 2
        self.motion_head = nn.Sequential(
            nn.Linear(motion_in, 64),
            nn.Mish(inplace=True),
            nn.Linear(64, 2),
        )

    def forward(
        self,
        p_h: torch.Tensor,
        goal: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
        goal_entropy: Optional[torch.Tensor],
        social_radius: Optional[float] = None,
    ) -> torch.Tensor:
        
        B, N, T, _ = p_h.shape
        F_ = self.cfg.pred_len

        if valid_mask is None:
            valid_mask = torch.ones(B, N, device=p_h.device)

        h_t = self.history_proj(p_h[:, :, -1, :]).reshape(B * N, -1)
        c_t = torch.zeros_like(h_t)

        if goal_entropy is None:
            goal_entropy = torch.zeros(B, N, 1, device=p_h.device)

        cur_pos = p_h[:, :, -1, :].clone()
        pred = []
        prev_h_2d = None

        for _ in range(F_):
            lstm_in = self.history_proj(cur_pos).reshape(B * N, -1)
            h_t, c_t = self.lstm_cell(lstm_in, (h_t, c_t))
            h_2d = h_t.view(B, N, -1)

            if self.cpcp is not None:
                adj, _ = self.social_graph(cur_pos, valid_mask, radius=social_radius)
                cpcp_feat = self.cpcp(cur_pos, p_h, valid_mask, adj)
                fused = h_2d + cpcp_feat
            else:
                fused = h_2d

            if self.cfg.use_akgr:
                kan_out = self.kan(fused, goal_entropy, h_2d, prev_h_2d)
            else:
                kan_out = self.kan_fixed(fused.reshape(B * N, -1)).view(B, N, -1)

            prev_h_2d = h_2d.clone()

            goal_dir = goal - cur_pos
            motion_in = torch.cat([kan_out, h_2d, goal_dir], dim=-1)
            delta = self.motion_head(motion_in)
            cur_pos = cur_pos + delta
            pred.append(cur_pos)

        return torch.stack(pred, dim=2)  # [B, N, F, 2]
