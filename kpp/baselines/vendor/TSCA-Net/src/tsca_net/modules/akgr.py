
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def b_spline_basis(x: torch.Tensor, grid: torch.Tensor, order: int) -> torch.Tensor:
    
    x = x.unsqueeze(-1)
    grid = grid.view(*([1] * (x.dim() - 1)), -1)

    basis = ((x >= grid[..., :-1]) & (x < grid[..., 1:])).float()
    for k in range(1, order + 1):
        left = (x - grid[..., : -(k + 1)]) / (grid[..., k:-1] - grid[..., : -(k + 1)] + 1e-8)
        right = (grid[..., k + 1 :] - x) / (grid[..., k + 1 :] - grid[..., 1:-k] + 1e-8)
        basis = left * basis[..., :-1] + right * basis[..., 1:]
    return basis


class KANLayer(nn.Module):
   

    def __init__(self, in_dim: int, out_dim: int, grid: int = 5, order: int = 3) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid = grid
        self.order = order
        num_bases = grid + order

        self.coeff = nn.Parameter(torch.randn(out_dim, in_dim, num_bases) * 0.1)
        self.base_weight = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1)
        self.base_scale = nn.Parameter(torch.ones(out_dim, in_dim))

       
        h = 2.0 / grid
        knots = torch.arange(-order, grid + order + 1, dtype=torch.float32) * h - 1.0
        self.register_buffer("knots", knots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [*, in_dim]
        Returns:
            y: [*, out_dim]
        """
        x_c = x.clamp(self.knots[0], self.knots[-1])
        basis = b_spline_basis(x_c, self.knots, self.order)
        spline = torch.einsum("...ib,oib->...o", basis, self.coeff)
        base = F.silu(x) @ (self.base_weight * self.base_scale).T
        return spline + base


class AdaptiveKANController(nn.Module):
    

    def __init__(self, cfg) -> None:
        super().__init__()
        self.base = cfg.kan_base_grid
        self.lo = cfg.kan_min_grid
        self.hi = cfg.kan_max_grid

        self.unc_enc = nn.Sequential(nn.Linear(1, 16), nn.Mish(inplace=True), nn.Linear(16, 8))
        self.mot_enc = nn.Sequential(nn.Linear(1, 16), nn.Mish(inplace=True), nn.Linear(16, 8))
        self.predictor = nn.Sequential(nn.Linear(16, 16), nn.Mish(inplace=True), nn.Linear(16, 1), nn.Sigmoid())

    def forward(
        self,
        entropy: torch.Tensor,
        hidden: torch.Tensor,
        prev_hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        u_feat = self.unc_enc(entropy)
        if prev_hidden is not None:
            motion = (hidden - prev_hidden).norm(dim=-1, keepdim=True)
        else:
            motion = hidden.norm(dim=-1, keepdim=True)
        m_feat = self.mot_enc(motion)
        frac = self.predictor(torch.cat([u_feat, m_feat], -1))  # [B, N, 1] in (0, 1)
        g_soft = self.lo + (self.hi - self.lo) * frac  # continuous in [lo, hi]
        return g_soft


class AdaptiveKANLayer(nn.Module):
    

    def __init__(self, cfg) -> None:
        super().__init__()
        self.in_dim = cfg.kan_hidden_dim
        self.out_dim = cfg.kan_output_dim
        self.order = cfg.kan_spline_order
        self.lo = cfg.kan_min_grid
        self.hi = cfg.kan_max_grid

        self.layers = nn.ModuleDict(
            {str(g): KANLayer(self.in_dim, self.out_dim, g, self.order) for g in range(self.lo, self.hi + 1)}
        )
        self.controller = AdaptiveKANController(cfg)

    def forward(
        self,
        x: torch.Tensor,
        entropy: torch.Tensor,
        hidden: torch.Tensor,
        prev_hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns:
            out: [B, N, out_dim]
        """
        B, N, _ = x.shape
        g_soft = self.controller(entropy, hidden, prev_hidden)  # [B, N, 1], requires_grad
        g_flat = g_soft.reshape(B * N)

        # Discrete neighbour indices for dispatch (indexing must be integer,
        # so this step is legitimately non-differentiable -- only the mixing
        # weight below carries the gradient back to the controller).
        lo_idx = g_flat.floor().clamp(self.lo, self.hi - 1)
        hi_idx = lo_idx + 1
        frac = (g_flat - lo_idx.detach()).clamp(0.0, 1.0)  # differentiable weight

        lo_idx_int = lo_idx.detach().long()
        hi_idx_int = hi_idx.detach().long()

        x_flat = x.reshape(B * N, self.in_dim)
        out = x.new_zeros(B * N, self.out_dim)

        for g in range(self.lo, self.hi + 1):
            m_lo = lo_idx_int == g
            if m_lo.any():
                out[m_lo] = out[m_lo] + self.layers[str(g)](x_flat[m_lo]) * (1.0 - frac[m_lo]).unsqueeze(-1)
            m_hi = hi_idx_int == g
            if m_hi.any():
                out[m_hi] = out[m_hi] + self.layers[str(g)](x_flat[m_hi]) * frac[m_hi].unsqueeze(-1)

        return out.view(B, N, self.out_dim)
