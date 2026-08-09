

from __future__ import annotations

import torch


def positions_to_heatmap(
    positions: torch.Tensor,
    map_h: int,
    map_w: int,
    sigma: float = 4.0,
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
) -> torch.Tensor:
   
    B, N, _ = positions.shape
    device = positions.device

    px = ((positions[..., 0] - x_min) / (x_max - x_min) * (map_w - 1)).long().clamp(0, map_w - 1)
    py = ((positions[..., 1] - y_min) / (y_max - y_min) * (map_h - 1)).long().clamp(0, map_h - 1)

    gy = torch.arange(map_h, device=device).float().view(1, 1, 1, map_h, 1)
    gx = torch.arange(map_w, device=device).float().view(1, 1, 1, 1, map_w)
    cy = py.view(B, N, 1, 1, 1).float()
    cx = px.view(B, N, 1, 1, 1).float()

    heatmap = torch.exp(-((gy - cy) ** 2 + (gx - cx) ** 2) / (2 * sigma**2))
    heatmap = heatmap / (heatmap.amax(dim=(-2, -1), keepdim=True) + 1e-8)
    return heatmap  # [B, N, 1, H, W]


def sample_from_heatmap(heatmap: torch.Tensor, K: int = 20, temperature: float = 1.0) -> torch.Tensor:
    
    B, N, _, H, W = heatmap.shape
    flat = heatmap.view(B, N, H * W)

    gumbel = -torch.log(-torch.log(torch.rand(B, N, K, H * W, device=heatmap.device) + 1e-20) + 1e-20)
    logits = (flat.unsqueeze(2) + gumbel) / temperature
    indices = logits.argmax(dim=-1)

    row = (indices // W).float() / (H - 1)
    col = (indices % W).float() / (W - 1)
    return torch.stack([col, row], dim=-1)  # [B, N, K, 2]


def compute_entropy(heatmap: torch.Tensor) -> torch.Tensor:
    
    flat = heatmap.view(*heatmap.shape[:2], -1)
    probs = flat / (flat.sum(dim=-1, keepdim=True) + 1e-8)
    ent = -(probs * torch.log(probs + 1e-8)).sum(dim=-1, keepdim=True)
    return ent


def build_trajectory_heatmap(
    trajectories: torch.Tensor,
    map_h: int,
    map_w: int,
    sigma: float = 4.0,
) -> torch.Tensor:
    
    B, N, T, _ = trajectories.shape
    frames = [positions_to_heatmap(trajectories[:, :, t, :], map_h, map_w, sigma) for t in range(T)]
    stacked = torch.stack(frames, dim=2)  # [B, N, T, 1, H, W]
    return stacked.amax(dim=2)  # [B, N, 1, H, W]
