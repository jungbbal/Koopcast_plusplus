

from __future__ import annotations

from typing import Optional

import torch


def compute_ade(pred: torch.Tensor, gt: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    
    diff = pred - gt.unsqueeze(2)
    dist = diff.norm(dim=-1)  # [B, N, K, F]
    ade_k = dist.mean(dim=-1)  # [B, N, K]
    min_ade = ade_k.min(dim=-1).values  # [B, N]
    if valid_mask is None:
        return min_ade.mean()
    w = valid_mask.clamp(0, 1)
    return (min_ade * w).sum() / w.sum().clamp(min=1.0)


def compute_fde(pred: torch.Tensor, gt: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    
    diff = pred[..., -1, :] - gt[..., -1, :].unsqueeze(2)
    dist = diff.norm(dim=-1)  # [B, N, K]
    min_fde = dist.min(dim=-1).values  # [B, N]
    if valid_mask is None:
        return min_fde.mean()
    w = valid_mask.clamp(0, 1)
    return (min_fde * w).sum() / w.sum().clamp(min=1.0)
