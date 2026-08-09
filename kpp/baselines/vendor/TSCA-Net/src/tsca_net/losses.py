

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .functional import positions_to_heatmap
from .metrics import compute_ade, compute_fde


class TSCANetLoss(nn.Module):
   

    def __init__(self, cfg) -> None:
        super().__init__()
        self.lambda1 = cfg.lambda1
        self.lambda2 = cfg.lambda2

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        gt_traj: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        gt_heatmap: Optional[torch.Tensor] = None,
        cfg=None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        preds = outputs["predictions"]  # [B, N, K, F, 2]
        M1 = outputs["M1"]
        M2 = outputs["M2"]
        B, N = preds.shape[:2]

        if valid_mask is None:
            valid_mask = preds.new_ones(B, N)
        agent_w = valid_mask.clamp(0, 1)  # [B, N]
        denom = agent_w.sum().clamp(min=1.0)

        if gt_heatmap is None and cfg is not None:
            gt_final = gt_traj[:, :, -1, :]
            gt_heatmap = positions_to_heatmap(gt_final, cfg.map_h, cfg.map_w, cfg.heatmap_sigma)

        if gt_heatmap is not None:
            bce1 = F.binary_cross_entropy(M1.clamp(1e-6, 1 - 1e-6), gt_heatmap.expand_as(M1), reduction="none")
            bce2 = F.binary_cross_entropy(M2.clamp(1e-6, 1 - 1e-6), gt_heatmap.expand_as(M2), reduction="none")
            per_agent1 = bce1.mean(dim=(2, 3, 4))  # [B, N]
            per_agent2 = bce2.mean(dim=(2, 3, 4))
            l_goal1 = (per_agent1 * agent_w).sum() / denom
            l_goal2 = (per_agent2 * agent_w).sum() / denom
        else:
            l_goal1 = l_goal2 = preds.new_zeros(())

        diff = preds - gt_traj.unsqueeze(2)  # [B, N, K, F, 2]
        dist = diff.norm(dim=-1)
        ade_k = dist.mean(dim=-1)  # [B, N, K]
        min_ade = ade_k.min(dim=-1).values  # [B, N]
        l_traj = (min_ade * agent_w).sum() / denom

        total = l_traj + self.lambda1 * l_goal1 + self.lambda2 * l_goal2

        return total, {
            "loss_total": total.item(),
            "loss_traj": l_traj.item(),
            "loss_goal1": l_goal1.item() if isinstance(l_goal1, torch.Tensor) else 0.0,
            "loss_goal2": l_goal2.item() if isinstance(l_goal2, torch.Tensor) else 0.0,
        }


@torch.no_grad()
def evaluate_batch(
    preds: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[float, float]:
    """Compute best-of-K ADE and FDE for one batch, masking padded agents."""
    ade = compute_ade(preds, gt, valid_mask).item()
    fde = compute_fde(preds, gt, valid_mask).item()
    return ade, fde
