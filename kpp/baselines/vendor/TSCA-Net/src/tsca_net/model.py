

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .functional import compute_entropy, sample_from_heatmap
from .modules.common import InputModule
from .modules.goal import TSCAGoalModule
from .modules.predictor import CPCPAKGRPredictor


class TSCANet(nn.Module):
    

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_module = InputModule(cfg)
        self.goal_module = TSCAGoalModule(cfg)
        self.predictor = CPCPAKGRPredictor(cfg)

    def forward(
        self,
        obs_traj: torch.Tensor,
        heatmap: torch.Tensor,
        scene_img: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        social_radius: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        
        I, p_h = self.input_module(obs_traj, heatmap, scene_img)

        M1, M2, time_gates = self.goal_module(I, p_h)

        all_goals = sample_from_heatmap(M2, K=self.cfg.num_samples)
        entropy = compute_entropy(M2)

        preds = []
        for k in range(self.cfg.num_samples):
            traj = self.predictor(p_h, all_goals[:, :, k, :], valid_mask, entropy, social_radius=social_radius)
            preds.append(traj)

        predictions = torch.stack(preds, dim=2)

        return {
            "predictions": predictions,
            "M1": M1,
            "M2": M2,
            "time_gates": time_gates,
            "goal_entropy": entropy,
        }

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
