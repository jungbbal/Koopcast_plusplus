
from __future__ import annotations

from typing import Dict, List

import torch


def collate_scenes(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    
    n_max = max(item["obs"].shape[0] for item in batch)
    B = len(batch)

    T = batch[0]["obs"].shape[1]
    F = batch[0]["gt"].shape[1]
    H, W = batch[0]["heatmap"].shape[-2:]

    obs = torch.zeros(B, n_max, T, 2)
    gt = torch.zeros(B, n_max, F, 2)
    heatmap = torch.zeros(B, n_max, 1, H, W)
    valid_mask = torch.zeros(B, n_max)

    for b, item in enumerate(batch):
        n = item["obs"].shape[0]
        obs[b, :n] = item["obs"]
        gt[b, :n] = item["gt"]
        heatmap[b, :n] = item["heatmap"]
        valid_mask[b, :n] = item["valid_mask"]

    return {"obs": obs, "gt": gt, "heatmap": heatmap, "valid_mask": valid_mask}
