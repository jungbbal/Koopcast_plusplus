
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from ..functional import build_trajectory_heatmap
from .collate import collate_scenes
from .windowing import windows_from_raw

ETH_UCY_DATASETS = ["eth", "hotel", "univ", "zara1", "zara2"]


def _read_raw_txt(path: Path) -> np.ndarray:

    return np.loadtxt(str(path), dtype=np.float64)


class ETHUCYDataset(Dataset):
   

    def __init__(
        self,
        data_root: str,
        split: str,
        obs_len: int = 8,
        pred_len: int = 12,
        map_h: int = 256,
        map_w: int = 256,
        sigma: float = 4.0,
        augment: bool = False,
        synthetic: bool = False,
        skip: int = 1,
        min_ped: int = 1,
        max_agents: int = 32,
        n_synthetic: int = 64,
    ) -> None:
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.map_h, self.map_w = map_h, map_w
        self.sigma = sigma
        self.augment = augment
        self.max_agents = max_agents

        txt_path = Path(data_root) / f"{split}.txt"
        if txt_path.exists() and not synthetic:
            raw = _read_raw_txt(txt_path)
            self.scenes = windows_from_raw(raw, obs_len, pred_len, skip, min_ped)
            # World-space (metres) min/max for heatmap normalisation. A real
            # top-down scene image + homography (H.txt, part of the
            # official ETH/UCY release) should replace this approximation
            # -- see docs/REPRODUCIBILITY.md.
            all_xy = np.concatenate([s.reshape(-1, 2) for s in self.scenes], axis=0)
            self._xy_min = all_xy.min(axis=0)
            self._xy_max = all_xy.max(axis=0)
        else:
            self.scenes = self._make_synthetic(n_synthetic)
            self._xy_min = np.zeros(2, dtype=np.float32)
            self._xy_max = np.ones(2, dtype=np.float32)

        print(f"  [{split}] {len(self.scenes)} scenes  (obs={obs_len}, pred={pred_len})")

    @staticmethod
    def _make_synthetic(n_scenes: int = 64, min_n: int = 2, max_n: int = 6, total_len: int = 20) -> List[np.ndarray]:
        """Generate synthetic multi-agent scenes (random-walk trajectories)."""
        scenes = []
        for _ in range(n_scenes):
            n_ped = random.randint(min_n, max_n)
            starts = np.random.rand(n_ped, 1, 2).astype(np.float32) * 0.6 + 0.2
            steps = (np.random.randn(n_ped, total_len - 1, 2) * 0.02).astype(np.float32)
            traj = np.concatenate([starts, starts + np.cumsum(steps, axis=1)], axis=1)
            scenes.append(np.clip(traj, 0.0, 1.0))
        return scenes

    def _normalise(self, scene: np.ndarray) -> np.ndarray:
        span = (self._xy_max - self._xy_min)
        span[span == 0] = 1.0
        return (scene - self._xy_min) / (span + 1e-8)

    def _augment(self, traj: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            traj[..., 0] = 1.0 - traj[..., 0]
        if random.random() < 0.5:
            traj[..., 1] = 1.0 - traj[..., 1]
        scale = random.uniform(0.9, 1.1)
        centre = traj.mean(dim=(0, 1), keepdim=True)
        traj = (traj - centre) * scale + centre
        return traj.clamp(0.0, 1.0)

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        scene = self.scenes[idx]
        if scene.max() > 1.0 or scene.min() < 0.0:
            scene = self._normalise(scene)
        scene = torch.from_numpy(np.ascontiguousarray(scene)).float()
        scene = scene[: self.max_agents]

        if self.augment:
            scene = self._augment(scene)

        obs = scene[:, : self.obs_len]  # [N, T, 2]
        gt = scene[:, self.obs_len : self.obs_len + self.pred_len]  # [N, F, 2]
        N = obs.shape[0]

        heatmap = build_trajectory_heatmap(
            obs.unsqueeze(0), self.map_h, self.map_w, self.sigma
        ).squeeze(0)  # [N, 1, H, W]

        return {
            "obs": obs,  # [N, T, 2]
            "gt": gt,  # [N, F, 2]
            "heatmap": heatmap,  # [N, 1, H, W]
            "valid_mask": torch.ones(N),
        }


def build_eth_ucy_loaders(cfg, leave_out: str = "eth", synthetic: bool = True) -> Tuple[DataLoader, DataLoader]:
   
    train_splits = [d for d in ETH_UCY_DATASETS if d != leave_out]
    val_splits = [leave_out]

    def _loader(splits, augment):
        datasets = [
            ETHUCYDataset(
                cfg.data_root,
                s,
                cfg.obs_len,
                cfg.pred_len,
                cfg.map_h,
                cfg.map_w,
                cfg.heatmap_sigma,
                augment=augment,
                synthetic=synthetic,
                max_agents=cfg.max_neighbors,
            )
            for s in splits
        ]
        combined = ConcatDataset(datasets)
        return DataLoader(
            combined,
            batch_size=cfg.batch_size,
            shuffle=augment,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_scenes,
        )

    return _loader(train_splits, augment=True), _loader(val_splits, augment=False)
