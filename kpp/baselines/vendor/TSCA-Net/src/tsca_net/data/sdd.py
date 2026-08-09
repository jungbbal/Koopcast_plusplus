
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


def _read_sdd_annotations(path: Path, label: str = "Pedestrian") -> np.ndarray:
  
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 10:
                continue
            track_id, xmin, ymin, xmax, ymax, frame = (float(p) for p in parts[:6])
            lost = int(parts[6])
            lbl = " ".join(parts[9:]).strip('"')
            if lost or lbl != label:
                continue
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            rows.append([frame, track_id, cx, cy])
    return np.array(rows, dtype=np.float64) if rows else np.zeros((0, 4))


class SDDDataset(Dataset):
  

    def __init__(
        self,
        data_root: str,
        scene: str,
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

        ann_path = Path(data_root) / scene / "annotations.txt"
        if ann_path.exists() and not synthetic:
            raw = _read_sdd_annotations(ann_path)
            self.scenes = windows_from_raw(raw, obs_len, pred_len, skip, min_ped) if len(raw) else []
            if self.scenes:
                all_xy = np.concatenate([s.reshape(-1, 2) for s in self.scenes], axis=0)
                self._xy_min = all_xy.min(axis=0)
                self._xy_max = all_xy.max(axis=0)
            else:
                self._xy_min, self._xy_max = np.zeros(2), np.ones(2)
        else:
            self.scenes = self._make_synthetic(n_synthetic)
            self._xy_min = np.zeros(2, dtype=np.float32)
            self._xy_max = np.ones(2, dtype=np.float32)

        print(f"  [SDD:{scene}] {len(self.scenes)} scenes  (obs={obs_len}, pred={pred_len})")

    @staticmethod
    def _make_synthetic(n_scenes: int = 64, min_n: int = 2, max_n: int = 8, total_len: int = 20) -> List[np.ndarray]:
        scenes = []
        for _ in range(n_scenes):
            n_ped = random.randint(min_n, max_n)
            starts = np.random.rand(n_ped, 1, 2).astype(np.float32) * 0.6 + 0.2
            steps = (np.random.randn(n_ped, total_len - 1, 2) * 0.015).astype(np.float32)
            traj = np.concatenate([starts, starts + np.cumsum(steps, axis=1)], axis=1)
            scenes.append(np.clip(traj, 0.0, 1.0))
        return scenes

    def _normalise(self, scene: np.ndarray) -> np.ndarray:
        span = self._xy_max - self._xy_min
        span[span == 0] = 1.0
        return (scene - self._xy_min) / (span + 1e-8)

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        scene = self.scenes[idx]
        if scene.max() > 1.0 or scene.min() < 0.0:
            scene = self._normalise(scene)
        scene = torch.from_numpy(np.ascontiguousarray(scene)).float()
        scene = scene[: self.max_agents]

        obs = scene[:, : self.obs_len]
        gt = scene[:, self.obs_len : self.obs_len + self.pred_len]
        N = obs.shape[0]

        heatmap = build_trajectory_heatmap(obs.unsqueeze(0), self.map_h, self.map_w, self.sigma).squeeze(0)

        return {"obs": obs, "gt": gt, "heatmap": heatmap, "valid_mask": torch.ones(N)}


def build_sdd_loaders(cfg, scenes: Tuple[str, ...], val_scene: str, synthetic: bool = True) -> Tuple[DataLoader, DataLoader]:
  
    train_scenes = [s for s in scenes if s != val_scene]

    def _loader(names, augment):
        datasets = [
            SDDDataset(
                cfg.data_root, s, cfg.obs_len, cfg.pred_len, cfg.map_h, cfg.map_w,
                cfg.heatmap_sigma, augment=augment, synthetic=synthetic, max_agents=cfg.max_neighbors,
            )
            for s in names
        ]
        combined = ConcatDataset(datasets)
        return DataLoader(
            combined, batch_size=cfg.batch_size, shuffle=augment,
            num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
            collate_fn=collate_scenes,
        )

    return _loader(train_scenes, True), _loader([val_scene], False)
