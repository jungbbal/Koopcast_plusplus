

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Config:
    

    #  Data 
    obs_len: int = 8  # T  - 3.2 s @ 2.5 fps
    pred_len: int = 12  # F  - 4.8 s
    num_samples: int = 20  # K  - multimodal samples
    map_h: int = 256  # spatial heatmap height
    map_w: int = 256  # spatial heatmap width
    max_neighbors: int = 16  # cap on neighbours encoded per pedestrian per scene

    #  Scene / Input 
    scene_channels: int = 2  # ETH/UCY: walkable + obstacle
    heatmap_channels: int = 1  # trajectory heatmap
    heatmap_sigma: float = 4.0  # Gaussian kernel std for heatmap rendering

    #  Goal Network 
    goal_enc_ch: List[int] = field(default_factory=lambda: [32, 32, 64, 64, 64])
    goal_dec_ch: List[int] = field(default_factory=lambda: [64, 64, 64, 32, 32])

    #  TSCA (Novel #1) å
    clique_enc_ch: List[int] = field(default_factory=lambda: [64, 32, 32, 96])
    temporal_embed_dim: int = 16
    tsca_sample_K: int = 20  # candidate goals fed to TSCA

    #  CPCP (Novel #2) å
    social_radius: float = 3.0  # metres - neighbourhood radius (ETH/UCY)
    social_radius_px: float = 50.0  # pixels - neighbourhood radius (SDD)
    social_embed_dim: int = 64

    #  AKGR / KAN (Novel #3) å
    kan_base_grid: int = 5
    kan_spline_order: int = 3
    kan_hidden_dim: int = 256
    kan_output_dim: int = 16
    kan_min_grid: int = 3
    kan_max_grid: int = 10

    #  LSTM å
    lstm_hidden_dim: int = 256

    #  Training å
    lr_goal: float = 1e-4
    lr_pred: float = 3e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    epochs: int = 100
    patience: int = 15
    batch_size: int = 8
    num_workers: int = 0

    #  Loss weights (paper Eq. 16) å
    lambda1: float = 1.0
    lambda2: float = 1.0

    #  Ablation flags å
    use_tsca: bool = True
    use_cpcp: bool = True
    use_akgr: bool = True

    #  Paths å
    data_root: str = "./data"
    ckpt_dir: str = "./checkpoints"
    log_dir: str = "./runs/tsca_net"

    def __post_init__(self) -> None:
        Path(self.ckpt_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

    @property
    def fusion_channels(self) -> int:
        return self.scene_channels + self.heatmap_channels

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load from one of the reference configs under ``configs/`` (e.g.
        ``configs/eth_ucy.yaml``, ``configs/sdd.yaml``)."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
