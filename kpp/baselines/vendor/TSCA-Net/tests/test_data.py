

from __future__ import annotations

import numpy as np
import torch

from tsca_net.data import collate_scenes, windows_from_raw
from tsca_net.data.eth_ucy import ETHUCYDataset


def test_windows_from_raw_groups_multi_agent_scenes():
    # 3 agents, 10 frames each, all present in every frame -> one scene
    # of shape [3, seq_len, 2] should be extracted per valid window.
    rows = []
    for frame in range(10):
        for ped in range(3):
            rows.append([frame, ped, float(frame + ped), float(frame - ped)])
    raw = np.array(rows, dtype=np.float64)

    scenes = windows_from_raw(raw, obs_len=4, pred_len=4, skip=1, min_ped=1)
    assert len(scenes) == 10 - 8 + 1  # 3 windows for seq_len=8 over 10 frames
    for s in scenes:
        assert s.shape == (3, 8, 2)


def test_windows_from_raw_drops_incomplete_pedestrians():
    # Pedestrian 2 only appears in frames 0-4 (not the full window).
    rows = []
    for frame in range(8):
        rows.append([frame, 0, 0.0, 0.0])
        rows.append([frame, 1, 1.0, 1.0])
        if frame < 5:
            rows.append([frame, 2, 2.0, 2.0])
    raw = np.array(rows, dtype=np.float64)

    scenes = windows_from_raw(raw, obs_len=4, pred_len=4, skip=1, min_ped=1)
    assert len(scenes) == 1
    assert scenes[0].shape[0] == 2  # ped 2 excluded (incomplete)


def test_eth_ucy_synthetic_dataset_multi_agent(tiny_cfg):
    ds = ETHUCYDataset(
        tiny_cfg.data_root,
        "eth",
        tiny_cfg.obs_len,
        tiny_cfg.pred_len,
        tiny_cfg.map_h,
        tiny_cfg.map_w,
        tiny_cfg.heatmap_sigma,
        synthetic=True,
        n_synthetic=8,
    )
    assert len(ds) == 8
    item = ds[0]
    N = item["obs"].shape[0]
    assert N >= 2  # multi-agent, not the original N=1 bug
    assert item["obs"].shape == (N, tiny_cfg.obs_len, 2)
    assert item["gt"].shape == (N, tiny_cfg.pred_len, 2)
    assert item["heatmap"].shape == (N, 1, tiny_cfg.map_h, tiny_cfg.map_w)
    assert item["valid_mask"].shape == (N,)


def test_collate_pads_variable_agent_counts():
    batch = [
        {
            "obs": torch.rand(2, 8, 2),
            "gt": torch.rand(2, 12, 2),
            "heatmap": torch.rand(2, 1, 16, 16),
            "valid_mask": torch.ones(2),
        },
        {
            "obs": torch.rand(5, 8, 2),
            "gt": torch.rand(5, 12, 2),
            "heatmap": torch.rand(5, 1, 16, 16),
            "valid_mask": torch.ones(5),
        },
    ]
    out = collate_scenes(batch)

    assert out["obs"].shape == (2, 5, 8, 2)
    assert out["valid_mask"].shape == (2, 5)
    assert out["valid_mask"][0].sum() == 2  # first scene only had 2 real agents
    assert out["valid_mask"][1].sum() == 5
    assert torch.all(out["obs"][0, 2:] == 0)  # padded slots are zero-filled
