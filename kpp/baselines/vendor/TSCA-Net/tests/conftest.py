import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsca_net import Config  # noqa: E402


@pytest.fixture
def tiny_cfg(tmp_path) -> Config:
    """A small Config for fast tests (small heatmaps, few samples)."""
    return Config(
        map_h=32,
        map_w=32,
        num_samples=4,
        tsca_sample_K=4,
        obs_len=8,
        pred_len=12,
        batch_size=2,
        kan_hidden_dim=256,
        ckpt_dir=str(tmp_path / "ckpt"),
        log_dir=str(tmp_path / "runs"),
    )
