

from __future__ import annotations

import torch

from tsca_net import TSCANet


def test_forward_pass_shapes(tiny_cfg):
    B, N = 2, 4
    T, F, K = tiny_cfg.obs_len, tiny_cfg.pred_len, tiny_cfg.num_samples
    H, W = tiny_cfg.map_h, tiny_cfg.map_w

    obs = torch.rand(B, N, T, 2)
    hm = torch.rand(B, N, 1, H, W)
    valid_mask = torch.ones(B, N)
    valid_mask[0, -1] = 0  # exercise padding path

    model = TSCANet(tiny_cfg)
    model.train()
    out = model(obs, hm, valid_mask=valid_mask)

    assert out["predictions"].shape == (B, N, K, F, 2)
    assert out["M1"].shape == (B, N, 1, H, W)
    assert out["M2"].shape == (B, N, 1, H, W)
    assert out["goal_entropy"].shape == (B, N, 1)
    assert out["time_gates"].shape == (B, N, tiny_cfg.tsca_sample_K, T)
    assert torch.isfinite(out["predictions"]).all()


def test_ablation_flags_disable_modules(tiny_cfg):
    from copy import deepcopy

    cfg = deepcopy(tiny_cfg)
    cfg.use_tsca = cfg.use_cpcp = cfg.use_akgr = False

    model = TSCANet(cfg)
    obs = torch.rand(1, 3, cfg.obs_len, 2)
    hm = torch.rand(1, 3, 1, cfg.map_h, cfg.map_w)
    with torch.no_grad():
        out = model(obs, hm)

    assert out["time_gates"] is None
    assert out["predictions"].shape == (1, 3, cfg.num_samples, cfg.pred_len, 2)


def test_akgr_disabled_falls_back_to_fixed_kan(tiny_cfg):
    from copy import deepcopy

    cfg = deepcopy(tiny_cfg)
    cfg.use_akgr = False

    model = TSCANet(cfg)
    assert hasattr(model.predictor, "kan_fixed")
    assert not hasattr(model.predictor, "kan")

    obs = torch.rand(1, 2, cfg.obs_len, 2)
    hm = torch.rand(1, 2, 1, cfg.map_h, cfg.map_w)
    with torch.no_grad():
        out = model(obs, hm)
    assert torch.isfinite(out["predictions"]).all()
