
from __future__ import annotations

import torch

from tsca_net import TSCANet, TSCANetLoss


def test_akgr_controller_receives_gradient(tiny_cfg):
    B, N = 2, 3
    obs = torch.rand(B, N, tiny_cfg.obs_len, 2)
    gt = torch.rand(B, N, tiny_cfg.pred_len, 2)
    hm = torch.rand(B, N, 1, tiny_cfg.map_h, tiny_cfg.map_w)

    model = TSCANet(tiny_cfg)
    loss_fn = TSCANetLoss(tiny_cfg)

    model.train()
    out = model(obs, hm)
    loss, _ = loss_fn(out, gt, cfg=tiny_cfg)
    loss.backward()

    controller = model.predictor.kan.controller
    checked = 0
    for name, param in controller.named_parameters():
        assert param.grad is not None, f"controller.{name} has no gradient at all"
        grad_norm = param.grad.norm().item()
        assert grad_norm > 0, f"controller.{name} received a zero gradient (regression of KNOWN_ISSUES #1)"
        checked += 1
    assert checked > 0, "no controller parameters were found to check"


def test_akgr_grid_soft_is_continuous_and_bounded(tiny_cfg):
    model = TSCANet(tiny_cfg)
    controller = model.predictor.kan.controller

    entropy = torch.rand(2, 3, 1) * 5
    hidden = torch.randn(2, 3, tiny_cfg.lstm_hidden_dim)

    g_soft = controller(entropy, hidden)
    assert g_soft.requires_grad
    assert (g_soft >= tiny_cfg.kan_min_grid).all()
    assert (g_soft <= tiny_cfg.kan_max_grid).all()
    
    assert g_soft.unique().numel() > 1
