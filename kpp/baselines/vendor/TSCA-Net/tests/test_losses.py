

from __future__ import annotations

import pytest
import torch

from tsca_net import TSCANet, TSCANetLoss, compute_ade, compute_fde


def test_loss_is_finite_and_backprops(tiny_cfg):
    B, N = 2, 3
    obs = torch.rand(B, N, tiny_cfg.obs_len, 2)
    gt = torch.rand(B, N, tiny_cfg.pred_len, 2)
    hm = torch.rand(B, N, 1, tiny_cfg.map_h, tiny_cfg.map_w)

    model = TSCANet(tiny_cfg)
    loss_fn = TSCANetLoss(tiny_cfg)

    out = model(obs, hm)
    loss, breakdown = loss_fn(out, gt, cfg=tiny_cfg)

    assert torch.isfinite(loss)
    expected_total = breakdown["loss_traj"] + breakdown["loss_goal1"] + breakdown["loss_goal2"]
    assert breakdown["loss_total"] == pytest.approx(expected_total, abs=1e-5)

    loss.backward()
    any_grad = any(p.grad is not None and p.grad.norm() > 0 for p in model.parameters())
    assert any_grad


def test_padded_agents_do_not_affect_loss(tiny_cfg):
    
    B, N = 1, 3
    obs = torch.rand(B, N, tiny_cfg.obs_len, 2)
    gt = torch.rand(B, N, tiny_cfg.pred_len, 2)
    hm = torch.rand(B, N, 1, tiny_cfg.map_h, tiny_cfg.map_w)

    model = TSCANet(tiny_cfg)
    loss_fn = TSCANetLoss(tiny_cfg)

    valid_mask = torch.tensor([[1.0, 1.0, 0.0]])
    out = model(obs, hm, valid_mask=valid_mask)
    loss_masked, _ = loss_fn(out, gt, valid_mask=valid_mask, cfg=tiny_cfg)

    # Corrupt the padded agent's prediction wildly; loss should be identical.
    out["predictions"][:, -1] = 1e6
    loss_corrupted, _ = loss_fn(out, gt, valid_mask=valid_mask, cfg=tiny_cfg)

    assert torch.isclose(loss_masked, loss_corrupted, atol=1e-4)


def test_metrics_masking_matches_manual_computation():
    B, N, K, F = 1, 3, 2, 4
    pred = torch.rand(B, N, K, F, 2)
    gt = torch.rand(B, N, F, 2)
    valid_mask = torch.tensor([[1.0, 1.0, 0.0]])

    ade_masked = compute_ade(pred, gt, valid_mask)
    ade_manual = compute_ade(pred[:, :2], gt[:, :2])
    assert torch.isclose(ade_masked, ade_manual, atol=1e-5)

    fde_masked = compute_fde(pred, gt, valid_mask)
    fde_manual = compute_fde(pred[:, :2], gt[:, :2])
    assert torch.isclose(fde_masked, fde_manual, atol=1e-5)
