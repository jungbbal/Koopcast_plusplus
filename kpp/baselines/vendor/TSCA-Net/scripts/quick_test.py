
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsca_net import Config, TSCANet, TSCANetLoss, compute_ade, compute_fde  # noqa: E402


def run_quick_test(cfg: Config, device: torch.device) -> None:
    print("=" * 60)
    print("  TSCA-Net Quick Test")
    print("=" * 60)

    B, N, T, F, K = 2, 4, cfg.obs_len, cfg.pred_len, cfg.num_samples
    H, W = cfg.map_h, cfg.map_w

    obs = torch.rand(B, N, T, 2, device=device)
    gt = torch.rand(B, N, F, 2, device=device)
    hm = torch.rand(B, N, 1, H, W, device=device)
    valid_mask = torch.ones(B, N, device=device)
    valid_mask[0, -1] = 0  # exercise the padding path

    model = TSCANet(cfg).to(device)
    loss_fn = TSCANetLoss(cfg)

    model.train()
    outputs = model(obs, hm, valid_mask=valid_mask)

    print("\n  Shape checks:")
    expected_shapes = {
        "predictions": (B, N, K, F, 2),
        "M1": (B, N, 1, H, W),
        "M2": (B, N, 1, H, W),
        "goal_entropy": (B, N, 1),
    }
    all_ok = True
    for key, exp in expected_shapes.items():
        got = tuple(outputs[key].shape)
        ok = got == exp
        all_ok &= ok
        print(f"    {key:20s} {str(got):25s} {'OK' if ok else f'MISMATCH expected {exp}'}")
    assert all_ok, "Shape mismatch detected!"

    print("\n  Loss & gradient check:")
    loss, breakdown = loss_fn(outputs, gt, valid_mask=valid_mask, cfg=cfg)
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    print(f"    total_loss = {loss.item():.6f}  (finite)")
    for k, v in breakdown.items():
        print(f"    {k:15s} = {v:.6f}")

    loss.backward()

    grad_checks = {
        "TSCA.temporal_emb.emb.weight": model.goal_module.tsca.temporal_emb.emb.weight,
        "TSCA.gate_net[0].weight": model.goal_module.tsca.gate_net[0].weight,
        "TSCA.spatial_attn.weight": model.goal_module.tsca.spatial_attn.weight,
        "CPCP.potential_mlp[0].weight": model.predictor.cpcp.potential_mlp[0].weight,
        "AKGR.controller.predictor[0].weight": model.predictor.kan.controller.predictor[0].weight,
        "AKGR.controller.unc_enc[0].weight": model.predictor.kan.controller.unc_enc[0].weight,
    }
    print("\n  Gradient flow through novel modules:")
    all_grad_ok = True
    for name, param in grad_checks.items():
        ok = param.grad is not None and param.grad.norm().item() > 0
        all_grad_ok &= ok
        g_norm = param.grad.norm().item() if param.grad is not None else 0.0
        print(f"    {name:42s}  grad_norm={g_norm:.4e}  {'OK' if ok else 'ZERO / MISSING'}")
    assert all_grad_ok, (
        "A novel-module parameter received zero gradient. If this is "
        "AKGR.controller.*, the soft grid-interpolation fix in "
        "src/tsca_net/modules/akgr.py has regressed -- see docs/KNOWN_ISSUES.md #1."
    )

    print("\n  ADE/FDE sanity (random model, masked by valid_mask):")
    with torch.no_grad():
        ade = compute_ade(outputs["predictions"].detach(), gt, valid_mask)
        fde = compute_fde(outputs["predictions"].detach(), gt, valid_mask)
    print(f"    ADE = {ade.item():.4f}")
    print(f"    FDE = {fde.item():.4f}")
    assert ade > 0 and fde > 0, "ADE/FDE should be positive for a random model"

    print("\n  Ablation flag check (MCK-Net baseline: all novel modules off):")
    cfg_abl = deepcopy(cfg)
    cfg_abl.use_tsca = cfg_abl.use_cpcp = cfg_abl.use_akgr = False
    model_abl = TSCANet(cfg_abl).to(device)
    with torch.no_grad():
        out_abl = model_abl(obs, hm, valid_mask=valid_mask)
    print(f"    predictions shape: {tuple(out_abl['predictions'].shape)}  OK")
    print(f"    time_gates: {out_abl['time_gates']}  (should be None)  OK")

    print("\n  Model parameters:")
    print(f"    Full TSCA-Net : {model.count_params():>12,}")
    print(f"    MCK-Net base  : {model_abl.count_params():>12,}")
    print(f"    Delta (+novel): {model.count_params() - model_abl.count_params():>+12,}")

    print("\n" + "=" * 60)
    print("  All sanity checks PASSED")
    print("=" * 60)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_quick_test(Config(), device)
