

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..losses import TSCANetLoss, evaluate_batch
from ..model import TSCANet


class EarlyStopper:
   

    def __init__(self, patience: int = 15, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = float("inf")

    def step(self, value: float) -> bool:
        """Returns True if training should stop."""
        if value < self.best - self.min_delta:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def build_optimizers(model: TSCANet, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    
    goal_params = list(model.input_module.parameters()) + list(model.goal_module.parameters())
    pred_params = list(model.predictor.parameters())

    opt_goal = torch.optim.Adam(goal_params, lr=cfg.lr_goal, weight_decay=cfg.weight_decay)
    opt_pred = torch.optim.Adam(pred_params, lr=cfg.lr_pred, weight_decay=cfg.weight_decay)
    return opt_goal, opt_pred


def train_one_epoch(
    model: TSCANet,
    loader: DataLoader,
    loss_fn: TSCANetLoss,
    opt_goal: torch.optim.Optimizer,
    opt_pred: torch.optim.Optimizer,
    cfg,
    device: torch.device,
    writer: Optional[SummaryWriter] = None,
    global_step: int = 0,
) -> Tuple[float, int]:
    
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="  train", leave=False):
        obs = batch["obs"].to(device)
        gt = batch["gt"].to(device)
        heatmap = batch["heatmap"].to(device)
        valid_mask = batch["valid_mask"].to(device)

        opt_goal.zero_grad()
        opt_pred.zero_grad()

        outputs = model(obs, heatmap, valid_mask=valid_mask)
        loss, breakdown = loss_fn(outputs, gt, valid_mask=valid_mask, cfg=cfg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        opt_goal.step()
        opt_pred.step()

        total_loss += loss.item()
        global_step += 1

        if writer is not None:
            for k, v in breakdown.items():
                writer.add_scalar(f"train/{k}", v, global_step)

    return total_loss / max(len(loader), 1), global_step


@torch.no_grad()
def validate(model: TSCANet, loader: DataLoader, loss_fn: TSCANetLoss, cfg, device: torch.device) -> Dict[str, float]:
    
    model.eval()
    tot_loss = tot_ade = tot_fde = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="  val  ", leave=False):
        obs = batch["obs"].to(device)
        gt = batch["gt"].to(device)
        heatmap = batch["heatmap"].to(device)
        valid_mask = batch["valid_mask"].to(device)

        outputs = model(obs, heatmap, valid_mask=valid_mask)
        loss, _ = loss_fn(outputs, gt, valid_mask=valid_mask, cfg=cfg)
        ade, fde = evaluate_batch(outputs["predictions"], gt, valid_mask)

        tot_loss += loss.item()
        tot_ade += ade
        tot_fde += fde
        n_batches += 1

    return {
        "val_loss": tot_loss / max(n_batches, 1),
        "val_ADE": tot_ade / max(n_batches, 1),
        "val_FDE": tot_fde / max(n_batches, 1),
    }


def train(
    cfg, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, leave_out: str = "eth"
) -> TSCANet:
    
    print(f"\n{'=' * 60}")
    print(f"  Training TSCA-Net  |  leave-out: {leave_out}")
    print(f"  TSCA={cfg.use_tsca}  CPCP={cfg.use_cpcp}  AKGR={cfg.use_akgr}")
    print(f"{'=' * 60}\n")

    model = TSCANet(cfg).to(device)
    print(f"  Trainable parameters: {model.count_params():,}")

    loss_fn = TSCANetLoss(cfg)
    opt_goal, opt_pred = build_optimizers(model, cfg)
    sched_goal = ReduceLROnPlateau(opt_goal, patience=5, factor=0.5)
    sched_pred = ReduceLROnPlateau(opt_pred, patience=5, factor=0.5)
    stopper = EarlyStopper(cfg.patience)

    writer = SummaryWriter(cfg.log_dir)
    best_ade = float("inf")
    step = 0

    cfg.save(str(Path(cfg.ckpt_dir) / "config.json"))

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss, step = train_one_epoch(model, train_loader, loss_fn, opt_goal, opt_pred, cfg, device, writer, step)
        metrics = validate(model, val_loader, loss_fn, cfg, device)
        elapsed = time.time() - t0

        sched_goal.step(metrics["val_loss"])
        sched_pred.step(metrics["val_loss"])

        for k, v in metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)
        writer.add_scalar("train/epoch_loss", train_loss, epoch)

        print(
            f"  Epoch {epoch:3d}/{cfg.epochs}  loss={train_loss:.4f}  "
            f"ADE={metrics['val_ADE']:.4f}  FDE={metrics['val_FDE']:.4f}  ({elapsed:.1f}s)"
        )

        if metrics["val_ADE"] < best_ade:
            best_ade = metrics["val_ADE"]
            ckpt = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "opt_goal": opt_goal.state_dict(),
                "opt_pred": opt_pred.state_dict(),
                "val_ADE": best_ade,
                "val_FDE": metrics["val_FDE"],
                "config": cfg.to_dict(),
            }
            torch.save(ckpt, Path(cfg.ckpt_dir) / f"best_{leave_out}.pt")
            print(f"    Saved best checkpoint  ADE={best_ade:.4f}")

        if stopper.step(metrics["val_loss"]):
            print(f"  Early stopping at epoch {epoch}.")
            break

    writer.close()
    return model
