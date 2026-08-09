"""Qualitative visualisation utilities (paper Fig. 3-4)."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_trajectory_predictions(
    obs: torch.Tensor,
    gt: torch.Tensor,
    preds: torch.Tensor,
    M2: torch.Tensor,
    title: str = "TSCA-Net Trajectory Predictions",
    n_show: int = 3,
    save_path: Optional[str] = None,
) -> None:
    """Publication-quality trajectory overlay: one column per pedestrian,
    goal heatmap (M2) with predicted trajectories overlaid.

    Args:
        obs:   [N, T, 2]
        gt:    [N, F, 2]
        preds: [N, K, F, 2]
        M2:    [N, 1, H, W]
    """
    N = min(n_show, obs.shape[0])
    fig, axes = plt.subplots(1, N, figsize=(5 * N, 5), dpi=150)
    if N == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        hm = M2[i, 0].cpu().numpy()
        ax.imshow(hm, cmap="hot", origin="lower", alpha=0.6, extent=[0, 1, 0, 1])

        ox, oy = obs[i, :, 0].cpu().numpy(), obs[i, :, 1].cpu().numpy()
        ax.plot(ox, oy, "o-", color="royalblue", lw=2, markersize=4, label="Observed", zorder=3)
        ax.plot(ox[-1], oy[-1], "s", color="royalblue", markersize=8, zorder=4)

        gx, gy = gt[i, :, 0].cpu().numpy(), gt[i, :, 1].cpu().numpy()
        ax.plot(gx, gy, "^-", color="green", lw=2, markersize=4, label="Ground truth", zorder=3)

        K = preds.shape[1]
        for k in range(K):
            px, py = preds[i, k, :, 0].cpu().numpy(), preds[i, k, :, 1].cpu().numpy()
            ax.plot(px, py, "-", color="salmon", lw=0.8, alpha=0.4)

        fde = ((preds[i, :, -1, :] - gt[i, -1, :]) ** 2).sum(-1).sqrt()
        best_k = fde.argmin().item()
        bx, by = preds[i, best_k, :, 0].cpu().numpy(), preds[i, best_k, :, 1].cpu().numpy()
        ax.plot(bx, by, "-", color="red", lw=2, label="Best pred.", zorder=3)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"Pedestrian {i + 1}", fontsize=11)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_xlabel("x (normalised)")
        ax.set_ylabel("y (normalised)")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_temporal_gates(
    time_gates: torch.Tensor,
    obs_len: int = 8,
    title: str = "TSCA Temporal Attention Gates",
    save_path: Optional[str] = None,
) -> None:
    """Visualise per-pedestrian, per-goal temporal gate weights.

    Args:
        time_gates: [N, K, T]
    """
    N, K, T = time_gates.shape
    n_plot = min(N, 3)
    fig, axes = plt.subplots(1, n_plot, figsize=(5 * n_plot, 4), dpi=150)
    if n_plot == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        gates_avg = time_gates[i].mean(dim=0).cpu().numpy()
        ax.bar(range(T), gates_avg, color="steelblue", edgecolor="white")
        ax.set_xlabel("Observed timestep t")
        ax.set_ylabel("Gate weight g_t")
        ax.set_title(f"Ped {i + 1} - avg gate across K goals")
        ax.set_xticks(range(T))
        ax.set_xticklabels([f"t{t}" for t in range(T)], fontsize=8)
        ax.set_ylim(0, 1)
        ax.axhline(1 / T, ls="--", color="grey", alpha=0.7, label="Uniform baseline")
        ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_goal_heatmaps(
    M1: torch.Tensor,
    M2: torch.Tensor,
    gt_traj: torch.Tensor,
    save_path: Optional[str] = None,
) -> None:
    """Compare first- and second-stage goal heatmaps side-by-side.

    Args:
        M1, M2:  [N, 1, H, W]
        gt_traj: [N, F, 2]
    """
    N = min(M1.shape[0], 3)
    fig, axes = plt.subplots(N, 2, figsize=(8, 3 * N), dpi=150)
    if N == 1:
        axes = axes[np.newaxis]

    for i in range(N):
        for j, (M, label) in enumerate([(M1, "First Goal (M1)"), (M2, "Second Goal / TSCA (M2)")]):
            ax = axes[i, j]
            hm = M[i, 0].cpu().numpy()
            ax.imshow(hm, cmap="viridis", origin="lower", extent=[0, 1, 0, 1])
            gx, gy = gt_traj[i, -1, 0].item(), gt_traj[i, -1, 1].item()
            ax.scatter(gx, gy, c="red", s=80, marker="*", zorder=5, label="GT goal")
            ax.set_title(f"Ped {i + 1}: {label}", fontsize=9)
            ax.legend(fontsize=7)

    fig.suptitle("Goal Distribution Comparison", fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
