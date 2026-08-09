

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from ..data import ETH_UCY_DATASETS, build_eth_ucy_loaders
from ..losses import TSCANetLoss
from ..model import TSCANet
from ..training.trainer import validate


@torch.no_grad()
def benchmark_eth_ucy(cfg, device: torch.device, synthetic: bool = False) -> pd.DataFrame:
    
    results = []
    for split in ETH_UCY_DATASETS:
        print(f"\n  Evaluating on [{split}] ...")
        _, val_loader = build_eth_ucy_loaders(cfg, leave_out=split, synthetic=synthetic)

        model = TSCANet(cfg).to(device)
        ckpt_path = Path(cfg.ckpt_dir) / f"best_{split}.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["state_dict"])
            print(f"    Loaded checkpoint (epoch {ckpt['epoch']}, val_ADE={ckpt['val_ADE']:.4f})")
        else:
            print("    No checkpoint found -- using random weights (dev mode; results not meaningful).")

        model.eval()
        loss_fn = TSCANetLoss(cfg)
        metrics = validate(model, val_loader, loss_fn, cfg, device)
        results.append(
            {"Dataset": split, "Min20ADE": round(metrics["val_ADE"], 4), "Min20FDE": round(metrics["val_FDE"], 4)}
        )
        print(f"    ADE={metrics['val_ADE']:.4f}  FDE={metrics['val_FDE']:.4f}")

    df = pd.DataFrame(results)
    avg = {
        "Dataset": "Average",
        "Min20ADE": round(df["Min20ADE"].mean(), 4),
        "Min20FDE": round(df["Min20FDE"].mean(), 4),
    }
    df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)
    print("\n", df.to_string(index=False))
    return df


def results_to_latex(df: pd.DataFrame) -> str:
    """Export benchmark results to a LaTeX table for direct paper inclusion."""
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{TSCA-Net on ETH/UCY (Min20ADE / Min20FDE, metres)}",
        r"  \label{tab:eth_ucy}",
        r"  \begin{tabular}{lcc}",
        r"    \toprule",
        r"    Dataset & Min20ADE $\downarrow$ & Min20FDE $\downarrow$ \\",
        r"    \midrule",
    ]
    for _, row in df.iterrows():
        bold = "\\textbf" if row["Dataset"] == "Average" else ""
        lines.append(
            f"    {bold}{{{row['Dataset']}}} & {bold}{{{row['Min20ADE']}}} & {bold}{{{row['Min20FDE']}}} \\\\"
        )
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)
