

from __future__ import annotations

from copy import deepcopy
from typing import Dict

import pandas as pd
import torch

from ..data import build_eth_ucy_loaders
from ..losses import TSCANetLoss
from ..model import TSCANet
from .trainer import build_optimizers, train_one_epoch, validate

ABLATION_VARIANTS: Dict[str, Dict[str, bool]] = {
    "Full TSCA-Net": {"use_tsca": True, "use_cpcp": True, "use_akgr": True},
    "w/o TSCA": {"use_tsca": False, "use_cpcp": True, "use_akgr": True},
    "w/o CPCP": {"use_tsca": True, "use_cpcp": False, "use_akgr": True},
    "w/o AKGR (fixed KAN)": {"use_tsca": True, "use_cpcp": True, "use_akgr": False},
    "MCK-Net baseline": {"use_tsca": False, "use_cpcp": False, "use_akgr": False},
}


def run_ablation(
    base_cfg,
    device: torch.device,
    leave_out: str = "eth",
    epochs: int = 5,
    synthetic: bool = True,
) -> pd.DataFrame:
    
    rows = []
    for name, flags in ABLATION_VARIANTS.items():
        print(f"\n{'-' * 50}\n  Variant: {name}\n{'-' * 50}")

        cfg_v = deepcopy(base_cfg)
        for k, v in flags.items():
            setattr(cfg_v, k, v)
        cfg_v.epochs = epochs
        cfg_v.log_dir = f"./runs/ablation_{name.replace(' ', '_').replace('/', '-')}"

        train_loader, val_loader = build_eth_ucy_loaders(cfg_v, leave_out, synthetic=synthetic)
        model = TSCANet(cfg_v).to(device)
        loss_fn = TSCANetLoss(cfg_v)
        opt_g, opt_p = build_optimizers(model, cfg_v)

        for _ in range(epochs):
            train_one_epoch(model, train_loader, loss_fn, opt_g, opt_p, cfg_v, device)

        metrics = validate(model, val_loader, loss_fn, cfg_v, device)
        rows.append(
            {
                "Variant": name,
                "TSCA": "yes" if flags["use_tsca"] else "no",
                "CPCP": "yes" if flags["use_cpcp"] else "no",
                "AKGR": "yes" if flags["use_akgr"] else "no",
                "Params(M)": round(model.count_params() / 1e6, 2),
                "Min20ADE": round(metrics["val_ADE"], 4),
                "Min20FDE": round(metrics["val_FDE"], 4),
            }
        )

    df = pd.DataFrame(rows)
    print("\n", df.to_string(index=False))
    return df


def ablation_to_latex(df: pd.DataFrame) -> str:
    
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Ablation study on ETH-UCY}",
        r"  \label{tab:ablation}",
        r"  \begin{tabular}{lcccccc}",
        r"    \toprule",
        r"    Variant & TSCA & CPCP & AKGR & Params(M) & ADE$\downarrow$ & FDE$\downarrow$ \\",
        r"    \midrule",
    ]
    for _, row in df.iterrows():
        bold = row["Variant"] == "Full TSCA-Net"

        def b(s):
            return f"\\textbf{{{s}}}" if bold else str(s)

        lines.append(
            f"    {b(row['Variant'])} & {b(row['TSCA'])} & {b(row['CPCP'])} & "
            f"{b(row['AKGR'])} & {b(row['Params(M)'])} & {b(row['Min20ADE'])} & {b(row['Min20FDE'])} \\\\"
        )
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)
