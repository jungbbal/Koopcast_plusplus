
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsca_net import Config  # noqa: E402
from tsca_net.training import ablation_to_latex, run_ablation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leave-out", default="eth")
    parser.add_argument("--data-root", default="./data/eth_ucy")
    parser.add_argument(
        "--epochs", type=int, default=20, help="Paper uses full training per variant; lower this for a smoke run."
    )
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    cfg = Config(data_root=args.data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = run_ablation(cfg, device, leave_out=args.leave_out, epochs=args.epochs, synthetic=args.synthetic)
    print("\nAblation LaTeX table:\n")
    print(ablation_to_latex(df))


if __name__ == "__main__":
    main()
