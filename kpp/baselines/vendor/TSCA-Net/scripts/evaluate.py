
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsca_net import Config  # noqa: E402
from tsca_net.eval import benchmark_eth_ucy, results_to_latex  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="./data/eth_ucy")
    parser.add_argument("--ckpt-dir", default="./checkpoints")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Evaluate against synthetic data instead of requiring real ETH/UCY files on disk.",
    )
    args = parser.parse_args()

    cfg = Config(data_root=args.data_root, ckpt_dir=args.ckpt_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = benchmark_eth_ucy(cfg, device, synthetic=args.synthetic)
    print("\nLaTeX table:\n")
    print(results_to_latex(df))


if __name__ == "__main__":
    main()
