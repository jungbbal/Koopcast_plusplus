
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsca_net import Config  # noqa: E402
from tsca_net.data import build_eth_ucy_loaders  # noqa: E402
from tsca_net.training import train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a YAML config (e.g. configs/eth_ucy.yaml); CLI flags below override its values.",
    )
    parser.add_argument("--leave-out", default="eth", choices=["eth", "hotel", "univ", "zara1", "zara2"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--ckpt-dir", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-tsca", action="store_true")
    parser.add_argument("--no-cpcp", action="store_true")
    parser.add_argument("--no-akgr", action="store_true")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic multi-agent scenes instead of real data on disk "
        "(smoke test / CI; results are not meaningful for the paper claims).",
    )
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    if args.data_root is not None:
        cfg.data_root = args.data_root
    if args.ckpt_dir is not None:
        cfg.ckpt_dir = args.ckpt_dir
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    cfg.log_dir = f"{args.log_dir or cfg.log_dir}_{args.leave_out}"
    if args.no_tsca:
        cfg.use_tsca = False
    if args.no_cpcp:
        cfg.use_cpcp = False
    if args.no_akgr:
        cfg.use_akgr = False
    # Config.__post_init__ only creates directories at construction time;
    # re-create them in case --ckpt-dir/--log-dir overrode the yaml/default.
    Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader = build_eth_ucy_loaders(cfg, leave_out=args.leave_out, synthetic=args.synthetic)
    train(cfg, train_loader, val_loader, device, leave_out=args.leave_out)


if __name__ == "__main__":
    main()
