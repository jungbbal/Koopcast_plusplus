# TSCA-Net

**Temporal-Spatial Clique Attention for Interpretable Multimodal Pedestrian Trajectory Prediction**

Md Mustafizur Rahman, Guangchao Yang*, A F M Abdun Noor, Md Imam Ahasan, Md Mahfuzur Rahman, Md Ariful Islam
Chongqing University · Daffodil International University
*Corresponding author: gchao_yang@cqu.edu.cn*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](pyproject.toml)
[![Tests](https://github.com/imamahasane/TSCA-Net/actions/workflows/ci.yml/badge.svg)](https://github.com/imamahasane/TSCA-Net/actions/workflows/ci.yml)

TSCA-Net predicts multimodal pedestrian trajectories with a hierarchical
predict-then-refine architecture built on three complementary modules that
extend the MCK-Net backbone:

- **TSCA** (Temporal-Spatial Clique Attention) — learnable time-gated
  weighting of goal-history interactions, replacing static
  equal-weighted clique matrices.
- **CPCP** (Cross-Pedestrian Clique Potential) — a dynamic, asymmetric
  social graph over neighbouring agents, replacing symmetric graph
  attention.
- **AKGR** (Adaptive KAN Grid Refinement) — a Kolmogorov-Arnold Network
  decoder whose B-spline grid resolution adapts per-agent to goal-entropy
  and motion complexity.

> **Before you read the numbers below:** this release fixes several bugs
> found in the code originally submitted with the paper — including three
> that made the original notebook crash on its own first forward pass, and
> two that silently prevented two of the three novel modules (AKGR, CPCP)
> from receiving a real training signal. Every fix is documented with root
> cause and verification evidence in **[`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md)**.
> The architecture and reported paper numbers below are as submitted;
> reproducing them exactly requires retraining on real data with the fixed
> code (see [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)).

## Results (as reported in the paper)

Best-of-20, leave-one-out protocol. ETH/UCY in metres, SDD in pixels.

| Method | ETH | HOTEL | UNIV | ZARA1 | ZARA2 | AVG |
|---|---|---|---|---|---|---|
| MCK-Net (baseline) | 0.23/0.27 | 0.08/0.13 | 0.19/0.33 | 0.14/0.24 | 0.11/0.20 | 0.15/0.23 |
| **TSCA-Net (ours)** | **0.21/0.26** | 0.08/0.14 | **0.17/0.29** | **0.12/0.21** | **0.10/0.18** | **0.13/0.20** |

| Dataset | MCK-Net | TSCA-Net (ours) |
|---|---|---|
| SDD (ADE/FDE, px) | 7.39 / 11.27 | **6.95 / 10.43** |

See the paper for the full comparison against 15 additional baselines and
the component-wise / grid-resolution ablation studies (Tables I-IV).

## Installation

```bash
git clone https://github.com/imamahasane/TSCA-Net.git
cd TSCA-Net
python -m venv .venv && source .venv/bin/activate
pip install -e .            # editable install, see pyproject.toml
# or: pip install -r requirements.txt   for the exact pinned/verified versions
```

Verify the install with the dependency-free sanity check (no dataset
required, runs on CPU in seconds):

```bash
python scripts/quick_test.py
```

## Quickstart

```bash
# 1. One leave-one-out split, on synthetic data (no dataset needed -- smoke test)
python scripts/train.py --leave-out eth --synthetic --epochs 3

# 2. Real data: place raw ETH/UCY files as data/eth_ucy/<split>.txt, then
python scripts/train.py --leave-out eth --data-root data/eth_ucy --epochs 100 --batch-size 32

# 3. Full leave-one-out benchmark across all five splits
python scripts/evaluate.py --data-root data/eth_ucy --ckpt-dir checkpoints

# 4. Component-wise ablation study (paper Table III)
python scripts/run_ablation.py --leave-out eth --data-root data/eth_ucy --epochs 100
```

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for dataset
acquisition, exact hyperparameters, and what has/hasn't been verified in
this repository.

### Library usage

```python
import torch
from tsca_net import Config, TSCANet, TSCANetLoss

cfg = Config()
model = TSCANet(cfg)
loss_fn = TSCANetLoss(cfg)

obs = torch.rand(2, 5, cfg.obs_len, 2)        # [B, N, T, 2]
heatmap = torch.rand(2, 5, 1, cfg.map_h, cfg.map_w)
gt = torch.rand(2, 5, cfg.pred_len, 2)

out = model(obs, heatmap)
loss, breakdown = loss_fn(out, gt, cfg=cfg)
loss.backward()
```

## Repository structure

```
TSCA-Net/
├── src/tsca_net/          # the package
│   ├── config.py           # Config dataclass (all hyperparameters)
│   ├── functional.py       # heatmap rendering, goal sampling, entropy
│   ├── losses.py           # TSCANetLoss (paper Eq. 16-17)
│   ├── metrics.py          # best-of-K ADE / FDE
│   ├── model.py            # TSCANet (full pipeline)
│   ├── modules/            # TSCA, CPCP, AKGR, and shared CNN/U-Net blocks
│   ├── data/                # ETH/UCY + SDD datasets, windowing, collate
│   ├── training/            # train loop, ablation runner
│   ├── eval/                 # leave-one-out benchmarking
│   └── viz/                  # qualitative plots
├── scripts/                # train.py, evaluate.py, run_ablation.py, quick_test.py
├── configs/                 # YAML configs for ETH/UCY and SDD
├── tests/                   # pytest suite (19 tests, incl. regression guards)
├── docs/
│   ├── ARCHITECTURE.md      # module <-> paper-equation map
│   ├── REPRODUCIBILITY.md   # dataset setup, hyperparameters, verification status
│   └── KNOWN_ISSUES.md      # every bug found + fixed, with root cause and evidence
├── notebooks/
│   └── quickstart_demo.ipynb # short usage demo against the installed package
└── TSCA_Net_ICDM2026.ipynb   # original notebook submitted with the paper,
                               # unmodified -- kept for the historical record;
                               # see docs/KNOWN_ISSUES.md for what changed and why
```

## Citation

```bibtex
@inproceedings{rahman2026tscanet,
  title     = {{TSCA-Net}: Temporal-Spatial Clique Attention for Interpretable
               Multimodal Pedestrian Trajectory Prediction},
  author    = {Rahman, Md Mustafizur and Yang, Guangchao and Noor, A F M Abdun
               and Ahasan, Md Imam and Rahman, Md Mahfuzur and Islam, Md Ariful},
  booktitle = {IEEE International Conference on Data Mining (ICDM)},
  year      = {2026}
}
```

See [`CITATION.cff`](CITATION.cff) for a machine-readable citation record.

## License

MIT — see [`LICENSE`](LICENSE). This applies to the code in this
repository; the paper text and figures are subject to the publisher's
copyright terms.

## Acknowledgements

This work was supported by the National Natural Science Foundation of
China under Grant 62572086. TSCA-Net extends the MCK-Net backbone
(Dong et al., *IEEE T-ITS*, 2025).

## Contributing

Contributions are welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md).
