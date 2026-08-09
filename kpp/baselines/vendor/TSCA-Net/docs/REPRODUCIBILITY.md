# Reproducibility Guide

## What has and hasn't been verified

**Verified in this environment** (CPU, Python 3.9, `torch==2.8.0` — see
`requirements.txt`):
- A full forward + backward pass through every module, including the three
  novel contributions (`scripts/quick_test.py`, `tests/`).
- A complete train + validate + checkpoint epoch on synthetic multi-agent
  scenes, through the real `DataLoader` / collate / padding path
  (`scripts/train.py --synthetic`).
- The component-wise ablation runner (`scripts/run_ablation.py`).
- 19/19 unit and regression tests, including gradient-flow checks for the
  AKGR controller and an entropy-variance check (`pytest tests/`).

**Not verified — no dataset access in this environment:**
- `ETHUCYDataset` and `SDDDataset`'s real-data parsing paths. They target
  the standard raw formats used across the trajectory-prediction
  literature (see below), but have not been run against an actual
  ETH/UCY or SDD release. **Run the smoke check below against one real
  file before trusting any number produced from real data.**
- The specific ADE/FDE values in the paper's Tables I-IV. Reproducing them
  requires real data, full-length training (100 epochs on ETH/UCY per
  split, per the paper's schedule), and a GPU for practical runtime — none
  of which were available here. Fixing the bugs in `docs/KNOWN_ISSUES.md`
  changes the numerics of AKGR and CPCP relative to the original
  submission; **retrain and re-verify ADE/FDE against Tables I-IV before
  camera-ready** if those exact numbers must stand.

## Obtaining the datasets

No download URL is hardcoded in this repository — dataset mirrors move and
an unverified link is worse than none. Obtain the data from the official
sources:

- **ETH/UCY**: the ETH ("Biwi Walking Pedestrians") and UCY ("Crowds by
  Example") datasets, as pre-processed and commonly redistributed by the
  trajectory-prediction community (Social-GAN / Social-STGCNN / SR-LSTM /
  Trajectron++ all ship a compatible `raw/` split). Any of those repos'
  `raw/<scene>/*.txt` files work directly with `ETHUCYDataset` here — the
  parser expects whitespace-separated `frame_id pedestrian_id x y` rows,
  one file per scene, placed at `<data_root>/<split>.txt` for
  `split in {eth, hotel, univ, zara1, zara2}`.
- **Stanford Drone Dataset (SDD)**: from the Stanford Computational
  Vision & Geometry Lab's official release page. Place each scene's
  native `annotations.txt` at `<data_root>/<scene>/annotations.txt`.

### Smoke-checking a real file before trusting it

```bash
python -c "
from tsca_net.data.eth_ucy import ETHUCYDataset
ds = ETHUCYDataset('data/eth_ucy', 'eth', synthetic=False)
print(len(ds), 'scenes')
item = ds[0]
print({k: v.shape for k, v in item.items()})
"
```

If `len(ds) == 0`, the file wasn't found or didn't parse into any complete
window — check the path and column layout against
`src/tsca_net/data/eth_ucy.py::_read_raw_txt` before proceeding to train.

## Hyperparameters (paper Sec. III-B)

| Setting | Value | Config field |
|---|---|---|
| Optimiser | AdamW, weight decay 1e-4 | `Config.weight_decay` |
| LR (ETH/UCY) | 3e-4, cosine annealing, 100 epochs | override `Config.lr_goal` / `lr_pred` |
| LR (SDD) | 5e-4, step decay x0.5 / 30 epochs | same |
| Batch size | 32 | `Config.batch_size` (default here: 8 — lower for CPU/dev; raise for the paper setting) |
| Scene image resolution | 600x600 | this codebase renders heatmaps at `Config.map_h/map_w = 256` by default; raise to match if reproducing exactly |
| TSCA temporal embedding dim | 64 (paper text) | `Config.temporal_embed_dim` defaults to 16 here, inherited from the original notebook — see `docs/KNOWN_ISSUES.md` #7 regarding other paper/code architecture mismatches worth reconciling |
| CPCP social radius | 3.0m (Sec. II-D-1) vs. 2m (Sec. III-B) — inconsistent in the paper | `Config.social_radius` (ETH/UCY, metres), `Config.social_radius_px` (SDD, pixels) |
| AKGR grid range | [3, 10], base 5 | `Config.kan_min_grid/kan_max_grid/kan_base_grid` |
| K (best-of-K) | 20 | `Config.num_samples` |

`Config.epochs`, `batch_size`, and the LR schedule are left as
constructor overrides in `scripts/train.py` rather than baked-in paper
defaults, since the dev/CI default (small, fast) and the paper's reported
setting (100 epochs, batch 32, GPU) are different use cases.

## Reproducing the leave-one-out ETH/UCY protocol

```bash
for split in eth hotel univ zara1 zara2; do
    python scripts/train.py --leave-out $split --data-root data/eth_ucy \
        --epochs 100 --batch-size 32
done
python scripts/evaluate.py --data-root data/eth_ucy --ckpt-dir checkpoints
```

`scripts/evaluate.py` requires `best_<split>.pt` checkpoints from the loop
above; without real data and without `--synthetic`, it will refuse
meaningful benchmarking (falls back to random-weight evaluation and prints
a warning — see `src/tsca_net/eval/benchmark.py`).

## Hardware

The paper reports a single NVIDIA RTX 3070 (8GB), 3.4M trainable
parameters, ~0.078s/forward pass. This release's `TSCANet` reports
**3,557,502** trainable parameters with default `Config()` — within ~5% of
the paper's figure, the difference coming from the architectural fixes in
`docs/KNOWN_ISSUES.md` (encoder/decoder symmetry, CPCP redesign, AKGR knot
padding). No GPU was available in the environment this repository was
built in; all verification above ran on CPU.
