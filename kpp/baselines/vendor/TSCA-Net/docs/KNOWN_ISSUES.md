# Known Issues: Original Submission vs. This Release

This document records every substantive gap between the code originally
submitted alongside the ICDM 2026 paper (`TSCA_Net_ICDM2026.ipynb`, kept for
the historical record — see repository root) and this public release, why
each one matters, how it was fixed, and how it was verified. Ordered
roughly by severity / discovery order.

If you are a reviewer trying to reconcile paper claims with this code,
start here.

---

## #0 — The original notebook crashes on its own first forward pass (blocking)

**Severity: critical.** Discovered by literally re-running the notebook's
own "Quick Test & Sanity Checks" cell and its "Experiment Runner" cell
against the ported code. Both raise the *same* `RuntimeError` in the
original notebook's saved execution output (`TSCA_Net_ICDM2026.ipynb`,
cells 32 and 34):

```
RuntimeError: shape '[2, 3, 1, 256, 256]' is invalid for input of size 98304
RuntimeError: shape '[4, 1, 1, 256, 256]' is invalid for input of size 65536
```

**Root cause:** `FirstGoalNetwork`'s encoder is built as
`[in_ch] + cfg.goal_enc_ch` (6 entries → 5 stride-2 conv layers → 32x
spatial downsampling, 256→8), while its decoder uses `cfg.goal_dec_ch`
directly (5 entries → 4 stride-2 transpose-conv layers → 16x upsampling,
8→128). The encoder and decoder are asymmetric, so the very first module in
the entire pipeline (the M1 goal heatmap network) cannot produce a tensor
of the expected shape. This means **no cell in the original notebook that
calls the model ever completed successfully** — every recorded
execution that reaches `model(...)` ends in this same crash.

**Fix:** `src/tsca_net/modules/goal.py`, `FirstGoalNetwork.__init__` now
drops the duplicated leading channel (`cfg.goal_enc_ch[1:]`), giving the
encoder 4 layers (16x downsample) to match the decoder's 4 layers (16x
upsample) — the same pattern `SecondGoalNetwork` already used correctly.

**Verified:** `scripts/quick_test.py` and `tests/test_model_shapes.py` both
run a real forward pass through the full model and assert exact output
shapes.

---

## #1 — Two more bugs were hidden behind #0 and had never been reached

Because the original notebook crashed at the very first module, none of
its later code paths were ever actually exercised — not even once. Fixing
#0 immediately exposed two more crashes on the next re-run:

### #1a — `TSCAModule.gate_net` applies `GroupNorm` to channels-last data

`nn.GroupNorm` requires the channel dimension at index 1 (NCHW-style), but
this MLP runs on `[B, N, K, T, 32]` tensors straight out of a `Linear`
layer — channels-last, as every attention/transformer-style MLP produces.
Calling `GroupNorm(8, 32)` on such a tensor crashes with
`Expected number of channels in input to be divisible by num_groups, but
got input of shape [2, 4, 20, 8, 32] and num_groups=8`.

**Fix:** `src/tsca_net/modules/tsca.py` replaces `GroupNorm(8, 32)` with
`LayerNorm(32)`, the dimensionally-correct and standard choice for
channels-last features.

### #1b — `KANLayer`'s B-spline knot vector isn't padded

The Cox–de Boor recursion in `b_spline_basis` loses one basis function per
recursion step (`order` steps total). With the original's unpadded
`torch.linspace(-1, 1, grid + 1)` knot vector, the recursion actually
produces `grid - order` basis functions, but `self.coeff` was allocated
assuming `grid + order` — a shape mismatch that crashes `torch.einsum` the
moment any KAN sub-layer is evaluated with a grid size other than one that
happens to coincide.

**Fix:** `src/tsca_net/modules/akgr.py` pads the knot vector with `order`
extra points on each side at the same spacing (the standard construction
used by reference KAN implementations such as `efficient-kan`), which
makes the recursion produce exactly `grid + order` basis functions.

**Verified:** `tests/test_modules.py::test_b_spline_basis_count_matches_coeff_shape`
and the full `quick_test.py` run (see below).

---

## #2 — AKGR's controller could not learn (non-differentiable dispatch)

**Severity: critical — undermines novelty #3.** The original
`AdaptiveKANLayer.forward` computed
`grid_sizes = (base + delta).clamp(lo, hi).round().long()` and used the
result purely for boolean-mask indexing to select a KAN sub-layer.

Casting to an integer dtype detaches a tensor from autograd in PyTorch.
Since `grid_sizes` was *only* used for indexing (never combined with any
float value that reaches the loss), **no gradient could ever reach
`AdaptiveKANController`'s parameters** — it behaved as a fixed random
function of goal-entropy and motion-complexity for the entire training
run, no matter how long training continued. The original notebook's own
`run_quick_test` even printed a per-parameter gradient check for exactly
this case, but only printed `"✗ ZERO!"` without asserting, so this
silently produced no visible failure.

This directly contradicts the paper's own description (Sec. II-D-3): *"we
implement a soft interpolation between a set of precomputed KAN basis
functions at discrete grid sizes, enabling differentiable backpropagation
through the grid selection mechanism."* The code did not do this; the
paper's intended design and the shipped code diverged.

**Fix:** `AdaptiveKANController` now predicts a *continuous* grid size
`g_soft ∈ [g_min, g_max]`. `AdaptiveKANLayer` evaluates the two
neighbouring integer-grid KAN sub-layers (`floor(g_soft)`, `ceil(g_soft)`)
and linearly blends them with weight `frac(g_soft)`. The blend weight is a
differentiable function of `g_soft`, so gradients now reach the
controller; only the *choice of which two sub-layers to evaluate* remains
non-differentiable, which is unavoidable for a discrete lookup and is the
same pattern used by soft mixture-of-experts routing. This is, in effect,
implementing what the paper already claimed the code did.

**Verified:**
`tests/test_akgr_gradient.py::test_akgr_controller_receives_gradient`
asserts every controller parameter has `grad is not None` and
`grad.norm() > 0` after a real backward pass. Measured gradient norms from
one `quick_test.py` run: `controller.predictor[0].weight` grad_norm =
`5.02e-01`, `controller.unc_enc[0].weight` grad_norm = `1.80e-01` — both
comfortably non-zero.

---

## #3 — Goal-heatmap entropy was numerically almost input-invariant

**Severity: high — undermines the signal driving AKGR.** The original
`compute_entropy` applied `softmax` over the full flattened
`H*W = 256*256 = 65,536`-pixel heatmap of *raw sigmoid values* (not
logits, range `[0, 1]`). Softmax over that many narrow-range values
collapses to a near-uniform distribution regardless of the input's actual
shape. Verified empirically with a pure-Python simulation (no ML
framework needed to demonstrate this):

```
peaky heatmap (one small confident region near 1.0):  entropy = 11.0890
uniform/flat heatmap:                                  entropy = 11.0896
theoretical max, log(65536):                           entropy = 11.0904
```

Both cases land within 0.06% of the theoretical maximum — the "goal
distribution uncertainty" signal that's supposed to drive AKGR's grid
adaptation (paper Eq. 11) barely varies with how confident or diffuse the
prediction actually is.

**Fix:** `src/tsca_net/functional.py::compute_entropy` now normalises the
heatmap into a proper probability distribution by dividing by its own sum
(rather than exponentiating and re-normalising via softmax), then takes
Shannon entropy directly. This stays in the same `[0, log(H*W)]` range but
actually varies with concentration.

**Verified:**
`tests/test_entropy.py::test_entropy_distinguishes_peaky_from_uniform`
asserts a peaked heatmap's entropy falls below 90% of the theoretical
maximum while a uniform one sits at the maximum — the opposite of the
near-constant behaviour measured above.

---

## #4 — CPCP never saw a real social signal during training

**Severity: critical — undermines novelty #2.** `TrajectoryDataset` treated
every sequence as a single isolated pedestrian (`N=1` per scene), and
`train_one_epoch` / `validate` / `benchmark_eth_ucy` all called
`model(obs, heatmap)` with no neighbour argument, so `CPCPModule` only ever
received an all-zero placeholder neighbour tensor. The architecture for
"asymmetric social interaction" was present; the actual training signal
behind it was not.

**Fix:** `src/tsca_net/data/eth_ucy.py` (`ETHUCYDataset`) and
`src/tsca_net/data/sdd.py` (`SDDDataset`) now group raw annotation rows
into genuine multi-agent scenes — every pedestrian visible in a time
window is included together, not windowed independently. `CPCPModule` was
also rewritten (`src/tsca_net/modules/cpcp.py`) to operate directly over
all agents present in a scene, encoding each agent's history exactly once
and reusing it for every query agent (previously it duplicated the
encoding per neighbour pair — `O(N × neighbours)` — now `O(N)`).
`CPCPAKGRPredictor` also rebuilds the social graph from the *current*
predicted positions at every decoding step, matching the paper's explicit
claim (Sec. II-D-1) that the graph is rebuilt at every step rather than
fixed from the last observed position.

**Verified:** `tests/test_data.py` confirms multi-agent scenes are
produced with `N ≥ 2`; `tests/test_modules.py::test_cpcp_module_shapes`
and `tests/test_losses.py::test_padded_agents_do_not_affect_loss` exercise
the real multi-agent path end-to-end, including variable-N batching with
zero-padding masked out of the loss; `scripts/train.py --synthetic` was
run for a full epoch (data loading → forward → backward → optimizer step →
validation → checkpointing) and completed successfully.

---

## #5 — No SDD loader existed at all

The original notebook had no code path for the Stanford Drone Dataset
despite the paper reporting SDD results in Table II.

**Fix:** `src/tsca_net/data/sdd.py` adds an `SDDDataset` parsing the
standard SDD `annotations.txt` format (bounding-box center, pixel
coordinates, filtered to the `Pedestrian` label to match the paper's
evaluation subset).

**Caveat — not yet run against real data.** Per user direction, this
loader (and `ETHUCYDataset`'s real-data path) was written against the
standard, widely-used raw annotation formats from the trajectory-prediction
literature, but **has not been executed against an actual ETH/UCY or SDD
release** (no dataset files were available in this environment, and no
download URL is hardcoded — see `docs/REPRODUCIBILITY.md`). Everything
that *was* runnable in this environment (the model, losses, metrics,
training loop, ablation runner, and the multi-agent windowing/collate
logic against synthetic data) was actually executed and is covered by
`tests/`. Validate the real-data path against one real annotation file
before trusting benchmark numbers produced with it.

---

## #6 — Paper-internal inconsistency: CPCP social radius

Paper Sec. II-D-1 states the social radius is fixed at 3.0m "following
standard convention," but Sec. III-B (Implementation Details) states
"neighbourhood radius of 2m for ETH/UCY." The code (both original and
here) uses `social_radius: float = 3.0`. Reconcile the paper text before
camera-ready; `Config.social_radius` / `Config.social_radius_px` are the
single source of truth for what the code actually does.

## #7 — Paper claims 8 attention heads; none exist in the code

Sec. III-B states "the number of attention heads within the clique encoder
is fixed at 8." No multi-head attention mechanism exists anywhere in
`TSCAModule` or `CliqueEncoder` — the temporal gate is a small
scalar-output MLP, and the clique encoder is a CNN + Squeeze-Excitation
block. Either correct the paper text, or the architecture needs an actual
multi-head formulation to match the claim. This release does not attempt
to guess which was intended and does not add an unrequested multi-head
mechanism; flagging for the authors to decide.

## #8 — Leftover template text

The original notebook's title cell read *"Publication-Ready Implementation
for CVPR Submission"* — leftover from a different venue. Not present here.
