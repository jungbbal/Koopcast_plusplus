# Primitive-gated (selective) Koopman adaptation — prototype results (2026-10-27)

## Setup
- Base: repo KoopCast++ weights, unchanged (`kpp/predictors/koopcastpp/data/*.pt`).
- Primitives (rule-based, from last 8 observed steps, thresholds = training-split quantiles): stop (speed < q10), turn_L/turn_R (|Δheading over 3 steps| > q80), accel/decel (|Δspeed| > q80), else straight.
- `lib`: per-primitive operator K_m = EDMD on that primitive's training pairs, shrunk toward K0 (λ grid 1e-6…1e-3 × data scale; auto-increase λ until ρ(K_m) ≤ max(1, ρ(K0))).
- Gate: a pair updates K (rank-1, as in repo) only if its one-step residual exceeds τ_m = q-quantile of that primitive's training residuals (q ∈ {0.9, 0.95, 0.99}).
- `gate0` = gate with K0 only; `lib_gate` = K_m(t0) + gate; `always` = repo behaviour (all pairs); `always_safe` = always + reject adapted K if spectral radius grows past max(1, ρ(start)).
- Neighbour time alignment fixed in all adaptive variants (repo `_adapt` indexes neighbours with `o[c]`).
- Protocol: all hyperparameters (η ∈ {0, .005, .01, .02, .05, .1}, q, λ) chosen on val, scored on test. OOD uses lobby val (no OOD val). `full_history=True`.

## Test ADE (val-selected configs)
| scene | static | always | always_safe | lib | gate0 | lib_gate |
|---|---|---|---|---|---|---|
| eth | **1.0649** | 1.0649 | 1.0666 | 1.0735 | 1.0695 | 1.0819 |
| hotel | 0.4882 | 0.4807 | **0.4685** | 0.4997 | 0.4882 | 0.4997 |
| univ | **0.6271** | 0.6308 | 0.6339 | 0.6333 | 0.6276 | 0.6333 |
| zara1 | 0.4316 | 0.4316 | 0.4327 | **0.4310** | 0.4319 | 0.4319 |
| zara2 | 0.3544 | 0.3544 | **0.3500** | 0.3591 | 0.3544 | 0.3591 |
| ETH/UCY avg | 0.5932 | 0.5925 | **0.5903** | 0.5993 | 0.5943 | 0.6012 |
| snu-asri | **0.1339** | 0.1339 | 0.1339 | 0.1345 | 0.1344 | 0.1350 |
| snu-asri-ood | **0.1562** | 0.1562 | 0.1562 | 0.1599 | 0.1562 | 0.1599 |

Sanity: static matches repo numbers (zara1 0.4316, snu-asri 0.1339, OOD 0.1562); always @ η=0.05 on zara1 = 0.4597 as before.

## Findings
1. The primitive-gated variants do not beat static or always-on adaptation anywhere. Per-primitive operators (`lib`) are slightly worse on 6/7 test sets even though they were selected on val.
2. Per-primitive EDMD barely lowers the one-step residual vs K0 (τ quantiles almost identical): with the time-delay observable, primitive differences live in the history block K0 already uses, so splitting K adds variance, not information.
3. Residual magnitude cannot detect the OOD shift: OOD one-step residuals are *smaller* than in-distribution (median 0.0073 vs 0.0104; 0% of OOD pairs exceed the in-dist 99th-pct threshold). The OOD gain of always-on adaptation (0.1562 → 0.1496 at η=0.01, test-chosen) comes from accumulating a small consistent bias, not from rare large errors. Even test-chosen gated configs only reach ~0.153.
4. With honest val tuning, always-on adaptation picks η=0 on snu-asri, so the OOD improvement in docs/RESULTS.md is not reachable under this protocol.
5. The one ingredient that helped: the spectral safety check on always-on adaptation (`always_safe`), hotel −4.0%, zara2 −1.2%, ETH/UCY avg 0.5932 → 0.5903. Small but the only consistent gain.

## Suggested next steps
- Trigger on systematic error instead of magnitude: e.g. CUSUM/sign-consistency of the residual vector, or multi-step back-cast error over the observed window.
- Keep the spectral safety check in any adaptive variant.
- Larger OOD evaluation (snu-asri-ood has 2 usable agents); e.g. cross-domain ETH/UCY → lobby.

Code: selective.py, eval_selective.py, diag_ood.py (sent in chat 2026-10-27).
