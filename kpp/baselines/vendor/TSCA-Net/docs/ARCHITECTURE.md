# Architecture

TSCA-Net follows a hierarchical predict-then-refine paradigm (paper Sec.
II-A, Fig. 1) extending the MCK-Net backbone with three modules. This
document maps paper sections and equations to source files.

```
                      +-------------------+
  scene image  ------>|                   |
  obs. trajectories -->|   InputModule     |---> I  (fused map)
                      +-------------------+
                                |
                                v
                      +-------------------+
                      | TSCAGoalModule    |
                      |  FirstGoalNetwork |---> M1 (coarse goal heatmap)
                      |  TSCAModule       |---> time-gated clique matrix
                      |  TSCACliqueEncoder|---> I_C~
                      |  SecondGoalNetwork|---> M2 (TSCA-refined goal heatmap)
                      +-------------------+
                                |
                    sample K goals from M2
                                |
                                v
                      +-----------------------+
                      | CPCPAKGRPredictor     |
                      |  (per decoding step)  |
                      |   LSTM update          |
                      |   SocialGraphBuilder   |
                      |   CPCPModule           |---> social clique potential
                      |   AdaptiveKANLayer     |---> AKGR: entropy-driven KAN
                      |   motion_head          |---> delta(x, y)
                      +-----------------------+
                                |
                                v
                   predictions [B, N, K, F, 2]
```

## Module -> file -> paper section

| Module | File | Paper section | Novelty |
|---|---|---|---|
| `InputModule` | `src/tsca_net/modules/common.py` | Sec. II-B, Eq. 1 | — (MCK-Net) |
| `TemporalEmbedding`, `TSCAModule` | `src/tsca_net/modules/tsca.py` | Sec. II-C-2, Eq. 3-6 | **#1 TSCA** |
| `TSCACliqueEncoder` | `src/tsca_net/modules/tsca.py` | Sec. II-C-2 | **#1 TSCA** |
| `FirstGoalNetwork`, `SecondGoalNetwork`, `TSCAGoalModule` | `src/tsca_net/modules/goal.py` | Sec. II-C, Eq. 2, 7 | — / #1 |
| `SocialGraphBuilder`, `CPCPModule` | `src/tsca_net/modules/cpcp.py` | Sec. II-D-1/2, Eq. 8-10 | **#2 CPCP** |
| `b_spline_basis`, `KANLayer` | `src/tsca_net/modules/akgr.py` | Sec. II-D-3, Eq. 13 | — (KAN backbone) |
| `AdaptiveKANController`, `AdaptiveKANLayer` | `src/tsca_net/modules/akgr.py` | Sec. II-D-3, Eq. 11-12 | **#3 AKGR** |
| `CPCPAKGRPredictor` | `src/tsca_net/modules/predictor.py` | Sec. II-D-4, Eq. 14-15 | #2 + #3 |
| `TSCANet` | `src/tsca_net/model.py` | Sec. II-A, Fig. 1 | full pipeline |
| `TSCANetLoss` | `src/tsca_net/losses.py` | Sec. II-E, Eq. 16-17 | — |
| `compute_ade`, `compute_fde` | `src/tsca_net/metrics.py` | Sec. III-B | — |

Every deviation from the originally submitted notebook (bug fixes, and
the reasoning behind each one) is documented in `docs/KNOWN_ISSUES.md`,
cross-referenced from the relevant module's docstring.

## Data flow shapes

- `B` = batch size (number of scenes per batch)
- `N` = number of agents in a scene (padded to the batch max; see
  `src/tsca_net/data/collate.py`, `valid_mask` marks real vs. padded slots)
- `T` = `obs_len` (8, i.e. 3.2s at 2.5Hz)
- `F` = `pred_len` (12, i.e. 4.8s)
- `K` = `num_samples` (20, best-of-K protocol)
- `H, W` = spatial heatmap resolution (256x256 default)

```
obs_traj      [B, N, T, 2]
heatmap       [B, N, 1, H, W]
I (fused)     [B, N, fusion_channels, H, W]
M1, M2        [B, N, 1, H, W]
goal_pts      [B, N, K, 2]
tsca_matrix   [B, N, K, T, 2]
I_C_tilde     [B, N, 64, H, W]
predictions   [B, N, K, F, 2]
```

## Why CPCP encodes each agent's history once, not once per neighbour pair

The paper's social-graph formulation (Eq. 8-9) is naturally
`O(N x neighbours)`: for every agent i, sum a function of every neighbour
j's encoded history. The original notebook implemented this literally by
re-encoding each neighbour's trajectory with an LSTM for every querying
agent. Since a pedestrian's own history is identical regardless of who is
querying it, `src/tsca_net/modules/cpcp.py` encodes every agent's history
exactly once per scene (`O(N)`) and reuses that encoding as both the query
agent's own context and every other agent's neighbour context, weighted by
the (asymmetric) adjacency matrix. This is mathematically equivalent for
the encoding step and meaningfully cheaper for scenes with many agents.
