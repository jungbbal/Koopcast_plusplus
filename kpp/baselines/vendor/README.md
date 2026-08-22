# kpp/baselines/vendor
## Last update

| update | revisor | contexts |
|---|---|---|
| 26. 08. 09 | Jeewon | __(addition)__ EqMotion, MoFlow, SingularTrajectory, TSCA-Net |
| 26. 08. 22 | Jeewon | __(addition)__ SMEMO |

```
baselines/vendor     # UPSTREAM CODE — byte-for-byte unmodified
   └─ canvas/predictors/
      ├─ linear_predictor.py         CANVAS constant-turn-rate baseline
      ├─ gp_predictor.py             CANVAS Gaussian-process baseline
      ├─ Social_STGCNN/  (+checkpoint/)   Social-STGCNN + weights
      ├─ SocialVAE/      (+config/ models/) SocialVAE + weights
      ├─ eigen/          (+models/ json_files/) EigenTrajectory + weights
      └─ trajectron/                 Trajectron++ (weights in ../assets/)
   └─ assets/models/trajectron/       Trajectron++ pretrained checkpoints
   ───────────────────────────── [Addition] ─────────────────────────────
   └─ EqMotion         
   └─ MoFlow
   └─ SingularTrajectory
   └─ SMEMO
   └─ TSCA-Net
```

## Addition

### 1) Summary
| baseline | year | paper | git |
|---|---|---|---|
| EqMotion | 2023 | https://ieeexplore.ieee.org/document/10205349 | https://github.com/MediaBrain-SJTU/EqMotion |
| MoFlow | 2025 | https://ieeexplore.ieee.org/document/11095073 | https://github.com/DSL-Lab/MoFlow |
| SingularTrajectory | 2024 | https://ieeexplore.ieee.org/document/10657121 | https://github.com/InhwanBae/SingularTrajectory |
| SMEMO | 2024 | https://ieeexplore.ieee.org/document/10411104 | https://github.com/Marchetz/SMEMO_trajectory_forecasting |
| TSCA-Net | 2026 | https://arxiv.org/abs/2607.11939 | https://github.com/imamahasane/TSCA-Net |

| baseline | notes (_by Claude_) |
|---|---|
| EqMotion | The first sequence-to-sequence motion prediction network that is theoretically equivariant under Euclidean transforms (rotation, translation, reflection) |
| MoFlow | Generates K correlated trajectories jointly via flow matching, then distills to one-step inference with IMLE |
| SingularTrajectory | A universal diffusion-based predictor that unifies five trajectory tasks into one Singular space |
| SMEMO | Models social interaction as data manipulation in an end-to-end trainable external working memory, yielding predictions explainable by design |
| TSCA-Net | Strengthens goal-conditioned prediction with temporal gating, asymmetric social graphs, and adaptive KAN grids |

### 2) Considerations
|checklists| | exceptions |
|---|---|---|
| Does it provide open source for project codes? _(github repository)_ | ✅ | OTTP (2026), HiGoalNet (2026), MMFlow (2026), TOTP (2025), FuseFormer (2025) |
| Does it deal with BEV (bird's eye view) datasets? _(ETH/UCY in common)_ | ✅ | TrajiCL (2025) |
| Is it the latest trend regarding with trajectory prediction? _(2023~)_ | ✅ | Sophie (2019), PECNet (2020) |