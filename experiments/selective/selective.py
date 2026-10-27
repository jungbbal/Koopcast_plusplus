"""
Primitive-gated (selective) Koopman adaptation for KoopCast++ -- prototype.

Idea
----
* A rule-based motion-primitive label is computed from the last 8 observed
  positions: stop / straight / turn_L / turn_R / accel / decel.
  Thresholds are training-split quantiles (no test data involved), because
  ETH/UCY and the snu-asri lobby move at very different speeds.
* Library: one Koopman operator K_m per primitive, fitted by EDMD on the training
  pairs of that primitive (frozen trained encoder), shrunk toward the global K0:
      K_m = (Psi' Psi^T + lam K0)(Psi Psi^T + lam I)^-1
* Gate: per-primitive residual threshold tau_m = q-quantile of the one-step
  residual ||psi' - K_m psi|| on that primitive's training pairs.
* Inference: start from K_{m(t0)}. Walk the observed pairs; a pair is used for the
  rank-1 update ONLY if its residual under its own primitive's operator exceeds
  tau_{m(s)}, i.e. no primitive in the library explains it. A spectral safety
  check rejects an adapted K whose spectral radius grows past max(1, rho(start)).

Also fixes the neighbour time-alignment of the repo's ``KoopCastPP._adapt``
(histories are right-aligned at t0, so a neighbour's position ``back`` steps before
t0 is ``o[len(o)-1-back]``, not ``o[c]``).

Everything heavy (social histograms, encoder) is precomputed once per split, so a
whole eta / quantile grid is evaluated in pure numpy.
"""
from __future__ import annotations

import sys
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]

from kpp.data.windows import scene_windows                     # noqa: E402
from kpp.predictors import KoopCastPP                          # noqa: E402
from kpp.predictors.koopcastpp import _core                    # noqa: E402

PRIMS = ["stop", "straight", "turn_L", "turn_R", "accel", "decel"]
DT = 0.4          # both ETH/UCY (frame step 10 @ 25 fps) and snu-asri (2.5 Hz)


# --------------------------------------------------------------------------- #
# Motion primitives
# --------------------------------------------------------------------------- #
def motion_features(H: np.ndarray):
    """H: (N, 8, 2) positions, oldest -> newest. Returns speed, dtheta(deg), dv."""
    v = np.diff(H, axis=1) / DT                         # (N, 7, 2)
    sp = np.linalg.norm(v, axis=2)                      # (N, 7)
    speed = sp[:, -3:].mean(1)
    head = np.arctan2(v[..., 1], v[..., 0])
    dth = np.degrees(np.angle(np.exp(1j * (head[:, -1] - head[:, -4]))))
    dv = sp[:, -3:].mean(1) - sp[:, :3].mean(1)
    return speed, dth, dv


@dataclass
class PrimitiveRule:
    stop_speed: float
    turn_deg: float
    acc_dv: float

    @classmethod
    def fit(cls, H_train: np.ndarray, q_stop=10, q_turn=80, q_acc=80):
        speed, dth, dv = motion_features(H_train)
        moving = speed >= np.percentile(speed, q_stop)
        return cls(stop_speed=float(np.percentile(speed, q_stop)),
                   turn_deg=float(np.percentile(np.abs(dth[moving]), q_turn)),
                   acc_dv=float(np.percentile(np.abs(dv[moving]), q_acc)))

    def label(self, H: np.ndarray) -> np.ndarray:
        speed, dth, dv = motion_features(H)
        lab = np.full(len(H), PRIMS.index("straight"))
        acc = np.abs(dv) > self.acc_dv
        lab[acc & (dv > 0)] = PRIMS.index("accel")
        lab[acc & (dv < 0)] = PRIMS.index("decel")
        turn = np.abs(dth) > self.turn_deg                # turn overrides accel/decel
        lab[turn & (dth > 0)] = PRIMS.index("turn_L")
        lab[turn & (dth < 0)] = PRIMS.index("turn_R")
        lab[speed < self.stop_speed] = PRIMS.index("stop")  # stop overrides all
        return lab


def hist_from_block(Hflat: np.ndarray) -> np.ndarray:
    """(N, 16) newest-first history block -> (N, 8, 2) oldest -> newest."""
    return Hflat.reshape(-1, 8, 2)[:, ::-1]


# --------------------------------------------------------------------------- #
# Library (per-primitive K_m + residual thresholds)
# --------------------------------------------------------------------------- #
def edmd_shrink(P: np.ndarray, P2: np.ndarray, K0: np.ndarray, ridge: float, lam: float):
    """Row-stacked P, P2 (N, d). K = (P2^T P + lam K0)(P^T P + (ridge+lam) I)^-1."""
    d = P.shape[1]
    G = P.T @ P + (ridge + lam) * np.eye(d)
    A = P2.T @ P + lam * K0
    return np.linalg.solve(G.T, A.T).T


@dataclass
class Library:
    rule: PrimitiveRule
    K0: np.ndarray
    K: Dict[int, np.ndarray]
    resid: Dict[int, np.ndarray]          # sorted training residuals per primitive (under K_m)
    resid0: Dict[int, np.ndarray]         # ... under K0 (for the K0+gate ablation)
    counts: Dict[int, int]
    rho: Dict[int, float]

    def tau(self, m: int, q: float, base: bool = False) -> float:
        r = (self.resid0 if base else self.resid)[m]
        return float(np.quantile(r, q))


def build_library(scene: str, model: KoopCastPP, ridge: float, lam_frac: float = 0.1,
                  min_count: int = 500) -> Library:
    import train_koopcastpp as T
    (H_t, G_t, H_tp, G_tp), _ = T.build_data(T.load_train_split(scene))
    with torch.no_grad():
        P = torch.cat([H_t, model.enc(G_t)], 1).double().numpy()
        P2 = torch.cat([H_tp, model.enc(G_tp)], 1).double().numpy()
    H = hist_from_block(H_t.numpy().astype(float))
    rule = PrimitiveRule.fit(H)
    lab = rule.label(H)
    K0 = model.K
    # shrinkage strength relative to the data scale (absolute coordinates)
    scale = np.trace(P.T @ P) / len(P) / P.shape[1]
    K, resid, resid0, counts, rho = {}, {}, {}, {}, {}
    for m in range(len(PRIMS)):
        idx = lab == m
        counts[m] = int(idx.sum())
        if counts[m] < min_count:
            Km = K0.copy()
        else:
            # increase shrinkage until K_m is no less stable than K0
            rho_cap = max(1.0, float(np.abs(np.linalg.eigvals(K0)).max())) + 1e-3
            lam = lam_frac * scale * counts[m]
            for _ in range(8):
                Km = edmd_shrink(P[idx], P2[idx], K0, ridge, lam)
                if np.abs(np.linalg.eigvals(Km)).max() <= rho_cap:
                    break
                lam *= 10
            else:
                Km = K0.copy()
        K[m] = Km
        rho[m] = float(np.abs(np.linalg.eigvals(Km)).max())
        resid[m] = np.sort(np.linalg.norm(P2[idx] - P[idx] @ Km.T, axis=1))
        resid0[m] = np.sort(np.linalg.norm(P2[idx] - P[idx] @ K0.T, axis=1))
    return Library(rule, K0, K, resid, resid0, counts, rho)


# --------------------------------------------------------------------------- #
# Precomputed evaluation samples
# --------------------------------------------------------------------------- #
@dataclass
class Sample:
    seq: np.ndarray        # (L, d) observed lifted states, time-aligned; seq[-1] = psi0
    hist8: np.ndarray      # (L, 8, 2) the 8-step window behind each seq entry
    fut: np.ndarray        # (12, 2)
    lab: np.ndarray = field(default=None)   # (L,) primitive labels (set later)


def prepare(model: KoopCastPP, ds) -> List[Sample]:
    """One Sample per scored (agent, t0) target, with full observed history."""
    raw = []
    G_all = []
    for sw in scene_windows(ds, full_history=True):
        if not sw.targets:
            continue
        obs = {a: np.asarray(o, float) for a, o in sw.obs.items()}
        for t in sw.targets:
            hist = obs[t.agent_id]
            n = len(hist)
            Hs, Gs, H8 = [], [], []
            for c in range(7, n):
                back = n - 1 - c
                neigh = [o[len(o) - 1 - back] for a, o in obs.items()
                         if a != t.agent_id and len(o) - 1 - back >= 0]
                neigh = np.asarray(neigh, float).reshape(-1, 2)
                Gs.append(_core.social_histogram(hist[c], neigh, model.radius, model.num_sectors))
                Hs.append(_core.history_block(hist[:c + 1], 8))
                H8.append(hist[c - 7:c + 1])
            raw.append((np.asarray(Hs), len(G_all), len(Gs), np.asarray(H8), np.asarray(t.pred)))
            G_all.extend(Gs)
    with torch.no_grad():
        E = model.enc(torch.from_numpy(np.asarray(G_all)).float()).double().numpy()
    out = []
    for Hs, i0, L, H8, fut in raw:
        seq = np.concatenate([Hs, E[i0:i0 + L]], axis=1)
        out.append(Sample(seq=seq, hist8=H8, fut=fut))
    return out


def label_samples(samples: List[Sample], rule: PrimitiveRule):
    for s in samples:
        s.lab = rule.label(s.hist8)


# --------------------------------------------------------------------------- #
# Methods
# --------------------------------------------------------------------------- #
def spectral_radius(K):
    return float(np.abs(np.linalg.eigvals(K)).max())


def rank1(K, x, xn, eta):
    return K + eta * np.outer(xn - K @ x, x) / (float(x @ x) + 1e-8)


def run(samples: List[Sample], lib: Library, method: str, eta: float = 0.0,
        q: float = 0.95, safety: bool = True):
    """Returns ADE, FDE, fraction of targets with >=1 update, mean updates/target.

    methods:
      static        global K0, no adaptation
      always        K0 + rank-1 on every observed pair (repo behaviour, aligned)
      lib           K_{m(t0)}, no adaptation
      gate0         K0 + update only pairs with residual(K0) > tau0_{m(s)}
      lib_gate      K_{m(t0)} + update only pairs with residual(K_{m(s)}) > tau_{m(s)}
    """
    ade, fde, upd_any, upd_n = [], [], 0, 0
    tau = {m: lib.tau(m, q) for m in lib.K}
    tau0 = {m: lib.tau(m, q, base=True) for m in lib.K}
    for s in samples:
        psi0 = s.seq[-1]
        m0 = int(s.lab[-1])
        K = lib.K[m0] if method in ("lib", "lib_gate") else lib.K0
        Kstart = K
        n_upd = 0
        if eta > 0 and method in ("always", "gate0", "lib_gate") and len(s.seq) >= 2:
            for i in range(len(s.seq) - 1):
                x, xn = s.seq[i], s.seq[i + 1]
                if method == "always":
                    use = True
                else:
                    m = int(s.lab[i])
                    Kref = lib.K0 if method == "gate0" else lib.K[m]
                    thr = tau0[m] if method == "gate0" else tau[m]
                    use = np.linalg.norm(xn - Kref @ x) > thr
                if use:
                    K = rank1(K, x, xn, eta)
                    n_upd += 1
            if safety and n_upd and spectral_radius(K) > max(1.0, spectral_radius(Kstart)) + 1e-3:
                K, n_upd = Kstart, 0          # reject an adaptation that destabilises
        pred = _core.rollout(K, psi0, 12)
        err = np.linalg.norm(pred - s.fut, axis=1)
        ade.append(err.mean()); fde.append(err[-1])
        upd_any += n_upd > 0; upd_n += n_upd
    N = len(samples)
    return float(np.mean(ade)), float(np.mean(fde)), upd_any / N, upd_n / N
