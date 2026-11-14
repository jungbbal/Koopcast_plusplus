"""
Exception-handling Koopman operators for KoopCast++ -- prototype.

Default: the static global operator K0 (repo weights), unchanged.
Exception: only when the last 8 observed steps show an *extreme* manoeuvre
(sharp turn L/R, hard acceleration, hard deceleration -- beyond the q_hi quantile
of the training split) is a dedicated operator K_c used for the 12-step rollout.
Samples that are not flagged are predicted exactly as by the static model.

Two ways to fit the exception operators, both on extreme *training* samples only:

* ``edmd``  one-step EDMD on the flagged training pairs, shrunk toward K0
            (same estimator as the primitive library in ../selective).
* ``ms``    12-step rollout loss on the flagged training windows. Only the
            position-update rows (0:2) of K are changed, through a
            translation-invariant correction
                dp_{t+1} = sum_k D_k (p_{t-k} - p_t) + F e_t ,
            regularised by beta * ||[D, F]||^2.  36 parameters per class.

Both keep rho(K_c) <= max(1, rho(K0)) + 1e-3 (shrink the correction otherwise).
"""
from __future__ import annotations

import sys
import pathlib
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "selective"))
import selective as S                                    # noqa: E402  (adds repo to sys.path)

CLASSES = ["turn_L", "turn_R", "hard_accel", "hard_decel"]
NORMAL = -1


# --------------------------------------------------------------------------- #
# Extreme-manoeuvre detector
# --------------------------------------------------------------------------- #
@dataclass
class ExceptionRule:
    q: float
    stop_speed: float
    turn_deg: float
    acc_dv: float

    @classmethod
    def fit(cls, H_train: np.ndarray, q: float, q_stop: float = 10):
        speed, dth, dv = S.motion_features(H_train)
        stop = np.percentile(speed, q_stop)
        moving = speed >= stop
        return cls(q=q, stop_speed=float(stop),
                   turn_deg=float(np.quantile(np.abs(dth[moving]), q)),
                   acc_dv=float(np.quantile(np.abs(dv[moving]), q)))

    def label(self, H: np.ndarray) -> np.ndarray:
        """H (N, 8, 2) oldest->newest -> class index in CLASSES, or NORMAL."""
        speed, dth, dv = S.motion_features(H)
        lab = np.full(len(H), NORMAL)
        acc = np.abs(dv) > self.acc_dv
        lab[acc & (dv > 0)] = CLASSES.index("hard_accel")
        lab[acc & (dv < 0)] = CLASSES.index("hard_decel")
        turn = np.abs(dth) > self.turn_deg
        lab[turn & (dth > 0)] = CLASSES.index("turn_L")
        lab[turn & (dth < 0)] = CLASSES.index("turn_R")
        lab[speed < self.stop_speed] = NORMAL           # heading/dv meaningless when standing
        return lab


# --------------------------------------------------------------------------- #
# Training data (frozen encoder)
# --------------------------------------------------------------------------- #
@dataclass
class TrainData:
    P: np.ndarray       # (Np, 20) one-step pairs psi_t
    P2: np.ndarray      # (Np, 20) psi_{t+1}
    Hp: np.ndarray      # (Np, 8, 2) window behind psi_t
    X0: np.ndarray      # (Nm, 20) multistep start states
    FUT: np.ndarray     # (Nm, 12, 2)
    Hm: np.ndarray      # (Nm, 8, 2)


def train_data(scene: str, model) -> TrainData:
    import train_koopcastpp as T
    (H_t, G_t, H_tp, G_tp), (H0, G0, FUT) = T.build_data(T.load_train_split(scene))
    with torch.no_grad():
        P = torch.cat([H_t, model.enc(G_t)], 1).double().numpy()
        P2 = torch.cat([H_tp, model.enc(G_tp)], 1).double().numpy()
        X0 = torch.cat([H0, model.enc(G0)], 1).double().numpy()
    return TrainData(P, P2, S.hist_from_block(H_t.numpy().astype(float)), X0,
                     FUT.numpy().astype(float).reshape(-1, 12, 2),
                     S.hist_from_block(H0.numpy().astype(float)))


# --------------------------------------------------------------------------- #
# Exception operators
# --------------------------------------------------------------------------- #
def _rho(K):
    return float(np.abs(np.linalg.eigvals(K)).max())


def _cap_stability(K0, K, cap):
    """Shrink K toward K0 until rho(K) <= cap."""
    for a in [1.0, 0.5, 0.25, 0.1, 0.0]:
        Kc = K0 + a * (K - K0)
        if _rho(Kc) <= cap:
            return Kc, a
    return K0.copy(), 0.0


def fit_edmd(td: TrainData, rule: ExceptionRule, K0, ridge, lam_frac, min_count=100):
    lab = rule.label(td.Hp)
    scale = np.trace(td.P.T @ td.P) / len(td.P) / td.P.shape[1]
    cap = max(1.0, _rho(K0)) + 1e-3
    ops, info = {}, {}
    for c in range(len(CLASSES)):
        idx = lab == c
        n = int(idx.sum())
        if n < min_count:
            ops[c], info[c] = K0.copy(), dict(n=n, used=False)
            continue
        K = S.edmd_shrink(td.P[idx], td.P2[idx], K0, ridge, lam_frac * scale * n)
        K, a = _cap_stability(K0, K, cap)
        ops[c] = K
        info[c] = dict(n=n, used=True, shrink=a, dK=float(np.linalg.norm(K - K0) / np.linalg.norm(K0)))
    return ops, info


def _delta_rows(D, F):
    """Translation-invariant correction of rows 0:2. D (2,14) on p_{t-k}-p_t (k=1..7), F (2,4)."""
    Dk = D.reshape(2, 7, 2)                              # (out, k, coord)
    first = -Dk.sum(1)                                   # coefficient on p_t
    rest = Dk.permute(0, 1, 2).reshape(2, 14)
    return torch.cat([first, rest, F], dim=1)            # (2, 20)


def fit_ms(td: TrainData, rule: ExceptionRule, K0, beta, iters=300, lr=1e-3, min_count=100):
    lab = rule.label(td.Hm)
    cap = max(1.0, _rho(K0)) + 1e-3
    K0t = torch.from_numpy(K0).double()
    ops, info = {}, {}
    for c in range(len(CLASSES)):
        idx = lab == c
        n = int(idx.sum())
        if n < min_count:
            ops[c], info[c] = K0.copy(), dict(n=n, used=False)
            continue
        X = torch.from_numpy(td.X0[idx])
        Y = torch.from_numpy(td.FUT[idx])
        D = torch.zeros(2, 14, dtype=torch.float64, requires_grad=True)
        F = torch.zeros(2, 4, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([D, F], lr=lr)

        def loss_fn():
            K = K0t.clone()
            K[:2] = K0t[:2] + _delta_rows(D, F)
            psi, err = X, 0.0
            for t in range(12):
                psi = psi @ K.T
                err = err + ((psi[:, :2] - Y[:, t]) ** 2).sum(1).mean()
            return err / 12 + beta * (D.pow(2).sum() + F.pow(2).sum()), K

        with torch.no_grad():
            l0 = float(loss_fn()[0])
        for _ in range(iters):
            opt.zero_grad()
            l, _K = loss_fn()
            l.backward()
            opt.step()
        with torch.no_grad():
            l1, K = loss_fn()
        K, a = _cap_stability(K0, K.numpy(), cap)
        ops[c] = K
        info[c] = dict(n=n, used=True, shrink=a, loss0=l0, loss1=float(l1),
                       dK=float(np.linalg.norm(K - K0) / np.linalg.norm(K0)))
    return ops, info


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def rollout_batch(K, X):
    """X (N, 20) -> (N, 12, 2)."""
    out = np.empty((len(X), 12, 2))
    psi = X.copy()
    for t in range(12):
        psi = psi @ K.T
        out[:, t] = psi[:, :2]
    return out


def predict(X, labels, K0, ops=None):
    """Static K0 for NORMAL samples, ops[c] for exception class c."""
    pred = rollout_batch(K0, X)
    if ops:
        for c, K in ops.items():
            idx = labels == c
            if idx.any():
                pred[idx] = rollout_batch(K, X[idx])
    return pred


def errors(pred, fut):
    e = np.linalg.norm(pred - fut, axis=2)          # (N, 12)
    return e.mean(1), e[:, -1]                       # per-sample ADE, FDE
