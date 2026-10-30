"""
OOD diagnostic for the residual gate (snu-asri model, lobby val/test vs snu-asri-ood).

Hyperparameters (eta, q, lam) are read from eval_selective.py so this script always
matches the configuration that produced the main results. Run eval_selective.py
snu-asri first; this script only reads its cache.

    python diag_ood.py

Part 1  where do the one-step gate residuals come from, and how often do they cross
        the gate threshold, in-distribution vs OOD?
Part 2  OOD scores of all methods at the fixed configuration.
Part 3  gate sensitivity to q at the fixed eta (chosen on test -> diagnostic only).
"""
import sys, glob, pickle, pathlib, warnings
warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import selective as S
import eval_selective as E

ETA, Q, LAM = E.ETAS[0], E.QS[0], E.LAMS[0]
CACHE = HERE / "cache"


def load_cache(name):
    p = CACHE / f"{name}.pkl"
    if not p.exists():
        sys.exit(f"missing {p}. Run first:  python eval_selective.py snu-asri")
    return pickle.loads(p.read_bytes())


lib_path = CACHE / f"lib_snu-asri_{LAM}.pkl"
if not lib_path.exists():                       # fall back to any snu-asri library
    found = sorted(glob.glob(str(CACHE / "lib_snu-asri_*.pkl")))
    if not found:
        sys.exit("no snu-asri library in cache. Run first:  python eval_selective.py snu-asri")
    lib_path = pathlib.Path(found[0])
lib = pickle.loads(lib_path.read_bytes())

print(f"config from eval_selective.py: eta={ETA}  q={Q}  lam={LAM}   (library: {lib_path.name})")
if len(E.ETAS) > 1 or len(E.QS) > 1 or len(E.LAMS) > 1:
    print("  note: grids have >1 value; this script uses the first of each")

# --------------------------------------------------------------------------- #
print("\n[1] one-step gate residual  r = ||psi_{s+1} - K0 psi_s||  (all observed pairs)")
QREF = sorted({0.5, 0.9, 0.95, 0.99, Q})
print(f"  {'data':<20}{'pairs':>9}{'median r':>10}   share of r^2: {'pos':>5}{'social':>8}   "
      + "  ".join(f"r>tau(q={q})" for q in QREF))
datasets = {"lobby val": load_cache("val_snu-asri"),
            "lobby test": load_cache("test_snu-asri"),
            "OOD test": load_cache("test_snu-asri-ood")}
for name, ss in datasets.items():
    S.label_samples(ss, lib.rule)
    R = np.concatenate([s.seq[1:] - s.seq[:-1] @ lib.K0.T for s in ss])
    m = np.concatenate([s.lab[:-1] for s in ss]).astype(int)
    r = np.linalg.norm(R, axis=1)
    sq = R ** 2; tot = sq.sum()
    pos, soc = sq[:, :2].sum() / tot, sq[:, 16:].sum() / tot
    exc = []
    for q in QREF:
        thr = np.array([lib.tau(k, q, base=True) for k in range(len(S.PRIMS))])
        exc.append(f"{100 * np.mean(r > thr[m]):>11.1f}%")
    print(f"  {name:<20}{len(r):>9}{np.median(r):>10.4f}   {'':>13}{100*pos:>4.0f}%{100*soc:>7.0f}%   "
          + "  ".join(exc))

# --------------------------------------------------------------------------- #
ood = datasets["OOD test"]
print(f"\n[2] OOD test, fixed config (eta={ETA}, q={Q}, lam={LAM}),  n={len(ood)}")
print(f"  {'method':<13}{'ADE':>8}{'FDE':>8}{'upd%':>7}{'upd/tgt':>9}")
rows = [("static",      "static",   {}),
        ("always",      "always",   dict(eta=ETA, safety=False)),
        ("always_safe", "always",   dict(eta=ETA, safety=True)),
        ("lib",         "lib",      {}),
        ("gate0",       "gate0",    dict(eta=ETA, q=Q)),
        ("lib_gate",    "lib_gate", dict(eta=ETA, q=Q))]
for label, method, kw in rows:
    a, f, u, k = S.run(ood, lib, method, **kw)
    print(f"  {label:<13}{a:8.4f}{f:8.4f}{100*u:6.1f}%{k:9.2f}")

# --------------------------------------------------------------------------- #
print(f"\n[3] gate0 sensitivity to q on OOD (eta={ETA})   ** chosen on test -> diagnostic only **")
print(f"  {'q':>6}{'ADE':>8}{'upd%':>7}{'upd/tgt':>9}")
for q in [0.5, 0.75, 0.9, 0.95, 0.99]:
    a, f, u, k = S.run(ood, lib, "gate0", ETA, q)
    print(f"  {q:>6}{a:8.4f}{100*u:6.1f}%{k:9.2f}" + ("   <- fixed q" if q == Q else ""))
