import sys, pickle, numpy as np, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,'.')
import selective as S
lib = pickle.load(open("cache/lib_snu-asri_0.001.pkl","rb"))
for name in ["val_snu-asri", "test_snu-asri", "test_snu-asri-ood"]:
    ss = pickle.load(open(f"cache/{name}.pkl","rb")); S.label_samples(ss, lib.rule)
    r, m = [], []
    for s in ss:
        r.append(np.linalg.norm(s.seq[1:] - s.seq[:-1] @ lib.K0.T, axis=1)); m.append(s.lab[:-1])
    r, m = np.concatenate(r), np.concatenate(m).astype(int)
    exc = {}
    for q in (0.5, 0.9, 0.95, 0.99):
        thr = np.array([lib.tau(k, q, True) for k in range(6)])
        exc[q] = float(np.mean(r > thr[m]))
    print(f"{name:20s} pairs={len(r):7d} median resid={np.median(r):.4f}  frac above tau_q:", {q: round(v,3) for q,v in exc.items()})
ood = pickle.load(open("cache/test_snu-asri-ood.pkl","rb")); S.label_samples(ood, lib.rule)
print("\nOOD diagnostic grid (chosen on test -> NOT reportable)")
for eta in [0.01, 0.05]:
    print(f" always   eta={eta}: ADE {S.run(ood, lib, 'always', eta, safety=False)[0]:.4f}")
    for q in [0.5, 0.75, 0.9]:
        a,f,u,k = S.run(ood, lib, 'gate0', eta, q); print(f" gate0    eta={eta} q={q}: ADE {a:.4f} upd {100*u:.0f}% {k:.1f}/tgt")
