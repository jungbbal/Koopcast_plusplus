"""
Exception-handling evaluation: static K0 by default, dedicated operators only for
extreme manoeuvres. Selection on val, scores on test.

    python eval_exception.py zara1 eth hotel univ zara2 snu-asri

snu-asri-ood is scored after snu-asri with the configs chosen on lobby val.
Observed-sample caches are shared with ../selective/cache.
"""
import sys, time, json, pickle, pathlib, itertools, warnings
warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "selective"))
import numpy as np
import exception as X
import selective as S
import eval_selective as E                    # splits(), cached(), ridge, always-eta, lib-lambda
from kpp.data import load
from kpp.predictors import KoopCastPP

Q_HI = [0.9, 0.95, 0.99]                      # exception threshold quantile (val-selected)
EDMD_LAMS = [1e-6, 1e-5, 1e-4, 1e-3]          # shrinkage for exc_edmd (val-selected)
MS_BETAS = [0.01, 0.1, 1.0]                   # regulariser for exc_ms (val-selected)
REPORT_Q = 0.95                               # fixed, pre-registered exception slice for reporting
ETA = E.ETAS[0]                               # always-adapt baseline uses eval_selective's fixed eta
LAM_LIB = E.LAMS[0]
OWN_CACHE = HERE / "cache"; OWN_CACHE.mkdir(exist_ok=True)


def own_cached(name, fn):
    p = OWN_CACHE / f"{name}.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    v = fn(); p.write_bytes(pickle.dumps(v)); return v


def arrays(samples):
    Xs = np.stack([s.seq[-1] for s in samples])
    F = np.stack([s.fut for s in samples]).astype(float)
    H = np.stack([s.hist8[-1] for s in samples]).astype(float)
    return Xs, F, H


def always_pred(samples, K0, eta):
    out = np.empty((len(samples), 12, 2))
    for i, s in enumerate(samples):
        K = K0
        if eta > 0:
            for j in range(len(s.seq) - 1):
                K = S.rank1(K, s.seq[j], s.seq[j + 1], eta)
        out[i] = X.rollout_batch(K, s.seq[-1][None])[0]
    return out


def lib_pred(Xs, H, lib):
    lab = lib.rule.label(H)
    return X.predict(Xs, lab, lib.K0, {m: lib.K[m] for m in lib.K})


def summarize(ade, fde, sl, cls):
    r = dict(ade=float(ade.mean()), fde=float(fde.mean()),
             p90=float(np.percentile(ade, 90)), p95=float(np.percentile(ade, 95)),
             exc_ade=float(ade[sl].mean()) if sl.any() else float("nan"),
             exc_fde=float(fde[sl].mean()) if sl.any() else float("nan"),
             normal_ade=float(ade[~sl].mean()))
    turn = (cls == 0) | (cls == 1); spd = (cls == 2) | (cls == 3)
    r["turn_ade"] = float(ade[turn].mean()) if turn.any() else float("nan")
    r["speed_ade"] = float(ade[spd].mean()) if spd.any() else float("nan")
    return r


def main():
    scenes = sys.argv[1:] or ["zara1"]
    allres = {}
    for scene in scenes:
        t0 = time.time()
        model = KoopCastPP(scene)
        K0 = model.K
        ridge = E.RIDGE.get(scene, 1e-4)
        td = own_cached(f"train_{scene}", lambda: X.train_data(scene, model))
        val_ds, test_ds = E.splits(scene)
        val = E.cached(f"val_{scene}", lambda: S.prepare(model, val_ds))
        tests = {scene: E.cached(f"test_{scene}", lambda: S.prepare(model, test_ds))}
        if scene == "snu-asri":
            tests["snu-asri-ood"] = E.cached("test_snu-asri-ood", lambda: S.prepare(model, load("snu-asri-ood")))
        lib = E.cached(f"lib_{scene}_{LAM_LIB}", lambda: S.build_library(scene, model, ridge=ridge, lam_frac=LAM_LIB))
        Xv, Fv, Hv = arrays(val)
        print(f"\n[{scene}] train pairs={len(td.P)} ms-windows={len(td.X0)}  n_val={len(val)}  "
              f"prep {time.time()-t0:.0f}s", flush=True)

        # ---- fit exception operators for every (q, reg) and pick on val ----
        rules = {q: X.ExceptionRule.fit(td.Hp, q) for q in Q_HI}
        for q, r in rules.items():
            print(f"  q={q}: turn>{r.turn_deg:.1f} deg, |dv|>{r.acc_dv:.3f} m/s, "
                  f"flagged val {100*np.mean(r.label(Hv) != X.NORMAL):.1f}%")
        cand = {"exc_edmd": [], "exc_ms": []}
        for q in Q_HI:
            for lam in EDMD_LAMS:
                ops, info = X.fit_edmd(td, rules[q], K0, ridge, lam)
                cand["exc_edmd"].append((dict(q=q, lam=lam), ops, info))
            for beta in MS_BETAS:
                ops, info = X.fit_ms(td, rules[q], K0, beta)
                cand["exc_ms"].append((dict(q=q, beta=beta), ops, info))
        chosen = {}
        for name, lst in cand.items():
            def vscore(item):
                cfg, ops, _ = item
                return X.errors(X.predict(Xv, rules[cfg["q"]].label(Hv), K0, ops), Fv)[0].mean()
            chosen[name] = min(lst, key=vscore)
            cfg, _, info = chosen[name]
            print(f"  {name}: chosen {cfg}  ops: " +
                  ", ".join(f"{X.CLASSES[c]} n={i['n']} dK={i.get('dK', 0):.3f}" for c, i in info.items()), flush=True)

        # ---- test ----
        rep_rule = X.ExceptionRule.fit(td.Hp, REPORT_Q)
        for tname, ts in tests.items():
            Xt, Ft, Ht = arrays(ts)
            cls = rep_rule.label(Ht); sl = cls != X.NORMAL
            preds = {"static": X.predict(Xt, cls, K0),
                     "always": always_pred(ts, K0, ETA),
                     "lib": lib_pred(Xt, Ht, lib)}
            cover = {}
            for name in ["exc_edmd", "exc_ms"]:
                cfg, ops, _ = chosen[name]
                lab = rules[cfg["q"]].label(Ht)
                p = X.predict(Xt, lab, K0, ops)
                assert np.allclose(p[lab == X.NORMAL], preds["static"][lab == X.NORMAL]), "normal samples must equal static"
                preds[name] = p; cover[name] = float(np.mean(lab != X.NORMAL))
            res = {}
            per_sample = {name: X.errors(p, Ft)[0] for name, p in preds.items()}
            np.savez(HERE / f"per_sample_{tname}.npz", slice=sl, cls=cls, **per_sample)
            print(f"  -> {tname}  n={len(ts)}  exception slice (q={REPORT_Q}): n={int(sl.sum())} "
                  f"({100*sl.mean():.1f}%; turn {int(((cls==0)|(cls==1)).sum())}, speed {int(((cls==2)|(cls==3)).sum())})")
            print(f"    {'method':<10}{'ADE':>8}{'FDE':>8}{'p90':>8}{'p95':>8} | {'excADE':>8}{'excFDE':>8}{'turn':>8}{'speed':>8}{'normal':>8}  cover")
            for name, p in preds.items():
                a, f = X.errors(p, Ft)
                r = summarize(a, f, sl, cls); res[name] = r
                cv = f"{100*cover[name]:.1f}%" if name in cover else ""
                print(f"    {name:<10}{r['ade']:8.4f}{r['fde']:8.4f}{r['p90']:8.4f}{r['p95']:8.4f} | "
                      f"{r['exc_ade']:8.4f}{r['exc_fde']:8.4f}{r['turn_ade']:8.4f}{r['speed_ade']:8.4f}{r['normal_ade']:8.4f}  {cv}", flush=True)
            res["_chosen"] = {k: v[0] for k, v in chosen.items()}
            res["_n"] = len(ts); res["_n_exc"] = int(sl.sum())
            allres[tname] = res
        print(f"  ({time.time()-t0:.0f}s)", flush=True)
    out = HERE / f"results_{'_'.join(scenes)}.json"
    out.write_text(json.dumps(allres, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
