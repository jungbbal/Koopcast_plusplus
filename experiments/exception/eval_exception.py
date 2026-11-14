"""
Exception-handling evaluation: static K0 by default, dedicated operators only for
extreme manoeuvres. Selection on val, scores on test.

    python eval_exception.py zara1 eth hotel univ zara2 snu-asri
    python eval_exception.py ... --no-baselines      # KoopCast++ variants only

At the end a comparison table is printed for every test scene:
KoopCast++ (exc_ms), KoopCast++ (static), KoopCast++ (lib), SocialVAE, Social-STGCNN, Constant Velocity,
EigenTrajectory, Trajectron++  x  ADE / FDE / excADE / excFDE
(exc* = exception slice: sharp turn or hard accel/decel in the last 8 observed steps,
q = REPORT_Q of the scene's training windows; identical slice for every model).
Baselines use the vendored checkpoints of the same scene, single deterministic
prediction, same targets (matched by scene/agent/start frame). Trajectron++ is scored
through its own Environment (repo's evaluate_trajectron path; ETH/UCY only) and its
exception slice is computed from the same 8-step history with the same rule.

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
from kpp.data import load, ETHUCY_SCENES
from kpp.data.windows import scene_windows
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


# --------------------------------------------------------------------------- #
# Baselines on the same targets
# --------------------------------------------------------------------------- #
BASELINES = [("socialvae", "SocialVAE"), ("stgcnn", "Social-STGCNN"), ("eigen", "EigenTrajectory")]
TABLE_ROWS = ["KoopCast++ (exc_ms)", "KoopCast++ (static)", "KoopCast++ (lib)", "SocialVAE", "Social-STGCNN",
              "Constant Velocity", "EigenTrajectory", "Trajectron++"]


def target_keys(ds):
    """(scene_id, agent_id, start_frame) per target, in the order S.prepare produces samples."""
    return [(int(sw.scene_id), int(t.agent_id), int(t.start_frame))
            for sw in scene_windows(ds, full_history=True) for t in sw.targets]


def baseline_preds(name, model_scene, ds, keys, tag):
    """(N, 12, 2) predictions aligned with keys (NaN where the model returned nothing)."""
    p = OWN_CACHE / f"baseline_{name}_{tag}.pkl"
    if p.exists():
        out = pickle.loads(p.read_bytes())
    else:
        from kpp.baselines import make_baseline
        pred = make_baseline(name, model_scene, pred_len=12, history_len=8, dt=0.4)
        out = {}
        for sw in scene_windows(ds):
            if not sw.targets:
                continue
            P = pred.predict_scene(sw.obs)
            for t in sw.targets:
                if t.agent_id in P:
                    out[(int(sw.scene_id), int(t.agent_id), int(t.start_frame))] = np.asarray(P[t.agent_id], float)[:12]
        p.write_bytes(pickle.dumps(out))
    nan = np.full((12, 2), np.nan)
    return np.stack([out.get(k, nan) for k in keys])


def trajectron_scores(scene, rule):
    """Per-sample ADE/FDE of pretrained Trajectron++ (repo path) + exception flag from its own history."""
    p = OWN_CACHE / f"trajectron_{scene}.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    import kpp.baselines.trajectron_eval as TE
    env = TE.load_split(scene, "test")
    stg, _ = TE.load_trajectron(str(TE.DEFAULT_MODEL_ROOT / TE.MODEL_DIRS[scene]), env, ts=100, device="cpu")
    ade, fde, hist = [], [], []
    for sc in env.scenes:
        for t in range(sc.timesteps):
            preds = stg.predict(sc, np.array([t]), 12, num_samples=1, min_history_timesteps=7,
                                min_future_timesteps=12, z_mode=True, gmm_mode=True, full_dist=False)
            for _t, per_node in (preds or {}).items():
                for node, pr in per_node.items():
                    pr = np.asarray(pr, float).reshape(-1, 12, 2)[0]
                    gt = np.asarray(node.get(np.array([t + 1, t + 12]), TE._STATE), float)
                    if gt.shape[0] != 12 or not np.isfinite(gt).all():
                        continue
                    d = np.linalg.norm(pr - gt, axis=-1)
                    ade.append(d.mean()); fde.append(d[-1])
                    hist.append(np.asarray(node.get(np.array([t - 7, t]), TE._STATE), float))
    H = np.stack(hist)
    ok = np.isfinite(H).all(axis=(1, 2))
    lab = np.full(len(H), X.NORMAL); lab[ok] = rule.label(H[ok])
    r = dict(ade=np.asarray(ade), fde=np.asarray(fde), slice=lab != X.NORMAL)
    p.write_bytes(pickle.dumps(r))
    return r


def table_row(ade, fde, sl):
    ok = np.isfinite(ade)
    return dict(ade=float(ade[ok].mean()), fde=float(fde[ok].mean()),
                exc_ade=float(ade[ok & sl].mean()) if (ok & sl).any() else float("nan"),
                exc_fde=float(fde[ok & sl].mean()) if (ok & sl).any() else float("nan"),
                n=int(ok.sum()), n_exc=int((ok & sl).sum()))


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
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    with_baselines = "--no-baselines" not in sys.argv
    scenes = args or ["zara1"]
    allres = {}
    compare = {}
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
            # ---- comparison with baselines (same targets, same exception slice) ----
            if with_baselines and tname != "snu-asri-ood":
                tds = test_ds if tname == scene else load("snu-asri-ood")
                keys = target_keys(tds)
                assert len(keys) == len(ts), "target order mismatch"
                rows = {}
                for m, label in [("exc_ms", "KoopCast++ (exc_ms)"), ("static", "KoopCast++ (static)"),
                                 ("lib", "KoopCast++ (lib)")]:
                    a, f = X.errors(preds[m], Ft); rows[label] = table_row(a, f, sl)
                cvp = Ht[:, -1][:, None] + np.arange(1, 13)[None, :, None] * (Ht[:, -1] - Ht[:, -2])[:, None]
                a, f = X.errors(cvp, Ft); rows["Constant Velocity"] = table_row(a, f, sl)
                for name, label in BASELINES:
                    bp = baseline_preds(name, scene, tds, keys, tname)
                    e = np.linalg.norm(bp - Ft, axis=2)
                    rows[label] = table_row(e.mean(1), e[:, -1], sl)
                if scene in ETHUCY_SCENES:
                    tr = trajectron_scores(scene, rep_rule)
                    rows["Trajectron++"] = table_row(tr["ade"], tr["fde"], tr["slice"])
                compare[tname] = rows
                print(f"    [compared with baselines: {', '.join(r for r in rows if not r.startswith('KoopCast'))}]", flush=True)
            res["_chosen"] = {k: v[0] for k, v in chosen.items()}
            res["_n"] = len(ts); res["_n_exc"] = int(sl.sum())
            allres[tname] = res
        print(f"  ({time.time()-t0:.0f}s)", flush=True)
    out = HERE / f"results_{'_'.join(scenes)}.json"
    out.write_text(json.dumps(allres, indent=1))
    print("saved", out)
    if compare:
        print_comparison(compare)
        (HERE / f"comparison_{'_'.join(scenes)}.json").write_text(json.dumps(compare, indent=1))
        lines = ["scene,model,ADE,FDE,excADE,excFDE,n,n_exc"]
        for sc, rows in compare.items():
            for m in TABLE_ROWS:
                if m in rows:
                    r = rows[m]
                    lines.append(f"{sc},{m},{r['ade']:.4f},{r['fde']:.4f},{r['exc_ade']:.4f},{r['exc_fde']:.4f},{r['n']},{r['n_exc']}")
        (HERE / f"comparison_{'_'.join(scenes)}.csv").write_text("\n".join(lines) + "\n")
        print("saved", HERE / f"comparison_{'_'.join(scenes)}.csv")


def print_comparison(compare):
    """One block per scene: model x ADE / FDE / excADE / excFDE (best in column marked *)."""
    print("\n" + "=" * 78)
    print("Comparison with baselines  (exc* = exception slice, q=%.2f, same for all models)" % REPORT_Q)
    print("=" * 78)
    for sc, rows in compare.items():
        best = {c: min(r[c] for r in rows.values() if np.isfinite(r[c])) for c in ("ade", "fde", "exc_ade", "exc_fde")}
        any_row = next(iter(rows.values()))
        print(f"\n[{sc}]  n={any_row['n']}  exception slice n={any_row['n_exc']}")
        print(f"  {'model':<22}{'ADE':>10}{'FDE':>10}{'excADE':>10}{'excFDE':>10}")
        for m in TABLE_ROWS:
            if m not in rows:
                print(f"  {m:<22}{'-':>10}{'-':>10}{'-':>10}{'-':>10}   (not available for this scene)")
                continue
            r = rows[m]
            cell = lambda c: (f"{r[c]:.4f}" + ("*" if np.isclose(r[c], best[c]) else " ")).rjust(10)
            note = "" if (r["n"], r["n_exc"]) == (any_row["n"], any_row["n_exc"]) else f"   (n={r['n']}, exc n={r['n_exc']})"
            print(f"  {m:<22}" + "".join(cell(c) for c in ("ade", "fde", "exc_ade", "exc_fde")) + note)


if __name__ == "__main__":
    main()
