"""
Val-tuned comparison of adaptation strategies (selection on val, scored on test).

    python eval_selective.py zara1 eth hotel univ zara2 snu-asri

snu-asri-ood is scored automatically after snu-asri with the configs chosen on the
in-distribution lobby val (there is no OOD val).
"""
import sys, time, json, pickle, itertools, pathlib, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
import selective as S
from kpp.data import load_ethucy, load, ETHUCY_SCENES
from kpp.data.loaders import load_taa_npy
from kpp.predictors import KoopCastPP

ETAS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]
QS = [0.9, 0.95, 0.99]
LAMS = [1e-6, 1e-5, 1e-4, 1e-3]
RIDGE = {"snu-asri": 0.1}
HERE = pathlib.Path(__file__).parent
CACHE = HERE / "cache"; CACHE.mkdir(exist_ok=True)
SNU_VAL = S.REPO / "data/raw/snu-asri-train/val_1.npy"


def splits(scene):
    if scene in ETHUCY_SCENES:
        return load_ethucy(scene, "val"), load_ethucy(scene, "test")
    return load_taa_npy(SNU_VAL, title="snu-asri/val", fps=2.5), load("snu-asri")


def cached(name, fn):
    p = CACHE / f"{name}.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    v = fn(); p.write_bytes(pickle.dumps(v)); return v


def configs(method):
    if method == "static":   return [dict()]
    if method == "always":   return [dict(eta=e, safety=False) for e in ETAS]
    if method == "always_safe": return [dict(eta=e) for e in ETAS]
    if method == "lib":      return [dict(lam=l) for l in LAMS]
    if method == "gate0":    return [dict(eta=e, q=q) for e, q in itertools.product(ETAS, QS)]
    if method == "lib_gate": return [dict(eta=e, q=q, lam=l) for e, q, l in itertools.product(ETAS, QS, LAMS)]


METHODS = ["static", "always", "always_safe", "lib", "gate0", "lib_gate"]


def score(samples, libs, method, cfg):
    cfg = dict(cfg); lam = cfg.pop("lam", LAMS[-1])
    m = "always" if method == "always_safe" else method
    return S.run(samples, libs[lam], m, **cfg)


def main():
    scenes = sys.argv[1:] or ["zara1"]
    results = {}
    for scene in scenes:
        t0 = time.time()
        model = KoopCastPP(scene)
        libs = {l: cached(f"lib_{scene}_{l}", lambda l=l: S.build_library(
                    scene, model, ridge=RIDGE.get(scene, 1e-4), lam_frac=l)) for l in LAMS}
        rule = libs[LAMS[0]].rule
        val_ds, test_ds = splits(scene)
        val = cached(f"val_{scene}", lambda: S.prepare(model, val_ds))
        test = cached(f"test_{scene}", lambda: S.prepare(model, test_ds))
        S.label_samples(val, rule); S.label_samples(test, rule)
        tests = {scene: test}
        if scene == "snu-asri":
            ood = cached("test_snu-asri-ood", lambda: S.prepare(model, load("snu-asri-ood")))
            S.label_samples(ood, rule); tests["snu-asri-ood"] = ood
        print(f"\n[{scene}] n_val={len(val)} n_test={len(test)}  prep {time.time()-t0:.0f}s", flush=True)
        print("  primitive counts (train pairs):", {S.PRIMS[k]: v for k, v in libs[LAMS[0]].counts.items()})

        chosen = {}
        for method in METHODS:
            best = min(configs(method), key=lambda c: score(val, libs, method, c)[0])
            chosen[method] = best
        for name, ts in tests.items():
            results[name] = {}
            print(f"  -> {name}")
            print(f"    {'method':<13}{'ADE':>8}{'FDE':>8}{'upd%':>7}{'upd/tgt':>9}  config (val-selected)")
            for method in METHODS:
                a, f, u, k = score(ts, libs, method, chosen[method])
                results[name][method] = dict(ade=a, fde=f, upd_frac=u, upd_per_target=k, cfg=chosen[method])
                print(f"    {method:<13}{a:8.4f}{f:8.4f}{100*u:6.1f}%{k:9.2f}  {chosen[method]}", flush=True)
        print(f"  ({time.time()-t0:.0f}s)")
    out = HERE / f"results_{'_'.join(scenes)}.json"
    out.write_text(json.dumps(results, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
