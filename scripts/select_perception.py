"""Perception hyper-parameters by coordinate descent on the validation Cat-MRR (BPR, fixed kappa).

Usage:  python scripts/select_perception.py <dataset>
Output: config/params/<dataset>.json, outputs/param_closure_<dataset>.csv
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from xsage.descriptor import build_descriptor, membership_from_assign, GAMMA, DEPTH, N, BETA, H as H0
from xsage.selection import select_K, select_eps, SEED
from xsage.l2_comprehension import _assign, fit_rough_kmeans
from xsage.recommendation import ALPHA, fit_situation_biases_z, situational_nudge
from select_kappa import val_catmrr

KAPPA_FIX = 0.5
GRID = {"depth": [2, 3, 4], "n": [2, 3, 5], "gamma": [0.3, 0.4, 0.5, 0.6], "beta": [0.5, 0.7, 0.9], "H": [1, 2, 3]}
ALPHA_GRID = [10.0, 50.0, 100.0, 200.0]


def val_mrr(city, p, rng):
    """Build the descriptor with parameters p, re-select K and epsilon, score X-SAGE on validation (fixed kappa)."""
    D0 = build_descriptor(city, splits=("train", "val"), gamma=p["gamma"], depth=p["depth"],
                          n=p["n"], beta=p["beta"], h_hops=p["H"])
    vtr, vva = D0["vs"]["train"], D0["vs"]["val"]; icm = D0["icm"]; excl = D0["excl"]; sb = D0["sb"]
    cmt = D0["cmt"]; nmac = D0["n_macros"]; dfv = D0["ds"]["df_val"]
    K = select_K(vtr, int(D0["attractors"].sum()) + 2, rng); eps = select_eps(vtr, vva, K)
    fit = fit_rough_kmeans(vtr, K=K, eps=eps, seed=SEED, max_iter=80); z_tr = fit.core_label.astype(np.int64)
    _, k_va, comp, isb = _assign(vva, fit.prototypes, eps)
    mem = membership_from_assign(k_va, comp, isb, K); gam = (1. / np.maximum(comp.sum(1), 1).astype(np.float32))
    nudge = situational_nudge(mem, fit_situation_biases_z(z_tr, cmt, K, nmac, alpha=p["alpha"]))
    uv = dfv["u_idx"].values.astype(np.int64); iv = dfv["i_idx"].values.astype(np.int64)
    cm = val_catmrr(lambda idx: sb[uv[idx]], nudge, KAPPA_FIX, gam, uv, iv, icm, excl)
    return float(cm.mean()), K, eps


def main():
    city = sys.argv[1] if len(sys.argv) > 1 else "ml1m"; rng = np.random.default_rng(SEED)
    cur = {"gamma": GAMMA, "depth": DEPTH, "n": N, "beta": BETA, "H": H0, "alpha": ALPHA}
    rows = []
    print(f"=== {city}: perception parameters on validation (kappa={KAPPA_FIX}); start {cur}", flush=True)
    for param in ["depth", "n", "gamma", "beta", "H"]:
        best = (-1, cur[param])
        for v in GRID[param]:
            p = dict(cur); p[param] = v
            m, K, eps = val_mrr(city, p, rng)
            rows.append({"phase": "select", "param": param, "value": v, "val_CatMRR": round(m, 5), "K": K, "eps": eps})
            print(f"    {param}={v}: validation Cat-MRR={m:.5f} (K={K} eps={eps})", flush=True)
            if m > best[0]: best = (m, v)
        cur[param] = best[1]
        print(f"  -> {param} = {best[1]} (validation Cat-MRR={best[0]:.5f})", flush=True)
    for a in ALPHA_GRID:   # sensitivity only: alpha is not written to the configuration
        p = dict(cur); p["alpha"] = a; m, K, eps = val_mrr(city, p, rng)
        rows.append({"phase": "sensitivity", "param": "alpha", "value": a, "val_CatMRR": round(m, 5), "K": K, "eps": eps})
        print(f"    alpha={a}: validation Cat-MRR={m:.5f}", flush=True)

    (ROOT / "outputs").mkdir(parents=True, exist_ok=True); (ROOT / "config" / "params").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(ROOT / "outputs" / f"param_closure_{city}.csv", index=False)
    sel = {k: cur[k] for k in ("gamma", "depth", "n", "beta", "H")}
    open(ROOT / "config" / "params" / f"{city}.json", "w").write(json.dumps(sel, indent=2) + "\n")
    print(f"-> config/params/{city}.json {sel}; outputs/param_closure_{city}.csv", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
