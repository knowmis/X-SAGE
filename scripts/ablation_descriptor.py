"""Descriptor ablation on BPR: fused, context-only and intent-only descriptor; fused vs each half, user-level CI, mark.

Usage:  python scripts/ablation_descriptor.py [dataset ...]   (default: the five datasets)
Output: outputs/ablation.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import run_pipeline as rr
from xsage.decision import mark
from xsage.descriptor import build_descriptor, membership_from_assign
from xsage.selection import select_K, select_eps
from xsage.l2_comprehension import _assign, fit_rough_kmeans
from xsage.recommendation import ALPHA, fit_situation_biases_z, situational_nudge

VARIANTS = ["joint", "ctx", "intent"]


def run(city, bk="B_blind"):
    csvb = pd.read_csv(ROOT / "config" / f"hparams_{city}.csv")

    def kget(seed):
        return float(csvb[(csvb.seed == seed) & (csvb.backbone == bk) & (csvb.method == "SIT")]["kstar"].iloc[0])

    D0 = build_descriptor(city, splits=("train", "val", "test"))
    ds = D0["ds"]; nmac = D0["n_macros"]; icm = D0["icm"]; excl = D0["excl"]; sb = D0["sb"]; cmt = D0["cmt"]
    G1 = D0["G1"].astype(np.float32)
    vtr, vva, vte = D0["vs"]["train"], D0["vs"]["val"], D0["vs"]["test"]
    A = vtr.shape[1] - nmac   # context dimensions; the remaining K_mac are the intent vector
    sl = {"joint": slice(None), "ctx": slice(0, A), "intent": slice(A, None)}
    dft = ds["df_test"]; ute = dft["u_idx"].values.astype(np.int64); ite = dft["i_idx"].values.astype(np.int64)
    sfn = (lambda idx, u=ute: sb[u[idx]])
    eB = rr.per_request_eval(sfn, None, 0., np.ones(len(ute), np.float32), ute, ite, icm, excl, G1, nmac)
    base = rr.macro_msupp(eB["cm"], eB["tm"], nmac, rr.MSUPP)
    per = {v: [] for v in VARIANTS}; Ks = {v: [] for v in VARIANTS}; cm42 = {}
    for seed in rr.SEEDS:
        rng = np.random.default_rng(seed); kap = kget(seed)
        for v in VARIANTS:
            vt, vv, ve = vtr[:, sl[v]], vva[:, sl[v]], vte[:, sl[v]]
            K = select_K(vt, int(D0["attractors"].sum()) + 2, rng); eps = select_eps(vt, vv, K)
            fit = fit_rough_kmeans(vt, K=K, eps=eps, seed=seed, max_iter=80); z_tr = fit.core_label.astype(np.int64)
            _, kte, compte, isbte = _assign(ve, fit.prototypes, eps)
            mem = membership_from_assign(kte, compte, isbte, K); gam = 1.0 / np.maximum(compte.sum(1), 1).astype(np.float32)
            nudge = situational_nudge(mem, fit_situation_biases_z(z_tr, cmt, K, nmac, alpha=ALPHA))
            eS = rr.per_request_eval(sfn, nudge, kap, gam, ute, ite, icm, excl, G1, nmac)
            per[v].append(rr.macro_msupp(eS["cm"], eS["tm"], nmac, rr.MSUPP)); Ks[v].append(K)
            if seed == 42: cm42[v] = eS["cm"]
        print(f"  [{city}] seed {seed} (K joint/ctx/intent = {Ks['joint'][-1]}/{Ks['ctx'][-1]}/{Ks['intent'][-1]})", flush=True)
    print(f"\n### {city} (BPR): macro-Cat-MRR, backbone = {base:.5f}")
    print(f"{'variant':8s} {'X-SAGE (mean+-sd)':20s} {'delta vs backbone':18s} {'seeds delta>0':14s} {'mean K':6s}")
    for v in VARIANTS:
        a = np.array(per[v]); d = a - base
        print(f"{v:8s} {a.mean():.5f}+-{a.std(ddof=1):.5f}   {d.mean():+.5f}            {int((d > 0).sum())}/5            {np.mean(Ks[v]):.1f}")
    j = np.array(per["joint"]); tm = eB["tm"]; blk = rr.user_blocks(ute)
    better = "ctx" if np.mean(per["ctx"]) >= np.mean(per["intent"]) else "intent"
    rows = []
    for v in ("ctx", "intent"):
        d = j - np.array(per[v]); rng = np.random.default_rng(2024); bd = np.empty(rr.BOOT)
        for b in range(rr.BOOT):
            ix = rr.cluster_sample(*blk, rng)
            bd[b] = rr.macro_msupp(cm42["joint"][ix], tm[ix], nmac, rr.MSUPP) - rr.macro_msupp(cm42[v][ix], tm[ix], nmac, rr.MSUPP)
        lo, hi = np.percentile(bd, [2.5, 97.5])
        c = dict(d=float(d.mean()), lo=float(lo), hi=float(hi), seeds=int((d > 0).sum()))
        rows.append(dict(dataset=city, contrast=f"joint-{v}", better_half=(v == better), joint=float(j.mean()),
                         other=float(np.mean(per[v])), delta=round(c["d"], 5), sd_seed=round(float(d.std(ddof=1)), 5),
                         ci_lo=round(c["lo"], 5), ci_hi=round(c["hi"], 5), seeds=c["seeds"], mark=mark(c, "macroCatMRR"),
                         K=f"{Ks['joint'][0]}/{Ks['ctx'][0]}/{Ks['intent'][0]}"))
        print(f"joint - {v:6s}: {c['d']:+.5f} (sd over seeds {d.std(ddof=1):.5f}), user-level 95% CI (seed 42) "
              f"[{lo:+.5f}, {hi:+.5f}], seeds > 0: {c['seeds']}/5{'  <- better half' if v == better else ''}")
    return rows


def main():
    rows = []
    for c in sys.argv[1:] or rr.PRIMARY_DATASETS:
        rows += run(c)
    df = pd.DataFrame(rows); out = ROOT / "outputs"; out.mkdir(exist_ok=True)
    df.to_csv(out / "ablation.csv", index=False)
    print(f"-> {out / 'ablation.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
