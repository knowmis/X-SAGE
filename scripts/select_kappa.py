"""kappa per dataset, backbone, method and seed from KAPPA_GRID on the validation Cat-MRR (strict rank).

Usage:  python scripts/select_kappa.py <dataset> [--skip-focal] [--out DIR]
Output: DIR/hparams_<dataset>.csv (default DIR = config/)
"""
import argparse, copy, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from xsage import data as D
from xsage.backbones import default_device
from xsage.descriptor import build_descriptor, membership_from_assign
from xsage.selection import select_K_eps
from xsage.l2_comprehension import _assign, fit_rough_kmeans
from xsage.recommendation import ALPHA, fit_situation_biases_z, situational_nudge, combine_scores
from xsage.backbones.fm import feats_from_df, score_test, ContextAwareFM, FeatureSpec, train_b_full

K_TOP, BATCH = 20, 1024
KAPPA_GRID = [0.05, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00]
BF_GRID = [(32, 5e-3), (64, 5e-3)]; BF_EPOCHS, BF_CKPT, VAL_SUB = 9, 3, 12000
SEEDS = [42, 43, 44, 45, 46]
BACKBONES = ["B_blind", "B_full", "EASE", "SASRec", "FPMC", "DeepFM", "AFM"]
METHODS = ["SIT", "Steck-b"]


def val_catmrr(bb_rows, nudge, kappa, gam, u, i_t, icm, excl):
    """Per-request Cat-MRR@K_TOP (strict rank) of the re-ranked list; nudge=None scores the backbone."""
    tm = icm[i_t]; n = len(u); cm = np.zeros(n)
    for bs in range(0, n, BATCH):
        be = min(n, bs + BATCH); idx = np.arange(bs, be); u_b = u[idx]
        S = bb_rows(idx).astype(np.float32, copy=True)
        if nudge is not None:
            S = combine_scores(S, nudge[idx], gam[idx], icm, kappa)
        for j in range(be - bs):
            cc = excl.indices[excl.indptr[int(u_b[j])]:excl.indptr[int(u_b[j]) + 1]]
            if len(cc): S[j, cc] = -np.inf
        part = np.argpartition(-S, K_TOP - 1, axis=1)[:, :K_TOP]
        order = np.argsort(-np.take_along_axis(S, part, 1), axis=1); topk = np.take_along_axis(part, order, 1)
        match = icm[topk] == tm[idx, None]; has = match.any(1); first = np.where(has, match.argmax(1) + 1, 0)
        cm[idx] = np.where(first > 0, 1. / np.maximum(first, 1), 0.)
    return cm


def focal_val_scores(ds, dfv, icm, excl, nmac, seed, dev):
    """Train the focal FM (configuration chosen on a validation subsample) and score validation."""
    nU = int(ds["n_users"]); nI = int(ds["n_items"])
    dfa = pd.concat([ds["df_train"], ds["df_val"], ds["df_test"]], ignore_index=True)
    icmF = (dfa.groupby("i_idx")["cat_macro"].first().map(ds["macro_to_idx"]).reindex(np.arange(nI), fill_value=0).values.astype(np.int64))
    ftr = feats_from_df(ds["df_train"], icmF, nmac); fva = feats_from_df(dfv, icmF, nmac)
    mask = (ds["urm_train"] + ds["urm_val"]).tocsr(); mask.data[:] = 1.
    sub = np.random.default_rng(seed).choice(len(dfv), min(VAL_SUB, len(dfv)), replace=False)
    fvs = {k: v[sub] for k, v in fva.items()}
    us = dfv["u_idx"].values.astype(np.int64)[sub]; its = dfv["i_idx"].values.astype(np.int64)[sub]
    spec = FeatureSpec(n_users=nU, n_items=nI, n_macros=nmac, n_fine=1, n_geo=0, n_intent_last=nmac)
    best = {"val": -1, "st": None, "cfg": None}
    for emb, lr in BF_GRID:
        torch.manual_seed(seed); mdl = ContextAwareFM(spec, d=emb).to(dev)
        for st in range(0, BF_EPOCHS, BF_CKPT):
            train_b_full(mdl, ftr, mask, icmF, np.zeros(nI, np.int64), dev, lr=lr, n_epochs=BF_CKPT, verbose=False)
            sc = score_test(mdl, fvs, spec, icmF, dev)
            vm = val_catmrr(lambda idx, S=sc: S[idx], None, 0., None, us, its, icm, excl).mean()
            if vm > best["val"]: best = {"val": vm, "st": copy.deepcopy(mdl.state_dict()), "cfg": (emb, lr, st + BF_CKPT)}
    bm = ContextAwareFM(spec, d=best["cfg"][0]).to(dev); bm.load_state_dict(best["st"])
    return score_test(bm, fva, spec, icmF, dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--skip-focal", action="store_true", help="do not train the focal FM (its rows are omitted)")
    ap.add_argument("--out", default=str(ROOT / "config"))
    a = ap.parse_args(); city = a.dataset
    dev = torch.device(default_device()); bdir = D.DEFAULT_DATA_ROOT / "data" / city / "backbone"
    D0 = build_descriptor(city, splits=("train", "val"))
    ds = D0["ds"]; nmac = D0["n_macros"]; icm = D0["icm"]; excl = D0["excl"]; sb = D0["sb"]; cmt = D0["cmt"]
    vtr, vva = D0["vs"]["train"], D0["vs"]["val"]
    K, eps = select_K_eps(D0)
    print(f"[{city}] K={K} eps={eps} device={dev}", flush=True)
    dfv = ds["df_val"]; uv = dfv["u_idx"].values.astype(np.int64); iv = dfv["i_idx"].values.astype(np.int64)
    nU = int(ds["n_users"]); umac = ds["df_train"]["u_idx"].values.astype(np.int64)
    nudge_stb = fit_situation_biases_z(umac, cmt, nU, nmac, alpha=ALPHA)[uv]
    rows = []
    for seed in SEEDS:
        fit = fit_rough_kmeans(vtr, K=K, eps=eps, seed=seed, max_iter=80); z_tr = fit.core_label.astype(np.int64)
        _, kva, comp, isb = _assign(vva, fit.prototypes, eps)
        mem = membership_from_assign(kva, comp, isb, K); gam = 1. / np.maximum(comp.sum(1), 1).astype(np.float32)
        nudge = {"SIT": situational_nudge(mem, fit_situation_biases_z(z_tr, cmt, K, nmac, alpha=ALPHA)), "Steck-b": nudge_stb}
        for bk in BACKBONES:
            if bk == "B_blind":
                vfn = (lambda idx, u=uv: sb[u[idx]])
            elif bk == "B_full":
                if a.skip_focal: continue
                sv = focal_val_scores(ds, dfv, icm, excl, nmac, seed, dev); vfn = (lambda idx, S=sv: S[idx])
            else:
                fu = bdir / f"{bk}.scores_user.npy"; fv = bdir / f"{bk}.scores_val.npy"
                if fu.exists(): M = np.load(fu, mmap_mode="r"); vfn = (lambda idx, M=M, u=uv: M[u[idx]])
                elif fv.exists(): Mv = np.load(fv, mmap_mode="r"); vfn = (lambda idx, Mv=Mv: Mv[idx])
                else:
                    print(f"  [{city}] no validation scores for {bk}: skipped", flush=True); continue
            rows.append(dict(seed=seed, backbone=bk, method="BASE", kstar=""))
            for m in METHODS:
                kstar = max(KAPPA_GRID, key=lambda kap: val_catmrr(vfn, nudge[m], kap, gam, uv, iv, icm, excl).mean())
                rows.append(dict(seed=seed, backbone=bk, method=m, kstar=kstar))
            print(f"  [{city}] seed {seed} {bk}: kappa* SIT={rows[-2]['kstar']} Steck-b={rows[-1]['kstar']}", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"hparams_{city}.csv", index=False)
    print(f"-> {out / f'hparams_{city}.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
