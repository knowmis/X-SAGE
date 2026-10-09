"""X-SAGE head and evaluation on the datasets: five seeds, user-level bootstrap on seed 42, Holm on macro-Cat-MRR, marks.

Usage:  python scripts/run_pipeline.py [dataset ...]   (no argument: the five datasets, as the Holm families require)
Output: outputs/results.csv, outputs/selection.csv, outputs/cache/raw_<dataset>.npz
"""
import sys, copy, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import stats
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from xsage import data as D
from xsage.backbones import default_device
from xsage.decision import mark
from xsage.descriptor import build_descriptor, membership_from_assign
from xsage.selection import select_K_eps
from xsage.l2_comprehension import _assign, fit_rough_kmeans
from xsage.recommendation import ALPHA, fit_situation_biases_z, situational_nudge, combine_scores
from xsage.backbones.fm import feats_from_df, score_test
from xsage.backbones.fm import ContextAwareFM, FeatureSpec, train_b_full

KTOP, KSAVE, BOOT, TOST_M = 20, 50, 1500, 0.005   # KTOP = reported K; KSAVE = cached top-list depth (any @K up to KSAVE is re-derivable)
MSUPP = int(os.environ.get("MSUPP", "20"))        # min-support |R_c| >= MSUPP for macro-Cat-MRR
BK = ["B_blind", "B_full", "EASE", "DeepFM", "AFM", "FPMC", "SASRec"]   # B_blind = BPR, B_full = focal FM
METHODS = ["BASE", "SIT", "Steck-b"]              # BASE = backbone, SIT = X-SAGE, Steck-b = static per-user profile
PRIMARY_DATASETS = ["ml1m", "nyc_tist", "saopaulo", "yelp", "kuairand"]   # the datasets entering the Holm families
FOCAL_BACKBONE = "B_full"                          # focal backbone, declared a priori
SEEDS = [42, 43, 44, 45, 46]
CATMET = ["CatMRR", "CatNDCG", "macroCatMRR"]
ACCMET = ["MRR", "HR5", "HR10", "HR20", "NDCG5", "NDCG10", "NDCG20"]   # primary: averaged per request
ACCMET_U = [m + "_u" for m in ACCMET]                                  # secondary: averaged per user
EXPMET = ["Coverage", "Gini", "LT20"]
COSTMET = ["JS"]
METRICS = CATMET + ACCMET + ACCMET_U + EXPMET + COSTMET
PERREQ = {"CatMRR", "CatNDCG", "LT20", "JS"}      # metrics that are means of a per-request quantity
HRK = {"HR5": 5, "HR10": 10, "HR20": 20}; NDK = {"NDCG5": 5, "NDCG10": 10, "NDCG20": 20}


def gini(c):
    x = np.sort(c.astype(np.float64)); n = len(x); s = x.sum()
    return 0. if s <= 0 else float((2 * np.sum(np.arange(1, n + 1) * x) / (n * s)) - (n + 1) / n)


def js(p, q, e=1e-9):
    """Row-wise Jensen-Shannon divergence (bits) between the user profile p and the top-K category histogram q."""
    p = p + e; p = p / p.sum(1, keepdims=True); q = q + e; q = q / q.sum(1, keepdims=True)
    m = .5 * (p + q); kl = lambda a, b: np.sum(a * np.log2(a / b), 1); return .5 * kl(p, m) + .5 * kl(q, m)


def _prefix(nI):
    """Prefix sums for the expected rank (McSherry-Najork): H[k]=sum 1/p, Hlog[k]=sum 1/log2(p+1), p=1..k."""
    p = np.arange(1, nI + 2)
    return np.concatenate([[0.], np.cumsum(1. / p)]), np.concatenate([[0.], np.cumsum(1. / np.log2(p + 1.))])


def _exp_trunc(rk, g, K, Hpre):
    """E[weight(p) * 1(p <= K)] over the tie block [rk, rk+g-1]; weights from Hpre."""
    b = rk - 1; L = len(Hpre) - 1
    return np.where(b < K, (Hpre[np.clip(np.minimum(b + g, K), 0, L)] - Hpre[np.clip(b, 0, L)]) / np.maximum(g, 1), 0.)


def _exp_mrr(rk, g, H):
    """E[1/p] over the tie block (untruncated)."""
    b = rk - 1; L = len(H) - 1
    return (H[np.clip(b + g, 0, L)] - H[np.clip(b, 0, L)]) / np.maximum(g, 1)


def _exp_hr(rk, g, K):
    """E[1(p<=K)] = fraction of the tie block within K."""
    return np.clip((K - (rk - 1)) / np.maximum(g, 1), 0., 1.)


def per_request_eval(scores_fn, nudge, kappa, gam, u, i_t, icm, excl, G1, nmac, pur=None):
    """Score, re-rank with the combiner of Eq. (14) and evaluate every request (raw arrays, then derive())."""
    n = len(u); tm = icm[i_t]; nI = len(icm)
    rk = np.zeros(n, np.int32); catrk = np.zeros(n, np.int32)
    g = np.zeros(n, np.int32); gc = np.zeros(n, np.int32); tk50 = np.zeros((n, KSAVE), np.int32)
    cat_cols = [np.where(icm == c)[0] for c in range(nmac)]   # items per category (vectorised category rank)
    for bs in range(0, n, 1024):
        be = min(n, bs + 1024); idx = np.arange(bs, be)
        S = np.asarray(scores_fn(idx)).astype(np.float32, copy=True)
        if nudge is not None:
            S = combine_scores(S, nudge[idx], gam[idx], icm, kappa)
        for j in range(be - bs):
            cc = excl.indices[excl.indptr[int(u[bs + j])]:excl.indptr[int(u[bs + j]) + 1]]
            if len(cc): S[j, cc] = -np.inf
        part = np.argpartition(-S, KSAVE - 1, axis=1)[:, :KSAVE]
        order = np.argsort(-np.take_along_axis(S, part, 1), axis=1); tk = np.take_along_axis(part, order, 1)
        tk50[idx] = tk
        s_t = S[np.arange(be - bs), i_t[idx]]
        rk[idx] = (S > s_t[:, None]).sum(1) + 1; g[idx] = (S == s_t[:, None]).sum(1)
        tmb = tm[idx]
        catmax = np.full((be - bs, nmac), -np.inf, np.float32)   # max score per category (loop over categories, not requests)
        for c in range(nmac):
            cc2 = cat_cols[c]
            if cc2.size: catmax[:, c] = S[:, cc2].max(1)
        best = catmax[np.arange(be - bs), tmb]
        catrk[idx] = (S > best[:, None]).sum(1) + 1; gc[idx] = (S == best[:, None]).sum(1)
    return derive(dict(rk=rk, catrk=catrk, g=g, gc=gc, tk50=tk50, u=u, tm=tm), icm, G1, nmac, nI, pur)


def derive(e, icm, G1, nmac, nI, pur=None):
    """Per-request metrics from the raw arrays, with the expected rank over ties, plus JS."""
    H, Hlog = _prefix(nI); e["H"] = H; e["Hlog"] = Hlog
    e["cm"] = _exp_trunc(e["catrk"], e["gc"], KTOP, H)        # Cat-MRR@KTOP (expected rank)
    e["cn"] = _exp_trunc(e["catrk"], e["gc"], KTOP, Hlog)     # Cat-NDCG@KTOP (expected rank)
    topk = e["tk50"][:, :KTOP]; e["topk"] = topk
    e["lt"] = G1[topk].mean(1)
    if pur is not None:
        n = len(e["u"]); q = np.zeros((n, nmac)); rows = np.repeat(np.arange(n), KTOP)
        np.add.at(q, (rows, icm[topk].ravel()), 1.)
        e["js"] = js(pur, q); e["pur"] = pur
    return e


def pu_mean(v, u):
    uq, inv = np.unique(u, return_inverse=True); s = np.zeros(len(uq)); c = np.zeros(len(uq)); np.add.at(s, inv, v); np.add.at(c, inv, 1); return s / c


def per_req_quantity(e, metric):
    """Per-request quantity with the expected rank; the _u variants share it and are averaged per user."""
    if metric.endswith("_u"): metric = metric[:-2]
    rk, g = e["rk"], e["g"]
    if metric == "MRR": return _exp_mrr(rk, g, e["H"])
    if metric in HRK: return _exp_hr(rk, g, HRK[metric])
    if metric in NDK: return _exp_trunc(rk, g, NDK[metric], e["Hlog"])
    return None


def macro_msupp(cm, tm, nmac, m=0):
    """macro-Cat-MRR with min-support: unweighted mean over the categories with |R_c| >= m present."""
    per = [cm[tm == c].mean() for c in range(nmac) if (tm == c).sum() >= max(m, 1)]
    return float(np.mean(per)) if per else 0.


def metrics_from(e, nmac, nI):
    u = e["u"]; out = {"CatMRR": e["cm"].mean(), "CatNDCG": e["cn"].mean(),
                       "macroCatMRR": macro_msupp(e["cm"], e["tm"], nmac, MSUPP), "LT20": e["lt"].mean()}
    if "js" in e: out["JS"] = float(e["js"].mean())
    for m in ACCMET: out[m] = float(per_req_quantity(e, m).mean())                  # per request
    for m in ACCMET_U: out[m] = float(pu_mean(per_req_quantity(e, m), u).mean())    # per user
    expo = np.bincount(e["topk"].ravel(), minlength=nI).astype(float)
    out["Coverage"] = float((expo > 0).mean()); out["Gini"] = gini(expo)
    return out


def user_blocks(u):
    """Indices of the requests grouped by user (blocks), for the user-level bootstrap."""
    uq, inv = np.unique(u, return_inverse=True); nu = len(uq)
    order = np.argsort(inv, kind="stable")
    st = np.searchsorted(inv[order], np.arange(nu))
    cnt = np.searchsorted(inv[order], np.arange(nu), side="right") - st
    return order, st, cnt


def cluster_sample(order, st, cnt, rng):
    """Draw users with replacement and return the indices of all their requests."""
    nu = len(cnt); ks = rng.integers(0, nu, nu); L = cnt[ks]
    off = np.concatenate([[0], np.cumsum(L)[:-1]])
    return order[np.repeat(st[ks], L) + (np.arange(int(L.sum())) - np.repeat(off, L))]


def boot_paired(eB, eS, metric, nmac, nI, rng):
    """One paired user-level bootstrap: intervals and SE of the two methods, p-value and interval of the difference."""
    blk = user_blocks(eB["u"])
    if metric in ACCMET:
        qB = per_req_quantity(eB, metric); qS = per_req_quantity(eS, metric)
        bB = np.empty(BOOT); bS = np.empty(BOOT)
        for b in range(BOOT):
            ix = cluster_sample(*blk, rng); bB[b] = qB[ix].mean(); bS[b] = qS[ix].mean()
    elif metric in ACCMET_U:
        u = eB["u"]; uq, inv = np.unique(u, return_inverse=True); nu = len(uq)
        qB = per_req_quantity(eB, metric); qS = per_req_quantity(eS, metric)
        sB = np.zeros(nu); cc = np.zeros(nu); np.add.at(sB, inv, qB); np.add.at(cc, inv, 1); pB = sB / cc
        sS = np.zeros(nu); np.add.at(sS, inv, qS); pS = sS / cc
        bB = np.empty(BOOT); bS = np.empty(BOOT)
        for b in range(BOOT):
            ix = rng.integers(0, nu, nu); bB[b] = pB[ix].mean(); bS[b] = pS[ix].mean()
    elif metric in PERREQ:
        k = {"CatMRR": "cm", "CatNDCG": "cn", "LT20": "lt", "JS": "js"}[metric]; aB, aS = eB[k], eS[k]
        bB = np.empty(BOOT); bS = np.empty(BOOT)
        for b in range(BOOT):
            ix = cluster_sample(*blk, rng); bB[b] = aB[ix].mean(); bS[b] = aS[ix].mean()
    elif metric == "macroCatMRR":
        aB, aS, tm = eB["cm"], eS["cm"], eB["tm"]; bB = np.empty(BOOT); bS = np.empty(BOOT)
        for b in range(BOOT):
            ix = cluster_sample(*blk, rng); bB[b] = macro_msupp(aB[ix], tm[ix], nmac, MSUPP); bS[b] = macro_msupp(aS[ix], tm[ix], nmac, MSUPP)
    else:  # Coverage / Gini
        tkB, tkS = eB["topk"], eS["topk"]; bB = np.empty(BOOT); bS = np.empty(BOOT)
        for b in range(BOOT):
            ix = cluster_sample(*blk, rng)
            eB_ = np.bincount(tkB[ix].ravel(), minlength=nI).astype(float); eS_ = np.bincount(tkS[ix].ravel(), minlength=nI).astype(float)
            if metric == "Coverage": bB[b] = (eB_ > 0).mean(); bS[b] = (eS_ > 0).mean()
            else: bB[b] = gini(eB_); bS[b] = gini(eS_)
    bd = bS - bB
    loB, hiB = np.percentile(bB, [2.5, 97.5]); loS, hiS = np.percentile(bS, [2.5, 97.5]); loD, hiD = np.percentile(bd, [2.5, 97.5])
    p = 2. * min((bd <= 0).mean(), (bd >= 0).mean())
    return (float(loB), float(hiB), float(bB.std())), (float(loS), float(hiS), float(bS.std())), float(min(p, 1.)), (float(loD), float(hiD))


def bfull_scores(ds, icm, excl, nmac, dev, seed):
    """Focal FM trained at each seed (embedding and epochs chosen on validation Cat-MRR); test scores."""
    nU = int(ds["n_users"]); nI = int(ds["n_items"])
    dfa = pd.concat([ds["df_train"], ds["df_val"], ds["df_test"]], ignore_index=True)
    icmF = (dfa.groupby("i_idx")["cat_macro"].first().map(ds["macro_to_idx"]).reindex(np.arange(nI), fill_value=0).values.astype(np.int64))
    ftr = feats_from_df(ds["df_train"], icmF, nmac); fva = feats_from_df(ds["df_val"], icmF, nmac); fte = feats_from_df(ds["df_test"], icmF, nmac)
    mask = (ds["urm_train"] + ds["urm_val"]).tocsr(); mask.data[:] = 1.
    uv = ds["df_val"]["u_idx"].values.astype(np.int64); iv = ds["df_val"]["i_idx"].values.astype(np.int64)
    G1d = np.zeros(nI, np.float32)
    spec = FeatureSpec(n_users=nU, n_items=nI, n_macros=nmac, n_fine=1, n_geo=0, n_intent_last=nmac); best = (-1, None, 64)
    for emb in (32, 64):
        torch.manual_seed(seed); mdl = ContextAwareFM(spec, d=emb).to(dev)
        for _ in range(3):
            train_b_full(mdl, ftr, mask, icmF, np.zeros(nI, np.int64), dev, lr=5e-3, n_epochs=3, verbose=False)
            sv = score_test(mdl, fva, spec, icmF, dev)
            vm = per_request_eval((lambda idx, S=sv: S[idx]), None, 0., np.ones(len(uv), np.float32), uv, iv, icm, excl, G1d, nmac)["cm"].mean()
            if vm > best[0]: best = (vm, copy.deepcopy(mdl.state_dict()), emb)
    mdl = ContextAwareFM(spec, d=best[2]).to(dev); mdl.load_state_dict(best[1])
    return score_test(mdl, fte, spec, icmF, dev)


def run_city(city, dev):
    csvb = pd.read_csv(ROOT / "config" / f"hparams_{city}.csv"); bdir = D.DEFAULT_DATA_ROOT / "data" / city / "backbone"
    per_seed = {(bk, m, met): [] for bk in BK for m in METHODS for met in METRICS}; seed42 = {}
    cache = {}    # raw cache to re-derive metrics/@K without re-running the evaluation
    # Descriptor, K and epsilon: once per dataset (fixed seed); the clustering and kappa change with the seed.
    D0 = build_descriptor(city, splits=("train", "val", "test"))
    ds = D0["ds"]; nmac = D0["n_macros"]; icm = D0["icm"]; excl = D0["excl"]; sb = D0["sb"]; cmt = D0["cmt"]; G1 = D0["G1"].astype(np.float32); nI = int(ds["n_items"])
    vtr, vte = D0["vs"]["train"], D0["vs"]["test"]
    K, eps = select_K_eps(D0)
    dft = ds["df_test"]; ute = dft["u_idx"].values.astype(np.int64); ite = dft["i_idx"].values.astype(np.int64)
    # Static per-user category profile (reference), estimated with the same estimator; no training
    nU = int(ds["n_users"]); umac = ds["df_train"]["u_idx"].values.astype(np.int64)
    b_z_user = fit_situation_biases_z(umac, cmt, nU, nmac, alpha=ALPHA); nudge_stb = b_z_user[ute]
    Pu = np.zeros((nU, nmac)); np.add.at(Pu, (umac, cmt), 1.); Pu[Pu.sum(1) == 0] = 1.; Pu /= Pu.sum(1, keepdims=True); pur = Pu[ute]
    selection = dict(dataset=city, K=K, eps=eps, D=int(vtr.shape[1]), n_macros=int(nmac), n_users=nU)
    print(f"  [{city}] K={K} eps={eps}", flush=True)
    for seed in SEEDS:
        fit = fit_rough_kmeans(vtr, K=K, eps=eps, seed=seed, max_iter=80); z_tr = fit.core_label.astype(np.int64)
        _, kte, compte, isbte = _assign(vte, fit.prototypes, eps)
        mem_te = membership_from_assign(kte, compte, isbte, K); gam_te = 1.0 / np.maximum(compte.sum(1), 1).astype(np.float32)
        b_z = fit_situation_biases_z(z_tr, cmt, K, nmac, alpha=ALPHA); nudge = situational_nudge(mem_te, b_z)
        bft = bfull_scores(ds, icm, excl, nmac, dev, seed)
        print(f"  [{city}] seed {seed}: focal FM trained", flush=True)
        for bk in BK:
            if bk == "B_blind": sfn = (lambda idx, u=ute: sb[u[idx]])
            elif bk == "B_full": sfn = (lambda idx: bft[idx])
            else:
                fu = bdir / f"{bk}.scores_user.npy"; ft = bdir / f"{bk}.scores_test.npy"
                if fu.exists(): M = np.load(fu, mmap_mode="r"); sfn = (lambda idx, M=M, u=ute: M[u[idx]])
                elif ft.exists(): Mt = np.load(ft, mmap_mode="r"); sfn = (lambda idx, Mt=Mt: Mt[idx])
                else:
                    print(f"  [{city}] no scores for {bk} in {bdir}: skipped", flush=True); continue

            def kget(mth, bk=bk, seed=seed):
                r = csvb[(csvb.seed == seed) & (csvb.backbone == bk) & (csvb.method == mth)]["kstar"]
                return float(r.iloc[0])
            ev = {"BASE": per_request_eval(sfn, None, 0., gam_te, ute, ite, icm, excl, G1, nmac, pur),
                  "SIT": per_request_eval(sfn, nudge, kget("SIT"), gam_te, ute, ite, icm, excl, G1, nmac, pur),
                  "Steck-b": per_request_eval(sfn, nudge_stb, kget("Steck-b"), gam_te, ute, ite, icm, excl, G1, nmac, pur)}
            for m in METHODS:
                mm = metrics_from(ev[m], nmac, nI)
                for met in METRICS: per_seed[(bk, m, met)].append(mm[met])
            if seed == 42: seed42[bk] = ev
            i16 = lambda a: np.clip(a, 0, 32767).astype(np.int16)
            for m in METHODS:
                e = ev[m]; cache[(bk, m, seed)] = (i16(e["rk"]), i16(e["catrk"]), i16(e["g"]), i16(e["gc"]), e["tk50"].astype(np.int16))
    shared = dict(u=ute.astype(np.int32), tm=icm[ite].astype(np.int16), G1=G1.astype(np.float32),
                  icm=icm.astype(np.int32), Pu=Pu.astype(np.float32), nI=np.int32(nI), nmac=np.int32(nmac))
    cdir = ROOT / "outputs" / "cache"; cdir.mkdir(parents=True, exist_ok=True)
    arrs = {}
    for (bk, m, s), (rk, catrk, g, gc, tk) in cache.items():
        pre = f"{bk}|{m}|{s}"; arrs[f"{pre}|rk"] = rk; arrs[f"{pre}|catrk"] = catrk; arrs[f"{pre}|g"] = g; arrs[f"{pre}|gc"] = gc; arrs[f"{pre}|tk50"] = tk
    for k, v in shared.items(): arrs[f"_shared|{k}"] = v
    np.savez_compressed(cdir / f"raw_{city}.npz", **arrs); print(f"  [{city}] raw cache saved: outputs/cache/raw_{city}.npz", flush=True)
    return aggregate(per_seed, seed42, city, nmac, nI), selection


def aggregate(per_seed, seed42, city, nmac, nI):
    """CSV rows per (backbone, metric, method): five-seed mean, bootstrap CI, contrasts L1 and L2, mark."""
    rng = np.random.default_rng(2024); rows = []
    for bk in seed42:   # only the backbones actually run
        ev = seed42[bk]; m42 = {m: metrics_from(ev[m], nmac, nI) for m in METHODS}
        for met in METRICS:
            (loBA, hiBA, seBA), (loSI, hiSI, seSI), pb1, (loD1, hiD1) = boot_paired(ev["BASE"], ev["SIT"], met, nmac, nI, rng)
            (loST, hiST, seST), _, pb2, (loD2, hiD2) = boot_paired(ev["Steck-b"], ev["SIT"], met, nmac, nI, rng)
            ci = {"BASE": (loBA, hiBA, seBA), "SIT": (loSI, hiSI, seSI), "Steck-b": (loST, hiST, seST)}
            sv = np.array(per_seed[(bk, "SIT", met)])

            def contrast(ma, loD, hiD, pboot):
                """Five-seed difference, bootstrap interval and p, paired t across seeds, seeds with a positive difference."""
                va = np.array(per_seed[(bk, ma, met)]); sd = sv - va; sdd = sd.std(ddof=1)
                pseed = round(float(stats.ttest_rel(sv, va).pvalue), 6) if sdd > 1e-9 else ""
                spos = int((sd > 0).sum())
                return dict(d=round(float(sv.mean() - va.mean()), 5), lo=round(loD, 5), hi=round(hiD, 5),
                            p=round(pboot, 6), pseed=pseed, seeds=spos, passpos=(loD > 0 and spos == 5))
            c1 = contrast("BASE", loD1, hiD1, pb1); c2 = contrast("Steck-b", loD2, hiD2, pb2)
            # regime on macro-Cat-MRR: winner on a reliable improvement over the backbone, null otherwise
            box = ("winner" if c1["passpos"] else "null") if met == "macroCatMRR" else ""
            tost1 = 1 if (loD1 > -TOST_M and hiD1 < TOST_M) else 0
            for m in METHODS:
                v = np.array(per_seed[(bk, m, met)]); lo, hi, se = ci[m]
                r = dict(dataset=city, backbone=bk, metric=met, method=m,
                         mean=round(float(v.mean()), 5), sd_seed=round(float(v.std(ddof=1)), 7), mean_s42=round(m42[m][met], 5),
                         se_boot=round(se, 5), ci_lo=round(lo, 5), ci_hi=round(hi, 5), box="",
                         delta_l1="", ci_l1_lo="", ci_l1_hi="", p_l1="", p_l1_seed="", seeds_l1="", holm_l1="", sig_l1="",
                         delta_l2="", ci_l2_lo="", ci_l2_hi="", p_l2="", p_l2_seed="", seeds_l2="", holm_l2="", sig_l2="", tost_equiv="", mark="")
                if m == "SIT":
                    r.update(box=box, tost_equiv=tost1, mark=mark(c1, met),
                             delta_l1=c1["d"], ci_l1_lo=c1["lo"], ci_l1_hi=c1["hi"], p_l1=c1["p"], p_l1_seed=c1["pseed"], seeds_l1=c1["seeds"],
                             delta_l2=c2["d"], ci_l2_lo=c2["lo"], ci_l2_hi=c2["hi"], p_l2=c2["p"], p_l2_seed=c2["pseed"], seeds_l2=c2["seeds"])
                rows.append(r)
    return rows


def finalize(df):
    """Holm correction of macro-Cat-MRR across datasets in two families: focal backbone, other backbones."""
    df = df.reset_index(drop=True)

    def holm(mask):
        items = []
        for idx in df.index[mask]:
            for lab in ("l1", "l2"):
                p = df.at[idx, f"p_{lab}"]
                if p != "" and pd.notna(p): items.append((idx, lab, float(p)))
        items.sort(key=lambda x: x[2]); mtot = len(items); prev = 0.
        for rank, (idx, lab, p) in enumerate(items):
            ph = min(1.0, max(prev, (mtot - rank) * p)); prev = ph
            df.at[idx, f"holm_{lab}"] = round(ph, 6)
            df.at[idx, f"sig_{lab}"] = "***" if ph < .001 else "**" if ph < .01 else "*" if ph < .05 else ""
        return mtot
    sit = df.method == "SIT"; mac = df.metric == "macroCatMRR"; prim = df.dataset.isin(PRIMARY_DATASETS)
    nprim = holm(sit & mac & (df.backbone == FOCAL_BACKBONE) & prim)
    nrob = holm(sit & mac & (df.backbone != FOCAL_BACKBONE) & prim)
    ndp = df[prim].dataset.nunique()
    print(f"  [Holm] primary ({FOCAL_BACKBONE} x macroCatMRR x {{L1,L2}} x {ndp} datasets) = {nprim} tests, "
          f"robustness (other backbones) = {nrob} tests", flush=True)
    if ndp < len(PRIMARY_DATASETS):
        print(f"  [Holm] warning: {ndp} of the {len(PRIMARY_DATASETS)} datasets are present; "
              f"the families are those of the paper only when all of them run in one call", flush=True)
    return df


def main():
    cities = sys.argv[1:] or PRIMARY_DATASETS
    dev = torch.device(default_device())
    outdir = ROOT / "outputs"; outdir.mkdir(parents=True, exist_ok=True)
    allrows = []; sel = []
    for city in cities:
        print(f"=== {city} ===", flush=True); rows, s = run_city(city, dev); allrows += rows; sel.append(s)
    df = finalize(pd.DataFrame(allrows))   # Holm across datasets: all datasets in a single call
    df.to_csv(outdir / "results.csv", index=False)
    pd.DataFrame(sel).to_csv(outdir / "selection.csv", index=False)
    print(f"-> {outdir / 'results.csv'} ({len(df)} rows, {df.dataset.nunique()} datasets, MSUPP={MSUPP})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
