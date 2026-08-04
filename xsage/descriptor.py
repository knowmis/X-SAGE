"""Build the situational descriptor v = [c_tilde || e] for every request.

c_tilde is the per-attribute context state (L1a) and e the propagated intent
vector (L1b); their concatenation is the input to the comprehension level.
Perception hyper-parameters are read from config/params/<city>.json (selected on
the validation split), falling back to the defaults below.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from xsage.l0_sensing import build_recent_window
from xsage.l1_perception import (compute_intent, compute_profile,
                                 estimate_macro_transition,
                                 fit_contribution_functions, find_attractors)
from xsage import data as D
from xsage.metrics import long_tail_groups

ROOT = Path(__file__).resolve().parents[1]

# Context attributes: temporal-only, or temporal plus a coarse previous-location geohash.
TEMPORAL_ATTRS = ("c_hour", "c_dow", "c_isweekend", "c_month", "intent_last_cat_idx")
GEO_ATTRS = ("c_hour", "c_dow", "c_isweekend", "c_month", "prev_geohash5", "intent_last_cat_idx")
GEO_CITIES = {"nyc_tist", "saopaulo"}   # check-in datasets carrying a previous-location geohash

# Default perception hyper-parameters (overridden by config/params/<city>.json).
H, BETA, ALPHA, SHORT_HEAD, MAX_ITER = 2, 0.7, 50.0, 0.20, 80
GAMMA, DEPTH, N = 0.4, 3, 3


def city_attrs(city):
    """Geo-enabled datasets use the previous-location geohash; the others are temporal-only."""
    return GEO_ATTRS if city in GEO_CITIES else TEMPORAL_ATTRS


def membership_from_assign(k_star, comp, isb, K):
    """Soft situation membership: 1 on the single core situation, 1/|T| split over competing ones."""
    B = len(k_star); mem = np.zeros((B, K), np.float32)
    core = ~isb; mem[core, k_star[core]] = 1.0
    bnd = np.where(isb)[0]
    if bnd.size:
        T = comp[bnd].sum(1).astype(np.float32); mem[bnd] = comp[bnd].astype(np.float32) / T[:, None]
    return mem


def closed_params(city):
    """Perception hyper-parameters selected on validation (config/params/<city>.json), else defaults."""
    import json
    pj = ROOT / "config" / "params" / f"{city}.json"
    P = json.load(open(pj)) if pj.exists() else {}
    return {"gamma": P.get("gamma", GAMMA), "depth": P.get("depth", DEPTH), "n": P.get("n", N),
            "beta": P.get("beta", BETA), "H": P.get("H", H), "alpha": P.get("alpha", ALPHA)}


def build_descriptor(city="ml1m", splits=("train", "val", "test"),
                     gamma=None, depth=None, n=None, beta=None, h_hops=None,
                     raw_context=False, raw_intent=False):
    """Build v = [c_tilde || e] per split. Perception parameters come from (in order):
    explicit arguments -> config/params/<city>.json -> defaults.

    Ablation switches (used only for the fusion analysis, not the main pipeline):
    raw_context=True replaces the context state with a raw one-hot of the context;
    raw_intent=True replaces the propagated intent with the raw recency profile.
    """
    P = closed_params(city)
    gamma = P["gamma"] if gamma is None else gamma; depth = P["depth"] if depth is None else depth
    n = P["n"] if n is None else n; beta = P["beta"] if beta is None else beta
    h_hops = P["H"] if h_hops is None else h_hops
    ds = D.load_city(city, data_root=str(ROOT))
    m2i = ds["macro_to_idx"]; n_macros = ds["n_macros"]; n_items = ds["n_items"]
    for k in ("df_train", "df_val", "df_test"):
        ds[k] = ds[k].copy()
        ds[k]["cat_target"] = ds[k]["cat_macro"].map(m2i).astype(np.int64)
        ds[k]["user_id"] = ds[k]["u_idx"]
    contrib = fit_contribution_functions(ds["df_train"], m2i, attributes=city_attrs(city),
                                         max_depth=depth, min_leaf=200)
    W = estimate_macro_transition(ds["df_train"], m2i, transit_macros=[], transit_mode="keep")
    attractors = find_attractors(W, exclude_indices=None)
    hist = {"train": ds["df_train"], "val": ds["df_train"],
            "test": pd.concat([ds["df_train"], ds["df_val"]], ignore_index=True)}

    attrs = city_attrs(city)
    # train vocabularies for the raw one-hot (only when raw_context)
    voc = {a: {v: i for i, v in enumerate(sorted(ds["df_train"][a].unique()))} for a in attrs} if raw_context else {}

    def onehot_ctx(tgt):
        blocks = []
        for a in attrs:
            vmap = voc[a]; nv = len(vmap)
            idx = tgt[a].map(lambda x: vmap.get(x, nv)).values.astype(np.int64)   # OOV -> extra bucket
            oh = np.zeros((len(tgt), nv + 1), np.float32); oh[np.arange(len(tgt)), idx] = 1.0
            blocks.append(oh)
        return np.concatenate(blocks, axis=1)

    def build(split):
        tgt = ds[f"df_{split}"]
        l0 = build_recent_window(tgt, hist[split], m2i, n=n)
        c = onehot_ctx(tgt) if raw_context else contrib.transform(tgt)
        m = compute_profile(l0.recent_macro, l0.n_prior, n_macros, gamma=gamma)
        e = m.astype(np.float32) if raw_intent else compute_intent(m, W, attractors, H=h_hops, beta=beta, mode="hard")
        return np.concatenate([c, e], axis=1).astype(np.float32)

    vs = {s: build(s) for s in splits}
    cmt = ds["df_train"]["cat_macro"].map(m2i).values.astype(np.int64)
    df_all = pd.concat([ds["df_train"], ds["df_val"], ds["df_test"]], ignore_index=True)
    icm = (df_all.groupby("i_idx")["cat_macro"].first().map(m2i)
           .reindex(np.arange(n_items), fill_value=0).values.astype(np.int64))
    pop = np.asarray((ds["urm_train"] + ds["urm_val"]).sum(0)).ravel()
    _, G1 = long_tail_groups(pop, short_head_share=SHORT_HEAD)
    sb, _ = D.load_backbone_scores(city, data_root=str(ROOT))
    excl = D.load_excluded_mask(city, n_items, data_root=str(ROOT))
    return dict(ds=ds, vs=vs, m2i=m2i, n_macros=n_macros, n_items=n_items, cmt=cmt,
                icm=icm, G1=G1, sb=sb, excl=excl, attractors=attractors)
