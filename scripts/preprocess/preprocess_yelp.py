"""Yelp Open Dataset -> X-SAGE schema (with geo). Review = interaction, item category = Yelp root
category; temporal context + previous-review geohash (causal, anti-leakage); per-user 80/10/10; 20-core."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
import numpy as np
import pandas as pd
import scipy.sparse as sps

RAW = (ROOT / "data" / "raw" / "yelp_dataset")
CITY = sys.argv[1] if len(sys.argv) > 1 else "Philadelphia"
KCORE = int(sys.argv[2]) if len(sys.argv) > 2 else 10   # protocollo UNIFORME (come ml-1m)
BAL = len(sys.argv) > 3 and sys.argv[3] == "bal"          # ribilanciamento categorie (diagnostico)
CAPMULT = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0  # per-macro cap = CAPMULT * median
OUT = (ROOT / "data" / "processed" / ("yelp_bal" if BAL else "yelp"))
GH_PREC = 6   # geohash6 ~ 1.2 km intra-metro resolution (column kept as prev_geohash5)

# Yelp root categories in priority order (consumption/leisure first: more situationally informative)
ROOTS = ["Restaurants", "Food", "Nightlife", "Arts & Entertainment", "Active Life", "Beauty & Spas",
         "Shopping", "Hotels & Travel", "Health & Medical", "Automotive", "Event Planning & Services",
         "Home Services", "Local Services", "Pets", "Education", "Professional Services",
         "Financial Services", "Public Services & Government", "Religious Organizations",
         "Mass Media", "Local Flavor"]
ROOTSET = set(ROOTS)

_B32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def geohash(lat, lon, prec=GH_PREC):
    lat_r = [-90.0, 90.0]; lon_r = [-180.0, 180.0]; gh = []; bits = [16, 8, 4, 2, 1]
    bit = 0; ch = 0; even = True
    while len(gh) < prec:
        if even:
            mid = (lon_r[0] + lon_r[1]) / 2
            if lon > mid: ch |= bits[bit]; lon_r[0] = mid
            else: lon_r[1] = mid
        else:
            mid = (lat_r[0] + lat_r[1]) / 2
            if lat > mid: ch |= bits[bit]; lat_r[0] = mid
            else: lat_r[1] = mid
        even = not even
        if bit < 4: bit += 1
        else: gh.append(_B32[ch]); bit = 0; ch = 0
    return "".join(gh)


def assign_macro(cats):
    """root category by priority; fall back to the first category if no root is present."""
    s = {c.strip() for c in cats.split(",")} if cats else set()
    for r in ROOTS:
        if r in s: return r
    return next(iter(s)) if s else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # 1. metro businesses -> macro category + geohash
    biz = {}  # bid -> (macro, geohash)
    nseen = 0
    for ln in open(RAW / "yelp_academic_dataset_business.json"):
        b = json.loads(ln)
        if b.get("city") != CITY or b.get("state") not in ("PA", b.get("state")):
            pass
        if b.get("city") != CITY: continue
        nseen += 1
        m = assign_macro(b.get("categories"))
        lat, lon = b.get("latitude"), b.get("longitude")
        if m is None or lat is None or lon is None: continue
        biz[b["business_id"]] = (m, geohash(lat, lon))
    print(f"[1] {CITY}: {len(biz)}/{nseen} business con macro+geo", flush=True)

    # 2. stream review.json -> interactions in the metro
    rows = []
    n = 0
    for ln in open(RAW / "yelp_academic_dataset_review.json"):
        n += 1
        if n % 2000000 == 0: print(f"    ...{n//1000000}M review, {len(rows)} in-metro", flush=True)
        r = json.loads(ln)
        b = r["business_id"]
        if b in biz:
            rows.append((r["user_id"], b, r["date"]))
    df = pd.DataFrame(rows, columns=["user", "item", "date"])
    df["time_local"] = pd.to_datetime(df["date"])
    macro = {b: v[0] for b, v in biz.items()}; gh = {b: v[1] for b, v in biz.items()}
    df["cat_macro"] = df.item.map(macro); df["geohash"] = df.item.map(gh)
    df = df.drop(columns="date")
    print(f"[2] review in-metro: {len(df)}  utenti={df.user.nunique()}  business={df.item.nunique()}", flush=True)

    # 2b. (diagnostic) category rebalancing: per-macro cap = CAPMULT * median, random downsample
    if BAL:
        vc = df.cat_macro.value_counts(); cap = int(CAPMULT * vc.iloc[1])  # CAPMULT * 2nd macro (brings the dominant macro down to the runner-up level)
        df = df.groupby("cat_macro", group_keys=False).apply(
            lambda g: g.sample(min(len(g), cap), random_state=42)).reset_index(drop=True)
        print(f"[2b BAL] cap={cap}/macro (dominante prima={int(vc.iloc[0])}) -> {len(df)} interazioni", flush=True)

    # 3. k-core
    while True:
        uc = df.user.value_counts(); ic = df.item.value_counts()
        m = df.user.isin(uc[uc >= KCORE].index) & df.item.isin(ic[ic >= KCORE].index)
        if m.all(): break
        df = df[m]
    print(f"[3] dopo k-core={KCORE}: {len(df)}  utenti={df.user.nunique()}  business={df.item.nunique()}", flush=True)

    # 4. remap
    uu = {u: i for i, u in enumerate(sorted(df.user.unique()))}
    ii = {it: i for i, it in enumerate(sorted(df.item.unique()))}
    df["u_idx"] = df.user.map(uu).astype(np.int64); df["i_idx"] = df.item.map(ii).astype(np.int64)
    n_users, n_items = len(uu), len(ii)
    macros = sorted(df.cat_macro.unique()); m2i = {mm: k for k, mm in enumerate(macros)}

    # 5. per-user chronological 80/10/10 split
    df = df.sort_values(["u_idx", "time_local"], kind="stable").reset_index(drop=True)
    rk = df.groupby("u_idx").cumcount(); nu = df.groupby("u_idx")["u_idx"].transform("size")
    frac = rk / nu
    df["split"] = np.where(frac < 0.8, "train", np.where(frac < 0.9, "val", "test"))

    # 6. temporal context + geo (prev_geohash5, anti-leakage) + intent
    df["c_hour"] = df.time_local.dt.hour.astype(np.int64)
    df["c_dow"] = df.time_local.dt.dayofweek.astype(np.int64)
    df["c_isweekend"] = (df.c_dow >= 5).astype(np.int64)
    df["c_month"] = df.time_local.dt.month.astype(np.int64)
    df["prev_geohash5"] = df.groupby("u_idx")["geohash"].shift(1).fillna("none")  # geohash of the PREVIOUS review
    df["_cmi"] = df.cat_macro.map(m2i).astype(np.int64)
    df["intent_last_cat_idx"] = df.groupby("u_idx")["_cmi"].shift(1).fillna(len(macros)).astype(np.int64)

    cols = ["u_idx", "i_idx", "cat_macro", "time_local", "c_hour", "c_dow", "c_isweekend",
            "c_month", "prev_geohash5", "intent_last_cat_idx"]
    for sp in ("train", "val", "test"):
        d = df[df.split == sp][cols].reset_index(drop=True)
        d.to_parquet(OUT / f"df_{sp}.parquet"); print(f"[6] df_{sp}: {len(d)}", flush=True)

    def urm(s):
        d = df[df.split.isin(s)]
        return sps.csr_matrix((np.ones(len(d)), (d.u_idx, d.i_idx)), shape=(n_users, n_items))
    sps.save_npz(OUT / "URM_train.npz", urm(["train"])); sps.save_npz(OUT / "URM_val.npz", urm(["val"]))

    # 7. interaction dump for training the BPR backbone
    tr = df[df.split == "train"]
    test_users = np.array(sorted(df[df.split == "test"].u_idx.unique()), dtype=np.int64)
    np.savez(OUT / "_backbone_input.npz", train_u=tr.u_idx.values.astype(np.int64),
             train_i=tr.i_idx.values.astype(np.int64), n_users=np.int64(n_users),
             n_items=np.int64(n_items), test_users=test_users)

    print(f"\n[OK] yelp({CITY}) -> {OUT}")
    print(f"  utenti={n_users} business={n_items} macro={len(macros)}")
    print(f"  macro: {macros}")
    print(f"  reviews/user mean={df.groupby('u_idx').size().mean():.1f} min={df.groupby('u_idx').size().min()}")
    print(f"  geohash6 distinti={df.prev_geohash5.nunique()} (risoluzione geo intra-metro)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
