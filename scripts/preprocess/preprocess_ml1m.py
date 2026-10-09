"""MovieLens-1M -> X-SAGE schema: ratings, first genre as category, 10-core, per-user 80/10/10 split.

Raw input:  data/raw/ml-1m/{movies,ratings}.dat
Output:     data/processed/ml1m/df_{train,val,test}.parquet, URM_{train,val}.npz
Usage:      python scripts/preprocess/preprocess_ml1m.py
"""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
import numpy as np
import pandas as pd
import scipy.sparse as sps

DATA = Path(os.environ.get("XSAGE_DATA_ROOT", str(ROOT))) / "data"
ML = DATA / "raw" / "ml-1m"
OUT = DATA / "processed" / "ml1m"
KCORE = 10


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # 1. movie -> primary genre
    gen = {}
    for ln in open(ML / "movies.dat", encoding="latin-1"):
        p = ln.rstrip("\n").split("::")
        if len(p) >= 3:
            g = p[2].split("|")[0].strip()
            if g and g != "(no genres listed)": gen[p[0]] = g
    print(f"[1] films with a genre: {len(gen)}", flush=True)

    # 2. ratings -> interactions
    rows = []
    for ln in open(ML / "ratings.dat", encoding="latin-1"):
        p = ln.rstrip("\n").split("::")
        if len(p) < 4: continue
        if p[1] in gen:
            rows.append((p[0], p[1], int(p[3])))
    df = pd.DataFrame(rows, columns=["user", "item", "ts"])
    df["time_local"] = pd.to_datetime(df["ts"], unit="s")
    df["cat_macro"] = df["item"].map(gen)
    df = df.drop(columns="ts")
    print(f"[2] ratings: {len(df)}  users={df.user.nunique()}  films={df.item.nunique()}", flush=True)

    # 3. k-core=10
    while True:
        uc = df.user.value_counts(); ic = df.item.value_counts()
        m = df.user.isin(uc[uc >= KCORE].index) & df.item.isin(ic[ic >= KCORE].index)
        if m.all(): break
        df = df[m]
    print(f"[3] after {KCORE}-core: {len(df)}  users={df.user.nunique()}  films={df.item.nunique()}", flush=True)

    # 4. remap + macro
    uu = {u: i for i, u in enumerate(sorted(df.user.unique()))}
    ii = {it: i for i, it in enumerate(sorted(df.item.unique()))}
    df["u_idx"] = df.user.map(uu).astype(np.int64); df["i_idx"] = df.item.map(ii).astype(np.int64)
    n_users, n_items = len(uu), len(ii)
    macros = sorted(df.cat_macro.unique()); m2i = {m: k for k, m in enumerate(macros)}

    # 5. per-user chronological 80/10/10 split
    df = df.sort_values(["u_idx", "time_local"], kind="stable").reset_index(drop=True)
    r = df.groupby("u_idx").cumcount(); nu = df.groupby("u_idx")["u_idx"].transform("size")
    frac = r / nu
    df["split"] = np.where(frac < 0.8, "train", np.where(frac < 0.9, "val", "test"))

    # 6. context + intent proxy (category of the previous interaction of the user)
    df["c_hour"] = df.time_local.dt.hour.astype(np.int64)
    df["c_dow"] = df.time_local.dt.dayofweek.astype(np.int64)
    df["c_isweekend"] = (df.c_dow >= 5).astype(np.int64)
    df["c_month"] = df.time_local.dt.month.astype(np.int64)
    df["_cmi"] = df.cat_macro.map(m2i).astype(np.int64)
    df["intent_last_cat_idx"] = df.groupby("u_idx")["_cmi"].shift(1).fillna(len(macros)).astype(np.int64)

    cols = ["u_idx", "i_idx", "cat_macro", "time_local", "c_hour", "c_dow",
            "c_isweekend", "c_month", "intent_last_cat_idx"]
    for sp in ("train", "val", "test"):
        d = df[df.split == sp][cols].reset_index(drop=True)
        d.to_parquet(OUT / f"df_{sp}.parquet"); print(f"[5] df_{sp}: {len(d)}", flush=True)

    def urm(s):
        d = df[df.split.isin(s)]
        return sps.csr_matrix((np.ones(len(d)), (d.u_idx, d.i_idx)), shape=(n_users, n_items))
    sps.save_npz(OUT / "URM_train.npz", urm(["train"])); sps.save_npz(OUT / "URM_val.npz", urm(["val"]))

    print(f"\n[OK] ml1m -> {OUT}")
    print(f"  users={n_users} items={n_items} categories={len(macros)} ({macros})")
    print(f"  ratings/user mean={df.groupby('u_idx').size().mean():.1f} min={df.groupby('u_idx').size().min()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
