"""KuaiRand-Pure -> X-SAGE schema (no geo). Video = interaction, item category = primary tag; temporal
context; per-user chronological 80/10/10 split; 10-core filtering."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
import numpy as np
import pandas as pd
import scipy.sparse as sps

RAW = (ROOT / "data" / "raw" / "KuaiRand-Pure/data")
OUT = (ROOT / "data" / "processed" / "kuairand")
KCORE = int(sys.argv[1]) if len(sys.argv) > 1 else 10   # protocollo UNIFORME (come ml-1m)
LOGS = ["log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # 1. video -> primary tag (macro)
    vf = pd.read_csv(RAW / "video_features_basic_pure.csv", usecols=["video_id", "tag"])
    vf["cat_macro"] = vf.tag.astype(str).str.split(r"[,;]").str[0].str.strip()
    vf = vf[(vf.cat_macro != "") & (vf.cat_macro.str.lower() != "nan")][["video_id", "cat_macro"]]
    print(f"[1] video con tag: {len(vf)}  tag distinti: {vf.cat_macro.nunique()}", flush=True)

    # 2. standard log -> click
    parts = []
    for f in LOGS:
        d = pd.read_csv(RAW / f, usecols=["user_id", "video_id", "date", "hourmin", "time_ms", "is_click"])
        parts.append(d[d.is_click == 1])
    df = pd.concat(parts, ignore_index=True).drop(columns="is_click")
    df = df.merge(vf, on="video_id", how="inner")
    print(f"[2] click con macro: {len(df)}  utenti={df.user_id.nunique()}  video={df.video_id.nunique()}", flush=True)

    # 3. k-core
    while True:
        uc = df.user_id.value_counts(); ic = df.video_id.value_counts()
        m = df.user_id.isin(uc[uc >= KCORE].index) & df.video_id.isin(ic[ic >= KCORE].index)
        if m.all(): break
        df = df[m]
    print(f"[3] dopo k-core={KCORE}: {len(df)}  utenti={df.user_id.nunique()}  video={df.video_id.nunique()}", flush=True)

    # 4. remap
    uu = {u: i for i, u in enumerate(sorted(df.user_id.unique()))}
    ii = {it: i for i, it in enumerate(sorted(df.video_id.unique()))}
    df["u_idx"] = df.user_id.map(uu).astype(np.int64); df["i_idx"] = df.video_id.map(ii).astype(np.int64)
    n_users, n_items = len(uu), len(ii)
    macros = sorted(df.cat_macro.unique()); m2i = {mm: k for k, mm in enumerate(macros)}

    # 5. time + per-user chronological 80/10/10 split (ordered by time_ms)
    df["time_local"] = pd.to_datetime(df.time_ms, unit="ms")
    df["c_hour"] = (df.hourmin // 100).astype(np.int64)
    dloc = pd.to_datetime(df.date, format="%Y%m%d")
    df["c_dow"] = dloc.dt.dayofweek.astype(np.int64)
    df["c_isweekend"] = (df.c_dow >= 5).astype(np.int64)
    df["c_month"] = dloc.dt.month.astype(np.int64)
    df = df.sort_values(["u_idx", "time_ms"], kind="stable").reset_index(drop=True)
    r = df.groupby("u_idx").cumcount(); nu = df.groupby("u_idx")["u_idx"].transform("size")
    frac = r / nu
    df["split"] = np.where(frac < 0.8, "train", np.where(frac < 0.9, "val", "test"))
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

    tr = df[df.split == "train"]
    test_users = np.array(sorted(df[df.split == "test"].u_idx.unique()), dtype=np.int64)
    np.savez(OUT / "_backbone_input.npz", train_u=tr.u_idx.values.astype(np.int64),
             train_i=tr.i_idx.values.astype(np.int64), n_users=np.int64(n_users),
             n_items=np.int64(n_items), test_users=test_users)

    print(f"\n[OK] kuairand -> {OUT}")
    print(f"  utenti={n_users} video={n_items} macro={len(macros)}")
    print(f"  clicks/user mean={df.groupby('u_idx').size().mean():.1f} min={df.groupby('u_idx').size().min()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
