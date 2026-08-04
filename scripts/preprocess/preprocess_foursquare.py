"""Foursquare check-in city -> X-SAGE schema, matching the ml1m/yelp format. Keeps the user/item indices
and derives the two context columns (temporal + previous-category intent proxy) on the full timeline
for causal correctness of the 'previous' interaction."""
import shutil, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
import numpy as np
import pandas as pd

OLD = (ROOT / "data" / "processed")
OUTROOT = (ROOT / "data" / "raw" / "processed")
CLEAN_COLS = ["u_idx", "i_idx", "cat_macro", "time_local", "c_hour", "c_dow",
              "c_isweekend", "c_month", "prev_geohash5", "intent_last_cat_idx"]


def main():
    city = sys.argv[1] if len(sys.argv) > 1 else "nyc_tist"
    src = OLD / city; out = OUTROOT / city; out.mkdir(parents=True, exist_ok=True)
    parts = []
    for sp in ("train", "val", "test"):
        d = pd.read_parquet(src / f"df_{sp}.parquet"); d["split"] = sp; parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df = df.dropna(subset=["cat_macro"]).copy()
    n_users = int(df.u_idx.max()) + 1; n_items = int(df.i_idx.max()) + 1
    macros = sorted(df.cat_macro.unique()); m2i = {m: k for k, m in enumerate(macros)}

    # full per-user timeline -> derive the 'previous' interaction (causal)
    df = df.sort_values(["u_idx", "time_local"], kind="stable").reset_index(drop=True)
    gh_src = "geohash5" if "geohash5" in df.columns else ("geohash4" if "geohash4" in df.columns else None)
    if gh_src is None:
        raise SystemExit("ERRORE: nessun geohash nel df OLD")
    df["prev_geohash5"] = df.groupby("u_idx")[gh_src].shift(1).fillna("none").astype(str)
    df["_cmi"] = df.cat_macro.map(m2i).astype(np.int64)
    df["intent_last_cat_idx"] = df.groupby("u_idx")["_cmi"].shift(1).fillna(len(macros)).astype(np.int64)

    for sp in ("train", "val", "test"):
        d = df[df.split == sp][CLEAN_COLS].reset_index(drop=True)
        d.to_parquet(out / f"df_{sp}.parquet"); print(f"  df_{sp}: {len(d)}", flush=True)

    # URM: source of truth for popularity / excluded_mask
    for f in ("URM_train.npz", "URM_val.npz"):
        shutil.copy(src / f, out / f)

    # interaction dump for training the BPR backbone (same indices)
    tr = df[df.split == "train"]
    test_users = np.array(sorted(df[df.split == "test"].u_idx.unique()), dtype=np.int64)
    np.savez(out / "_backbone_input.npz", train_u=tr.u_idx.values.astype(np.int64),
             train_i=tr.i_idx.values.astype(np.int64), n_users=np.int64(n_users),
             n_items=np.int64(n_items), test_users=test_users)

    print(f"[OK] {city} -> {out}")
    print(f"  utenti={n_users} item={n_items} macro={len(macros)}")
    print(f"  geohash sorgente={gh_src}; prev_geohash5 distinti={df.prev_geohash5.nunique()}")
    print(f"  reviews/user mean={df.groupby('u_idx').size().mean():.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
