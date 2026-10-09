"""Foursquare TIST 2015 -> X-SAGE schema for New York or Sao Paulo: carving, de-duplication, 10-core,
per-user 80/10/10 split, cold filter, legacy-taxonomy categories, previous-check-in geohash.

Raw input:  data/raw/foursquare/dataset_TIST2015_{Checkins,POIs,Cities}.txt
Output:     data/processed/<city>/df_{train,val,test}.parquet, URM_{train,val}.npz
Usage:      python scripts/preprocess/preprocess_foursquare.py <nyc_tist|saopaulo>
"""
import json, math, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
import numpy as np
import pandas as pd
import pygeohash
import scipy.sparse as sps

DATA = Path(os.environ.get("XSAGE_DATA_ROOT", str(ROOT))) / "data"
RAW = DATA / "raw" / "foursquare"
TAXO = ROOT / "config" / "foursquare_legacy_taxonomy.json"
CITIES = {"nyc_tist": "New York", "saopaulo": "Sao Paulo"}   # names in dataset_TIST2015_Cities.txt
RADIUS_KM, WINDOW, KCORE = 35.0, ("2012-04", "2013-02"), 10
TRAIN_RATIO, VAL_RATIO = 0.8, 0.1
OTHER_CAT_ID = "00000000000000000000ffff"                   # placeholder id for unknown category names
TSV_COLS = ["user_id", "venue_id", "cat_id", "cat_name", "lat", "lon", "tz_offset", "utc_time"]
UTC_FORMAT = "%a %b %d %H:%M:%S %z %Y"
CLEAN_COLS = ["u_idx", "i_idx", "cat_macro", "time_local", "c_hour", "c_dow",
              "c_isweekend", "c_month", "prev_geohash5", "intent_last_cat_idx"]


# 1. Carving

def taxonomy_nodes():
    data = json.loads(TAXO.read_text(encoding="latin-1"))
    return data if isinstance(data, list) else data["response"]["categories"]


def build_name_to_id():
    """{category name (lower case) -> category id}, over the whole taxonomy tree."""
    out = {}

    def walk(node):
        if node.get("name") and node.get("id"):
            out[node["name"].lower()] = node["id"]
        for ch in node.get("categories") or []:
            walk(ch)
    for top in taxonomy_nodes():
        walk(top)
    return out


def build_cat_id_to_macro():
    """{category id -> name of its top-level category}."""
    out = {}

    def walk(nodes, top_name):
        for n in nodes:
            if "id" in n:
                out[n["id"]] = top_name
            for child in n.get("categories", []):
                walk([child], top_name)
    for c in taxonomy_nodes():
        if "id" in c:
            out[c["id"]] = c["name"]
        walk(c.get("categories", []), c["name"])
    return out


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def in_window(utc, ws, we):
    """`utc` like 'Tue Apr 03 18:00:06 +0000 2012'; `ws`/`we` like '2012-04' (inclusive, month level)."""
    parts = utc.split()
    if len(parts) < 6:
        return False
    months = {"Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04", "May": "05", "Jun": "06",
              "Jul": "07", "Aug": "08", "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"}
    ym = f"{parts[5]}-{months.get(parts[1], '00')}"
    return ws <= ym <= we


def carve(city, out_tsv):
    centres = {}
    for line in open(RAW / "dataset_TIST2015_Cities.txt", encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 6:
            try: centres[p[0]] = (float(p[1]), float(p[2]))
            except ValueError: pass
    clat, clon = centres[CITIES[city]]
    name_to_id = build_name_to_id()
    venues = {}
    for line in open(RAW / "dataset_TIST2015_POIs.txt", encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) < 5:
            continue
        v_id, lat_s, lon_s, cat_name = p[:4]
        try: lat = float(lat_s); lon = float(lon_s)
        except ValueError: continue
        if haversine_km(lat, lon, clat, clon) < RADIUS_KM + 1.0:
            venues[v_id] = (name_to_id.get(cat_name.lower(), OTHER_CAT_ID), cat_name, lat, lon)
    n = 0
    with open(out_tsv, "w", encoding="utf-8") as fo:
        for line in open(RAW / "dataset_TIST2015_Checkins.txt", encoding="utf-8"):
            p = line.rstrip("\n").split("\t")
            if len(p) < 4:
                continue
            u_id, v_id, utc, tz = p[:4]
            hit = venues.get(v_id)
            if hit is None or not in_window(utc, *WINDOW):
                continue
            cid, cat_name, lat, lon = hit
            fo.write(f"{u_id}\t{v_id}\t{cid}\t{cat_name}\t{lat:.15f}\t{lon:.15f}\t{tz}\t{utc}\n"); n += 1
    print(f"[1] {city}: {len(venues)} venues within {RADIUS_KM + 1.0:.0f} km, {n} check-ins in {WINDOW[0]}..{WINDOW[1]}", flush=True)


# 2-6. Cleaning, k-core, split, features

def load_and_localize(tsv):
    df = pd.read_csv(tsv, sep="\t", header=None, names=TSV_COLS, encoding="latin-1",
                     dtype={"user_id": np.int32, "venue_id": str, "cat_id": str, "cat_name": str,
                            "lat": np.float64, "lon": np.float64, "tz_offset": np.int32, "utc_time": str})
    df["utc_time_parsed"] = pd.to_datetime(df["utc_time"], format=UTC_FORMAT, utc=True)
    df["time_local"] = df["utc_time_parsed"] + pd.to_timedelta(df["tz_offset"].astype(int), unit="m")
    df["time_local"] = df["time_local"].dt.tz_convert(None)
    return df.drop_duplicates(subset=["user_id", "venue_id", "utc_time"])


def iterative_kcore(df, k):
    cur = df
    while True:
        n_before = len(cur)
        user_n = cur.groupby("user_id")["venue_id"].nunique()
        item_n = cur.groupby("venue_id")["user_id"].nunique()
        cur = cur[cur["user_id"].isin(user_n[user_n >= k].index) & cur["venue_id"].isin(item_n[item_n >= k].index)]
        if len(cur) == n_before:
            return cur.reset_index(drop=True)


def per_user_split(df):
    df = df.sort_values(["user_id", "time_local"], kind="stable").reset_index(drop=True)
    pos = df.groupby("user_id").cumcount(); total = df.groupby("user_id")["user_id"].transform("count")
    frac = (pos + 1) / total
    df["split"] = frac.map(lambda f: "train" if f <= TRAIN_RATIO else ("val" if f <= TRAIN_RATIO + VAL_RATIO else "test"))
    return df


def cold_filter(df):
    train = df[df["split"] == "train"]
    is_eval = df["split"].isin(("val", "test"))
    cold = is_eval & (~df["user_id"].isin(set(train["user_id"].unique())) | ~df["venue_id"].isin(set(train["venue_id"].unique())))
    return df[~cold].reset_index(drop=True), int((cold & (df["split"] == "val")).sum()), int((cold & (df["split"] == "test")).sum())


def add_features(df):
    t = df["time_local"]; df = df.copy()
    df["c_hour"] = t.dt.hour.astype(np.int8)
    df["c_dow"] = t.dt.dayofweek.astype(np.int8)
    df["c_isweekend"] = (df["c_dow"] >= 5).astype(np.int8)
    df["c_month"] = t.dt.month.astype(np.int8)
    df["geohash5"] = [pygeohash.encode(la, lo, precision=5) for la, lo in zip(df["lat"].values, df["lon"].values)]
    df = df.sort_values(["user_id", "time_local"], kind="stable").reset_index(drop=True)
    df["cat_macro"] = df["cat_id"].map(build_cat_id_to_macro()).fillna("Other")
    return df


def main():
    city = sys.argv[1] if len(sys.argv) > 1 else "nyc_tist"
    if city not in CITIES:
        raise SystemExit(f"unknown city {city!r}: expected one of {sorted(CITIES)}")
    tsv = RAW / f"{city}_carved.tsv"
    carve(city, tsv)
    df = load_and_localize(tsv); n_raw = len(df)
    df = iterative_kcore(df, KCORE); n_core = len(df)
    df = per_user_split(df)
    df, cold_val, cold_test = cold_filter(df)
    df = add_features(df)
    print(f"[2-6] check-ins: {n_raw} after deduplication, {n_core} after {KCORE}-core, {len(df)} after dropping "
          f"{cold_val} validation and {cold_test} test rows with an unseen user or venue", flush=True)

    # indices and the per-split views
    user2id = {u: i for i, u in enumerate(sorted(df["user_id"].unique()))}
    item2id = {v: i for i, v in enumerate(sorted(df["venue_id"].unique()))}
    df["u_idx"] = df["user_id"].map(user2id).astype(np.int32)
    df["i_idx"] = df["venue_id"].map(item2id).astype(np.int32)
    n_users, n_items = len(user2id), len(item2id)
    out = DATA / "processed" / city; out.mkdir(parents=True, exist_ok=True)
    parts = []
    for sp in ("train", "val", "test"):
        sub = df[df["split"] == sp]
        if sp in ("train", "val"):
            urm = sps.csr_matrix((np.ones(len(sub), dtype=np.float32), (sub["u_idx"].values, sub["i_idx"].values)),
                                 shape=(n_users, n_items))
            urm.sum_duplicates(); urm.data[:] = 1.0
            sps.save_npz(out / f"URM_{sp}.npz", urm)
        parts.append(sub[["u_idx", "i_idx", "time_local", "cat_macro", "c_hour", "c_dow", "c_isweekend",
                          "c_month", "geohash5", "split"]].reset_index(drop=True))

    # 7. previous-check-in geohash and category over the whole timeline of each user
    df = pd.concat(parts, ignore_index=True).dropna(subset=["cat_macro"])
    macros = sorted(df.cat_macro.unique()); m2i = {m: k for k, m in enumerate(macros)}
    df = df.sort_values(["u_idx", "time_local"], kind="stable").reset_index(drop=True)
    df["prev_geohash5"] = df.groupby("u_idx")["geohash5"].shift(1).fillna("none").astype(str)
    df["_cmi"] = df.cat_macro.map(m2i).astype(np.int64)
    df["intent_last_cat_idx"] = df.groupby("u_idx")["_cmi"].shift(1).fillna(len(macros)).astype(np.int64)
    for sp in ("train", "val", "test"):
        d = df[df.split == sp][CLEAN_COLS].reset_index(drop=True)
        d.to_parquet(out / f"df_{sp}.parquet"); print(f"[7] df_{sp}: {len(d)}", flush=True)
    print(f"\n[OK] {city} -> {out}")
    print(f"  users={n_users} venues={n_items} categories={len(macros)} ({macros})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
