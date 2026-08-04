"""Re-derive outputs/results.csv from the raw cache (outputs/cache/raw_<city>.npz)
without re-running the expensive evaluation. To change metrics / @K / formulas / min-support,
edit the block below and re-run (seconds). The cache is produced by run_pipeline.py.

Usage:  python scripts/aggregate.py [city ...]
Output: outputs/results.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import run_pipeline as rr
ROOT = rr.ROOT

# >>> EDIT HERE (then re-run: no re-eval) <<<
rr.KTOP = 20                                       # @K for Cat-MRR/Cat-NDCG/LT/Coverage/Gini/JS
rr.MSUPP = 20                                      # min-support |R_c|>=MSUPP for macro-Cat-MRR
rr.HRK = {"HR5": 5, "HR10": 10, "HR20": 20}        # e.g. add "HR50": 50  (re-derivable up to 50)
rr.NDK = {"NDCG5": 5, "NDCG10": 10, "NDCG20": 20}  # e.g. add "NDCG50": 50
rr.ACCMET = ["MRR"] + list(rr.HRK) + list(rr.NDK)  # primary: per-request (standard)
rr.ACCMET_U = [m + "_u" for m in rr.ACCMET]        # secondary: per-user (user-balanced)
rr.METRICS = rr.CATMET + rr.ACCMET + rr.ACCMET_U + rr.EXPMET + rr.COSTMET
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


def edict(z, bk, m, s, sh):
    e = dict(rk=z[f"{bk}|{m}|{s}|rk"].astype(np.int64), catrk=z[f"{bk}|{m}|{s}|catrk"].astype(np.int64),
             g=z[f"{bk}|{m}|{s}|g"].astype(np.int64), gc=z[f"{bk}|{m}|{s}|gc"].astype(np.int64),
             tk50=z[f"{bk}|{m}|{s}|tk50"].astype(np.int64), u=sh["u"], tm=sh["tm"])
    pur = sh["Pu"][sh["u"]]
    return rr.derive(e, sh["icm"], sh["G1"], sh["nmac"], sh["nI"], pur)


def run(city):
    z = np.load(ROOT / "outputs" / "cache" / f"raw_{city}.npz")
    sh = dict(u=z["_shared|u"].astype(np.int64), tm=z["_shared|tm"].astype(np.int64), G1=z["_shared|G1"],
              icm=z["_shared|icm"].astype(np.int64), Pu=z["_shared|Pu"], nI=int(z["_shared|nI"]), nmac=int(z["_shared|nmac"]))
    present = [bk for bk in rr.BK if f"{bk}|SIT|42|rk" in z.files]   # backbone effettivamente in cache
    per_seed = {(bk, m, met): [] for bk in present for m in rr.METHODS for met in rr.METRICS}; seed42 = {}
    for s in rr.SEEDS:
        for bk in present:
            ev = {m: edict(z, bk, m, s, sh) for m in rr.METHODS}
            for m in rr.METHODS:
                mm = rr.metrics_from(ev[m], sh["nmac"], sh["nI"])
                for met in rr.METRICS: per_seed[(bk, m, met)].append(mm[met])
            if s == 42: seed42[bk] = ev
    return rr.aggregate(per_seed, seed42, city, sh["nmac"], sh["nI"])


def main():
    cities = sys.argv[1:] or ["nyc_tist", "saopaulo", "ml1m"]
    out = ROOT / "outputs" / "results.csv"; allrows = []
    for c in cities:
        f = ROOT / "outputs" / "cache" / f"raw_{c}.npz"
        if not f.exists(): print(f"  manca cache/raw_{c}.npz — salta {c}"); continue
        allrows += run(c); print(f"  {c}: ri-derivato dalla cache (MSUPP={rr.MSUPP})", flush=True)
    if allrows:
        df = rr.finalize(pd.DataFrame(allrows)); df.to_csv(out, index=False)
        print(f"-> {out} ({len(df)} righe, {df.dataset.nunique()} dataset)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
