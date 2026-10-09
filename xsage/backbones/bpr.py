"""Context-blind BPR backbone (Rendle et al., 2009, Cornac); writes data/<city>/backbone/BPR.scores_user.npy.

Usage:  python -m xsage.backbones.bpr <city>
"""
import sys

import numpy as np
import pandas as pd
from cornac.data import Dataset
from cornac.models import BPR

from xsage.data import DEFAULT_DATA_ROOT


def main():
    city = sys.argv[1] if len(sys.argv) > 1 else "ml1m"
    P = DEFAULT_DATA_ROOT / "data" / "processed" / city
    dtr = pd.read_parquet(P / "df_train.parquet")
    alld = pd.concat([dtr, pd.read_parquet(P / "df_val.parquet"), pd.read_parquet(P / "df_test.parquet")], ignore_index=True)
    tu = dtr["u_idx"].values.astype(np.int64); ti = dtr["i_idx"].values.astype(np.int64)
    nU = int(alld.u_idx.max()) + 1; nI = int(alld.i_idx.max()) + 1
    print(f"[{city}] BPR | train pairs={len(tu)} n_users={nU} n_items={nI}", flush=True)

    triples = list(zip(tu.astype(str).tolist(), ti.astype(str).tolist(),
                       np.ones(len(tu), dtype=np.float32).tolist()))
    ds = Dataset.from_uir(triples)
    m = BPR(k=64, max_iter=150, learning_rate=0.01, lambda_reg=0.001, seed=42, verbose=False)
    m.fit(ds)

    # Cornac internal item index -> dataset i_idx
    citem_to_our = np.full(ds.num_items, -1, np.int64)
    for s, c in ds.iid_map.items():
        citem_to_our[int(c)] = int(s)
    cidx_known = np.where(citem_to_our >= 0)[0]
    our_known = citem_to_our[cidx_known]
    uid_map = ds.uid_map

    scores = np.zeros((nU, nI), dtype=np.float32)
    miss = 0
    for u in range(nU):
        us = str(u)
        if us not in uid_map:
            miss += 1; continue
        sc = np.asarray(m.score(int(uid_map[us])), dtype=np.float32)
        scores[u, our_known] = sc[cidx_known]
    out = DEFAULT_DATA_ROOT / "data" / city / "backbone"; out.mkdir(parents=True, exist_ok=True)
    np.save(out / "BPR.scores_user.npy", scores)
    print(f"[{city}] -> {out / 'BPR.scores_user.npy'}  shape={scores.shape}  (users without training pairs: {miss})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
