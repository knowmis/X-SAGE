"""EASE backbone (Steck, 2019, Cornac); writes data/<city>/backbone/EASE.scores_user.npy.

Usage:  python -m xsage.backbones.ease <city> [lambda=500]
"""
import sys
import numpy as np
import pandas as pd
from cornac.data import Dataset
from cornac.models import EASE

from xsage.data import DEFAULT_DATA_ROOT


def main():
    city = sys.argv[1] if len(sys.argv) > 1 else "nyc_tist"
    lamb = float(sys.argv[2]) if len(sys.argv) > 2 else 500.0
    P = DEFAULT_DATA_ROOT / "data" / "processed" / city
    dtr = pd.read_parquet(P / "df_train.parquet")
    alld = pd.concat([dtr, pd.read_parquet(P / "df_val.parquet"), pd.read_parquet(P / "df_test.parquet")], ignore_index=True)
    n_users = int(alld.u_idx.max()) + 1
    n_items = int(alld.i_idx.max()) + 1
    print(f"[{city}] EASE lamb={lamb} | n_users={n_users} n_items={n_items} | train_int={len(dtr)}", flush=True)

    data = list(zip(dtr.u_idx.astype(str), dtr.i_idx.astype(str), np.ones(len(dtr), np.float32)))
    ts = Dataset.from_uir(data)
    m = EASE(lamb=lamb, verbose=False)
    m.fit(ts)

    # Cornac index maps -> dataset indices
    uid = dict(ts.uid_map)                      # dataset u_idx (str) -> Cornac user index
    iid = dict(ts.iid_map)                       # dataset i_idx (str) -> Cornac item index
    inv_i = np.full(len(iid), -1, np.int64)
    for si, c in iid.items():
        inv_i[int(c)] = int(si)                  # Cornac item index -> dataset i_idx

    M = np.zeros((n_users, n_items), np.float16)
    for su, uc in uid.items():
        sc = np.asarray(m.score(int(uc)), np.float32)   # (n_cornac_items,)
        M[int(su), inv_i] = sc.astype(np.float16)
    out = DEFAULT_DATA_ROOT / "data" / city / "backbone"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "EASE.scores_user.npy", M)
    print(f"[{city}] -> {out/'EASE.scores_user.npy'}  shape={M.shape}  (users scored: {len(uid)})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
