"""EASE backbone (Steck 2019), a static per-user linear item-item model, trained with the Cornac
library. Saves the per-user score matrix to data/<city>/backbone/.
Usage:  python scripts/backbones/ease.py <city>"""
import sys
from pathlib import Path
import sys as _sys
import numpy as np
import pandas as pd
from cornac.data import Dataset
from cornac.models import EASE

ROOT = Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(ROOT))


def main():
    city = sys.argv[1] if len(sys.argv) > 1 else "nyc_tist"
    lamb = float(sys.argv[2]) if len(sys.argv) > 2 else 500.0
    P = ROOT / "data" / "processed" / city
    dtr = pd.read_parquet(P / "df_train.parquet")
    alld = pd.concat([dtr, pd.read_parquet(P / "df_val.parquet"), pd.read_parquet(P / "df_test.parquet")], ignore_index=True)
    n_users = int(alld.u_idx.max()) + 1
    n_items = int(alld.i_idx.max()) + 1
    print(f"[{city}] EASE lamb={lamb} | n_users={n_users} n_items={n_items} | train_int={len(dtr)}", flush=True)

    data = list(zip(dtr.u_idx.astype(str), dtr.i_idx.astype(str), np.ones(len(dtr), np.float32)))
    ts = Dataset.from_uir(data)
    m = EASE(lamb=lamb, verbose=False)
    m.fit(ts)

    # cornac -> our index maps
    uid = dict(ts.uid_map)                      # "our_u"(str) -> cornac_user_idx
    iid = dict(ts.iid_map)                       # "our_i"(str) -> cornac_item_idx
    inv_i = np.full(len(iid), -1, np.int64)
    for si, c in iid.items():
        inv_i[int(c)] = int(si)                  # cornac_item_idx -> our i_idx

    M = np.zeros((n_users, n_items), np.float16)
    for su, uc in uid.items():
        sc = np.asarray(m.score(int(uc)), np.float32)   # (n_cornac_items,)
        M[int(su), inv_i] = sc.astype(np.float16)
    out = ROOT / "data" / city / "backbone"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "EASE.scores_user.npy", M)
    print(f"[{city}] -> {out/'EASE.scores_user.npy'}  shape={M.shape}  (utenti scorati={len(uid)})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
