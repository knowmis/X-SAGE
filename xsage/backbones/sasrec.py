"""SASRec backbone (Kang & McAuley 2018), a self-attentive sequential model, trained with the Cornac
library. Each request is scored from its causal history. Saves validation/test score matrices to
data/<city>/backbone/. Usage:  python scripts/backbones/sasrec.py <city>"""
import sys
from collections import defaultdict
from pathlib import Path
import sys as _sys
import numpy as np
import pandas as pd
from cornac.data import SequentialDataset
from cornac.models import SASRec

ROOT = Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(ROOT))


def main():
    city = sys.argv[1] if len(sys.argv) > 1 else "nyc_tist"
    dev = sys.argv[2] if len(sys.argv) > 2 else "mps"
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else None   # optional per-seed suffix -> file .s<seed>
    TAG = f".s{seed}" if seed is not None else ""
    SEED = seed if seed is not None else 42
    P = ROOT / "data" / "processed" / city
    dtr = pd.read_parquet(P / "df_train.parquet"); dva = pd.read_parquet(P / "df_val.parquet"); dte = pd.read_parquet(P / "df_test.parquet")
    n_items = int(pd.concat([dtr, dva, dte]).i_idx.max()) + 1
    nv, nt = len(dva), len(dte)
    for d, s in [(dtr, "train"), (dva, "val"), (dte, "test")]:
        d["split"] = s; d["pos"] = np.arange(len(d))           # row position in the parquet (evaluation order)
    print(f"[{city}] SASRec fit(dev={dev}, ep={epochs}) | n_items={n_items} n_val={nv} n_test={nt}", flush=True)

    # train USIT (user, session=user, item, timestamp)
    dtr_s = dtr.sort_values(["u_idx", "time_local"], kind="stable")
    tcol = (dtr_s.groupby("u_idx").cumcount() + 1).values
    usit = list(zip(dtr_s.u_idx.astype(str), dtr_s.u_idx.astype(str), dtr_s.i_idx.astype(str), tcol))
    ds = SequentialDataset.build(usit, fmt="USIT")
    uid, iid = dict(ds.uid_map), dict(ds.iid_map)
    inv_i = np.full(len(iid), -1, np.int64)
    for si, c in iid.items():
        inv_i[int(c)] = int(si)                                 # cornac_item -> our i_idx

    m = SASRec(embedding_dim=64, n_epochs=epochs, max_len=50, num_blocks=2, num_heads=1,
               device=dev, seed=SEED, batch_size=256, verbose=False)
    m.fit(ds)

    Mv = np.zeros((nv, n_items), np.float16); Mt = np.zeros((nt, n_items), np.float16)
    alld = pd.concat([dtr, dva, dte], ignore_index=True).sort_values(["u_idx", "time_local"], kind="stable")
    hist = defaultdict(list)                                     # u -> [cornac_item_idx]  (causale)
    nsv = nst = 0
    for r in alld.itertuples():
        u = r.u_idx
        if r.split in ("val", "test"):
            uc = uid.get(str(u))
            if uc is not None and hist[u]:
                sc = np.asarray(m.score(int(uc), hist[u]), np.float32)   # (n_cornac_items,)
                row = np.zeros(n_items, np.float16); row[inv_i] = sc.astype(np.float16)
                if r.split == "val": Mv[r.pos] = row; nsv += 1
                else: Mt[r.pos] = row; nst += 1
        ic = iid.get(str(r.i_idx))
        if ic is not None: hist[u].append(int(ic))              # aggiorna DOPO lo score (causale)
    out = ROOT / "data" / city / "backbone"; out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"SASRec{TAG}.scores_val.npy", Mv); np.save(out / f"SASRec{TAG}.scores_test.npy", Mt)
    print(f"[{city}] -> SASRec{TAG}.scores_{{val,test}}.npy  val{Mv.shape} test{Mt.shape}  scorate val={nsv}/{nv} test={nst}/{nt}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
