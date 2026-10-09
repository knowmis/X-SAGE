"""Train DeepFM or AFM with a BPR loss; writes data/<city>/backbone/<DeepFM|AFM>.scores_{val,test}.npy.

Usage:  python -m xsage.backbones.deepfm_afm <city> <DeepFM|AFM> [epochs=12]
"""
import sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fnn

from xsage import data as D
from xsage.backbones import default_device
from xsage.backbones.fm import feats_from_df
from xsage.backbones.fm import FeatureSpec
from xsage.backbones.deep import DeepFM, AFM

EMB_D, EPOCHS_DEF, BATCH, LR, L2 = 32, 12, 4096, 5e-3, 1e-5
FIELDS = ["u_idx", "i_idx", "macro_idx", "fine_idx", "c_hour", "c_dow", "c_isw", "c_month", "prev_geo_idx", "intent_last_idx"]
FOFF = ["user", "item", "macro", "fine", "hour", "dow", "isw", "month", "prev_geo", "intent_last"]
CTX = [("u_idx", "user"), ("c_hour", "hour"), ("c_dow", "dow"), ("c_isw", "isw"),
       ("c_month", "month"), ("prev_geo_idx", "prev_geo"), ("intent_last_idx", "intent_last")]


def rows_idx(model, feats, idx, i_arr, macro_arr, dev):
    """(B,10) offset indices for forward_row, given item i_arr (and its macro)."""
    offs = model.offsets
    cols = {"u_idx": feats["u_idx"][idx], "i_idx": i_arr, "macro_idx": macro_arr,
            "fine_idx": feats["fine_idx"][idx], "c_hour": feats["c_hour"][idx], "c_dow": feats["c_dow"][idx],
            "c_isw": feats["c_isw"][idx], "c_month": feats["c_month"][idx],
            "prev_geo_idx": feats["prev_geo_idx"][idx], "intent_last_idx": feats["intent_last_idx"][idx]}
    out = np.stack([cols[f] + offs[o] for f, o in zip(FIELDS, FOFF)], 1)
    return torch.from_numpy(out).to(dev)


def train(model, feats, mask, icm, dev, n_epochs, seed=42):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=L2)
    n = len(feats["u_idx"]); nI = model.spec.n_items; t0 = time.time()
    for ep in range(1, n_epochs + 1):
        model.train(); order = rng.permutation(n); tot = 0.0; nb = 0
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]; B = len(idx)
            u_np = feats["u_idx"][idx]; ipos = feats["i_idx"][idx]
            ineg = rng.integers(0, nI, B)
            for _ in range(4):
                bad = np.array([ineg[j] in mask.indices[mask.indptr[u_np[j]]:mask.indptr[u_np[j] + 1]] for j in range(B)])
                if not bad.any(): break
                ineg[bad] = rng.integers(0, nI, int(bad.sum()))
            ip = rows_idx(model, feats, idx, ipos, icm[ipos], dev)
            ng = rows_idx(model, feats, idx, ineg, icm[ineg], dev)
            yp = model.forward_row(ip); yn = model.forward_row(ng)
            loss = -Fnn.logsigmoid(yp - yn).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"    {model.__class__.__name__} e{ep:02d} loss={tot/max(1,nb):.4f}", flush=True)
    print(f"    train done in {time.time()-t0:.0f}s", flush=True)


@torch.no_grad()
def score_split(model, feats, icm, dev, bs=128):
    model.eval(); offs = model.offsets; nI = model.spec.n_items; n = len(feats["u_idx"])
    item_off = torch.arange(nI, device=dev) + offs["item"]
    macro_off = torch.from_numpy(icm).to(dev) + offs["macro"]
    fine_off = torch.zeros(nI, dtype=torch.long, device=dev) + offs["fine"]
    out = np.zeros((n, nI), np.float16)
    for s in range(0, n, bs):
        e = min(n, s + bs); idx = np.arange(s, e)
        ctx = torch.stack([torch.from_numpy(feats[c][idx]).to(dev) + offs[o] for c, o in CTX], -1)
        S = model.score_catalogue(ctx, item_off, macro_off, fine_off)
        out[s:e] = S.cpu().numpy().astype(np.float16)
    return out


def main():
    city = sys.argv[1] if len(sys.argv) > 1 else "nyc_tist"
    name = sys.argv[2] if len(sys.argv) > 2 else "DeepFM"
    n_ep = int(sys.argv[3]) if len(sys.argv) > 3 else EPOCHS_DEF
    SEED = 42
    torch.manual_seed(SEED); np.random.seed(SEED)             # model initialisation
    dev = torch.device(default_device())
    ds = D.load_city(city); nmac = int(ds["n_macros"]); nI = int(ds["n_items"]); nU = int(ds["n_users"])
    dfa = pd.concat([ds["df_train"], ds["df_val"], ds["df_test"]], ignore_index=True)
    icm = (dfa.groupby("i_idx")["cat_macro"].first().map(ds["macro_to_idx"]).reindex(np.arange(nI), fill_value=0).values.astype(np.int64))
    spec = FeatureSpec(n_users=nU, n_items=nI, n_macros=nmac, n_fine=1, n_geo=0, n_intent_last=nmac)
    Model = {"DeepFM": DeepFM, "AFM": AFM}[name]
    model = Model(spec, d=EMB_D).to(dev)
    print(f"[{city}] {name} (dev={dev}) nU={nU} nI={nI} nmac={nmac} ep={n_ep}", flush=True)
    ftr = feats_from_df(ds["df_train"], icm, nmac); fva = feats_from_df(ds["df_val"], icm, nmac); fte = feats_from_df(ds["df_test"], icm, nmac)
    mask = (ds["urm_train"] + ds["urm_val"]).tocsr(); mask.data[:] = 1.
    train(model, ftr, mask, icm, dev, n_ep, seed=SEED)
    Mv = score_split(model, fva, icm, dev); Mt = score_split(model, fte, icm, dev)
    out = D.DEFAULT_DATA_ROOT / "data" / city / "backbone"; out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{name}.scores_val.npy", Mv); np.save(out / f"{name}.scores_test.npy", Mt)
    print(f"[{city}] -> {name}.scores_{{val,test}}.npy  val{Mv.shape} test{Mt.shape}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
