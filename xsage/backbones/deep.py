"""Context-aware DeepFM (Guo et al. 2017) and AFM (Xiao et al. 2017) backbones.

Both share the factorization-machine core with the focal FM and consume the same
multi-hot features; DeepFM adds a DNN over the embeddings, AFM an attention over
the pairwise interactions. Trained with a BPR loss."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from xsage.backbones.fm import ContextAwareFM, FeatureSpec  # noqa: E402

N_FIELDS = 10  # user,item,macro,fine,hour,dow,isw,month,prev_geo,intent_last


# ---------------------------------------------------------------------------
# parte FM condivisa (identica a ContextAwareFM): linear + order-2 decomposto
# ---------------------------------------------------------------------------
class _FMShared(nn.Module):
    def __init__(self, spec: FeatureSpec, d: int):
        super().__init__()
        self.spec = spec; self.offsets = spec.offsets(); self.d = d
        F_total = self.offsets["_total"]
        self.emb = nn.Embedding(F_total, d)
        self.lin = nn.Embedding(F_total, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.emb.weight, std=0.05); nn.init.zeros_(self.lin.weight)

    def _fm_row(self, e, idx):  # e:(B,10,d) idx:(B,10) -> (linear(B,), order2(B,))
        w = self.lin(idx).squeeze(-1)
        linear = self.bias + w.sum(-1)
        se = e.sum(-2); ses = e.pow(2).sum(-2)
        order2 = 0.5 * (se.pow(2).sum(-1) - ses.sum(-1))
        return linear, order2

    def _fm_catalogue(self, ctx_idx, item_off, macro_off, fine_off):
        # identico a ContextAwareFM.score_full_catalogue -> (B,I) linear, (B,I) order2
        e_ctx = self.emb(ctx_idx); w_ctx = self.lin(ctx_idx).squeeze(-1).sum(-1)
        sum_ctx = e_ctx.sum(-2); sumsq_ctx = e_ctx.pow(2).sum(-2)
        e_i = self.emb(item_off); e_m = self.emb(macro_off); e_f = self.emb(fine_off)
        sum_it = e_i + e_m + e_f; sumsq_it = e_i.pow(2) + e_m.pow(2) + e_f.pow(2)
        w_it = (self.lin(item_off).squeeze(-1) + self.lin(macro_off).squeeze(-1) + self.lin(fine_off).squeeze(-1))
        sum_tot = sum_ctx.unsqueeze(1) + sum_it.unsqueeze(0)
        sumsq_tot = sumsq_ctx.unsqueeze(1) + sumsq_it.unsqueeze(0)
        order2 = 0.5 * (sum_tot.pow(2).sum(-1) - sumsq_tot.sum(-1))
        linear = self.bias + w_ctx.unsqueeze(-1) + w_it.unsqueeze(0)
        return linear, order2

    # per-field embeddings for the deep part: ctx (B,7,d) + item (I,3,d)
    def _field_embs(self, ctx_idx, item_off, macro_off, fine_off):
        e_ctx = self.emb(ctx_idx)                                  # (B,7,d)
        e_item = torch.stack([self.emb(item_off), self.emb(macro_off), self.emb(fine_off)], 1)  # (I,3,d)
        return e_ctx, e_item


def _row_embs(model, idx):
    return model.emb(idx)                                          # (B,10,d)


# ---------------------------------------------------------------------------
# DeepFM  [Guo et al. 2017]
# ---------------------------------------------------------------------------
class DeepFM(_FMShared):
    def __init__(self, spec, d=32, layers=(128, 128), dropout=0.1):
        super().__init__(spec, d)
        dims = [N_FIELDS * d, *layers]
        mlp = []
        for a, b in zip(dims[:-1], dims[1:]):
            mlp += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
        mlp += [nn.Linear(dims[-1], 1)]
        self.dnn = nn.Sequential(*mlp)
        nn.init.zeros_(self.dnn[-1].weight); nn.init.zeros_(self.dnn[-1].bias)  # all'init DNN~0 -> parte da FM

    def forward_row(self, idx, use_deep=True):
        e = _row_embs(self, idx); lin, o2 = self._fm_row(e, idx)
        s = lin + o2
        if use_deep:
            s = s + self.dnn(e.reshape(e.shape[0], -1)).squeeze(-1)
        return s

    @torch.no_grad()
    def score_catalogue(self, ctx_idx, item_off, macro_off, fine_off, use_deep=True, chunk=512):
        lin, o2 = self._fm_catalogue(ctx_idx, item_off, macro_off, fine_off)
        S = lin + o2                                               # (B,I) parte FM
        if use_deep:
            e_ctx, e_item = self._field_embs(ctx_idx, item_off, macro_off, fine_off)
            B = e_ctx.shape[0]; I = e_item.shape[0]; d = self.d
            ctx_flat = e_ctx.reshape(B, 7 * d)                     # (B,7d)
            for c0 in range(0, I, chunk):
                c1 = min(I, c0 + chunk); ic = e_item[c0:c1].reshape(c1 - c0, 3 * d)  # (c,3d)
                x = torch.cat([ctx_flat.unsqueeze(1).expand(B, c1 - c0, 7 * d),
                               ic.unsqueeze(0).expand(B, c1 - c0, 3 * d)], -1)        # (B,c,10d)
                S[:, c0:c1] = S[:, c0:c1] + self.dnn(x).squeeze(-1)
        return S


# ---------------------------------------------------------------------------
# AFM  [Xiao et al. 2017]   (Attentional Factorization Machine)
# ---------------------------------------------------------------------------
class AFM(_FMShared):
    def __init__(self, spec, d=32, attn=32, dropout=0.1):
        super().__init__(spec, d)
        self.att_W = nn.Linear(d, attn); self.att_h = nn.Linear(attn, 1, bias=False)
        self.p = nn.Linear(d, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.pairs = [(f, g) for f in range(N_FIELDS) for g in range(f + 1, N_FIELDS)]  # 45 pairs f<g

    def _pair_out(self, E, uniform=False):
        # E:(...,10,d) -> contributo AFM (...). Slicing semplice sui campi (NO advanced-index 4D -> mps-safe).
        prods = [E[..., f, :] * E[..., g, :] for f, g in self.pairs]   # 45 x (...,d)
        if uniform:
            v = torch.stack(prods, -2).sum(-2)                        # sum_{f<g} e_f*e_g -> order-2 FM
        else:
            scr = torch.stack([self.att_h(torch.relu(self.att_W(p))).squeeze(-1) for p in prods], -1)  # (...,45)
            a = torch.softmax(scr, -1)                                # (...,45)
            v = sum(a[..., k:k + 1] * prods[k] for k in range(len(prods)))  # (...,d)
        return self.p(self.drop(v)).squeeze(-1)                       # (...)

    def forward_row(self, idx, uniform=False):
        e = _row_embs(self, idx); lin, _ = self._fm_row(e, idx)
        return lin + self._pair_out(e, uniform=uniform)

    @torch.no_grad()
    def score_catalogue(self, ctx_idx, item_off, macro_off, fine_off, uniform=False, chunk=256):
        lin, _ = self._fm_catalogue(ctx_idx, item_off, macro_off, fine_off)
        e_ctx, e_item = self._field_embs(ctx_idx, item_off, macro_off, fine_off)
        B = e_ctx.shape[0]; I = e_item.shape[0]; d = self.d
        S = lin
        for c0 in range(0, I, chunk):
            c1 = min(I, c0 + chunk)
            E = torch.cat([e_ctx.unsqueeze(1).expand(B, c1 - c0, 7, d),
                           e_item[c0:c1].unsqueeze(0).expand(B, c1 - c0, 3, d)], 2)   # (B,c,10,d)
            S[:, c0:c1] = S[:, c0:c1] + self._pair_out(E, uniform=uniform)
        return S


# ---------------------------------------------------------------------------
# REDUCTION-TO-FM: prova di coerenza
# ---------------------------------------------------------------------------
@torch.no_grad()
def assert_reduces_to_fm(spec, device="cpu", d=16, B=64, tol=1e-4):
    """DeepFM (DNN off) equals the focal FM exactly; AFM (uniform attention) is proportional to its order-2 part (same ranking)."""
    torch.manual_seed(0)
    fm = ContextAwareFM(spec, d=d).to(device).eval()
    # ---- DeepFM: copia pesi FM, azzera DNN, confronta forward_row ----
    dfm = DeepFM(spec, d=d).to(device).eval()
    dfm.emb.weight.data.copy_(fm.emb.weight.data); dfm.lin.weight.data.copy_(fm.lin.weight.data); dfm.bias.data.copy_(fm.bias.data)
    sizes = spec.sizes(); rng = np.random.default_rng(0)
    idx = np.stack([rng.integers(0, s, B) for s in sizes], 1)
    offs_acc = np.cumsum([0, *sizes[:-1]]); idx = torch.from_numpy(idx + offs_acc).to(device)
    a = fm.forward_row(idx); b = dfm.forward_row(idx, use_deep=False)
    err_dfm = float((a - b).abs().max())
    # ---- AFM: copy FM weights; uniform attention + p=1 -> order-2 FM ----
    afm = AFM(spec, d=d).to(device).eval()
    afm.emb.weight.data.copy_(fm.emb.weight.data); afm.lin.weight.data.copy_(fm.lin.weight.data); afm.bias.data.copy_(fm.bias.data)
    nn.init.ones_(afm.p.weight)                                    # p = somma sulle dim -> <e_f,e_g>
    e = fm.emb(idx)                                                # order-2 di FM (formula diretta)
    se = e.sum(-2); ses = e.pow(2).sum(-2); o2_fm = 0.5 * (se.pow(2).sum(-1) - ses.sum(-1))
    o2_afm = afm._pair_out(e, uniform=True)                        # sum_{f<g} <e_f,e_g> == order-2 FM
    err_afm = float((o2_fm - o2_afm).abs().max())
    return {"DeepFM_vs_FM_maxerr": err_dfm, "AFM_order2_vs_FM_maxerr": err_afm,
            "DeepFM_ok": err_dfm < tol, "AFM_ok": err_afm < tol}


if __name__ == "__main__":
    spec = FeatureSpec(n_users=50, n_items=80, n_macros=8, n_fine=1, n_geo=20, n_intent_last=8)
    print(assert_reduces_to_fm(spec))
