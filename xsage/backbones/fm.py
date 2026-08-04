"""Context-aware factorization machine (Rendle FM) with multi-hot features.

This is the focal backbone that ingests the same context end-to-end.
Features (concatenated multi-hot, one active index per group per row):

    user           (n_users)
    item           (n_items)
    cat_macro      (n_macros)        target item's macro
    cat_fine       (n_fine)          target item's fine category
    c_hour         (24)              request time
    c_dow          (7)
    c_isweekend    (2)
    c_month        (12)
    prev_geohash5  (n_geo + 1)       previous check-in's geohash (causal),
                                      index 0 = no previous location
    intent_last    (n_macros + 1)    previous interaction's macro,
                                      index 0 = no previous

Order-2 FM trick:
    score = w_0 + sum_f w_f + 0.5 ( (sum_f e_f)² − sum_f e_f² )

Trained on (train ∪ val) with BPR loss: for each positive request
(u, t, i_pos), sample one negative item i_neg not in the user's URM history,
build the two feature rows, minimise -log σ(score(pos) - score(neg)).

For evaluation we score *all* items per request — vectorised over the
catalogue using the standard "context-block + item-block" decomposition so
the (B, I, d) tensor never has to be materialised.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sps
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class FeatureSpec:
    n_users: int
    n_items: int
    n_macros: int
    n_fine: int
    n_geo: int            # excludes the +1 sentinel
    n_intent_last: int    # excludes the +1 sentinel
    n_hours: int = 24
    n_dows: int = 7
    n_isw: int = 2
    n_months: int = 12

    def sizes(self) -> list[int]:
        """In the same order as ContextAwareFM uses them in forward()."""
        return [self.n_users, self.n_items, self.n_macros, self.n_fine,
                self.n_hours, self.n_dows, self.n_isw, self.n_months,
                self.n_geo + 1, self.n_intent_last + 1]

    def offsets(self) -> dict[str, int]:
        sz = self.sizes()
        names = ["user", "item", "macro", "fine", "hour", "dow", "isw",
                  "month", "prev_geo", "intent_last"]
        out = {}; acc = 0
        for n, s in zip(names, sz):
            out[n] = acc; acc += s
        out["_total"] = acc
        return out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ContextAwareFM(nn.Module):
    def __init__(self, spec: FeatureSpec, d: int = 32) -> None:
        super().__init__()
        self.spec = spec
        offs = spec.offsets()
        self.offsets = offs
        F_total = offs["_total"]
        self.d = d
        self.emb = nn.Embedding(F_total, d)
        self.lin = nn.Embedding(F_total, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.emb.weight, std=0.05)
        nn.init.zeros_(self.lin.weight)

    # -------- one positive / one negative item per row -----------------
    def _row_indices_for_item(self,
                                u: torch.Tensor, i: torch.Tensor,
                                macro_of_i: torch.Tensor,
                                fine_of_i: torch.Tensor,
                                c_hour, c_dow, c_isw, c_month,
                                prev_geo, intent_last) -> torch.Tensor:
        """Build the (B, F_active=10) tensor of active feature indices."""
        offs = self.offsets
        return torch.stack([
            u + offs["user"],
            i + offs["item"],
            macro_of_i + offs["macro"],
            fine_of_i + offs["fine"],
            c_hour + offs["hour"],
            c_dow + offs["dow"],
            c_isw + offs["isw"],
            c_month + offs["month"],
            prev_geo + offs["prev_geo"],
            intent_last + offs["intent_last"],
        ], dim=-1)                                            # (B, 10)

    def forward_row(self, idx: torch.Tensor) -> torch.Tensor:
        """``idx`` is (B, F_active) long. Returns (B,) score."""
        e = self.emb(idx)                                     # (B, F, d)
        w = self.lin(idx).squeeze(-1)                         # (B, F)
        linear = self.bias + w.sum(dim=-1)
        sum_e = e.sum(dim=-2)                                 # (B, d)
        sum_e_sq = e.pow(2).sum(dim=-2)                       # (B, d)
        order2 = 0.5 * (sum_e.pow(2).sum(dim=-1)
                          - sum_e_sq.sum(dim=-1))
        return linear + order2

    # -------- per-request catalogue scoring -----------------------------
    def score_full_catalogue(self,
                              context_idx: torch.Tensor,
                              item_idx_in_emb: torch.Tensor,
                              macro_idx_in_emb: torch.Tensor,
                              fine_idx_in_emb: torch.Tensor) -> torch.Tensor:
        """For each request build s(i) for every item i.

        Args:
            context_idx:    (B, 6) — active indices for the *non-item* feature
                            groups (user, hour, dow, isw, month, prev_geo, intent_last).
                            Note: that's 7, but `macro` and `fine` for the item
                            depend on the candidate, so we pass them separately.
                            Caller assembles ``context_idx`` with 7 columns.
            item_idx_in_emb:  (I,) — already offset
            macro_idx_in_emb: (I,) — already offset (per item's cat_macro)
            fine_idx_in_emb:  (I,) — already offset (per item's cat_fine)

        Returns:
            (B, I) score tensor.
        """
        # Context-side aggregates
        e_ctx = self.emb(context_idx)                         # (B, 7, d)
        w_ctx = self.lin(context_idx).squeeze(-1).sum(dim=-1)  # (B,)
        sum_ctx = e_ctx.sum(dim=-2)                           # (B, d)
        sumsq_ctx = e_ctx.pow(2).sum(dim=-2)                  # (B, d)

        # Item-side aggregates over (item, macro, fine)
        e_item = self.emb(item_idx_in_emb)                    # (I, d)
        e_macro = self.emb(macro_idx_in_emb)                  # (I, d)
        e_fine = self.emb(fine_idx_in_emb)                    # (I, d)
        sum_item = e_item + e_macro + e_fine                  # (I, d)
        sumsq_item = e_item.pow(2) + e_macro.pow(2) + e_fine.pow(2)

        w_item = (self.lin(item_idx_in_emb).squeeze(-1)
                    + self.lin(macro_idx_in_emb).squeeze(-1)
                    + self.lin(fine_idx_in_emb).squeeze(-1))   # (I,)

        # (B, I, d)
        sum_total = sum_ctx.unsqueeze(1) + sum_item.unsqueeze(0)
        sumsq_total = sumsq_ctx.unsqueeze(1) + sumsq_item.unsqueeze(0)
        # 0.5 * sum_d (sum_total² - sumsq_total)
        order2 = 0.5 * (sum_total.pow(2).sum(dim=-1)
                          - sumsq_total.sum(dim=-1))           # (B, I)

        linear = self.bias + w_ctx.unsqueeze(-1) + w_item.unsqueeze(0)
        return linear + order2


# ---------------------------------------------------------------------------
# Per-row feature extraction from a SplitArrays-like dataframe
# ---------------------------------------------------------------------------

def _build_request_features(df: pd.DataFrame,
                              spec: FeatureSpec,
                              macro_to_idx: dict[str, int],
                              fine_to_idx: dict[str, int],
                              geo_to_idx: dict[str, int]) -> dict[str, np.ndarray]:
    """Per row, extract integer-coded values for the FM features. Returns a
    dict with keys: u_idx, i_idx, macro_idx, fine_idx, c_hour, c_dow, c_isw,
    c_month, prev_geo_idx, intent_last_idx.
    """
    n_macros = len(macro_to_idx)
    macro_arr = np.array([macro_to_idx.get(m, 0) for m in df["cat_macro"].values],
                          dtype=np.int64)
    fine_arr = np.array([fine_to_idx.get(f, 0) for f in df["cat_fine"].values],
                          dtype=np.int64)
    prev_geo = np.array([geo_to_idx.get(str(g), 0) for g in df["prev_geohash5"].values],
                          dtype=np.int64)
    intent_last = np.array([macro_to_idx.get(m, -1) + 1
                              if m != "None" else 0
                              for m in df["intent_last_cat"].values],
                              dtype=np.int64)
    return {
        "u_idx": df["u_idx"].values.astype(np.int64),
        "i_idx": df["i_idx"].values.astype(np.int64),
        "macro_idx": macro_arr,
        "fine_idx": fine_arr,
        "c_hour": df["c_hour"].values.astype(np.int64),
        "c_dow": df["c_dow"].values.astype(np.int64),
        "c_isw": df["c_isweekend"].values.astype(np.int64),
        # months in parquet are 1..12 — shift to 0..11.
        "c_month": (df["c_month"].values - df["c_month"].values.min()).astype(np.int64),
        "prev_geo_idx": prev_geo,
        "intent_last_idx": intent_last,
    }


def _catalogue_indices(df_all: pd.DataFrame,
                         spec: FeatureSpec,
                         macro_to_idx: dict[str, int],
                         fine_to_idx: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """For each i_idx in [0, n_items), find the most-common (macro, fine).

    Returns ``(macro_per_item, fine_per_item)``, both ``(n_items,)`` int.
    """
    # Most common macro per item
    counts_m = df_all.groupby(["i_idx", "cat_macro"]).size().reset_index(name="n")
    best_m = counts_m.sort_values(["i_idx", "n"], ascending=[True, False]) \
                       .drop_duplicates("i_idx", keep="first")
    macro_per_item = np.zeros(spec.n_items, dtype=np.int64)
    for _, r in best_m.iterrows():
        macro_per_item[int(r["i_idx"])] = macro_to_idx[r["cat_macro"]]

    # Most common fine per item
    counts_f = df_all.groupby(["i_idx", "cat_fine"]).size().reset_index(name="n")
    best_f = counts_f.sort_values(["i_idx", "n"], ascending=[True, False]) \
                       .drop_duplicates("i_idx", keep="first")
    fine_per_item = np.zeros(spec.n_items, dtype=np.int64)
    for _, r in best_f.iterrows():
        fine_per_item[int(r["i_idx"])] = fine_to_idx[r["cat_fine"]]

    return macro_per_item, fine_per_item


# ---------------------------------------------------------------------------
# Training (BPR)
# ---------------------------------------------------------------------------

def train_b_full(model: ContextAwareFM,
                  feats: dict[str, np.ndarray],
                  mask_csr: sps.csr_matrix,
                  macro_per_item: np.ndarray,
                  fine_per_item: np.ndarray,
                  device: torch.device,
                  *, lr: float = 5e-3, weight_decay: float = 1e-5,
                  batch_size: int = 4096, n_epochs: int = 12,
                  seed: int = 42, verbose: bool = False) -> dict:
    """One-pass BPR training on the supplied (train ∪ val) features."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    optim = torch.optim.Adam(model.parameters(), lr=lr,
                              weight_decay=weight_decay)
    n = len(feats["u_idx"])
    n_items = model.spec.n_items
    macro_t = torch.from_numpy(macro_per_item).to(device)
    fine_t = torch.from_numpy(fine_per_item).to(device)
    t_start = time.time()
    history = []
    for epoch in range(1, n_epochs + 1):
        model.train()
        order = rng.permutation(n)
        epoch_loss = 0.0; nbatches = 0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            B = len(idx)
            u_np = feats["u_idx"][idx]
            i_pos_np = feats["i_idx"][idx]
            # sample negatives: uniform with rejection
            i_neg_np = rng.integers(0, n_items, size=B)
            tries = 0
            while tries < 6:
                bad = np.zeros(B, dtype=bool)
                for j, ur in enumerate(u_np):
                    cols = mask_csr.indices[mask_csr.indptr[ur]:mask_csr.indptr[ur + 1]]
                    if i_neg_np[j] in cols:
                        bad[j] = True
                if not bad.any():
                    break
                i_neg_np[bad] = rng.integers(0, n_items, size=int(bad.sum()))
                tries += 1
            u = torch.from_numpy(u_np).to(device)
            i_pos = torch.from_numpy(i_pos_np).to(device)
            i_neg = torch.from_numpy(i_neg_np).to(device)
            c_hour = torch.from_numpy(feats["c_hour"][idx]).to(device)
            c_dow = torch.from_numpy(feats["c_dow"][idx]).to(device)
            c_isw = torch.from_numpy(feats["c_isw"][idx]).to(device)
            c_month = torch.from_numpy(feats["c_month"][idx]).to(device)
            prev_geo = torch.from_numpy(feats["prev_geo_idx"][idx]).to(device)
            intent_last = torch.from_numpy(feats["intent_last_idx"][idx]).to(device)

            macro_pos = macro_t[i_pos]; macro_neg = macro_t[i_neg]
            fine_pos = fine_t[i_pos]; fine_neg = fine_t[i_neg]

            idx_pos = model._row_indices_for_item(
                u, i_pos, macro_pos, fine_pos,
                c_hour, c_dow, c_isw, c_month, prev_geo, intent_last)
            idx_neg = model._row_indices_for_item(
                u, i_neg, macro_neg, fine_neg,
                c_hour, c_dow, c_isw, c_month, prev_geo, intent_last)

            y_pos = model.forward_row(idx_pos)
            y_neg = model.forward_row(idx_neg)
            loss = -F.logsigmoid(y_pos - y_neg).mean()

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            epoch_loss += float(loss.item()); nbatches += 1

        avg = epoch_loss / max(1, nbatches)
        history.append({"epoch": epoch, "loss": avg})
        if verbose:
            print(f"    B_full e{epoch:02d}  loss={avg:.4f}")
    return {"history": history, "wallclock_s": time.time() - t_start,
              "n_epochs": n_epochs}


# ---------------------------------------------------------------------------
# Full per-request catalogue score dump
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_all_per_request(model: ContextAwareFM,
                              feats: dict[str, np.ndarray],
                              macro_per_item: np.ndarray,
                              fine_per_item: np.ndarray,
                              device: torch.device,
                              batch_size: int = 256) -> np.ndarray:
    """Returns ``(B, n_items)`` score matrix."""
    model.eval()
    n = len(feats["u_idx"])
    n_items = model.spec.n_items
    offs = model.offsets
    item_idx_in_emb = torch.arange(n_items, device=device) + offs["item"]
    macro_idx_in_emb = torch.from_numpy(macro_per_item).to(device) + offs["macro"]
    fine_idx_in_emb = torch.from_numpy(fine_per_item).to(device) + offs["fine"]

    scores = np.zeros((n, n_items), dtype=np.float32)
    for s in range(0, n, batch_size):
        idx = np.arange(s, min(s + batch_size, n))
        u = torch.from_numpy(feats["u_idx"][idx]).to(device)
        c_hour = torch.from_numpy(feats["c_hour"][idx]).to(device)
        c_dow = torch.from_numpy(feats["c_dow"][idx]).to(device)
        c_isw = torch.from_numpy(feats["c_isw"][idx]).to(device)
        c_month = torch.from_numpy(feats["c_month"][idx]).to(device)
        prev_geo = torch.from_numpy(feats["prev_geo_idx"][idx]).to(device)
        intent_last = torch.from_numpy(feats["intent_last_idx"][idx]).to(device)

        ctx_idx = torch.stack([
            u + offs["user"],
            c_hour + offs["hour"],
            c_dow + offs["dow"],
            c_isw + offs["isw"],
            c_month + offs["month"],
            prev_geo + offs["prev_geo"],
            intent_last + offs["intent_last"],
        ], dim=-1)                                            # (b, 7)

        sc = model.score_full_catalogue(ctx_idx,
                                           item_idx_in_emb,
                                           macro_idx_in_emb,
                                           fine_idx_in_emb)    # (b, I)
        scores[idx] = sc.detach().cpu().numpy()
    return scores


# ---------------------------------------------------------------------------
# Feature construction + full-catalogue scoring (shared by FM / DeepFM / AFM)
# ---------------------------------------------------------------------------

BATCH_SCORE = 512


def feats_from_df(df, icm, n_macros):
    """Integer feature dict (no geo, no fine). intent_last is remapped to 0=none, 1..n_macros."""
    il = df["intent_last_cat_idx"].values.astype(np.int64)
    il = np.where(il >= n_macros, 0, il + 1)              # 0 = none, 1..n_macros
    i_idx = df["i_idx"].values.astype(np.int64)
    return {
        "u_idx": df["u_idx"].values.astype(np.int64),
        "i_idx": i_idx,
        "macro_idx": icm[i_idx].astype(np.int64),
        "fine_idx": np.zeros(len(df), np.int64),          # fine categories disabled
        "c_hour": df["c_hour"].values.astype(np.int64),
        "c_dow": df["c_dow"].values.astype(np.int64),
        "c_isw": df["c_isweekend"].values.astype(np.int64),
        "c_month": (df["c_month"].values - df["c_month"].values.min()).astype(np.int64),
        "prev_geo_idx": np.zeros(len(df), np.int64),      # geohash disabled (sentinel)
        "intent_last_idx": il,
    }


def score_test(model, feats_te, spec, icm, device, dtype=np.float32):
    """Return the (n_test, n_items) score matrix via the model's full-catalogue scorer."""
    offs = spec.offsets(); I = spec.n_items
    items = torch.arange(I, device=device)
    item_off = items + offs["item"]
    macro_off = torch.from_numpy(icm.astype(np.int64)).to(device) + offs["macro"]
    fine_off = torch.zeros(I, dtype=torch.long, device=device) + offs["fine"]
    n = len(feats_te["u_idx"]); out = np.zeros((n, I), dtype)
    # context_idx (B, 7): user, hour, dow, isweekend, month, prev_geo, intent_last (offset)
    cols = [("u_idx", "user"), ("c_hour", "hour"), ("c_dow", "dow"), ("c_isw", "isw"),
            ("c_month", "month"), ("prev_geo_idx", "prev_geo"), ("intent_last_idx", "intent_last")]
    model.eval()
    with torch.no_grad():
        for bs in range(0, n, BATCH_SCORE):
            be = min(n, bs + BATCH_SCORE)
            ctx = torch.stack([torch.from_numpy(feats_te[c][bs:be]).to(device) + offs[o]
                               for c, o in cols], dim=-1)          # (B, 7)
            S = model.score_full_catalogue(ctx, item_off, macro_off, fine_off)   # (B, I)
            out[bs:be] = S.cpu().numpy()
    return out
