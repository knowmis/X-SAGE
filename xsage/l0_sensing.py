"""L0 — sensing.

Turn the raw check-in stream (already preprocessed by step01) into clean,
strictly-causal per-request inputs. For every request ``q = (u, t)`` we
extract the user's last ``n`` ``cat_macro`` moves before ``t``.

This level is intentionally thin: preprocessing already enforced the per-user
temporal 80/10/10 split, the k-core filter, and the cold-leakage drop.

Inputs:
    df_train, df_val, df_test (pd.DataFrame) — from ``data/processed/<city>/``.
Outputs (for any split):
    A dict with NumPy arrays:
        user_idx    (B,)        int32   — request user index
        time_local  (B,)        ns int  — request timestamp
        cat_target  (B,)        int32   — observed next macro (= row's
                                          ``cat_macro``)
        recent_macro (B, n)     int32   — last n macros strictly before t
                                          padded with -1 if history is short
        recent_dt   (B, n)      f32     — minutes elapsed (most recent first)
        n_prior     (B,)        int32   — number of valid entries in
                                          ``recent_macro`` (0..n)

History rule per the brief §2:
    train rows → history = train rows strictly before (self)
    val   rows → history = train rows (full)
    test  rows → history = train ∪ val rows (refit step)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class L0Output:
    user_idx: np.ndarray        # (B,) int32
    time_local: np.ndarray      # (B,) datetime64[ns]
    cat_target: np.ndarray      # (B,) int32
    recent_macro: np.ndarray    # (B, n) int32, -1 padding
    recent_dt_min: np.ndarray   # (B, n) float32, NaN where padded
    n_prior: np.ndarray         # (B,) int32

    def __len__(self) -> int:
        return len(self.user_idx)


def _macro_to_index(df: pd.DataFrame) -> dict[str, int]:
    """Stable cat_macro → integer mapping, alphabetical."""
    macros = sorted(df["cat_macro"].unique())
    return {m: i for i, m in enumerate(macros)}


def build_recent_window(target_df: pd.DataFrame,
                         history_df: pd.DataFrame,
                         macro_to_idx: dict[str, int],
                         n: int = 5) -> L0Output:
    """For every row in ``target_df`` build the strictly-causal last-n window
    from rows in ``history_df`` of the **same user** at ``time_local < t``.

    ``target_df`` and ``history_df`` may be the same dataframe (train→train).
    """
    h = (history_df[["user_id", "time_local", "cat_macro"]]
         .sort_values(["user_id", "time_local"])
         .reset_index(drop=True))
    by_user: dict[int, dict[str, np.ndarray]] = {}
    for u, g in h.groupby("user_id", sort=False):
        by_user[int(u)] = {
            "t": g["time_local"].values.astype("datetime64[ns]"),
            "m": np.array([macro_to_idx.get(x, -1) for x in g["cat_macro"].values],
                          dtype=np.int32),
        }

    B = len(target_df)
    out_macro = np.full((B, n), -1, dtype=np.int32)
    out_dt = np.full((B, n), np.nan, dtype=np.float32)
    out_n = np.zeros(B, dtype=np.int32)

    users = target_df["user_id"].values.astype(np.int64)
    times = target_df["time_local"].values.astype("datetime64[ns]")

    for b in range(B):
        u = int(users[b]); t = times[b]
        rec = by_user.get(u)
        if rec is None:
            continue
        cut = np.searchsorted(rec["t"], t, side="left")
        if cut == 0:
            continue
        start = max(0, cut - n)
        win_m = rec["m"][start:cut]
        win_t = rec["t"][start:cut]
        # most recent first
        win_m = win_m[::-1]
        win_t = win_t[::-1]
        L = len(win_m)
        out_macro[b, :L] = win_m
        # minutes elapsed since each historical timestamp w.r.t. request t
        deltas = (t - win_t).astype("timedelta64[s]").astype(np.float64) / 60.0
        out_dt[b, :L] = deltas.astype(np.float32)
        out_n[b] = L

    return L0Output(
        user_idx=target_df["u_idx"].values.astype(np.int32),
        time_local=times,
        cat_target=np.array(
            [macro_to_idx.get(x, -1) for x in target_df["cat_macro"].values],
            dtype=np.int32,
        ),
        recent_macro=out_macro,
        recent_dt_min=out_dt,
        n_prior=out_n,
    )
