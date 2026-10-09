"""L0 sensing: categories of the user's last n interactions strictly before each request."""
from __future__ import annotations

from dataclasses import dataclass

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


def build_recent_window(target_df: pd.DataFrame,
                        history_df: pd.DataFrame,
                        macro_to_idx: dict[str, int],
                        n: int = 5) -> L0Output:
    """Last-n window of each target row from the same user's history rows with time_local < t."""
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
