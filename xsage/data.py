"""Loaders of the processed splits and of the backbone scores; data root: repository root or XSAGE_DATA_ROOT."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sps


DEFAULT_DATA_ROOT = Path(os.environ.get("XSAGE_DATA_ROOT", str(Path(__file__).resolve().parent.parent)))
DEFAULT_CITIES = ("ml1m", "nyc_tist", "saopaulo", "yelp", "kuairand")


def load_city(city: str, data_root: Path | None = None) -> dict:
    root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
    proc = root / "data" / "processed" / city
    df_train = pd.read_parquet(proc / "df_train.parquet")
    df_val = pd.read_parquet(proc / "df_val.parquet")
    df_test = pd.read_parquet(proc / "df_test.parquet")

    macros = sorted(set(df_train["cat_macro"].unique()) |
                    set(df_val["cat_macro"].unique()) |
                    set(df_test["cat_macro"].unique()))
    macro_to_idx = {m: i for i, m in enumerate(macros)}
    idx_to_macro = {i: m for m, i in macro_to_idx.items()}
    n_macros = len(macros)

    # Item / user catalogues
    n_users = int(max(df_train["u_idx"].max(), df_val["u_idx"].max(),
                      df_test["u_idx"].max()) + 1)
    n_items = int(max(df_train["i_idx"].max(), df_val["i_idx"].max(),
                      df_test["i_idx"].max()) + 1)

    # URM files: source of item popularity and of the seen-item mask.
    urm_train = sps.load_npz(proc / "URM_train.npz").tocsr()
    urm_val = sps.load_npz(proc / "URM_val.npz").tocsr()

    return {
        "city": city,
        "df_train": df_train, "df_val": df_val, "df_test": df_test,
        "macro_to_idx": macro_to_idx, "idx_to_macro": idx_to_macro,
        "n_macros": n_macros, "n_users": n_users, "n_items": n_items,
        "urm_train": urm_train, "urm_val": urm_val,
    }


def load_bpr_scores(city: str, data_root: Path | None = None) -> np.ndarray:
    """Memory-mapped (n_users, n_items) scores of the context-blind BPR backbone."""
    root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
    return np.load(root / "data" / city / "backbone" / "BPR.scores_user.npy", mmap_mode="r")


def load_excluded_mask(city: str, n_items: int,
                       data_root: Path | None = None) -> sps.csr_matrix:
    """(n_users, n_items) mask of each user's already-seen items (URM_train + URM_val)."""
    root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
    p = root / "data" / "processed" / city
    mask = (sps.load_npz(p / "URM_train.npz")
            + sps.load_npz(p / "URM_val.npz")).tocsr()
    mask.data[:] = 1.0
    return mask
