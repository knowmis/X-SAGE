"""Lightweight per-dataset loader.

Reads the processed split from  data/processed/<city>/{df_train,df_val,df_test}.parquet
and  URM_{train,val}.npz , and the backbone score matrices from
data/<city>/backbone/ . The data root defaults to the repository root and can be
overridden with the XSAGE_DATA_ROOT environment variable or a data_root argument.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sps


DEFAULT_DATA_ROOT = Path(os.environ.get("XSAGE_DATA_ROOT", str(Path(__file__).resolve().parent.parent)))
DEFAULT_CITIES = ("ml1m", "nyc_tist", "saopaulo", "yelp", "kuairand")

# Backbone score matrices are read from data/<city>/backbone/ under the repo root.
LOCAL_BACKBONE_ROOT = Path(__file__).resolve().parent.parent / "data"


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

    # URM cached by step01 (already binarised under the floor's protocol).
    # Reading the .npz is the source of truth for popularity + excluded_mask;
    # rebuilding from df would double-count any repeated (u, i) interactions.
    urm_train = sps.load_npz(proc / "URM_train.npz").tocsr()
    urm_val = sps.load_npz(proc / "URM_val.npz").tocsr()

    return {
        "city": city,
        "df_train": df_train, "df_val": df_val, "df_test": df_test,
        "macro_to_idx": macro_to_idx, "idx_to_macro": idx_to_macro,
        "n_macros": n_macros, "n_users": n_users, "n_items": n_items,
        "urm_train": urm_train, "urm_val": urm_val,
    }


def load_backbone_scores(city: str,
                              data_root: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (B_blind (n_users, n_items), B_full (n_test, n_items, mmap)).

    Reads the checksum-verified LOCAL copy under data/<city>/backbone/ to keep
    the repo standalone; falls back to the old repo only if a local file is
    absent.
    """
    root = Path(data_root) if data_root else DEFAULT_DATA_ROOT

    def _resolve(fn: str) -> Path:
        local = LOCAL_BACKBONE_ROOT / city / "backbone" / fn
        if local.exists():
            return local
        return root / "outputs" / city / "xsage" / "backbone" / fn

    blind = np.load(_resolve("FM.scores.npy"), mmap_mode="r")
    full = np.load(_resolve("Bfull.scores.npy"), mmap_mode="r")
    return blind, full


def load_excluded_mask(city: str, n_items: int,
                            data_root: Path | None = None) -> sps.csr_matrix:
    """(n_users, n_items) mask of items to exclude from ranking per user —
    URM_train ∪ URM_val (the floor's exclude_seen protocol)."""
    root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
    p = root / "data" / "processed" / city
    mask = (sps.load_npz(p / "URM_train.npz")
              + sps.load_npz(p / "URM_val.npz")).tocsr()
    mask.data[:] = 1.0
    return mask
