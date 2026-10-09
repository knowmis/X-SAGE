"""L1 perception: context state c_tilde (per-attribute trees) and intent vector e (propagation on W)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# Context state c_tilde

# Context attributes; none depends on the request's item.
DEFAULT_ATTRIBUTES = (
    "c_hour", "c_dow", "c_isweekend", "c_month",
    "prev_geohash5",          # geohash of the previous interaction of the user
    "intent_last_cat_idx",    # category index of the previous interaction
)


@dataclass
class ContributionModel:
    """One shallow tree per attribute; theta of a leaf = 1 - normalised entropy of its category distribution."""
    attributes: tuple[str, ...]
    trees: dict                       # attribute -> fitted DecisionTreeClassifier
    leaf_theta: dict                  # attribute -> dict[leaf_id -> theta]
    vocab: dict                       # attribute -> dict[value -> int]
    n_macros: int

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Return c_tilde of shape (B, A)."""
        rows = []
        for a in self.attributes:
            col = self._encode_column(a, df)
            leaves = self.trees[a].apply(col.reshape(-1, 1))
            theta = np.array([self.leaf_theta[a].get(int(l), 0.0) for l in leaves],
                             dtype=np.float32)
            rows.append(theta)
        return np.stack(rows, axis=1)            # (B, A)

    def _encode_column(self, a: str, df: pd.DataFrame) -> np.ndarray:
        if a in self.vocab:
            voc = self.vocab[a]
            return np.array([voc.get(str(v), -1) for v in df[a].values],
                            dtype=np.int32)
        return df[a].values.astype(np.int32)


def fit_contribution_functions(df_train: pd.DataFrame,
                               macro_to_idx: dict[str, int],
                               attributes: tuple[str, ...] = DEFAULT_ATTRIBUTES,
                               max_depth: int = 3,
                               min_leaf: int = 200) -> ContributionModel:
    """Fit one shallow tree per attribute predicting the category of the request's item."""
    y = df_train["cat_target"].values.astype(np.int64)
    vocab: dict[str, dict[str, int]] = {}
    trees: dict = {}
    leaf_theta: dict[str, dict[int, float]] = {}
    n_macros = max(macro_to_idx.values()) + 1
    log2K = float(np.log2(n_macros))

    for a in attributes:
        col = df_train[a].values
        if col.dtype.kind in ("O", "U"):
            uniq = sorted(set(str(v) for v in col))
            voc = {v: i for i, v in enumerate(uniq)}
            vocab[a] = voc
            x = np.array([voc[str(v)] for v in col], dtype=np.int32)
        else:
            x = col.astype(np.int32)
        tree = DecisionTreeClassifier(max_depth=max_depth,
                                      min_samples_leaf=min_leaf,
                                      random_state=42)
        tree.fit(x.reshape(-1, 1), y)
        trees[a] = tree
        leaf_ids = tree.apply(x.reshape(-1, 1))
        theta = {}
        for leaf in np.unique(leaf_ids):
            mask = leaf_ids == leaf
            counts = np.bincount(y[mask], minlength=n_macros).astype(np.float64)
            p = counts / counts.sum()
            H = -np.sum(p * np.log2(p + 1e-12))
            theta[int(leaf)] = float(1.0 - H / log2K)
        leaf_theta[a] = theta

    return ContributionModel(
        attributes=tuple(attributes),
        trees=trees, leaf_theta=leaf_theta, vocab=vocab,
        n_macros=n_macros,
    )


# Category transition matrix W and attractors A

def estimate_macro_transition(df_train: pd.DataFrame,
                              macro_to_idx: dict[str, int],
                              add_one_smoothing: bool = True) -> np.ndarray:
    """W[c, c'] = P(next = c' | current = c) from successive categories of each user, add-one smoothing."""
    n = len(macro_to_idx)
    counts = np.zeros((n, n), dtype=np.float64)
    df = df_train.sort_values(["user_id", "time_local"]).reset_index(drop=True)
    m = np.array([macro_to_idx[c] for c in df["cat_macro"].values], dtype=np.int32)
    u = df["user_id"].values
    same = u[1:] == u[:-1]
    src = m[:-1][same]; dst = m[1:][same]
    np.add.at(counts, (src, dst), 1)
    if add_one_smoothing:
        counts += 1.0
    W = counts / counts.sum(axis=1, keepdims=True)
    return W


def find_attractors(W: np.ndarray) -> np.ndarray:
    """Attractors: categories whose column sum of W is at least the mean (Boolean mask)."""
    indeg = W.sum(axis=0)
    return indeg >= indeg.mean()


# Recency profile m and intent vector e

def compute_profile(recent_macro: np.ndarray,
                    n_prior: np.ndarray,
                    n_macros: int,
                    gamma: float = 0.6) -> np.ndarray:
    """Recency profile m_c proportional to sum_j gamma^j 1[macro_j = c]; uniform for empty histories."""
    B, n = recent_macro.shape
    m = np.zeros((B, n_macros), dtype=np.float32)
    g = np.array([gamma ** j for j in range(n)], dtype=np.float32)
    for j in range(n):
        valid = recent_macro[:, j] >= 0
        if not valid.any():
            continue
        row_ix = np.where(valid)[0]
        col_ix = recent_macro[valid, j]
        np.add.at(m, (row_ix, col_ix), g[j])
    rows = m.sum(axis=1, keepdims=True)
    # empty rows -> uniform; non-empty -> normalise
    empty = (n_prior == 0)
    if empty.any():
        m[empty] = 1.0 / n_macros
    nz = (~empty)
    m[nz] /= np.maximum(rows[nz], 1e-9)
    return m


def compute_intent(m: np.ndarray, W: np.ndarray, attractors: np.ndarray,
                   H: int = 2, beta: float = 0.7) -> np.ndarray:
    """e = normalise((m sum_{k=1..H} beta^k W^k) * 1[A]), shape (B, n_macros)."""
    K = W.shape[0]
    acc = np.zeros_like(W)
    Wk = np.eye(K, dtype=np.float64)
    for k in range(1, H + 1):
        Wk = Wk @ W
        acc += (beta ** k) * Wk
    raw = m.astype(np.float64) @ acc                       # (B, K)
    raw = raw * attractors.astype(np.float64)[None, :]
    s = raw.sum(axis=1, keepdims=True)
    safe = np.where(s > 0, s, 1.0)
    e = raw / safe
    return e.astype(np.float32)
