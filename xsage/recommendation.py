"""Situational bias (Eq. (13)) and additive combiner (Eq. (14)): s_hat = s_B + kappa * gamma_S * sum_k r_k b_tilde^(k)_c(i)."""
from __future__ import annotations

import numpy as np

ALPHA = 50.0   # Dirichlet pseudo-count of the bias estimator, fixed for every dataset


# Bias estimation (Eq. (13))

def fit_situation_biases_z(z_train: np.ndarray, cat_macro_train: np.ndarray,
                           K: int, n_macros: int,
                           alpha: float = ALPHA) -> np.ndarray:
    """Per-label category bias (Eq. (13)): smoothed log-ratio to the training frequency, z-scored per label."""
    z_train = z_train.astype(np.int64)
    cat_macro_train = cat_macro_train.astype(np.int64)
    p_global = np.bincount(cat_macro_train, minlength=n_macros).astype(np.float64)
    p_global /= p_global.sum()
    counts_k = np.zeros((K, n_macros), dtype=np.float64)
    np.add.at(counts_k, (z_train, cat_macro_train), 1)
    smoothed = counts_k + alpha * p_global[None, :]
    smoothed = smoothed / smoothed.sum(axis=1, keepdims=True)
    b = np.log(smoothed) - np.log(p_global + 1e-12)
    # z-score within situation
    mu = b.mean(axis=1, keepdims=True)
    sd = b.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return ((b - mu) / sd).astype(np.float32)


# Additive combiner (Eq. (14))

def situational_nudge(membership: np.ndarray, b_z: np.ndarray) -> np.ndarray:
    """Per-request category nudge sum_k r_k * b_tilde^(k)_c, shape (B, n_macros)."""
    return membership.astype(np.float32) @ b_z


def combine_scores(scores_B: np.ndarray,
                   nudge: np.ndarray,
                   gamma: np.ndarray,
                   item_cat_macro: np.ndarray,
                   kappa: float) -> np.ndarray:
    """s_hat = s_B + kappa * gamma_S * nudge_c(i) (Eq. (14)); kappa = 0 returns scores_B."""
    if kappa == 0.0:
        return scores_B.astype(np.float32)
    return scores_B + kappa * gamma[:, None].astype(np.float32) * nudge[:, item_cat_macro]
