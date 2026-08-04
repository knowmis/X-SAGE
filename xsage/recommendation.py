"""Recommendation — additive situational combiner (Eq. 13).

The final score adds a per-situation categorical bias to the backbone score,
gated only by the recognition confidence:

    s_hat(u, i) = s_B(u, i) + kappa * gamma_S(v) * sum_k r_k * b_tilde^(k)_{c(i)}

where
    b_tilde^(k)_c = per-situation z-score of the shrunk-log-odds category bias,
    gamma_S(v)    = 1 if the request is core, 1/|T(v)| if boundary,
    kappa         = per-backbone mixing coefficient (selected on validation).

kappa = 0 recovers the backbone exactly (safe fallback). The combiner is
additive and faithful by construction: the situational term is the exact
explanation of the adjustment applied to an item.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Bias estimation
# ---------------------------------------------------------------------------

def fit_situation_biases_z(z_train: np.ndarray, cat_macro_train: np.ndarray,
                              K: int, n_macros: int,
                              alpha: float = 50.0) -> np.ndarray:
    """Per-situation per-macro bias, standardised (z-scored) within situation.

    Shrunk-log-odds with Bayesian smoothing toward the global macro
    distribution, then z-scored so kappa is interpretable as "additive nudge of
    size ~ kappa standard deviations".

    Args:
        z_train:          (B_train,) int - situation label per training row.
        cat_macro_train:  (B_train,) int - macro of the training row's target.
        K:                number of situations.
        n_macros:         vocabulary size.
        alpha:            Dirichlet pseudo-count.

    Returns:
        (K, n_macros) z-scored bias.
    """
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


# ---------------------------------------------------------------------------
# Additive combiner (Eq. 13)
# ---------------------------------------------------------------------------

def combine_scores(scores_B: np.ndarray,
                   membership_per_request: np.ndarray,
                   b_z: np.ndarray,
                   gamma_per_request: np.ndarray,
                   item_cat_macro: np.ndarray,
                   kappa: float) -> np.ndarray:
    """s_hat = s_B + kappa * gamma_S * sum_k r_k * b_tilde^(k)_{c(i)}  (Eq. 13).

    Args:
        scores_B:               (B, I) backbone scores.
        membership_per_request: (B, K) soft membership r_k of the request.
        b_z:                    (K, n_macros) z-scored per-situation bias.
        gamma_per_request:      (B,) certainty gate: 1 on core, 1/|T| on boundary.
        item_cat_macro:         (n_items,) each item's macro index.
        kappa:                  scalar mixing coefficient.

    Returns:
        (B, n_items) re-ranked scores. kappa = 0 returns scores_B exactly.
    """
    if kappa == 0.0:
        return scores_B.astype(np.float32)
    nudge_per_macro = membership_per_request.astype(np.float32) @ b_z   # (B, M)
    nudge_per_item = nudge_per_macro[:, item_cat_macro]                 # (B, I)
    gamma = gamma_per_request.astype(np.float32)[:, None]
    return (scores_B + kappa * gamma * nudge_per_item).astype(np.float32)
