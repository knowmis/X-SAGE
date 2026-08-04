"""Fairness, ranking and consistency metrics used by the X-SAGE pipeline.

* ``long_tail_ratio`` — LT(R) = #(items in R ∈ G_1) / |R|, eq.11.
* ``kl_divergence``  — d_KL(D_1 || D_2), eq.12.
* ``list_displacement`` — Δ_K(q), eq.16.
* Standard next-item metrics (Recall@K, NDCG@K, …) reused via the floor's
  evaluator are intentionally NOT re-implemented here; they live in
  ``engine.Evaluation``.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Long-tail / fairness
# ---------------------------------------------------------------------------

def long_tail_groups(item_popularity: np.ndarray,
                       short_head_share: float = 0.20) -> tuple[np.ndarray, np.ndarray]:
    """Split items into short-head (G_0, top 20% by popularity) and long-tail
    (G_1, the rest). Returns (G0_mask, G1_mask) — Boolean arrays of length
    ``n_items``.
    """
    order = np.argsort(item_popularity)[::-1]
    n_head = int(np.ceil(len(item_popularity) * short_head_share))
    head_items = order[:n_head]
    G0 = np.zeros_like(item_popularity, dtype=bool); G0[head_items] = True
    G1 = ~G0
    return G0, G1


def long_tail_ratio(top_k_items: np.ndarray, long_tail_mask: np.ndarray) -> float:
    """``LT(R) = #(items in R ∈ long-tail) / |R|`` for ONE list."""
    if len(top_k_items) == 0:
        return 0.0
    return float(long_tail_mask[top_k_items].mean())


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """``KL(p || q) = Σ p log(p / q)`` for non-negative vectors summing to 1.

    Inputs are normalised inline (so callers can pass raw counts).
    """
    p = np.asarray(p, dtype=np.float64); q = np.asarray(q, dtype=np.float64)
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    p = np.clip(p, eps, 1.0); q = np.clip(q, eps, 1.0)
    return float((p * np.log(p / q)).sum())


# ---------------------------------------------------------------------------
# Ranking consistency
# ---------------------------------------------------------------------------

def list_displacement(L_on: np.ndarray, L_off: np.ndarray, K: int) -> float:
    """``Δ_K(q) = 1 − |L_on ∩ L_off| / K`` for one request (eq.16)."""
    on_set = set(map(int, L_on[:K]))
    off_set = set(map(int, L_off[:K]))
    if K == 0:
        return 0.0
    return 1.0 - len(on_set & off_set) / float(K)


def topk_from_scores(scores: np.ndarray, K: int,
                       exclude: np.ndarray | None = None) -> np.ndarray:
    """Argpartition-based top-K, optionally masking ``exclude`` (1d bool of
    length n_items, True = exclude).

    Stable tie-break: descending sort within the top-K candidates.
    """
    s = scores.copy()
    if exclude is not None:
        s[exclude] = -np.inf
    if K >= len(s):
        order = np.argsort(s)[::-1]
        return order
    idx = np.argpartition(s, -K)[-K:]
    return idx[np.argsort(s[idx])[::-1]]
