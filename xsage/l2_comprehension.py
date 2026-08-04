"""L2 — comprehension via rough k-means (Lingras–West, 2004).

For each request the descriptor :math:`\\mathbf v = [\\tilde c \\Vert \\mathbf e]`
is compared to ``K`` prototypes ``μ_k``. The closest prototype is the *core*
assignment; any prototype within ``ε`` of the closest distance is *boundary*.
On core requests the situation is certain (``r_{k^*} = 1``); on boundary
requests the mass is split uniformly among competing situations
(``r_k = 1/|T|`` for ``k ∈ T``).

Prototype updates weight the core (lower-approximation) members above the
boundary ones (eq.9):

    μ_k = w_l · mean(lower_k) + w_b · mean(upper_k \\ lower_k),  w_l > w_b.

Convergence: prototypes move less than ``tol`` in L2 OR ``max_iter`` reached.

The implementation is pure NumPy, single file, ~120 LOC. No library dep.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RoughKMeansResult:
    prototypes: np.ndarray         # (K, D)
    core_label: np.ndarray         # (B,) int — the unique argmin, even for boundary points
    competing_sets: np.ndarray     # (B, K) bool — True for each k in T(v)
    membership: np.ndarray         # (B, K) float — r_k
    is_boundary: np.ndarray        # (B,) bool — True if |T| > 1
    distances: np.ndarray          # (B, K) float — d_k = ‖v − μ_k‖
    n_iters: int
    converged: bool


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

def _init_kmeanspp(X: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding for the prototypes."""
    B, D = X.shape
    centers = np.empty((K, D), dtype=X.dtype)
    idx0 = int(rng.integers(0, B))
    centers[0] = X[idx0]
    d2 = np.sum((X - centers[0]) ** 2, axis=1)
    for k in range(1, K):
        probs = d2 / max(d2.sum(), 1e-12)
        i = int(rng.choice(B, p=probs))
        centers[k] = X[i]
        new_d2 = np.sum((X - centers[k]) ** 2, axis=1)
        d2 = np.minimum(d2, new_d2)
    return centers


# ---------------------------------------------------------------------------
# Assignment + update
# ---------------------------------------------------------------------------

def _distances(X: np.ndarray, mu: np.ndarray) -> np.ndarray:
    diff = X[:, None, :] - mu[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _assign(X: np.ndarray, mu: np.ndarray, eps: float):
    d = _distances(X, mu)                                  # (B, K)
    k_star = np.argmin(d, axis=1)                           # (B,)
    d_star = d[np.arange(len(X)), k_star]                   # (B,)
    competing = d - d_star[:, None] <= eps                  # (B, K) bool
    T_size = competing.sum(axis=1)
    is_boundary = T_size > 1
    return d, k_star, competing, is_boundary


def _update(X: np.ndarray, k_star: np.ndarray, competing: np.ndarray,
             is_boundary: np.ndarray, K: int,
             w_l: float = 0.7, w_b: float = 0.3) -> np.ndarray:
    """μ_k = w_l mean(lower_k) + w_b mean(upper_k \\ lower_k)."""
    D = X.shape[1]
    new_mu = np.zeros((K, D), dtype=X.dtype)
    for k in range(K):
        in_lower = (k_star == k) & (~is_boundary)
        in_upper_only = competing[:, k] & is_boundary
        lower_mean = X[in_lower].mean(axis=0) if in_lower.any() else None
        bound_mean = X[in_upper_only].mean(axis=0) if in_upper_only.any() else None
        if lower_mean is None and bound_mean is None:
            # dead cluster — keep previous (caller must handle)
            new_mu[k] = np.nan
        elif bound_mean is None:
            new_mu[k] = lower_mean
        elif lower_mean is None:
            new_mu[k] = bound_mean
        else:
            new_mu[k] = w_l * lower_mean + w_b * bound_mean
    return new_mu


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def fit_rough_kmeans(X: np.ndarray,
                       K: int = 6,
                       eps: float = 0.1,
                       max_iter: int = 50,
                       tol: float = 1e-4,
                       w_l: float = 0.7,
                       w_b: float = 0.3,
                       seed: int = 42) -> RoughKMeansResult:
    """Lingras–West rough k-means.

    Args:
        X:    (B, D) float matrix.
        K:    number of clusters.
        eps:  boundary threshold (distance-units, NOT relative).
        max_iter, tol: convergence.
        w_l, w_b: prototype-update weights, ``w_l + w_b = 1``, ``w_l > w_b``.
        seed: RNG seed for k-means++ init.

    Returns:
        :class:`RoughKMeansResult`.
    """
    if not (w_l > w_b and abs(w_l + w_b - 1.0) < 1e-9):
        raise ValueError("rough k-means requires w_l > w_b and w_l+w_b=1")
    rng = np.random.default_rng(seed)

    mu = _init_kmeanspp(X, K, rng)
    converged = False
    for it in range(1, max_iter + 1):
        d, k_star, competing, is_b = _assign(X, mu, eps)
        new_mu = _update(X, k_star, competing, is_b, K, w_l, w_b)
        # Heal dead clusters
        bad = np.isnan(new_mu).any(axis=1)
        if bad.any():
            new_mu[bad] = mu[bad]
        shift = float(np.linalg.norm(new_mu - mu))
        mu = new_mu
        if shift < tol:
            converged = True
            break

    d, k_star, competing, is_b = _assign(X, mu, eps)
    membership = np.zeros_like(d, dtype=np.float32)
    core_mask = ~is_b
    if core_mask.any():
        membership[core_mask, k_star[core_mask]] = 1.0
    bnd_idx = np.where(is_b)[0]
    if bnd_idx.size:
        T_sizes = competing[bnd_idx].sum(axis=1).astype(np.float32)
        membership[bnd_idx] = competing[bnd_idx].astype(np.float32) \
                              / T_sizes[:, None]

    return RoughKMeansResult(
        prototypes=mu,
        core_label=k_star.astype(np.int32),
        competing_sets=competing,
        membership=membership,
        is_boundary=is_b,
        distances=d.astype(np.float32),
        n_iters=it,
        converged=converged,
    )


# ---------------------------------------------------------------------------
# Helpers used by Stage A diagnostics
# ---------------------------------------------------------------------------

def auto_epsilon(X: np.ndarray, K: int, target_boundary: tuple[float, float] = (0.10, 0.30),
                  seed: int = 42, search_grid: int = 20) -> float:
    """Pick ``ε`` so the resulting boundary fraction lies in the target band.

    The first version of this routine measured margins on random k-means++
    prototypes, which under-estimated the boundary fraction by a wide margin
    on a low-variance descriptor (because the post-fit prototypes sit at the
    cluster centroids and the margin distribution shifts).

    Strategy now: do a *full* rough-k-means fit with ``ε = 0`` (hard assign)
    to obtain converged prototypes, measure post-fit margins, and pick the
    quantile of those margins whose induced boundary fraction is closest to
    the centre of the target band.
    """
    res0 = fit_rough_kmeans(X, K=K, eps=0.0, max_iter=50, seed=seed)
    mu = res0.prototypes
    d = _distances(X, mu)
    sorted_d = np.sort(d, axis=1)
    margin = sorted_d[:, 1] - sorted_d[:, 0]

    centre = 0.5 * (target_boundary[0] + target_boundary[1])
    qs = np.linspace(0.05, 0.95, search_grid)
    best_eps = float(np.quantile(margin, 0.5))
    best_diff = float("inf")
    in_band: list[float] = []
    for q in qs:
        eps = float(np.quantile(margin, q))
        is_b = (d - sorted_d[:, 0:1]) <= eps
        frac = float((is_b.sum(axis=1) > 1).mean())
        if target_boundary[0] <= frac <= target_boundary[1]:
            in_band.append(eps)
        if abs(frac - centre) < best_diff:
            best_diff = abs(frac - centre); best_eps = eps
    # If any ε hits the band, return its median (more stable than the first hit).
    if in_band:
        return float(np.median(in_band))
    return best_eps


def adjusted_rand_score(a: np.ndarray, b: np.ndarray) -> float:
    """Standard ARI (Hubert & Arabie, 1985). No sklearn import — keep it tight."""
    from math import comb
    assert a.shape == b.shape
    classes_a = np.unique(a); classes_b = np.unique(b)
    contingency = np.zeros((len(classes_a), len(classes_b)), dtype=np.int64)
    for i, ca in enumerate(classes_a):
        for j, cb in enumerate(classes_b):
            contingency[i, j] = int(((a == ca) & (b == cb)).sum())
    sum_comb_c = sum(comb(int(n), 2) for n in contingency.flatten() if n >= 2)
    sum_comb_a = sum(comb(int(n), 2) for n in contingency.sum(axis=1) if n >= 2)
    sum_comb_b = sum(comb(int(n), 2) for n in contingency.sum(axis=0) if n >= 2)
    N = comb(len(a), 2)
    expected = sum_comb_a * sum_comb_b / N if N else 0.0
    max_index = 0.5 * (sum_comb_a + sum_comb_b)
    if max_index == expected:
        return 1.0
    return float((sum_comb_c - expected) / (max_index - expected))
