"""L2 comprehension: rough k-means (Lingras and West, 2004) with core and boundary requests."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RoughKMeansResult:
    prototypes: np.ndarray         # (K, D)
    core_label: np.ndarray         # (B,) int - the unique argmin, even for boundary points
    competing_sets: np.ndarray     # (B, K) bool - True for each k in T(v)
    membership: np.ndarray         # (B, K) float - r_k
    is_boundary: np.ndarray        # (B,) bool - True if |T| > 1
    distances: np.ndarray          # (B, K) float - d_k = ||v - mu_k||
    n_iters: int
    converged: bool


# Init

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


# Assignment + update

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
    """mu_k = w_l * mean(lower_k) + w_b * mean(upper_k minus lower_k)."""
    D = X.shape[1]
    new_mu = np.zeros((K, D), dtype=X.dtype)
    for k in range(K):
        in_lower = (k_star == k) & (~is_boundary)
        in_upper_only = competing[:, k] & is_boundary
        lower_mean = X[in_lower].mean(axis=0) if in_lower.any() else None
        bound_mean = X[in_upper_only].mean(axis=0) if in_upper_only.any() else None
        if lower_mean is None and bound_mean is None:
            # empty cluster: marked here, the caller keeps the previous prototype
            new_mu[k] = np.nan
        elif bound_mean is None:
            new_mu[k] = lower_mean
        elif lower_mean is None:
            new_mu[k] = bound_mean
        else:
            new_mu[k] = w_l * lower_mean + w_b * bound_mean
    return new_mu


# Public entry

def fit_rough_kmeans(X: np.ndarray,
                       K: int = 6,
                       eps: float = 0.1,
                       max_iter: int = 50,
                       tol: float = 1e-4,
                       w_l: float = 0.7,
                       w_b: float = 0.3,
                       seed: int = 42) -> RoughKMeansResult:
    """Rough k-means on X (B, D) with K prototypes, boundary margin eps and update weights w_l > w_b."""
    if not (w_l > w_b and abs(w_l + w_b - 1.0) < 1e-9):
        raise ValueError("rough k-means requires w_l > w_b and w_l+w_b=1")
    rng = np.random.default_rng(seed)

    mu = _init_kmeanspp(X, K, rng)
    converged = False
    for it in range(1, max_iter + 1):
        d, k_star, competing, is_b = _assign(X, mu, eps)
        new_mu = _update(X, k_star, competing, is_b, K, w_l, w_b)
        # keep the previous prototype of an empty cluster
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
