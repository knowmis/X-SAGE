"""L3 — projection.

Estimates the situation transition matrix ``T_{kk'} = P̂(s_{t+1}=k' | s_t=k)``
from the sequence of core labels per user (eq.17), uses it to (i) predict the
next situation and (ii) build the dynamic-fairness curve
``LT̄_{t:t+τ} = Σ_k (T^τ)_{z_t,k} · LT(R_k)`` (eq.19).

Boundary disambiguation (eq.18) ``r̃_k ∝ r_k · T_{z_{t-1}, k}`` is implemented
as a thin helper consumed by :mod:`recommendation`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def estimate_transition(core_labels_per_user: list[np.ndarray],
                          K: int,
                          add_one_smoothing: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Count successive (z_t, z_{t+1}) pairs across users, return T and the raw
    count matrix.
    """
    counts = np.zeros((K, K), dtype=np.float64)
    for seq in core_labels_per_user:
        if len(seq) < 2:
            continue
        s = seq[:-1]; d = seq[1:]
        np.add.at(counts, (s, d), 1)
    raw = counts.copy()
    if add_one_smoothing:
        counts += 1.0
    T = counts / counts.sum(axis=1, keepdims=True)
    return T, raw


def predict_next_situation(T: np.ndarray, z_t: np.ndarray) -> np.ndarray:
    """Return argmax over rows of T for each ``z_t`` index."""
    pred = np.zeros_like(z_t)
    for i, z in enumerate(z_t):
        pred[i] = int(np.argmax(T[int(z)]))
    return pred


def time_only_prior(z_train: np.ndarray, hour_train: np.ndarray,
                     hour_test: np.ndarray) -> np.ndarray:
    """Predict next situation from ``c_hour`` alone using train P(z | hour).

    Equivalent to: for each test hour, output the most common z observed at
    that hour in train.
    """
    Hs = 24
    K = int(z_train.max() + 1)
    counts = np.zeros((Hs, K), dtype=np.int64)
    np.add.at(counts, (hour_train, z_train), 1)
    # tie-break by lower index when counts equal
    most = counts.argmax(axis=1)
    return most[hour_test]


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, K: int) -> float:
    """Macro-averaged F1 across K classes (so a class with 0 support gets F1=0
    counted in the mean)."""
    f1s = []
    for k in range(K):
        tp = int(((y_true == k) & (y_pred == k)).sum())
        fp = int(((y_true != k) & (y_pred == k)).sum())
        fn = int(((y_true == k) & (y_pred != k)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        f1s.append(f1)
    return float(np.mean(f1s))


def dynamic_fairness(T: np.ndarray, lt_per_situation: np.ndarray,
                       tau_max: int = 3) -> np.ndarray:
    """``LT̄_{t:t+τ}`` for each starting z, for τ ∈ {1..tau_max}.

    Returns a (K, τ) array. Row k, col τ-1 = expected LT τ steps ahead given
    we start in k.
    """
    K = T.shape[0]
    out = np.zeros((K, tau_max), dtype=np.float64)
    Tpow = np.eye(K, dtype=np.float64)
    for t in range(tau_max):
        Tpow = Tpow @ T
        out[:, t] = Tpow @ lt_per_situation
    return out


def boundary_disambiguate(r_membership: np.ndarray, T: np.ndarray,
                            z_prev: np.ndarray) -> np.ndarray:
    """Apply eq.18 row-wise: ``r̃_k = r_k · T_{z_prev, k}`` then renormalise.

    For rows with no previous situation (``z_prev == -1``) we leave ``r``
    unchanged.
    """
    out = r_membership.astype(np.float64).copy()
    valid = z_prev >= 0
    if valid.any():
        priors = T[z_prev[valid]]
        out[valid] = out[valid] * priors
        s = out[valid].sum(axis=1, keepdims=True)
        s = np.where(s > 0, s, 1.0)
        out[valid] = out[valid] / s
    return out.astype(np.float32)
