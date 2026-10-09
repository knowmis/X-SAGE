"""L3 projection: transition matrix between situations (descriptive, not used in scoring)."""
from __future__ import annotations

import numpy as np


def estimate_transition(core_labels_per_user: list[np.ndarray],
                        K: int,
                        add_one_smoothing: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Row-normalised transition matrix T from successive core labels of each user, and raw counts."""
    counts = np.zeros((K, K), dtype=np.float64)
    for seq in core_labels_per_user:
        if len(seq) < 2:
            continue
        np.add.at(counts, (seq[:-1], seq[1:]), 1)
    raw = counts.copy()
    if add_one_smoothing:
        counts += 1.0
    T = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    return T, raw


def predict_next_situation(T: np.ndarray, z_t: np.ndarray) -> np.ndarray:
    """Expected next situation ``argmax_k' T[z_t, k']`` for each current situation ``z_t``."""
    return np.argmax(T[np.asarray(z_t, dtype=np.int64)], axis=1)


def persistence(T: np.ndarray) -> np.ndarray:
    """Self-transition probability of each situation (diagonal of T)."""
    return np.diag(T).copy()
