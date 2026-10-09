"""Long-tail item groups for the exposure metrics."""
from __future__ import annotations

import numpy as np


def long_tail_groups(item_popularity: np.ndarray,
                     short_head_share: float = 0.20) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks of the short head (top short_head_share items by popularity) and of the long tail."""
    order = np.argsort(item_popularity)[::-1]
    n_head = int(np.ceil(len(item_popularity) * short_head_share))
    head_items = order[:n_head]
    G0 = np.zeros_like(item_popularity, dtype=bool); G0[head_items] = True
    G1 = ~G0
    return G0, G1
