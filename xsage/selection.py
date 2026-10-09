"""Selection of K (training silhouette, capped at |A| + 2) and epsilon (validation boundary fraction), once per dataset."""
import numpy as np
from sklearn.metrics import silhouette_score

from xsage.l2_comprehension import _assign, fit_rough_kmeans

SEED, SIL_N = 42, 20000
K_RANGE = [3, 4, 5, 6, 7, 8, 9]
EPS_GRID = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
BAND = (0.10, 0.30)


def select_K(vtr, ceil, rng):
    """Silhouette-optimal K over K_RANGE on a subsample of the training descriptors, capped at ceil."""
    n = len(vtr); sidx = rng.choice(n, SIL_N, replace=False) if n > SIL_N else np.arange(n)
    best_k, best_s = K_RANGE[0], -1
    for K in K_RANGE:
        r = fit_rough_kmeans(vtr, K=K, eps=0.0, seed=SEED, max_iter=60); lab = r.core_label
        if len(np.unique(lab)) < 2:
            continue
        s = silhouette_score(vtr[sidx], lab[sidx])
        if s > best_s:
            best_s, best_k = s, K
    return min(best_k, ceil)


def select_eps(vtr, vva, K):
    """Margin whose validation boundary fraction is closest to 0.20 within BAND."""
    cells = []
    for eps in EPS_GRID:
        r = fit_rough_kmeans(vtr, K=K, eps=eps, seed=SEED, max_iter=80)
        _, _, _, isb = _assign(vva, r.prototypes, eps)
        cells.append((eps, float(isb.mean())))
    inb = [(e, b) for e, b in cells if BAND[0] <= b <= BAND[1]]
    return (min(inb, key=lambda x: abs(x[1] - 0.20)) if inb else min(cells, key=lambda x: abs(x[1] - 0.20)))[0]


def select_K_eps(D0):
    """K and epsilon of a dataset, selected once with the fixed seed SEED."""
    vtr, vva = D0["vs"]["train"], D0["vs"]["val"]
    K = select_K(vtr, int(D0["attractors"].sum()) + 2, np.random.default_rng(SEED))
    return K, select_eps(vtr, vva, K)
