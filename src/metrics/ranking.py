"""Per-user ranking utility (§12).

`holdout_ndcg` scores a user's held-out item against sampled negatives under a given model — a
cheap per-(A) utility probe. Pairing this with the inference-probe AUC is what makes the method
comparison fair: a method that drives inferability to ~0 by *damaging* the model (e.g.
gradient-ascent) is exposed here as a utility collapse, not a win (the privacy–utility frontier).
"""
from __future__ import annotations

import numpy as np

from ..models.mf import MFModel


def holdout_ndcg(model: MFModel, user: int, pos_item: int, exclude: set[int], n_items: int,
                 n_neg: int = 99, seed: int = 0) -> float:
    """NDCG of the held-out `pos_item` vs `n_neg` sampled negatives (items the user never saw).
    Graded (1/log2(rank+2)); 1.0 if ranked first. Returns NaN if pos_item is invalid."""
    if pos_item is None or pos_item < 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    negs: list[int] = []
    while len(negs) < n_neg:
        c = int(rng.integers(0, n_items))
        if c != pos_item and c not in exclude:
            negs.append(c)
    cand = np.array([pos_item] + negs)
    sc = model.scores(user, cand)
    rank = int((sc > sc[0]).sum())          # 0 = top
    return float(1.0 / np.log2(rank + 2))
