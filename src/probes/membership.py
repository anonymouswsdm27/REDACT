"""Inference probe + redundancy stratification for the Phase 0 kill gate (§5, §12).

The probe is a membership-style test run *only on the post-unlearning model*: for a user A
whose interaction with item X was unlearned, does the model still score X above
**popularity-matched control items A never touched** (§4.2)? Per-pair AUC = P(score(A,X) >
score(A, matched control)). We stratify pairs by X's cross-user **redundancy** = (#users who
interacted with X) - 1, the count of *other* users reinforcing X's shared embedding (§2).

Mandatory controls baked in:
- redundancy-0 stratum (X touched only by A) must give AUC ~ 0.5 -> the §4.3 negative control
  and simultaneously the probe-sanity check (§4.4 / §13b).
- placebo: a random user who never touched X must give AUC ~ 0.5 -> isolates residue *specific
  to the forgetting user* from "X is just generally well-scored".
"""
from __future__ import annotations

import dataclasses

import numpy as np

from ..models.mf import MFModel

# redundancy bins on r = pop - 1 (number of OTHER users interacting with the item).
BIN_EDGES = [0, 1, 3, 6, 11, 21, 51, 101, 201, 501, 10**9]
BIN_LABELS = ["0", "1-2", "3-5", "6-10", "11-20", "21-50", "51-100", "101-200", "201-500", "501+"]


def redundancy_bin(r: int) -> int:
    for b in range(len(BIN_EDGES) - 1):
        if BIN_EDGES[b] <= r < BIN_EDGES[b + 1]:
            return b
    return len(BIN_EDGES) - 2


@dataclasses.dataclass
class Target:
    user: int
    item: int           # forgotten item X
    r: int              # redundancy = pop_X - 1
    bin: int


def build_targets(user_items: list[np.ndarray], item_pop: np.ndarray, per_stratum: int,
                  seed: int) -> list[Target]:
    """Sample (A, X) pairs balanced across redundancy strata, X in A's train history,
    preferring distinct users (so many fit in one batch-leave-out retrain)."""
    rng = np.random.default_rng(seed)
    n_bins = len(BIN_LABELS)
    buckets: list[list[tuple[int, int, int]]] = [[] for _ in range(n_bins)]
    users = rng.permutation(len(user_items))
    for u in users:
        items = user_items[u]
        if len(items) == 0:
            continue
        # one candidate item per user per bin (keeps users spread across strata)
        rs = item_pop[items] - 1
        for b in range(n_bins):
            mask = (rs >= BIN_EDGES[b]) & (rs < BIN_EDGES[b + 1])
            if mask.any():
                choices = items[mask]
                x = int(choices[rng.integers(0, len(choices))])
                buckets[b].append((int(u), x, int(item_pop[x] - 1)))
    targets: list[Target] = []
    for b in range(n_bins):
        rows = buckets[b]
        rng.shuffle(rows)
        for (u, x, r) in rows[:per_stratum]:
            targets.append(Target(u, x, r, b))
    return targets


def pack_batches(targets: list[Target]) -> list[list[Target]]:
    """Greedily pack targets into batches with no repeated user (so each batch is a valid
    one-removal-per-user leave-out retrain)."""
    remaining = list(targets)
    batches: list[list[Target]] = []
    while remaining:
        used: set[int] = set()
        batch, leftover = [], []
        for t in remaining:
            if t.user in used:
                leftover.append(t)
            else:
                used.add(t.user)
                batch.append(t)
        batches.append(batch)
        remaining = leftover
    return batches


class ControlSampler:
    """Draws popularity-matched control items (A never touched) for a forgotten item X.

    Uses **nearest-popularity** matching: from items A never touched, take a pool whose
    popularity is closest to pop_X (balanced above/below where possible) and sample from it.
    This minimises the residual popularity gap that a one-sided band leaves at the popular
    tail (which would otherwise inflate the placebo) — sharpening the §4.2 control. Controls
    are drawn under a *per-pair* seed so a single-pair retrain probes the identical controls
    the batch retrain did (apples-to-apples validation of the batch-leave-out approximation)."""

    def __init__(self, item_pop: np.ndarray, user_items: list[np.ndarray]):
        self.pop = item_pop
        self.order = np.argsort(item_pop, kind="stable")     # item ids sorted by pop
        self.sorted_pop = item_pop[self.order].astype(np.int64)
        self.hist = [set(x.tolist()) for x in user_items]

    def sample(self, user: int, item_x: int, n: int, pair_seed: int, pool: int = 6) -> np.ndarray:
        """n control items whose popularity is nearest pop_X, not in A's history, != X.
        `pool` = how many times n to gather as the nearest-pop candidate pool before sampling."""
        px = int(self.pop[item_x])
        hist = self.hist[user]
        target = n * pool
        # walk outward from pop_X's position in the pop-sorted item list, nearest-first.
        c = np.searchsorted(self.sorted_pop, px, side="left")
        lo, hi = c - 1, c
        cand: list[int] = []
        while len(cand) < target and (lo >= 0 or hi < len(self.order)):
            take_hi = hi < len(self.order) and (
                lo < 0 or abs(int(self.sorted_pop[hi]) - px) <= abs(int(self.sorted_pop[lo]) - px))
            j = hi if take_hi else lo
            it = int(self.order[j])
            if it != item_x and it not in hist:
                cand.append(it)
            if take_hi:
                hi += 1
            else:
                lo -= 1
        if not cand:
            return np.array([], dtype=np.int32)
        rng = np.random.default_rng(pair_seed)
        arr = np.array(cand, dtype=np.int32)
        if len(arr) <= n:
            return arr
        return arr[rng.choice(len(arr), size=n, replace=False)]


def probe_auc(model: MFModel, user: int, item_x: int, controls: np.ndarray) -> float:
    """Per-pair AUC = P(score(user, X) > score(user, control)) with 0.5 for ties."""
    sx = float(model.scores(user, np.array([item_x]))[0])
    sc = model.scores(user, controls)
    return float((sc < sx).mean() + 0.5 * (sc == sx).mean())


def probe_auc_variant(model: MFModel, user: int, item_x: int, controls: np.ndarray,
                      kind: str = "score") -> float:
    """Probe AUC under alternative scoring statistics — the probe-architecture ablation (§6-P5, F2),
    to show the residue is not an artifact of one score function:
      'score'  = full <p_u,q_i>+b_i  (the default probe),
      'nobias' = <p_u,q_i> only      (rules out item-bias / popularity driving it),
      'cosine' = cos(p_u, q_i)        (magnitude-invariant — rules out embedding-norm effects)."""
    pu = model.p[user]
    qx = model.q[item_x]; qc = model.q[controls]
    if kind == "score":
        sx = float(qx @ pu + model.b[item_x]); sc = qc @ pu + model.b[controls]
    elif kind == "nobias":
        sx = float(qx @ pu); sc = qc @ pu
    elif kind == "cosine":
        npu = np.linalg.norm(pu) + 1e-12
        sx = float(qx @ pu / (np.linalg.norm(qx) + 1e-12) / npu)
        sc = (qc @ pu) / (np.linalg.norm(qc, axis=1) + 1e-12) / npu
    else:
        raise ValueError(kind)
    return float((sc < sx).mean() + 0.5 * (sc == sx).mean())
