"""Sequential-residue probe (§2 secondary claim, §6-P1). After unlearning (A,X) from a
SEQUENTIAL recommender, is X still inferable as having-been-in-A's-history from A's OWN
autoregressive predictions?

Probe: feed the context that PRECEDED X in A's history and ask whether the post-unlearning model
still ranks X above popularity-matched controls A never touched. Two residue channels are present
and we separate them:
  * collaborative (cross-user): X's shared embedding is reinforced by OTHER users' sequences → the
    probe AUC rises with X's redundancy, exactly as in the MF case (stratify by r).
  * sequential within-user: the transition [A's pre-X context → X] was learned during training. We
    isolate it with the CONTEXT-SPECIFICITY gap: AUC under A's real pre-X context minus AUC under a
    random context for the same X. A positive gap at fixed redundancy = a within-user residue that
    a shared-embedding (collaborative-only) account cannot explain.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from ..models.sequential import SeqModel


@dataclasses.dataclass
class SeqTarget:
    user: int
    item: int            # forgotten item X
    pos: int             # index of X in the user's chronological sequence
    r: int               # redundancy = pop_X - 1
    bin: int


def seq_probe_auc(model: SeqModel, context: np.ndarray, item_x: int, controls: np.ndarray) -> float:
    """AUC = P(score(context, X) > score(context, control)) with 0.5 for ties."""
    if len(controls) == 0:
        return float("nan")
    sx = float(model.score(context, np.array([item_x]))[0])
    sc = model.score(context, controls)
    return float((sc < sx).mean() + 0.5 * (sc == sx).mean())


def build_seq_targets(sequences: list[np.ndarray], item_pop: np.ndarray, per_stratum: int,
                      seed: int, min_prefix: int = 2) -> list[SeqTarget]:
    """Sample (A, X, pos) where X sits at `pos` in A's sequence with >=min_prefix items before it,
    balanced across redundancy strata (reuses the MF binning)."""
    from .membership import BIN_EDGES, BIN_LABELS, redundancy_bin
    rng = np.random.default_rng(seed)
    buckets: list[list[SeqTarget]] = [[] for _ in range(len(BIN_LABELS))]
    for u in rng.permutation(len(sequences)):
        s = sequences[u]
        if len(s) < min_prefix + 2:
            continue
        # candidate positions: enough prefix, and a held-out tail (don't touch the last item)
        for _ in range(1):
            pos = int(rng.integers(min_prefix, len(s) - 1))
            x = int(s[pos]); r = int(item_pop[x] - 1)
            b = redundancy_bin(r)
            buckets[b].append(SeqTarget(int(u), x, pos, r, b))
    out: list[SeqTarget] = []
    for b in range(len(BIN_LABELS)):
        rng.shuffle(buckets[b])
        out.extend(buckets[b][:per_stratum])
    _ = BIN_EDGES
    return out
