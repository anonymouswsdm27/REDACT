"""Tests for the load-bearing Phase 0 pieces (§13).

- retrain determinism: training is a deterministic function of (data, seed), so a from-scratch
  retrain excluding an interaction is a well-defined, provably-correct unlearning oracle (§4.1).
- probe sanity: with no residue by construction the probe returns AUC ~ 0.5 (§4.4 / §13b).
- redundancy-0 clean-forget: an item touched only by user A becomes untrained after unlearning
  and is NOT inferable -> the §4.3 negative-control mechanism, on a controlled synthetic case.
- control sampler: popularity matching + history exclusion behave (§4.2).
"""
from __future__ import annotations

import numpy as np

from src.models.mf import train_mf
from src.probes.membership import ControlSampler, probe_auc


def _synth(seed: int = 0):
    """Two user clusters with disjoint item tastes -> real collaborative structure."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(20):
        items = range(0, 15) if u < 10 else range(15, 30)
        for i in items:
            if rng.random() < 0.7:
                rows.append((u, i))
    return np.array(rows, dtype=np.int32), 20, 30


def test_train_determinism():
    train, nu, ni = _synth()
    a = train_mf(train, nu, ni, dim=8, epochs=5, seed=3)
    b = train_mf(train, nu, ni, dim=8, epochs=5, seed=3)
    assert np.allclose(a.q, b.q) and np.allclose(a.p, b.p) and np.allclose(a.b, b.b)
    # different seed -> different params (sanity that the seed actually matters)
    c = train_mf(train, nu, ni, dim=8, epochs=5, seed=4)
    assert not np.allclose(a.q, c.q)


def test_probe_sanity_no_residue():
    """Random 'forgotten' item vs random controls for a fixed user -> mean AUC ~ 0.5."""
    train, nu, ni = _synth()
    m = train_mf(train, nu, ni, dim=8, epochs=10, seed=1)
    rng = np.random.default_rng(7)
    aucs = []
    for _ in range(400):
        u = int(rng.integers(0, nu))
        items = rng.choice(ni, size=11, replace=False)
        aucs.append(probe_auc(m, u, int(items[0]), items[1:]))
    assert abs(np.mean(aucs) - 0.5) < 0.05, np.mean(aucs)


def test_redundancy_zero_clean_forget():
    """Items touched ONLY by their user become untrained after unlearning and are not inferable.

    Averaged over several sole-held (redundancy-0) items to beat tiny-data noise: the post-
    unlearning probe must NOT make them more inferable than the pre-unlearning model, and must
    leave no strong residue. This is the §4.3 negative control's mechanism in miniature.
    """
    nu, ni_base = 10, 24
    rng = np.random.default_rng(0)
    rows = [(u, i) for u in range(nu) for i in range(ni_base) if rng.random() < 0.6]
    # add one private (redundancy-0) item per user, id = ni_base + u
    privates = [(u, ni_base + u) for u in range(nu)]
    rows += privates
    ni = ni_base + nu
    train = np.array(rows, dtype=np.int32)
    full = train_mf(train, nu, ni, dim=16, epochs=40, reg=0.0, seed=2)

    auc_full, auc_unlearned = [], []
    for (A, X) in privates:
        tr = train[~((train[:, 0] == A) & (train[:, 1] == X))]
        unlearned = train_mf(tr, nu, ni, dim=16, epochs=40, reg=0.0, seed=2)
        a_items = set(train[train[:, 0] == A, 1].tolist())
        controls = np.array([i for i in range(ni_base) if i not in a_items], dtype=np.int32)
        if len(controls) < 3:
            continue
        auc_full.append(probe_auc(full, A, X, controls))
        auc_unlearned.append(probe_auc(unlearned, A, X, controls))
    mf, mu = float(np.mean(auc_full)), float(np.mean(auc_unlearned))
    assert mu < mf                 # unlearning removes the direct fit (averaged)
    assert mu <= 0.6               # no strong residue at redundancy 0


def test_control_sampler():
    train, nu, ni = _synth()
    user_items = [np.sort(train[train[:, 0] == u, 1]) for u in range(nu)]
    pop = np.zeros(ni, dtype=np.int32)
    np.add.at(pop, train[:, 1], 1)
    cs = ControlSampler(pop, user_items)
    x = int(np.argmax(pop))            # a popular item
    ctrl = cs.sample(0, x, n=5, pair_seed=11)
    hist = set(user_items[0].tolist())
    assert all(c not in hist and c != x for c in ctrl)
    # popularity matched: control pops should be near pop[x]
    assert np.median(np.abs(pop[ctrl].astype(int) - int(pop[x]))) <= max(2, int(0.5 * pop[x]))
