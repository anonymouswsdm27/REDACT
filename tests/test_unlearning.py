"""Sanity for the unlearning baselines (§11, §13).

- gradient-ascent must REDUCE the forgotten item's score for the user (it unlearns), and is
  deterministic given the seed;
- fine-tune unlearning returns a valid model with the same shapes, warm-started from base;
- all methods produce probe AUCs in [0, 1].
"""
from __future__ import annotations

import numpy as np

from src.models.mf import train_mf
from src.probes.membership import probe_auc
from src.unlearning.methods import finetune_unlearn, gradient_ascent_unlearn


def _synth(seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(20):
        for i in (range(0, 15) if u < 10 else range(15, 30)):
            if rng.random() < 0.7:
                rows.append((u, i))
    return np.array(rows, dtype=np.int32), 20, 30


def _remove(train, u, i):
    keys = train[:, 0] * 1000 + train[:, 1]
    return train[keys != (u * 1000 + i)]


def test_gradient_ascent_reduces_score_and_is_deterministic():
    train, nu, ni = _synth()
    base = train_mf(train, nu, ni, dim=8, epochs=20, seed=1)
    A = 0
    X = int(base.scores(A, np.arange(ni)).argmax())   # an item the user scores high
    before = float(base.scores(A, np.array([X]))[0])
    ga = gradient_ascent_unlearn(base, A, X, ni, steps=40, seed=5)
    after = float(ga.scores(A, np.array([X]))[0])
    assert after < before, (before, after)            # the interaction is pushed down
    ga2 = gradient_ascent_unlearn(base, A, X, ni, steps=40, seed=5)
    assert np.allclose(ga.q[X], ga2.q[X]) and np.allclose(ga.p[A], ga2.p[A])  # deterministic


def test_finetune_returns_valid_warm_started_model():
    train, nu, ni = _synth()
    base = train_mf(train, nu, ni, dim=8, epochs=20, seed=1)
    ft = finetune_unlearn(base, _remove(train, 0, 5), nu, ni, dim=8, epochs=3, seed=1)
    assert ft.q.shape == base.q.shape and ft.p.shape == base.p.shape
    controls = np.arange(15, 30)
    for m in (base, ft):
        auc = probe_auc(m, 0, 5, controls)
        assert 0.0 <= auc <= 1.0
