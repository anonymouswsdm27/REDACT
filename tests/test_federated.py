"""Tests for the FedAvg loop + FRU/ours (§11, §13).

- FedAvg trains a working recommender (beats random on a clustered synthetic), is deterministic,
  and logs per-user contributions.
- FRU/ours roll back the user's logged contribution to the shared item (reducing it), give the
  same per-user correction (ours == FRU numerically), and ours is local-only by construction.
"""
from __future__ import annotations

import numpy as np

from src.federated.simulator import train_fedgmf
from src.models.mf import evaluate_ranking
from src.unlearning.methods import fru_unlearn, ours_unlearn


def _clustered(seed: int = 0):
    """Two user clusters with disjoint item tastes -> learnable collaborative structure."""
    rng = np.random.default_rng(seed)
    rows, test = [], []
    for u in range(40):
        items = list(range(0, 20)) if u < 20 else list(range(20, 40))
        rng.shuffle(items)
        for i in items[:-1]:
            rows.append((u, i))
        test.append((u, items[-1]))                      # held-out in-cluster item
    return (np.array(rows, dtype=np.int32), np.array([(u, -1) for u in range(40)], np.int32),
            np.array(test, dtype=np.int32), 40, 40)


def test_fedavg_trains_and_logs_contribs():
    train, _val, test, nu, ni = _clustered()
    user_items = [np.sort(train[train[:, 0] == u, 1]) for u in range(nu)]
    m = train_fedgmf(train, nu, ni, dim=16, rounds=30, local_epochs=3, lr=0.3, seed=0)
    hr = evaluate_ranking(m, test, user_items, ni, k=10, n_neg=20)["HR@10"]
    assert hr > 0.10, hr                                 # learns: well above 1/(1+20) ~ 0.048 random
    assert m.meta is not None and len(m.meta["contribs_q"]) > 0
    # determinism
    m2 = train_fedgmf(train, nu, ni, dim=16, rounds=5, local_epochs=2, lr=0.2, seed=1)
    m3 = train_fedgmf(train, nu, ni, dim=16, rounds=5, local_epochs=2, lr=0.2, seed=1)
    assert np.allclose(m2.q, m3.q) and np.allclose(m2.p, m3.p)


def test_fru_and_ours_rollback():
    train, _val, _test, nu, ni = _clustered()
    user_items = [np.sort(train[train[:, 0] == u, 1]) for u in range(nu)]
    m = train_fedgmf(train, nu, ni, dim=16, rounds=20, local_epochs=3, lr=0.2, seed=0)
    A = 0
    X = int(user_items[A][0])                            # an item A interacted with
    hist_minus = user_items[A][user_items[A] != X]
    fru = fru_unlearn(m, A, X, hist_minus, ni, dim=16, seed=3)
    ours = ours_unlearn(m, A, X, hist_minus, ni, dim=16, seed=3)
    # rollback actually changed the shared item embedding (A's contribution removed)
    assert np.linalg.norm(fru.q[X] - m.q[X]) > 0
    # same per-user correction (ours == FRU numerically; the difference is deployment/collateral)
    assert np.allclose(fru.q[X], ours.q[X]) and np.allclose(fru.p[A], ours.p[A])
    # the rolled-back q_X equals base minus A's logged contribution
    contrib = m.meta["contribs_q"][A][X]
    assert np.allclose(fru.q[X], m.q[X] - contrib, atol=1e-5)
