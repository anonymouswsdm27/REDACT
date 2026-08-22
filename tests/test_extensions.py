"""Sanity for the paper-hardening additions (§13): the limit fit, the tunable/neighbor
unlearning knobs, FedShare == FRU (same contribution rollback), the probe-architecture variants,
and the sequential backbone+probe. Small synthetic data, deterministic, fast.
"""
from __future__ import annotations

import numpy as np

from src.federated.simulator import train_fedgmf
from src.metrics.limit import bound_at, fit_bound
from src.models.sequential import train_gru4rec
from src.probes.membership import probe_auc, probe_auc_variant
from src.probes.sequential import seq_probe_auc
from src.unlearning.methods import (
    fedshare_unlearn,
    fru_unlearn,
    neighbor_delete_unlearn,
    ours_suppress,
    ours_unlearn,
)


def _fed_synth(seed: int = 0):
    """Two taste groups over disjoint item ranges → high-redundancy shared items + logged contribs."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(30):
        items = range(0, 12) if u < 15 else range(12, 24)
        for i in items:
            if rng.random() < 0.8:
                rows.append((u, i))
    return np.array(rows, dtype=np.int32), 30, 24


# ---------- F1. partial participation ----------
def test_participation_full_is_bit_identical_and_partial_runs():
    train, nu, ni = _fed_synth()
    base = train_fedgmf(train, nu, ni, dim=8, rounds=6, local_epochs=2, lr=0.2, seed=3)
    # participation=1.0 must reproduce the default (no extra RNG drawn) bit-for-bit
    full = train_fedgmf(train, nu, ni, dim=8, rounds=6, local_epochs=2, lr=0.2, seed=3,
                        participation=1.0)
    assert np.allclose(base.q, full.q) and np.allclose(base.p, full.p)
    # partial participation runs, stays finite/bounded, and is deterministic given the seed
    part = train_fedgmf(train, nu, ni, dim=8, rounds=6, local_epochs=2, lr=0.2, seed=3,
                        participation=0.5)
    part2 = train_fedgmf(train, nu, ni, dim=8, rounds=6, local_epochs=2, lr=0.2, seed=3,
                         participation=0.5)
    assert np.isfinite(part.q).all() and np.abs(part.q).max() < 20
    assert np.allclose(part.q, part2.q)              # deterministic
    assert (part.meta or {}).get("contribs_q") is not None   # logs still produced for FRU/ours


# ---------- A. limit fit ----------
def test_limit_fit_recovers_saturating_bound():
    rng = np.random.default_rng(0)
    r, auc, seed = [], [], []
    for s in range(4):
        for rr in [0, 1, 2, 5, 10, 30, 80, 200]:
            true = 0.5 + 0.4 * (1 - np.exp(-0.3 * np.log1p(rr)))
            r.append(rr); auc.append(true + rng.normal(0, 0.01)); seed.append(s)
    fit = fit_bound(np.array(r), np.array(auc), np.array(seed))
    assert fit["r2"] > 0.9, fit["r2"]
    assert 0.3 < fit["a"] < 0.5 and 0.15 < fit["c"] < 0.5
    # monotone increasing, pinned at chance for r=0, lower bound below the mean fit
    assert abs(bound_at(fit, 0) - 0.5) < 1e-6
    assert bound_at(fit, 200) > bound_at(fit, 5) > bound_at(fit, 0)
    assert bound_at(fit, 50, lower=True) <= bound_at(fit, 50) + 1e-9


# ---------- B/C. unlearning knobs + FedShare==FRU ----------
def test_ours_suppress_is_monotone_and_matches_endpoints():
    train, nu, ni = _fed_synth()
    base = train_fedgmf(train, nu, ni, dim=8, rounds=8, local_epochs=2, lr=0.2, seed=1)
    # pick A and a high-redundancy item X in A's history
    A = 0
    hist = np.array([i for (u, i) in train if u == A])
    X = int(hist[0])
    hist_minus = hist[hist != X]
    ctrl = np.arange(12, 24)                       # items A never touched (other group)
    aucs = [probe_auc(ours_suppress(base, A, X, hist_minus, ni, dim=8, alpha=a, seed=3), A, X, ctrl)
            for a in (0.0, 1.0, 2.0)]
    assert aucs[0] >= aucs[1] - 1e-9 >= aucs[2] - 1e-9 - 1e-9, aucs   # more suppression -> lower AUC
    # alpha=1 equals the plain ours rollback
    o1 = ours_suppress(base, A, X, hist_minus, ni, dim=8, alpha=1.0, seed=3)
    o = ours_unlearn(base, A, X, hist_minus, ni, dim=8, seed=3)
    assert np.allclose(o1.q[X], o.q[X])


def test_fedshare_equals_fru_rollback():
    train, nu, ni = _fed_synth()
    base = train_fedgmf(train, nu, ni, dim=8, rounds=8, local_epochs=2, lr=0.2, seed=1)
    A = 1
    hist = np.array([i for (u, i) in train if u == A]); X = int(hist[0])
    hm = hist[hist != X]
    fs = fedshare_unlearn(base, A, X, hm, ni, dim=8, seed=2)
    fr = fru_unlearn(base, A, X, hm, ni, dim=8, seed=2)
    assert np.allclose(fs.q[X], fr.q[X]) and np.allclose(fs.p[A], fr.p[A])   # same floor residue (F5)


def test_neighbor_delete_reduces_inferability_and_is_valid():
    train, nu, ni = _fed_synth()
    base = train_fedgmf(train, nu, ni, dim=8, rounds=8, local_epochs=2, lr=0.2, seed=1)
    A = 0
    hist = np.array([i for (u, i) in train if u == A]); X = int(hist[0])
    hm = hist[hist != X]
    others = [u for u in range(15) if u != A]      # same taste group also consumed X
    ctrl = np.arange(12, 24)
    m0, k0 = neighbor_delete_unlearn(base, A, X, hm, ni, other_users=others, frac=0.0, dim=8, seed=1)
    m1, k1 = neighbor_delete_unlearn(base, A, X, hm, ni, other_users=others, frac=1.0, dim=8, seed=1)
    assert k1 >= k0 and m1.q.shape == base.q.shape
    assert probe_auc(m1, A, X, ctrl) <= probe_auc(m0, A, X, ctrl) + 1e-9   # deleting others lowers it


# ---------- F2. probe-architecture variants ----------
def test_probe_variants_bounded_and_detect_planted_signal():
    train, nu, ni = _fed_synth()
    base = train_fedgmf(train, nu, ni, dim=8, rounds=10, local_epochs=2, lr=0.2, seed=1)
    A = 0
    hist = np.array([i for (u, i) in train if u == A]); X = int(hist[0])
    ctrl = np.arange(12, 24)                        # other-group items A never touched
    for kind in ("score", "nobias", "cosine"):
        auc = probe_auc_variant(base, A, X, ctrl, kind=kind)
        assert 0.0 <= auc <= 1.0
        assert auc > 0.5                            # A's own high-redundancy item outranks controls


# ---------- D. sequential model + probe ----------
def test_gru4rec_trains_and_probe_is_sane():
    rng = np.random.default_rng(0)
    seqs = [np.array([i % 20 for i in range(rng.integers(5, 12))], dtype=np.int64) for _ in range(40)]
    model = train_gru4rec(seqs, n_items=20, dim=16, hidden=16, epochs=5, seed=0)
    assert model.item_emb.shape == (20, 16)
    ctx = seqs[0][:3]
    controls = np.array([5, 7, 9, 11])
    auc = seq_probe_auc(model, ctx, int(seqs[0][3]), controls)
    assert 0.0 <= auc <= 1.0
    v = model.predict_vec(ctx)
    assert v.shape == (16,) and np.isfinite(v).all()
