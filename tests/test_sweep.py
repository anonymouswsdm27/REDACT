"""Sweep-level idempotency/determinism (§9 r4, §13).

A sweep unit is a pure function of (config, seed, A, X): recomputing it — in any order, with a
cold or warm retrain cache — yields identical results. That is exactly why a killed/resumed sweep
equals an uninterrupted one (the full hard-kill integration proof is scripts/kill_resume_demo.py).
Skipped automatically if the ML-1M data isn't present.
"""
from __future__ import annotations

import math

import pytest

from experiments.run import DEFAULT_CFG, _config_hash, build_dataset, build_units, compute_unit
from src.data.movielens import RAW
from src.probes.membership import ControlSampler
from src.runtime import RetrainCache

pytestmark = pytest.mark.skipif(not RAW.exists(), reason="ML-1M not downloaded")


def _cfg():
    c = dict(DEFAULT_CFG)
    c.update(n_users=150, dim=16, epochs=8, per_stratum=2, seeds=[0])
    c["config_hash"] = _config_hash(c)
    return c


def _eq(a: dict, b: dict) -> bool:
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, float) or isinstance(vb, float):
            va, vb = float(va), float(vb)
            if math.isnan(va) and math.isnan(vb):
                continue
            if abs(va - vb) > 1e-9:
                return False
        elif va != vb:
            return False
    return True


def test_unit_is_deterministic_and_cache_consistent(tmp_path):
    cfg = _cfg()
    ds = build_dataset(cfg)
    cs = ControlSampler(ds.item_pop, ds.user_items)
    units = build_units(cfg, ds)[:6]
    assert units, "no units built"

    iu = [[] for _ in range(ds.n_items)]
    for uu, ii in ds.train:
        iu[ii].append(int(uu))
    cache1 = RetrainCache(tmp_path / "c1")   # cold cache
    cache2 = RetrainCache(tmp_path / "c2")   # independent cold cache
    for u in units:
        r1 = compute_unit(cfg, ds, cs, u, cache1, {}, iu)
        r2 = compute_unit(cfg, ds, cs, u, cache2, {}, iu)   # different cache, fresh mem
        r1b = compute_unit(cfg, ds, cs, u, cache1, {}, iu)  # warm cache, recompute
        assert r1 is not None and _eq(r1, r2), f"nondeterministic across caches: {u['unit_id']}"
        assert _eq(r1, r1b), f"warm-cache recompute differs: {u['unit_id']}"
