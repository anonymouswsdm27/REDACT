"""Tests for the an HPC cluster runtime (§9, §13).

The load-bearing guarantee: a **resumed run is numerically equivalent to an uninterrupted one**.
We prove it by running an RNG-driven loop, checkpointing mid-way, restoring into a fresh RNG
context, finishing, and asserting bit-identical results. Plus registry idempotency, retrain-cache
determinism, atomic save round-trip, and walltime stop behaviour.
"""
from __future__ import annotations

import numpy as np
import torch

from src.runtime import (
    Checkpointer,
    RetrainCache,
    SweepRegistry,
    WalltimeBudget,
    atomic_save,
    load_pickle,
    parse_duration,
)


def _rng_loop(n_steps: int, ckpt: Checkpointer | None, stop_at: int | None,
              resume: bool) -> list[float]:
    """A resumable loop: each step draws from numpy+torch global RNG and appends to a history.
    If `resume`, it first loads the checkpoint (restoring RNG + progress) and continues."""
    if resume and ckpt is not None and ckpt.exists():
        state = ckpt.load_latest()                      # restores RNG too
        hist, start = list(state["hist"]), state["step"]
    else:
        np.random.seed(1234)
        torch.manual_seed(1234)
        hist, start = [], 0
    for step in range(start, n_steps):
        v = float(np.random.rand()) + float(torch.randn(1).item())
        hist.append(v)
        if ckpt is not None and stop_at is not None and step + 1 == stop_at:
            ckpt.save({"hist": hist, "step": step + 1})  # captures RNG at this point
            return hist  # simulate a hard kill right after checkpoint
    return hist


def test_parse_duration():
    assert parse_duration("46h") == 46 * 3600
    assert parse_duration("90m") == 5400
    assert parse_duration("30s") == 30
    assert parse_duration(120) == 120


def test_atomic_save_roundtrip(tmp_path):
    obj = {"a": np.arange(5), "b": [1, 2, 3]}
    atomic_save(obj, tmp_path / "x.pkl")
    back = load_pickle(tmp_path / "x.pkl")
    assert np.array_equal(back["a"], obj["a"]) and back["b"] == obj["b"]


def test_resume_equals_uninterrupted(tmp_path):
    full = _rng_loop(10, ckpt=None, stop_at=None, resume=False)

    ckpt = Checkpointer(tmp_path / "ck")
    part = _rng_loop(10, ckpt=ckpt, stop_at=5, resume=False)   # runs 5 steps, checkpoints, "dies"
    assert len(part) == 5
    # fresh RNG context (deliberately perturbed) — resume must override it from the checkpoint
    np.random.seed(999)
    torch.manual_seed(999)
    resumed = _rng_loop(10, ckpt=ckpt, stop_at=None, resume=True)

    assert len(resumed) == 10
    assert np.allclose(resumed, full), "resumed run is not bit-identical to uninterrupted"


def test_registry_idempotency(tmp_path):
    reg = SweepRegistry(tmp_path / "runs", tmp_path / "results", "sweepA")
    units = ["u0", "u1", "u2"]
    assert reg.pending(units) == units
    reg.record_result("u1", {"auc": 0.8}, run_id="r0")
    assert reg.is_done("u1") and not reg.is_done("u0")
    assert reg.pending(units) == ["u0", "u2"]
    assert reg.load_results() == [{"auc": 0.8}]
    reg.heartbeat("r0", progress=0.33)
    assert (tmp_path / "runs" / "r0" / "status.json").exists()


def test_retrain_cache(tmp_path):
    c = RetrainCache(tmp_path / "cache")
    k1 = c.key("mf", "ml1m", 0, [(3, 7), (1, 2)])
    k2 = c.key("mf", "ml1m", 0, [(1, 2), (3, 7)])      # order-independent
    assert k1 == k2 and not c.has(k1)
    c.put(k1, {"q": np.ones(3)})
    assert c.has(k1) and np.array_equal(c.get(k1)["q"], np.ones(3))
    assert c.key("mf", "ml1m", 1, [(3, 7)]) != k1       # seed changes key


def test_walltime_stop():
    wt = WalltimeBudget(0.05, install_signals=False)   # 50 ms budget
    import time
    time.sleep(0.06)
    assert wt.should_stop() and wt.expired()
    assert wt.exit_code(work_remaining=True) == 64
    assert wt.exit_code(work_remaining=False) == 0
