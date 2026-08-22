"""Idempotent run registry, heartbeats, and the retrain cache (§9 rule 4).

A sweep over (backbone × dataset × seed × pair) is made *idempotent and resumable*:
- a sweep **unit** is "done" iff its result file `results/units/<unit_id>.json` exists (written
  atomically) — so completed units are skipped on resume and never recomputed;
- the **from-scratch retrain** (the budget driver, §5/§9) is cached by content key and never run
  twice — even across jobs, even after a kill;
- each shard emits a **heartbeat** `runs/<run_id>/status.json` (atomic) for the dashboard;
- state transitions are appended to `runs/registry.jsonl` (append-only event log).

All writes are atomic; the registry never holds a long-lived handle, so it is robust to the scheduler
hard-kills and to many shards running concurrently as array indices.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

from .checkpoint import atomic_save, load_pickle


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


class SweepRegistry:
    def __init__(self, runs_dir: str | Path, results_dir: str | Path, sweep_id: str):
        self.runs_dir = Path(runs_dir)
        self.results_dir = Path(results_dir)
        self.sweep_id = sweep_id
        self.units_dir = self.results_dir / "units" / sweep_id
        self.events = self.runs_dir / "registry.jsonl"
        self.units_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    # ---- unit completion (the source of truth for idempotency) ----
    def unit_path(self, unit_id: str) -> Path:
        return self.units_dir / f"{unit_id}.json"

    def is_done(self, unit_id: str) -> bool:
        return self.unit_path(unit_id).exists()

    def record_result(self, unit_id: str, result: dict, run_id: str = "") -> None:
        _atomic_write_text(self.unit_path(unit_id), json.dumps(result))
        self.log_event(run_id, unit_id, "done")

    def load_results(self) -> list[dict]:
        out = []
        for p in sorted(self.units_dir.glob("*.json")):
            try:
                out.append(json.loads(p.read_text()))
            except Exception:
                continue
        return out

    def pending(self, units: Iterable[str]) -> list[str]:
        return [u for u in units if not self.is_done(u)]

    # ---- event log + heartbeat (for the dashboard) ----
    def log_event(self, run_id: str, unit_id: str, state: str) -> None:
        line = json.dumps(dict(ts=time.time(), run_id=run_id, sweep=self.sweep_id,
                               unit=unit_id, state=state))
        try:
            with open(self.events, "a") as f:
                f.write(line + "\n")  # POSIX append of a short line is atomic
        except Exception:
            pass

    def heartbeat(self, run_id: str, **info: Any) -> None:
        status = dict(run_id=run_id, sweep=self.sweep_id, last_heartbeat=time.time(),
                      pbs_id=os.environ.get("the scheduler_JOBID", ""), **info)
        _atomic_write_text(self.runs_dir / run_id / "status.json", json.dumps(status, indent=2))

    def summary(self) -> dict:
        return dict(sweep=self.sweep_id, units_done=len(list(self.units_dir.glob("*.json"))))


class RetrainCache:
    """Content-addressed cache of from-scratch retrains — never recompute one (§9 r4).

    Key = (backbone, dataset, seed, sorted removed-interaction set). Lives on persistent storage
    (<DATA>) so it survives scratch purges and is shared across all shards/jobs.
    """

    def __init__(self, cache_dir: str | Path):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(backbone: str, dataset: str, seed: int, removed: list[tuple[int, int]],
            config_hash: str = "") -> str:
        import hashlib
        h = hashlib.sha256()
        h.update(f"{backbone}|{dataset}|{seed}|{config_hash}|".encode())
        h.update(repr(sorted(removed)).encode())
        return h.hexdigest()[:20]

    def get(self, key: str) -> Any | None:
        p = self.dir / f"{key}.pkl"
        if p.exists():
            try:
                return load_pickle(p)
            except Exception:
                return None
        return None

    def put(self, key: str, model: Any) -> None:
        atomic_save(model, self.dir / f"{key}.pkl")

    def has(self, key: str) -> bool:
        return (self.dir / f"{key}.pkl").exists()
