"""Atomic, RNG-capturing checkpoints (§9 rule 1).

A checkpoint contains *everything needed to resume*: caller state (model/optim/round/progress/
metric history) plus the RNG states of `random`, `numpy`, and `torch` (+CUDA). Writes are atomic
(`*.tmp` then `os.replace`) so a kill mid-write never corrupts the checkpoint. On startup the
caller calls `load_latest()` and, if present, restores RNG so a **resumed run is numerically
equivalent to an uninterrupted one** (the §13 invariant; tests/test_runtime.py proves it).
"""
from __future__ import annotations

import os
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np


def capture_rng() -> dict:
    state: dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state()}
    try:
        import torch
        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
    except Exception:
        pass
    return state


def restore_rng(state: dict) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    try:
        import torch
        if "torch" in state:
            torch.set_rng_state(state["torch"])
        if "torch_cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    except Exception:
        pass


def atomic_save(obj: Any, path: str | Path) -> None:
    """Pickle `obj` to `path` atomically (tmp + fsync + os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)  # atomic on POSIX


def load_pickle(path: str | Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


class Checkpointer:
    """Single-slot atomic checkpoint with an optional persistent mirror.

    ckpt_dir lives on fast scratch (<SCRATCH> on a cluster); keep_dir, if given, is a persistent mirror
    (<DATA>) so a resumed job — possibly after scratch purge — can still recover (§7, §9).
    """

    NAME = "checkpoint.pkl"

    def __init__(self, ckpt_dir: str | Path, keep_dir: str | Path | None = None):
        self.ckpt_dir = Path(ckpt_dir)
        self.keep_dir = Path(keep_dir) if keep_dir else None

    def _paths(self) -> list[Path]:
        ps = [self.ckpt_dir / self.NAME]
        if self.keep_dir:
            ps.append(self.keep_dir / self.NAME)
        return ps

    def save(self, state: dict, with_rng: bool = True) -> None:
        payload = dict(state)
        if with_rng:
            payload["_rng"] = capture_rng()
        for p in self._paths():
            atomic_save(payload, p)

    def load_latest(self, restore_rng_state: bool = True) -> dict | None:
        for p in self._paths():
            if p.exists():
                try:
                    payload = load_pickle(p)
                except Exception:
                    continue
                if restore_rng_state and "_rng" in payload:
                    restore_rng(payload["_rng"])
                return {k: v for k, v in payload.items() if k != "_rng"}
        return None

    def exists(self) -> bool:
        return any(p.exists() for p in self._paths())
