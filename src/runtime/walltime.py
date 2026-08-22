"""Walltime budgeting + clean stop on the scheduler signals (§9 rule 2).

Every long run tracks wall-time against a *soft* budget (e.g. 46 h, under the the walltime hard cap) and
also listens for SIGTERM / SIGUSR1 that the scheduler sends before a hard kill. When either fires the run
checkpoints, writes status, and exits with the distinct **out-of-time code 64** so the wrapper
script knows to chain a successor (vs exit 0 = genuinely done).
"""
from __future__ import annotations

import re
import signal
import time

OUT_OF_TIME = 64  # exit code: stopped with work remaining -> resubmit a successor (§9 r3)
DONE = 0

_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_duration(s: str | float | int) -> float:
    """'46h' / '90m' / '3600s' / 3600 -> seconds."""
    if isinstance(s, (int, float)):
        return float(s)
    m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([smhd]?)\s*", str(s).lower())
    if not m:
        raise ValueError(f"bad duration: {s!r}")
    return float(m.group(1)) * _UNITS.get(m.group(2) or "s", 1)


class WalltimeBudget:
    """Soft walltime budget + cooperative stop flag.

    Usage:
        wt = WalltimeBudget("46h")
        for unit in units:
            if wt.should_stop(): break    # checkpoint + exit(wt.exit_code()) outside the loop
            ...work...
    """

    def __init__(self, max_runtime: str | float, install_signals: bool = True):
        self.start = time.monotonic()
        self.budget = parse_duration(max_runtime)
        self._signalled = False
        self._sig = None
        if install_signals:
            for s in (signal.SIGTERM, signal.SIGUSR1, signal.SIGUSR2):
                try:
                    signal.signal(s, self._on_signal)
                except (ValueError, OSError):
                    pass  # not main thread / unsupported — fall back to budget-only

    def _on_signal(self, signum, frame):  # noqa: ANN001
        self._signalled = True
        self._sig = signum

    def elapsed(self) -> float:
        return time.monotonic() - self.start

    def remaining(self) -> float:
        return self.budget - self.elapsed()

    def expired(self) -> bool:
        return self.elapsed() >= self.budget

    def signalled(self) -> bool:
        return self._signalled

    def should_stop(self) -> bool:
        """True if the soft budget is spent or the scheduler asked us to stop."""
        return self._signalled or self.expired()

    def exit_code(self, work_remaining: bool) -> int:
        """64 if we stopped early with work left (resubmit), else 0 (done)."""
        return OUT_OF_TIME if work_remaining else DONE

    def reason(self) -> str:
        if self._signalled:
            return f"signal:{self._sig}"
        if self.expired():
            return "budget-expired"
        return "ok"
