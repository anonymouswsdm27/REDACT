"""Fit + plot the collaborative-residue LOWER BOUND from an existing sweep's units (the design notes
§6-P2). No new training — pure analysis on results/units/<sweep>/*.json.

  python -m experiments.limit_fit --sweep-id mf_ml1m [mf_ml100k ...]

Writes results/limit_<sweep>.json (coefficients, R², r½) and results/figures/limit_<sweep>.png
(per-bin AUC±SEM, the fitted saturating curve, and the shaded 95% lower-bound region).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from src.metrics.limit import _model, fit_bound

ROOT = Path(__file__).resolve().parents[1]


def load(sweep: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = [json.loads(open(p).read()) for p in glob.glob(str(ROOT / "results" / "units" / sweep / "*.json"))]
    rows = [r for r in rows if r.get("auc_floor") is not None]
    r = np.array([x["r"] for x in rows], dtype=float)
    auc = np.array([x["auc_floor"] for x in rows], dtype=float)
    seed = np.array([x.get("seed", 0) for x in rows])
    return r, auc, seed


def figure(sweep: str, fit: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cv = fit["curve"]
    x = np.log1p(np.array(cv["r_med"]))
    grid = np.linspace(0, max(x.max(), 1) * 1.02, 200)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.axhline(0.5, ls=":", c="gray", lw=1, label="chance")
    ax.errorbar(x, cv["mean"], yerr=np.array(cv["sem"]) * 1.96, fmt="o", ms=6, capsize=3,
                color="#1f77b4", label="measured AUC (mean ±95% across seeds)")
    ax.plot(grid, _model(grid, fit["a"], fit["c"]), "-", c="#d62728", lw=2,
            label=f"fit  0.5+a(1-e^(-c·log1p r))  R²={fit['r2']:.2f}")
    lo = _model(grid, fit["a_lo"], fit["c_lo"])
    ax.fill_between(grid, 0.5, lo, color="#d62728", alpha=0.12,
                    label="95% lower bound L(r) — the limit")
    ax.plot(grid, lo, "--", c="#d62728", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(cv["label"], rotation=30, ha="right", fontsize=8)
    ax.set_xlabel("cross-user redundancy r (# other users who touched X)")
    ax.set_ylabel("probe AUC after verified unlearning")
    ax.set_title(f"Fundamental limit — {sweep}\n"
                 f"asymptote 0.5+a={fit['asymptote']:.2f}  r½={fit['r_half']:.1f}")
    ax.set_ylim(0.45, 1.0); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = ROOT / "results" / "figures" / f"limit_{sweep}.png"
    fig.savefig(out, dpi=140); print(f"  fig -> {out}")


def main() -> None:
    (ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-id", nargs="+", required=True)
    for sweep in ap.parse_args().sweep_id:
        r, auc, seed = load(sweep)
        if len(auc) == 0:
            print(f"[skip] {sweep}: no units"); continue
        fit = fit_bound(r, auc, seed)
        slim = {k: v for k, v in fit.items() if k != "curve"}
        (ROOT / "results" / f"limit_{sweep}.json").write_text(json.dumps(fit, indent=2))
        print(f"[{sweep}] a={fit['a']:.3f} c={fit['c']:.3f} R²={fit['r2']:.3f} "
              f"asymptote={fit['asymptote']:.3f} r½={fit['r_half']:.1f} "
              f"| lower-bound asymptote={fit['asymptote_lo']:.3f}")
        figure(sweep, fit)
        _ = slim


if __name__ == "__main__":
    main()
