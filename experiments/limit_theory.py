"""Analytical statement of the limit (§6-P2): in a simplified MF/co-occurrence model
the post-unlearning inferability of an item is a *closed-form* function of its redundancy r, which
we confirm by Monte-Carlo AND by real BPR-MF training on synthetic data with a KNOWN structure.

Model.  Users who interact with item X share a latent "taste" direction μ (|μ|²=m); each such user
is p_u = μ + ε_u, ε_u ~ N(0, σ²I). The shared item embedding is the FedAvg/mean fixed point
q_X = mean of the interacting users' embeddings. When user A unlearns (A,X), the server refits
without A, so q_X' = mean over the OTHER r users = μ + η, η ~ N(0, (σ²/r)I). Control items have
random taste, q_c ~ N(0, τ²/r I). The probe compares Δ = ⟨p_A,q_X'⟩ − ⟨p_A,q_c⟩:

    E[Δ] = m                          (the COLLABORATIVE RESIDUE — A's own μ-component, un-removable)
    Var[Δ] = m·σ² + (m·σ² + (m+dσ²)·τ²)/r
    ⇒  AUC(r) = Φ( m / sqrt(Var[Δ]) )  for r ≥ 1,   AUC(0) = 0.5.

This is the fundamental limit in closed form: the residue E[Δ]=m is INDEPENDENT of r (you cannot
remove other users' reinforcement), while redundancy r only sharpens it by averaging down the noise
— so inferability rises and SATURATES at Φ(√m/σ) < 1. r=0 ⇒ no reinforcement ⇒ chance.

  python -m experiments.limit_theory   # -> results/figures/limit_theory.png + prints the fit
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from src.models.mf import train_mf
from src.probes.membership import probe_auc

ROOT = Path(__file__).resolve().parents[1]


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def closed_form_auc(r: int, m: float, sigma: float, tau: float, d: int) -> float:
    if r <= 0:
        return 0.5
    var = m * sigma**2 + (m * sigma**2 + (m + d * sigma**2) * tau**2) / r
    return _phi(m / math.sqrt(var))


def montecarlo_auc(r: int, m: float, sigma: float, tau: float, d: int,
                   trials: int = 20000, n_ctrl: int = 50, seed: int = 0) -> float:
    """Simulate the linear model directly: draw p_A, the r other users, q_X' and controls, and
    measure P(⟨p_A,q_X'⟩ > ⟨p_A,q_c⟩). This is the idealized (linear fixed-point) residue."""
    if r <= 0:
        return 0.5
    g = np.random.default_rng(seed)
    mu = np.zeros(d); mu[0] = math.sqrt(m)                    # planted taste direction
    wins = 0.0
    for _ in range(trials):
        p_A = mu + sigma * g.standard_normal(d)
        q_X = mu + (sigma / math.sqrt(r)) * g.standard_normal(d)   # mean of r others: var σ²/r
        q_c = (tau / math.sqrt(r)) * g.standard_normal((n_ctrl, d))
        s_x = p_A @ q_X
        s_c = q_c @ p_A
        wins += float((s_c < s_x).mean())
    return wins / trials


def real_mf_auc(r: int, m: float, sigma: float, d: int, n_bg_users: int = 300,
                n_bg_items: int = 400, seed: int = 0) -> float:
    """Confirm with REAL BPR-MF: build synthetic interactions where item X is co-consumed by A and
    r other users who share the planted taste μ, plus background traffic, then unlearn A's (A,X) by
    retrain-from-scratch (the oracle) and probe. Returns per-item AUC for A on X vs 50 controls."""
    g = np.random.default_rng(seed + 7919)
    mu = np.zeros(d); mu[0] = math.sqrt(m)
    n_users = 1 + r + n_bg_users
    n_items = 1 + n_bg_items
    X = 0
    # user 0 = A (a taste-μ user); users 1..r = the other X-consumers (taste-μ); rest = background.
    taste = np.zeros((n_users, d))
    taste[: 1 + r] = mu
    taste[1 + r:] = sigma * g.standard_normal((n_bg_users, d))
    item_lat = np.zeros((n_items, d)); item_lat[X] = mu
    item_lat[1:] = g.standard_normal((n_bg_items, d))
    rows = [(u, X) for u in range(1 + r)]                     # everyone who consumes X
    for u in range(n_users):                                 # background interactions (score-ranked)
        sc = item_lat[1:] @ (taste[u] + 0.3 * sigma * g.standard_normal(d))
        for it in (1 + np.argsort(-sc)[:8]):
            rows.append((u, int(it)))
    train = np.array(rows, dtype=np.int64)
    hist_A = train[(train[:, 0] == 0)][:, 1]
    train_minus = train[~((train[:, 0] == 0) & (train[:, 1] == X))]
    mdl = train_mf(train_minus, n_users, n_items, dim=d, epochs=40, seed=seed)   # retrain oracle
    # popularity-matched controls: items A never touched (all background items here have similar pop)
    ctrl = np.array([it for it in g.choice(np.arange(1, n_items), 50, replace=False)
                     if it not in set(hist_A.tolist())])
    return probe_auc(mdl, 0, X, ctrl)


def main() -> None:
    (ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d, m, sigma, tau = 16, 1.0, 1.0, 1.0
    rs = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
    cf = [closed_form_auc(r, m, sigma, tau, d) for r in rs]
    mc = [montecarlo_auc(r, m, sigma, tau, d) for r in rs]
    rs_mf = [0, 2, 8, 32, 128]
    mf = [np.mean([real_mf_auc(r, m, sigma, d, seed=s) for s in range(4)]) for r in rs_mf]
    print("  r      closed-form   monte-carlo   real-MF")
    for i, r in enumerate(rs):
        extra = f"   {mf[rs_mf.index(r)]:.3f}" if r in rs_mf else ""
        print(f"  {r:4d}    {cf[i]:.3f}        {mc[i]:.3f}{extra}")

    x = np.log1p(rs); xm = np.log1p(rs_mf)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.axhline(0.5, ls=":", c="gray", lw=1)
    ax.axhline(_phi(math.sqrt(m) / sigma), ls="--", c="green", lw=1,
               label=f"Φ(√m/σ)={_phi(math.sqrt(m)/sigma):.2f}  (asymptote)")
    ax.plot(x, cf, "-", c="#d62728", lw=2, label="closed form  Φ(m/√Var)")
    ax.plot(x, mc, "o", c="#1f77b4", ms=6, label="Monte-Carlo (linear model)")
    ax.plot(xm, mf, "s", c="#ff7f0e", ms=7, label="real BPR-MF (synthetic, retrain-unlearn)")
    ax.set_xticks(x); ax.set_xticklabels(rs, fontsize=8)
    ax.set_xlabel("redundancy r (# other users who consume X)")
    ax.set_ylabel("probe AUC after verified unlearning")
    ax.set_title("Analytical limit: inferability = Φ(m/√Var(r)), saturating\n"
                 "residue E[Δ]=m is independent of r — the un-removable collaborative signal")
    ax.set_ylim(0.45, 1.0); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = ROOT / "results" / "figures" / "limit_theory.png"
    fig.savefig(out, dpi=140); print(f"  fig -> {out}")


if __name__ == "__main__":
    main()
