"""The *limit* result (§1, §6-P2, §15.4): characterize inferability as a function of an
item's cross-user redundancy and state it as a **lower bound**.

The empirical curve (probe AUC after a verified-correct unlearning, vs redundancy r = #other users)
is monotone and saturating, and — by construction of the residue — is ~chance at r=0. We fit the
one-parameter-family that encodes exactly that shape:

    AUC(r) = 0.5 + a · (1 - exp(-c · log1p(r))),     a in (0, 0.5],  c > 0

so AUC(0)=0.5 (the redundancy-0 control is pinned, §4.3) and AUC → 0.5+a as r → ∞ (saturation).
This gives:
  * a  = asymptotic *excess* inferability (how leaky a maximally-redundant item is),
  * c  = how fast redundancy buys inferability,
  * r½ = exp(ln2 / c) - 1, the redundancy at which half the excess is reached.

The **lower bound** L(r): we refit the same family to the per-bin lower confidence points
(mean - z·SEM across seeds). "With ~95% confidence, a verified-correct unlearning cannot drive
inferability of an item with redundancy r below L(r)" — the fundamental-limit statement.

Pure-numpy Gauss-Newton fit (no scipy dependency); deterministic.
"""
from __future__ import annotations

import numpy as np

from ..probes.membership import BIN_LABELS, redundancy_bin


def _model(z: np.ndarray, a: float, c: float) -> np.ndarray:
    return 0.5 + a * (1.0 - np.exp(-c * z))


def _sse(z, y, w, a, c):
    return float(np.sum(w * (y - _model(z, a, c)) ** 2))


def _fit_ac(z: np.ndarray, y: np.ndarray, w: np.ndarray | None = None,
            iters: int = 300) -> tuple[float, float]:
    """Weighted least-squares fit of AUC(r)=0.5+a(1-exp(-c·z)) by damped (Levenberg-Marquardt)
    Gauss-Newton on (a, c). z=log1p(r). Only steps that reduce weighted SSE are accepted, so it is
    stable on noisy/near-flat data (e.g. Gowalla). Bounded a in [1e-3,0.5], c in [1e-3,10]."""
    if w is None:
        w = np.ones_like(y)
    a = float(np.clip(2.0 * (max(y.max(), 0.5001) - 0.5), 1e-3, 0.5))   # init from the max, not mean
    c = 0.5
    lam = 1e-3
    sse = _sse(z, y, w, a, c)
    for _ in range(iters):
        e = np.exp(-c * z)
        r = y - (0.5 + a * (1.0 - e))
        da = 1.0 - e; dc = a * z * e
        Jaa = np.sum(w * da * da); Jac = np.sum(w * da * dc); Jcc = np.sum(w * dc * dc)
        ga = np.sum(w * da * r);   gc = np.sum(w * dc * r)
        # LM damping on the diagonal; shrink lam on success, grow on failure.
        Jaa_d, Jcc_d = Jaa * (1 + lam), Jcc * (1 + lam)
        det = Jaa_d * Jcc_d - Jac * Jac
        if abs(det) < 1e-14:
            lam *= 10; continue
        na = float(np.clip(a + (Jcc_d * ga - Jac * gc) / det, 1e-3, 0.5))
        nc = float(np.clip(c + (Jaa_d * gc - Jac * ga) / det, 1e-3, 10.0))
        nsse = _sse(z, y, w, na, nc)
        if nsse < sse:
            if abs(na - a) < 1e-9 and abs(nc - c) < 1e-9:
                a, c = na, nc; break
            a, c, sse, lam = na, nc, nsse, max(lam * 0.5, 1e-9)
        else:
            lam *= 4
            if lam > 1e9:
                break
    return a, c


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def per_bin_curve(r: np.ndarray, auc: np.ndarray, seed: np.ndarray) -> dict:
    """Per-redundancy-bin mean AUC and SEM across seeds (the plot points + the bound anchors)."""
    bins = np.array([redundancy_bin(int(v)) for v in r])
    out = {"label": [], "bin": [], "r_med": [], "mean": [], "sem": [], "n": []}
    for b in range(len(BIN_LABELS)):
        m = bins == b
        if not m.any():
            continue
        seeds = np.unique(seed[m])
        per_seed = np.array([np.nanmean(auc[m & (seed == s)]) for s in seeds])
        per_seed = per_seed[~np.isnan(per_seed)]
        if len(per_seed) == 0:
            continue
        sem = float(np.std(per_seed) / np.sqrt(len(per_seed))) if len(per_seed) > 1 else 0.0
        out["label"].append(BIN_LABELS[b]); out["bin"].append(b)
        out["r_med"].append(float(np.median(r[m])))
        out["mean"].append(float(np.mean(per_seed))); out["sem"].append(sem)
        out["n"].append(int(m.sum()))
    return out


def fit_bound(r: np.ndarray, auc: np.ndarray, seed: np.ndarray, z_conf: float = 1.96) -> dict:
    """Fit the mean curve and the lower-confidence bound L(r). Returns coefficients, R², r½,
    and the per-bin curve for plotting. `z_conf` sets the confidence band (1.96 ≈ 95%)."""
    ok = ~np.isnan(auc)
    r, auc, seed = r[ok], auc[ok], seed[ok]

    curve = per_bin_curve(r, auc, seed)
    zb = np.log1p(np.array(curve["r_med"]))
    ybar = np.array(curve["mean"]); sem = np.array(curve["sem"])
    wa = np.array(curve["n"], dtype=float)

    a, c = _fit_ac(zb, ybar, wa)                             # mean fit on per-bin anchors (weighted)
    r2 = _r2(ybar, _model(zb, a, c))                         # R² on the per-bin means

    # lower bound: fit the SAME family to the per-bin (mean - z_conf·SEM) anchors, weighted by n.
    lb_pts = np.clip(ybar - z_conf * sem, 0.5, 1.0)
    a_lo, c_lo = _fit_ac(zb, lb_pts, wa)

    r_half = float(np.exp(np.log(2.0) / c) - 1.0)
    return {
        "a": a, "c": c, "r2": r2, "r_half": r_half,
        "a_lo": a_lo, "c_lo": c_lo,
        "asymptote": 0.5 + a, "asymptote_lo": 0.5 + a_lo,
        "curve": curve, "n": int(len(auc)),
        "form": "AUC(r)=0.5+a*(1-exp(-c*log1p(r)))",
    }


def bound_at(fit: dict, r: float, lower: bool = False) -> float:
    """Evaluate the fitted mean (lower=False) or the lower-confidence bound (lower=True) at r."""
    a = fit["a_lo"] if lower else fit["a"]
    c = fit["c_lo"] if lower else fit["c"]
    return float(_model(np.log1p(np.array([float(r)])), a, c)[0])
