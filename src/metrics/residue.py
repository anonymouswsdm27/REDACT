"""Aggregation of per-unit residue measurements into the redundancy-stratified result (§12).

A "unit row" is one (A,X) measurement: redundancy bin + AUCs for {floor (retrain), naive-delete,
full (no unlearn), placebo}. We reduce many rows to per-bin means and the headline statistics:
Spearman(log redundancy, floor-AUC) and the mean leak (method − floor).
"""
from __future__ import annotations

import numpy as np

from ..probes.membership import BIN_LABELS


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0


HI_BINS = ("51-100", "101-200", "201-500", "501+")
# methods that are references, not practical unlearning methods (no "leak vs floor" reported)
NON_METHODS = ("auc_floor", "auc_full", "auc_placebo")


def aggregate(rows: list[dict]) -> dict:
    """Reduce per-(A,X) rows to per-bin statistics, **seed-aware**: the unit of replication is the
    SEED, so every per-bin value is the mean over seeds and `<name>_std` is the std ACROSS seeds
    (for avg±std error bands). Generic over any auc_*/util_*/collateral_*/oth_* columns present."""
    if not rows:
        return {"per_bin": [], "n": 0, "methods": []}
    auc_cols = sorted({k for row in rows for k in row if k.startswith("auc_")})
    util_cols = sorted({k for row in rows for k in row if k.startswith("util_")})
    coll_cols = sorted({k for row in rows for k in row if k.startswith("collateral_")})
    oth_cols = sorted({k for row in rows for k in row if k.startswith("oth_")})
    metric_cols = auc_cols + util_cols + coll_cols + oth_cols
    short = {c: c[4:] for c in auc_cols}                      # auc_floor -> floor
    out_name = {c: short[c] for c in auc_cols}               # auc_* shortened; others keep full name
    out_name.update({c: c for c in util_cols + coll_cols + oth_cols})
    practical = [c for c in auc_cols if c not in NON_METHODS]
    a = {"r": np.array([row.get("r", np.nan) for row in rows], dtype=float),
         "bin": np.array([row.get("bin", -1) for row in rows], dtype=float),
         "seed": np.array([row.get("seed", 0) for row in rows], dtype=float)}
    for c in metric_cols:
        a[c] = np.array([row.get(c, np.nan) for row in rows], dtype=float)
    all_seeds = np.unique(a["seed"])

    def seed_stats(col, mask):
        """(mean over seeds, std across seeds) of the per-seed means within `mask`."""
        pm = []
        for s in np.unique(a["seed"][mask]):
            v = a[col][mask & (a["seed"] == s)]; v = v[~np.isnan(v)]
            if len(v):
                pm.append(v.mean())
        if not pm:
            return float("nan"), float("nan")
        pm = np.array(pm)
        return float(pm.mean()), (float(pm.std(ddof=1)) if len(pm) > 1 else 0.0)

    per_bin = []
    for b, lab in enumerate(BIN_LABELS):
        m = a["bin"] == b
        if not m.any():
            per_bin.append({"bin": lab, "n": 0})
            continue
        row = {"bin": lab, "n": int(m.sum()), "n_seeds": int(len(np.unique(a["seed"][m])))}
        for c in metric_cols:
            mu, sd = seed_stats(c, m)
            row[out_name[c]] = mu
            row[f"{out_name[c]}_std"] = sd
        for c in practical:                     # leak = per-seed (method-floor), averaged over seeds
            ls = []
            for s in np.unique(a["seed"][m]):
                ms = m & (a["seed"] == s)
                mc = a[c][ms]; mc = mc[~np.isnan(mc)]
                mf = a["auc_floor"][ms]; mf = mf[~np.isnan(mf)]
                if len(mc) and len(mf):
                    ls.append(mc.mean() - mf.mean())
            row[f"leak_{short[c]}"] = float(np.mean(ls)) if ls else float("nan")
        per_bin.append(row)

    # headline stats per seed -> mean +/- std across seeds
    rhos, hifs = [], []
    hi_mask = np.isin(a["bin"], [BIN_LABELS.index(h) for h in HI_BINS])
    for s in all_seeds:
        sm = (a["seed"] == s) & (~np.isnan(a["auc_floor"]))
        if sm.sum() >= 3:
            rhos.append(spearman(np.log1p(a["r"][sm]), a["auc_floor"][sm]))
        hh = a["auc_floor"][sm & hi_mask]; hh = hh[~np.isnan(hh)]
        if len(hh):
            hifs.append(hh.mean())

    def ms(arr):
        arr = np.array(arr, dtype=float)
        return (float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0) if len(arr) \
            else (float("nan"), float("nan"))

    out = {"per_bin": per_bin, "n": len(rows), "n_seeds": int(len(all_seeds)),
           "seeds": [int(s) for s in all_seeds],
           "methods": [short[c] for c in auc_cols], "practical": [short[c] for c in practical],
           "util_methods": [c[5:] for c in util_cols],
           "collateral_methods": [c[11:] for c in coll_cols],
           "oth_methods": [c[4:] for c in oth_cols]}
    out["spearman_logr_floor"], out["spearman_logr_floor_std"] = ms(rhos)
    out["high_redundancy_floor"], out["high_redundancy_floor_std"] = ms(hifs)
    full = a["bin"] >= 0
    for c in auc_cols:
        out[f"mean_auc_{short[c]}"], out[f"mean_auc_{short[c]}_std"] = seed_stats(c, full)
    for c in practical:
        ls = []
        for s in all_seeds:
            ms_ = (a["seed"] == s)
            mc = a[c][ms_]; mc = mc[~np.isnan(mc)]
            mf = a["auc_floor"][ms_]; mf = mf[~np.isnan(mf)]
            if len(mc) and len(mf):
                ls.append(mc.mean() - mf.mean())
        out[f"mean_leak_{short[c]}"], out[f"mean_leak_{short[c]}_std"] = ms(ls)
    for c in util_cols + coll_cols + oth_cols:
        out[f"mean_{c}"], out[f"mean_{c}_std"] = seed_stats(c, full)
    return out
