"""Probe-architecture ablation (§6-P5, F2): show the collaborative residue is not an
artifact of one score function. On a single batched leave-one-out retrain oracle (one removal per
user), we measure the residue AUC vs redundancy under three probe statistics — full score, score
without item bias (rules out popularity/bias), and cosine (rules out embedding-magnitude). If all
three rise with redundancy, the residue is robust to the probe.

  python -m experiments.ablation_probe [--dataset ml1m --n-users 800 --per-stratum 60 --seeds 0,1]
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.data.movielens import preprocess
from src.models.mf import MFModel, refit_user, train_mf
from src.probes.membership import BIN_LABELS, ControlSampler, build_targets, probe_auc_variant

ROOT = Path(__file__).resolve().parents[1]
KINDS = ["score", "nobias", "cosine"]


def main() -> None:
    (ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ml1m")
    ap.add_argument("--n-users", type=int, default=800)
    ap.add_argument("--per-stratum", type=int, default=60)
    ap.add_argument("--seeds", default="0,1")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    ds = preprocess(n_users=args.n_users, seed=0, dataset=args.dataset)
    cs = ControlSampler(ds.item_pop, ds.user_items)
    targets = build_targets(ds.user_items, ds.item_pop, per_stratum=args.per_stratum, seed=0)
    seen: set[int] = set(); uniq = []                       # one removal per user -> one batched retrain
    for t in targets:
        if t.user not in seen:
            seen.add(t.user); uniq.append(t)
    print(f"[probe-abl] {args.dataset} users={ds.n_users} targets={len(uniq)} seeds={seeds}")

    by_bin: dict[tuple[int, str], list[float]] = defaultdict(list)
    for seed in seeds:
        remove = {t.user: t.item for t in uniq}
        drop = np.array([remove.get(int(u), -1) == int(i) for u, i in ds.train[:, :2]])
        floor = train_mf(ds.train[~drop], ds.n_users, ds.n_items, dim=32, epochs=30, seed=seed)
        for t in uniq:
            u, x = t.user, t.item
            ps = int(np.random.SeedSequence([seed, u, x]).generate_state(1)[0])
            ctrl = cs.sample(u, x, 50, ps)
            if len(ctrl) < 3:
                continue
            hist_minus = ds.user_items[u][ds.user_items[u] != x]
            m = MFModel(p=floor.p.copy(), q=floor.q, b=floor.b, dim=floor.dim)
            m.p[u] = refit_user(floor.q, floor.b, hist_minus, ds.n_items, dim=32, seed=ps)
            for kind in KINDS:
                by_bin[(t.bin, kind)].append(probe_auc_variant(m, u, x, ctrl, kind=kind))

    rows = []
    for b in range(len(BIN_LABELS)):
        if not by_bin[(b, "score")]:
            continue
        rows.append(dict(bin=BIN_LABELS[b], n=len(by_bin[(b, "score")]),
                         **{k: float(np.mean(by_bin[(b, k)])) for k in KINDS}))
    (ROOT / "results" / f"ablation_probe_{args.dataset}.json").write_text(
        json.dumps(dict(dataset=args.dataset, seeds=seeds, per_bin=rows), indent=2))
    print(f"  {'bin':>8} | {'n':>4} | " + " | ".join(f"{k:>7}" for k in KINDS))
    for r in rows:
        print(f"  {r['bin']:>8} | {r['n']:>4} | " + " | ".join(f"{r[k]:>7.3f}" for k in KINDS))
    _figure(args.dataset, rows)


def _figure(dataset: str, rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(7, 4.6))
    styles = {"score": ("-o", "#1f77b4", "full score  ⟨p,q⟩+b (default)"),
              "nobias": ("-s", "#ff7f0e", "no item bias  ⟨p,q⟩"),
              "cosine": ("-^", "#2ca02c", "cosine  cos(p,q)")}
    ax.axhline(0.5, ls=":", c="gray", lw=1, label="chance")
    for k in KINDS:
        st, col, lab = styles[k]
        ax.plot(x, [r[k] for r in rows], st, color=col, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([r["bin"] for r in rows], rotation=30, ha="right", fontsize=8)
    ax.set_xlabel("cross-user redundancy r")
    ax.set_ylabel("residue probe AUC after verified unlearning")
    ax.set_title(f"Probe-architecture ablation ({dataset}): residue is robust\n"
                 "all three probe statistics rise with redundancy")
    ax.set_ylim(0.4, 1.0); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = ROOT / "results" / "figures" / f"ablation_probe_{dataset}.png"
    fig.savefig(out, dpi=140); print(f"  fig -> {out}")


if __name__ == "__main__":
    main()
