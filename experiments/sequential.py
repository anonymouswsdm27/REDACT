"""SEQUENTIAL-residue experiment (§2 secondary claim, §6-P1). Trains GRU4Rec, unlearns
(A,X) by a from-scratch retrain that removes X from A's sequence (the §4.1 oracle, batched one
removal per user), and probes whether X is still inferable from A's own autoregressive predictions.

Shows BOTH residue channels in one model:
  * collaborative (cross-user): probe AUC (A's real pre-X context) rises with X's redundancy — the
    same shared-embedding signal as MF.
  * sequential within-user: the context-specificity GAP = AUC(A's real pre-X context) −
    AUC(random user's context) for the same X. A positive gap = the model still expects X after A's
    specific history, which a collaborative-only (shared-embedding) account cannot explain.

  python -m experiments.sequential [--dataset ml1m --n-users 500 --per-stratum 30 --seeds 0,1]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.data.movielens import preprocess
from src.models.sequential import train_gru4rec
from src.probes.membership import BIN_LABELS, ControlSampler
from src.probes.sequential import build_seq_targets, seq_probe_auc

ROOT = Path(__file__).resolve().parents[1]


def chrono_sequences(ds) -> list[np.ndarray]:
    """Per-user chronological train item ids (from `train`, which is grouped by user in time order —
    NOT ds.user_items, which is re-sorted by item id)."""
    tr = ds.train
    bounds = np.searchsorted(tr[:, 0], np.arange(ds.n_users + 1))
    return [tr[bounds[u]:bounds[u + 1], 1] for u in range(ds.n_users)]


def main() -> None:
    (ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ml1m")
    ap.add_argument("--n-users", type=int, default=500)
    ap.add_argument("--per-stratum", type=int, default=30)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    ds = preprocess(n_users=args.n_users, seed=0, dataset=args.dataset)
    seqs = chrono_sequences(ds)
    cs = ControlSampler(ds.item_pop, ds.user_items)
    targets = build_seq_targets(seqs, ds.item_pop, per_stratum=args.per_stratum, seed=0)
    print(f"[seq] {args.dataset} users={ds.n_users} items={ds.n_items} targets={len(targets)} seeds={seeds}")

    by_bin: dict[int, list[float]] = defaultdict(list)
    randctx: dict[int, list[float]] = defaultdict(list)
    for seed in seeds:
        # one removal per user -> a single batched retrain is a valid leave-one-out oracle (§4.1)
        removed = {t.user: t.item for t in targets}
        seqs_minus = [np.array([it for it in s if not (u == uu and it == removed.get(uu, -1))])
                      if (uu := u) in removed else s for u, s in enumerate(seqs)]
        model = train_gru4rec(seqs_minus, ds.n_items, epochs=args.epochs, seed=seed)
        rng = np.random.default_rng(seed ^ 0xBEEF)
        for t in targets:
            u, x, pos = t.user, t.item, t.pos
            ps = int(np.random.SeedSequence([seed, u, x]).generate_state(1)[0])
            ctrl = cs.sample(u, x, 50, ps)
            if len(ctrl) < 3:
                continue
            context = seqs[u][:pos]                      # A's real history BEFORE X
            if len(context) == 0:
                continue
            by_bin[t.bin].append(seq_probe_auc(model, context, x, ctrl))
            # random-context control: another user's pre-X-length context (collaborative-only)
            v = int(rng.integers(0, ds.n_users))
            rc = seqs[v][:max(1, min(len(context), len(seqs[v])))] if len(seqs[v]) else context
            randctx[t.bin].append(seq_probe_auc(model, rc, x, ctrl))

    rows = []
    for b in range(len(BIN_LABELS)):
        if by_bin[b]:
            real = float(np.mean(by_bin[b])); rnd = float(np.mean(randctx[b]))
            rows.append(dict(bin=BIN_LABELS[b], n=len(by_bin[b]), auc=real,
                             auc_randctx=rnd, seq_gap=real - rnd))
    summary = dict(dataset=args.dataset, n_users=ds.n_users, seeds=seeds, per_bin=rows)
    (ROOT / "results" / f"sequential_{args.dataset}.json").write_text(json.dumps(summary, indent=2))
    print(f"  {'bin':>8} | {'n':>4} | {'AUC(A ctx)':>10} | {'AUC(rand)':>9} | {'seq gap':>7}")
    for r in rows:
        print(f"  {r['bin']:>8} | {r['n']:>4} | {r['auc']:>10.3f} | "
              f"{r['auc_randctx']:>9.3f} | {r['seq_gap']:>+7.3f}")
    _figure(args.dataset, rows)


def _figure(dataset: str, rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.axhline(0.5, ls=":", c="gray", lw=1, label="chance")
    ax.plot(x, [r["auc"] for r in rows], "-o", color="#1f77b4",
            label="A's real pre-X context (both channels)")
    ax.plot(x, [r["auc_randctx"] for r in rows], "-s", color="#ff7f0e",
            label="random context (collaborative channel only)")
    ax.fill_between(x, [r["auc_randctx"] for r in rows], [r["auc"] for r in rows],
                    color="#1f77b4", alpha=0.12, label="within-user sequential residue (gap)")
    ax.set_xticks(x); ax.set_xticklabels([r["bin"] for r in rows], rotation=30, ha="right", fontsize=8)
    ax.set_xlabel("cross-user redundancy r")
    ax.set_ylabel("sequential probe AUC after verified unlearning")
    ax.set_title(f"Sequential model ({dataset}): collaborative residue (rises with r)\n"
                 "+ within-user sequential residue (blue−orange gap)")
    ax.set_ylim(0.4, 1.0); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = ROOT / "results" / "figures" / f"sequential_{dataset}.png"
    fig.savefig(out, dpi=140); print(f"  fig -> {out}")


if __name__ == "__main__":
    main()
