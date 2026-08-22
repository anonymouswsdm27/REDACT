"""The privacy-utility FRONTIER (§6-P3): show that ours reaches the retrain floor at
ZERO harm to other users, and that the only way *below* the floor is to delete other users' data —
which costs their utility. Two figures:

  (1) ours' own knob: requesting-user inferability (AUC) vs the user's OWN held-out utility, as the
      suppression strength alpha sweeps 0 (naive) -> 1 (floor) -> 2 (over-suppress). Ours hits the
      floor at ~no utility cost; sub-floor costs the user's own utility.
  (2) the fundamental frontier: requesting-user inferability vs OTHER users' utility loss. Ours is a
      single point at (floor AUC, 0 loss). Neighbor-deletion (SRU-style) traces the sub-floor curve
      at increasing others'-utility loss. Nothing reaches ours' corner without harming others.

  python -m experiments.frontier [--dataset ml1m --n-users 800 --pairs 60 --seeds 0,1]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.data.movielens import preprocess
from src.metrics.ranking import holdout_ndcg
from src.models.mf import MFModel, refit_user
from src.probes.membership import ControlSampler, build_targets, probe_auc
from src.unlearning.methods import neighbor_delete_unlearn, ours_suppress

ROOT = Path(__file__).resolve().parents[1]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
FRACS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]


def others_ndcg(base: MFModel, x: int, others: list[int], cs: ControlSampler, n_items: int,
                q_x: np.ndarray, b_x: float, seed: int) -> float:
    """Mean NDCG of item X for OTHER users, given X's deployed embedding (q_x, b_x)."""
    if not others:
        return float("nan")
    r2 = np.random.default_rng(seed ^ 0x5151); vals = []
    for o in others:
        negs = []
        while len(negs) < 50:
            c = int(r2.integers(0, n_items))
            if c != x and c not in cs.hist[o]:
                negs.append(c)
        negs = np.array(negs)
        s_x = float(q_x @ base.p[o] + b_x)
        s_neg = base.q[negs] @ base.p[o] + base.b[negs]
        vals.append(1.0 / np.log2(int((s_neg > s_x).sum()) + 2))
    return float(np.mean(vals))


def main() -> None:
    (ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ml1m")
    ap.add_argument("--n-users", type=int, default=800)
    ap.add_argument("--pairs", type=int, default=60)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--min-r", type=int, default=6)      # frontier is interesting at high redundancy
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    from src.federated.simulator import train_fedgmf

    ds = preprocess(n_users=args.n_users, seed=0, dataset=args.dataset)
    cs = ControlSampler(ds.item_pop, ds.user_items)
    item_users: dict[int, list[int]] = defaultdict(list)
    for u, i in ds.train[:, :2]:
        item_users[int(i)].append(int(u))
    targets = [t for t in build_targets(ds.user_items, ds.item_pop, per_stratum=40, seed=0)
               if t.r >= args.min_r][: args.pairs]
    print(f"[frontier] {args.dataset} n_users={ds.n_users} n_items={ds.n_items} "
          f"pairs={len(targets)} (r>={args.min_r}) seeds={seeds}")

    acc = defaultdict(list)
    for seed in seeds:
        base = train_fedgmf(ds.train, ds.n_users, ds.n_items, dim=32, rounds=15,
                            local_epochs=3, lr=0.2, seed=seed, log_contribs=True)
        for t in targets:
            u, x = t.user, t.item
            ps = int(np.random.SeedSequence([seed, u, x]).generate_state(1)[0])
            ctrl = cs.sample(u, x, 50, ps)
            if len(ctrl) < 3:
                continue
            hist_minus = ds.user_items[u][ds.user_items[u] != x]
            others = [o for o in item_users[x] if o != u]
            ex = cs.hist[u]
            pos = int(ds.test[u, 1])

            def _util(m):
                return holdout_ndcg(m, u, pos, ex, ds.n_items, seed=ps)  # noqa: B023

            # retrain-oracle floor (remove only (u,x)) for the reference line
            keys = ds.train[:, 0].astype(np.int64) * ds.n_items + ds.train[:, 1]
            tr_minus = ds.train[keys != (u * ds.n_items + x)]
            floor = train_fedgmf(tr_minus, ds.n_users, ds.n_items, dim=32, rounds=15,
                                local_epochs=3, lr=0.2, seed=seed, log_contribs=False)
            fm = MFModel(p=base.p.copy(), q=floor.q, b=floor.b, dim=base.dim)
            fm.p[u] = refit_user(floor.q, floor.b, hist_minus, ds.n_items, dim=32, seed=ps)
            acc["floor_auc"].append(probe_auc(fm, u, x, ctrl))
            oth_base = others_ndcg(base, x, others, cs, ds.n_items, base.q[x], base.b[x], ps)
            acc["oth_base"].append(oth_base)

            for a in ALPHAS:                              # ours' private knob (zero others-harm)
                m = ours_suppress(base, u, x, hist_minus, ds.n_items, dim=32, alpha=a, seed=ps)
                acc[f"ours_auc_{a}"].append(probe_auc(m, u, x, ctrl))
                acc[f"ours_util_{a}"].append(_util(m))
            for fr in FRACS:                              # neighbor-deletion (harms others)
                m, k = neighbor_delete_unlearn(base, u, x, hist_minus, ds.n_items,
                                               other_users=others, frac=fr, dim=32, seed=ps)
                acc[f"nd_auc_{fr}"].append(probe_auc(m, u, x, ctrl))
                oth = others_ndcg(base, x, [o for o in others], cs, ds.n_items, m.q[x], m.b[x], ps)
                acc[f"nd_oth_{fr}"].append(oth)

    def mean(k):
        v = np.array(acc[k], dtype=float); v = v[~np.isnan(v)]
        return float(v.mean()) if len(v) else float("nan")

    floor = mean("floor_auc"); oth_base = mean("oth_base")
    summary = dict(dataset=args.dataset, n_pairs=len(targets), seeds=seeds, floor_auc=floor,
                   oth_base=oth_base,
                   ours=[dict(alpha=a, auc=mean(f"ours_auc_{a}"), util=mean(f"ours_util_{a}"))
                         for a in ALPHAS],
                   neighbor=[dict(frac=fr, auc=mean(f"nd_auc_{fr}"),
                                  oth_loss=oth_base - mean(f"nd_oth_{fr}")) for fr in FRACS])
    (ROOT / "results" / f"frontier_{args.dataset}.json").write_text(json.dumps(summary, indent=2))
    print(f"  floor AUC={floor:.3f}   others' base NDCG={oth_base:.3f}")
    print("  ours:", [(a, round(mean(f'ours_auc_{a}'), 3), round(mean(f'ours_util_{a}'), 3)) for a in ALPHAS])
    print("  nbr :", [(fr, round(mean(f'nd_auc_{fr}'), 3),
                        round(oth_base - mean(f'nd_oth_{fr}'), 3)) for fr in FRACS])
    _figures(args.dataset, summary)


def _figures(dataset: str, s: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.7))
    floor = s["floor_auc"]
    naive = next(d["auc"] for d in s["ours"] if d["alpha"] == 0.0)
    umean = float(np.mean([d["util"] for d in s["ours"]]))

    # (a) ON-DEVICE threat: ours' alpha knob drops the user's inferability at ~no cost to the
    #     user's OWN feed (util ~flat) and ZERO collateral to others.
    al = [d["alpha"] for d in s["ours"]]; oa = [d["auc"] for d in s["ours"]]
    ax[0].plot(al, oa, "-o", color="#4c72b0", label="ours (private, 0 collateral)")
    ax[0].axhline(floor, ls="--", c="green", lw=1, label=f"retrain floor {floor:.2f}")
    ax[0].axhline(0.5, ls=":", c="gray", lw=1)
    ax[0].set_xlabel("suppression strength α  (0=naive · 1=full rollback · >1 over-suppress)")
    ax[0].set_ylabel("requesting user's inferability (AUC)")
    ax[0].set_title(f"(a) On-device threat: ours tunes privacy to the floor and below,\n"
                    f"at ~no cost to the user's own feed (NDCG≈{umean:.2f}, flat)")
    ax[0].legend(fontsize=8)

    # (b) SERVER / global-model threat: the fundamental frontier. At ~zero harm to others the best
    #     inferability is the FLOOR (the residue); going lower needs neighbor-deletion → others' loss.
    na = [d["auc"] for d in s["neighbor"]]; nl = [d["oth_loss"] for d in s["neighbor"]]
    ax[1].plot(nl, na, "-s", color="#c44e52", label="neighbor-deletion (SRU-style, global)")
    for d in s["neighbor"]:
        ax[1].annotate(f"{int(d['frac']*100)}%", (d["oth_loss"], d["auc"]), fontsize=7,
                       textcoords="offset points", xytext=(4, 4))
    ax[1].scatter([0.0], [naive], s=70, marker="o", color="#7f7f7f", zorder=5,
                  label=f"naive: X intact ({naive:.2f})")
    ax[1].scatter([0.0], [floor], s=140, marker="*", color="#2ca02c", zorder=6,
                  label=f"floor = FRU/FedShare/ours ({floor:.2f})")
    ax[1].axhline(floor, ls="--", c="green", lw=1)
    ax[1].axvspan(-0.02, 0.02, color="green", alpha=0.08)
    ax[1].set_xlabel("OTHER users' utility loss (NDCG drop for X)")
    ax[1].set_ylabel("requesting user's inferability (AUC)")
    ax[1].set_title("(b) Server threat: below the floor ONLY by harming others\n"
                    "(the collaborative-residue limit)")
    ax[1].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    out = ROOT / "results" / "figures" / f"frontier_{dataset}.png"
    fig.savefig(out, dpi=140); print(f"  fig -> {out}")


if __name__ == "__main__":
    main()
