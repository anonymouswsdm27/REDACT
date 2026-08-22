"""Robustness ablations (§6-P5). Two reviewer-critical knobs, one shared residue meter:

  --mode dp     F4: does DP / secure-aggregation noise remove the residue? (It should NOT — the
                noise perturbs the aggregate but not any single user's *contribution* to it, and the
                residue comes from OTHER users.) Sweeps dp_sigma.
  --mode scale  F3: how does the residue dilute with the number of clients? (One client = one user;
                each user's global footprint -> 0 as #clients grows.) Sweeps n_users.

Both report, after a verified (retrain) unlearning: high-redundancy probe AUC (the residue), the
naive-delete leak above floor, and ours' leak — so we can see the residue persist (dp) or dilute
(scale) while ours stays at the floor throughout.

  python -m experiments.ablation --mode dp    [--n-users 500 --pairs 60 --seeds 0,1]
  python -m experiments.ablation --mode scale  [--pairs 60 --seeds 0,1]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.data.movielens import preprocess
from src.federated.simulator import train_fedgmf
from src.models.mf import MFModel, evaluate_ranking, refit_user
from src.probes.membership import ControlSampler, build_targets, probe_auc
from src.unlearning.methods import ours_unlearn

ROOT = Path(__file__).resolve().parents[1]


def _hi_pairs(ds, min_r: int, n: int) -> list:
    """High-redundancy (r>=min_r) target pairs, capped at n — where the residue is worth measuring."""
    return [t for t in build_targets(ds.user_items, ds.item_pop, 40, 0) if t.r >= min_r][:n]


def _agg_seeds(acc: dict) -> dict:
    """Aggregate a metric->per-seed-list dict into mean AND std across seeds (unit of replication =
    seed). Keeps `<metric>` (mean) and `<metric>_std` so figures can draw error bands."""
    out = {}
    for k, vals in acc.items():
        v = np.array(vals, float); v = v[~np.isnan(v)]
        out[k] = float(v.mean()) if len(v) else float("nan")
        out[f"{k}_std"] = float(v.std()) if len(v) > 1 else 0.0
    return out


def measure_residue(ds, make_base, pairs, dim: int, seed: int) -> dict:
    """Train base + per-pair retrain oracles; return the redundant-item floor AUC (the residue),
    naive/ours leak vs floor, and the base model's held-out utility (to read the residue against
    whether the recommender is still useful)."""
    cs = ControlSampler(ds.item_pop, ds.user_items)
    base = make_base(ds.train, seed)
    util = evaluate_ranking(base, ds.test, ds.user_items, ds.n_items, k=10)["NDCG@10"]
    res_floor, naive_leak, ours_leak = [], [], []
    for t in pairs:
        u, x = t.user, t.item
        ps = int(np.random.SeedSequence([seed, u, x]).generate_state(1)[0])
        ctrl = cs.sample(u, x, 50, ps)
        if len(ctrl) < 3:
            continue
        hist_minus = ds.user_items[u][ds.user_items[u] != x]
        keys = ds.train[:, 0].astype(np.int64) * ds.n_items + ds.train[:, 1]
        retrain = make_base(ds.train[keys != (u * ds.n_items + x)], seed)

        def _refit(q, b):
            m = MFModel(p=base.p.copy(), q=q, b=b, dim=base.dim)
            m.p[u] = refit_user(q, b, hist_minus, ds.n_items, dim=dim, seed=ps)  # noqa: B023
            return m
        f = probe_auc(_refit(retrain.q, retrain.b), u, x, ctrl)      # floor (oracle) = the residue
        nv = probe_auc(_refit(base.q, base.b), u, x, ctrl)           # naive: X intact
        ours = ours_unlearn(base, u, x, hist_minus, ds.n_items, dim=dim, seed=ps)
        ou = probe_auc(ours, u, x, ctrl)
        res_floor.append(f); naive_leak.append(nv - f); ours_leak.append(ou - f)
    return dict(residue_floor=float(np.mean(res_floor)) if res_floor else float("nan"),
                naive_leak=float(np.mean(naive_leak)), ours_leak=float(np.mean(ours_leak)),
                utility=float(util), n=len(res_floor))


def main() -> None:
    (ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dp", "scale", "participation", "fedconfig"], required=True)
    ap.add_argument("--dataset", default="ml1m")
    ap.add_argument("--n-users", type=int, default=500)
    ap.add_argument("--pairs", type=int, default=60)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--min-r", type=int, default=6)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    dim = 32

    rows = []
    if args.mode == "dp":
        # fine low-σ grid: σ·lr·√rounds is the accumulated noise, so σ≳0.5 already destroys the model.
        # The honest question is whether the residue survives at UTILITY-PRESERVING DP (small σ).
        grid = [0.0, 0.02, 0.05, 0.1, 0.2]
        ds = preprocess(n_users=args.n_users, seed=0, dataset=args.dataset)
        pairs = _hi_pairs(ds, args.min_r, args.pairs)
        for sig in grid:
            acc = defaultdict(list)
            for seed in seeds:
                def mk(train, s, _sig=sig):
                    return train_fedgmf(train, ds.n_users, ds.n_items, dim=dim, rounds=15,
                                        local_epochs=3, lr=0.2, seed=s, dp_sigma=_sig, log_contribs=True)
                r = measure_residue(ds, mk, pairs, dim, seed)
                for k, v in r.items():
                    acc[k].append(v)
            rows.append(dict(knob=sig, **_agg_seeds(acc)))
            print(f"  dp_sigma={sig:<4} residue floor AUC={rows[-1]['residue_floor']:.3f} "
                  f"naive_leak={rows[-1]['naive_leak']:+.3f} ours_leak={rows[-1]['ours_leak']:+.3f} "
                  f"utility={rows[-1]['utility']:.3f}")
    elif args.mode == "participation":
        # partial client participation: fraction of clients selected EACH round (realistic non-IID
        # FL, F1). More rounds to compensate for slower convergence at low participation.
        grid = [1.0, 0.5, 0.25, 0.1]
        ds = preprocess(n_users=args.n_users, seed=0, dataset=args.dataset)
        pairs = _hi_pairs(ds, args.min_r, args.pairs)
        for frac in grid:
            acc = defaultdict(list)
            for seed in seeds:
                def mk(train, s, _frac=frac):
                    return train_fedgmf(train, ds.n_users, ds.n_items, dim=dim, rounds=30,
                                        local_epochs=3, lr=0.2, seed=s, participation=_frac,
                                        log_contribs=True)
                r = measure_residue(ds, mk, pairs, dim, seed)
                for k, v in r.items():
                    acc[k].append(v)
            rows.append(dict(knob=frac, **_agg_seeds(acc)))
            print(f"  participation={frac:<4} residue floor AUC={rows[-1]['residue_floor']:.3f} "
                  f"naive_leak={rows[-1]['naive_leak']:+.3f} ours_leak={rows[-1]['ours_leak']:+.3f} "
                  f"utility={rows[-1]['utility']:.3f}")
    elif args.mode == "fedconfig":
        # FedAvg training-configuration robustness: sweep communication rounds and local epochs E
        # (the two defining FedAvg knobs) to show the residue is not a training-setup artefact.
        # Each config is (label, rounds, local_epochs); R15/E3 is the default reference.
        grid = [("R5", 5, 3), ("R15", 15, 3), ("R30", 30, 3), ("E1", 15, 1), ("E5", 15, 5)]
        ds = preprocess(n_users=args.n_users, seed=0, dataset=args.dataset)
        pairs = _hi_pairs(ds, args.min_r, args.pairs)
        for label, rnds, le in grid:
            acc = defaultdict(list)
            for seed in seeds:
                def mk(train, s, _r=rnds, _e=le):
                    return train_fedgmf(train, ds.n_users, ds.n_items, dim=dim, rounds=_r,
                                        local_epochs=_e, lr=0.2, seed=s, log_contribs=True)
                r = measure_residue(ds, mk, pairs, dim, seed)
                for k, v in r.items():
                    acc[k].append(v)
            rows.append(dict(knob=label, rounds=rnds, local_epochs=le, **_agg_seeds(acc)))
            print(f"  {label:<4} (R={rnds},E={le}) residue floor AUC={rows[-1]['residue_floor']:.3f} "
                  f"naive_leak={rows[-1]['naive_leak']:+.3f} ours_leak={rows[-1]['ours_leak']:+.3f} "
                  f"utility={rows[-1]['utility']:.3f}")
    else:
        grid = [50, 200, 800, 3000]
        for nu in grid:
            ds = preprocess(n_users=nu, seed=0, dataset=args.dataset)
            pairs = _hi_pairs(ds, args.min_r, args.pairs)
            acc = defaultdict(list)
            for seed in seeds:
                def mk(train, s):
                    return train_fedgmf(train, ds.n_users, ds.n_items, dim=dim, rounds=15,  # noqa: B023
                                        local_epochs=3, lr=0.2, seed=s, log_contribs=True)
                r = measure_residue(ds, mk, pairs, dim, seed)
                for k, v in r.items():
                    acc[k].append(v)
            rows.append(dict(knob=nu, **_agg_seeds(acc)))
            print(f"  n_users={nu:<5} residue floor AUC={rows[-1]['residue_floor']:.3f} "
                  f"naive_leak={rows[-1]['naive_leak']:+.3f} ours_leak={rows[-1]['ours_leak']:+.3f} "
                  f"utility={rows[-1]['utility']:.3f}")

    out = dict(mode=args.mode, dataset=args.dataset, seeds=seeds, rows=rows)
    # participation and fedconfig are run per-dataset (multi-domain robustness) -> keep a per-dataset
    # file so looped datasets don't overwrite each other; dp/scale stay single-file (ML-1M).
    per_dataset = args.mode in ("participation", "fedconfig")
    suffix = f"_{args.dataset}" if per_dataset else ""
    (ROOT / "results" / f"ablation_{args.mode}{suffix}.json").write_text(json.dumps(out, indent=2))
    if not per_dataset:                         # per-dataset modes are reported as tables, not figures
        _figure(args.mode, rows)


def _figure(mode: str, rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = list(range(len(rows)))
    labs = [str(r["knob"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.axhline(0.5, ls=":", c="gray", lw=1, label="chance")
    ax.plot(x, [r["residue_floor"] for r in rows], "-o", color="#d62728",
            label="residue (floor AUC after verified unlearning)")
    if mode == "dp":
        ax.plot(x, [r["utility"] for r in rows], "-^", color="#2ca02c", alpha=0.8,
                label="recommender utility (NDCG@10)")
        xlabel = "DP-FedAvg noise σ (F4)"
        title = ("(F4) Residue PERSISTS under DP / secure-agg noise\n"
                 "(the noise hides no single user's contribution)")
    elif mode == "participation":
        ax.plot(x, [r["utility"] for r in rows], "-^", color="#2ca02c", alpha=0.8,
                label="recommender utility (NDCG@10)")
        xlabel = "client participation per round (F1)"
        title = ("(F1) Residue under partial client participation\n"
                 "(realistic non-IID FL; each round samples a client subset)")
    else:
        # naive AUC = floor + its excess; it converges DOWN to the (flat) floor as #clients grows.
        ax.plot(x, [r["residue_floor"] + r["naive_leak"] for r in rows], "-s", color="#7f7f7f",
                alpha=0.8, label="naive local-delete AUC (X intact)")
        xlabel = "# clients / users (F3)"
        title = ("(F3) The residue is SCALE-ROBUST for redundant items;\n"
                 "only naive-delete's excess over the floor dilutes with #clients")
    ax.set_xticks(x); ax.set_xticklabels(labs)
    ax.set_xlabel(xlabel); ax.set_ylabel("AUC / NDCG")
    ax.set_title(title); ax.set_ylim(0.0, 1.0); ax.legend(fontsize=8, loc="center left")
    fig.tight_layout()
    out = ROOT / "results" / "figures" / f"ablation_{mode}.png"
    fig.savefig(out, dpi=140); print(f"  fig -> {out}")


if __name__ == "__main__":
    main()
