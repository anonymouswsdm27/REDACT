"""Base recommender utility per dataset (§12): leave-one-out HR@10/NDCG@10 (+@20) for the
centralized MF and federated GMF backbones, BEFORE any unlearning. This is the setup-table number
that shows the models are competitive — so the collaborative residue is not an artifact of a broken
recommender (the first thing a reviewer checks).

    python scripts/base_utility.py --datasets ml1m ml100k gowalla lastfm steam yelp
      -> results/base_utility.json  +  paper/tables/base_utility.tex
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.movielens import preprocess  # noqa: E402
from src.federated.simulator import train_fedgmf  # noqa: E402
from src.models.mf import evaluate_ranking, train_mf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# n_users per dataset, matching the confirmatory configs (None = full).
NU = {"ml1m": None, "ml100k": None, "gowalla": 5000, "lastfm": None, "steam": 5000, "yelp": 5000}
NAME = {"ml1m": "ML-1M", "ml100k": "ML-100K", "gowalla": "Gowalla", "lastfm": "LastFM",
        "steam": "Steam", "yelp": "Yelp"}
METRICS = ["hr10", "ndcg10", "hr20", "ndcg20"]


def _eval(model, ds) -> dict:
    e10 = evaluate_ranking(model, ds.test, ds.user_items, ds.n_items, k=10)
    e20 = evaluate_ranking(model, ds.test, ds.user_items, ds.n_items, k=20)
    return {"hr10": e10["HR@10"], "ndcg10": e10["NDCG@10"],
            "hr20": e20["HR@20"], "ndcg20": e20["NDCG@20"]}


def _ms(per_seed: list[dict], key: str) -> list[float]:
    """[mean, std] of a metric across seeds (each seed = one trained model)."""
    v = np.array([d[key] for d in per_seed])
    return [float(v.mean()), float(v.std())]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(NU))
    ap.add_argument("--n-users", type=int, default=0, help="override for ALL datasets (0 = per-config)")
    ap.add_argument("--seeds", default="0,1,2,3,4")   # unit of replication = training seed (mean±std)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    out = {}
    for key in args.datasets:
        nu = args.n_users or NU.get(key)
        ds = preprocess(n_users=nu, seed=0, dataset=key)   # data split fixed; model seed varies
        mf_runs, fed_runs = [], []
        for s in seeds:
            mf_runs.append(_eval(train_mf(ds.train, ds.n_users, ds.n_items, dim=32, epochs=30,
                                          seed=s), ds))
            fed_runs.append(_eval(train_fedgmf(ds.train, ds.n_users, ds.n_items, dim=32, rounds=20,
                                               local_epochs=3, lr=0.2, seed=s, log_contribs=False), ds))
        out[key] = {"n_users": ds.n_users, "n_items": ds.n_items, "n_inter": int(ds.train.shape[0]),
                    "seeds": seeds,
                    "mf": {m: _ms(mf_runs, m) for m in METRICS},
                    "fed": {m: _ms(fed_runs, m) for m in METRICS}}
        m, f = out[key]["mf"], out[key]["fed"]
        print(f"  {NAME.get(key, key):9} users={ds.n_users:>5} items={ds.n_items:>6} | "
              f"MF HR@10={m['hr10'][0]:.3f}±{m['hr10'][1]:.3f} NDCG@10={m['ndcg10'][0]:.3f} | "
              f"FedGMF HR@10={f['hr10'][0]:.3f}±{f['hr10'][1]:.3f} NDCG@10={f['ndcg10'][0]:.3f}")
    (ROOT / "results" / "base_utility.json").write_text(json.dumps(out, indent=2))

    (ROOT / "paper" / "tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "paper" / "tables" / "base_utility.tex").write_text(_latex(out, args.datasets, len(seeds)))
    print("-> results/base_utility.json + paper/tables/base_utility.tex")


def _latex(out: dict, datasets: list, n_seeds: int) -> str:
    """Full-width booktabs table: HR@10/@20 + NDCG@10/@20 (mean$\\pm$std over seeds) for both
    backbones. Dataset-size columns dropped to keep the focus on utility."""
    def c(ms):
        return f"{ms[0]:.3f}{{\\scriptsize$\\pm${ms[1]:.3f}}}"
    L = [r"\begin{table*}[t]", r"\centering", r"\small",
         r"\caption{Base recommender utility (leave-one-out HR@$K$ / NDCG@$K$, "
         rf"mean$\pm$std over {n_seeds} seeds) before any unlearning, for the centralized MF and "
         r"federated GMF backbones. The models are competitive on ML-1M/ML-100K/Gowalla/Steam; on the "
         r"sparser LastFM (no timestamps $\Rightarrow$ order-based split) and Yelp accuracy is lower, "
         r"yet the residue still appears strongly (Fig.~\ref{fig:problem})---confirming it is driven "
         r"by cross-user redundancy, not ranking accuracy.}", r"\label{tab:base_utility}",
         r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l cccc cccc}", r"\toprule",
         r"& \multicolumn{4}{c}{MF (centralized)} & \multicolumn{4}{c}{GMF (federated)} \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         r"Dataset & HR@10 & NDCG@10 & HR@20 & NDCG@20 & HR@10 & NDCG@10 & HR@20 & NDCG@20 \\",
         r"\midrule"]
    for key in datasets:
        mf = out[key]["mf"]; fd = out[key]["fed"]
        L.append(f"{NAME.get(key, key)} & "
                 f"{c(mf['hr10'])} & {c(mf['ndcg10'])} & {c(mf['hr20'])} & {c(mf['ndcg20'])} & "
                 f"{c(fd['hr10'])} & {c(fd['ndcg10'])} & {c(fd['hr20'])} & {c(fd['ndcg20'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
