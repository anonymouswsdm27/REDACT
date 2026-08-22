# Collaborative Residue in Federated Recommender Unlearning

Anonymous code and reproducibility artifact for the paper
*"Forgotten by One, Remembered by Many: The Collaborative Residue in Federated Recommendation."*

We show that in a collaborative (federated) recommender, an item a user asks to forget can remain
**inferable** even after a provably-correct unlearning, because the item's signal was reinforced by
**other** users and persists in the **shared** item embeddings. We (1) measure this *collaborative
residue* with a redundancy-stratified membership-inference probe, (2) characterize the redundancy
lower bound (empirical + closed form), (3) provide an on-device residue-suppression method, and
(4) release this verified-unlearning benchmark.

## Requirements

- Python ≥ 3.10, PyTorch (CPU is sufficient for the small subsamples; a GPU speeds up full runs).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Data

All datasets are public. Download and preprocess into `data/` (run once):

```bash
python data/scripts/download.py --dataset ml-1m --out data
# also: ml-100k, gowalla, lastfm, steam, yelp
```

## Reproduce

**1. Run a measurement sweep** (one config = one backbone × dataset). Each *unit* is one forgotten
`(user, item)` pair: it trains the from-scratch-retrain oracle, runs the inference probe, and records
every method's privacy / utility / collateral. Runs are checkpointed and idempotent (completed units
are skipped on resume).

```bash
# centralized matrix-factorization backbone
python -m experiments.run --config configs/mf_ml1m.yaml
# federated (FedNCF) backbone — adds FRU / FedShare / our method
python -m experiments.run --config configs/fedncf_ml1m.yaml
```

Configs for all six datasets are in `configs/` (`mf_*.yaml`, `fedncf_*.yaml`). Subsample quickly with
`--n-users 500 --per-stratum 20` for a smoke test.

**2. Ablations & analyses** (differential-privacy-style noise, client scale, partial participation,
training config, probe architecture, the limit fits, the frontier, sequential residue):

```bash
python -m experiments.ablation      --mode dp
python -m experiments.ablation      --mode scale
python -m experiments.ablation      --mode participation
python -m experiments.ablation_probe
python -m experiments.limit_fit     --sweep-id mf_ml1m mf_ml100k
python -m experiments.limit_theory
python -m experiments.frontier
python -m experiments.sequential
python -m experiments.verified
```

**3. Figures & tables.** We ship our own run outputs so the paper's figures and tables can be
regenerated **without** re-running the sweeps. The per-sweep measurement units are stored compressed
under `results/units/*.tar.gz`; unpack them once, then run the generators:

```bash
bash scripts/extract_results.sh            # unpack results/units/*.tar.gz -> results/units/<sweep>/
python scripts/make_paper_figures.py       # all figures -> results/figures/
python scripts/make_comparison_table.py    # main comparison table
python scripts/base_utility.py             # base-utility table
```

To reproduce the numbers *from scratch* instead, delete `results/` and run the sweeps in steps 1–2.

## Repository layout

```
src/
  data/         dataset loaders, per-user (federated) partitioning, leave-one-out splits
  models/       matrix-factorization / GMF and sequential (GRU4Rec) backbones
  federated/    FedAvg simulator (one client = one user), contribution logging
  unlearning/   retrain-from-scratch (oracle), naive delete, gradient-ascent, FRU, FedShare, ours
  probes/       membership-style inference probe + popularity-matched controls
  metrics/      probe AUC, residue-vs-redundancy, the limit fit, ranking metrics
  runtime/      checkpoint/resume + idempotent sweep registry (numerically-equivalent resume)
experiments/    thin entrypoints for each result (run, ablation, limit, frontier, sequential, verified)
scripts/        figure and table generators
configs/        one YAML per experiment (backbone × dataset)
tests/          unlearning correctness, probe sanity, checkpoint round-trip
```

## Tests

```bash
pytest            # unlearning correctness, probe sanity, checkpoint round-trip
```

## License

Released under the MIT License (see `LICENSE`).
