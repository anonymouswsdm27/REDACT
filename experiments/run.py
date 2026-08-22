"""Resume-aware, walltime-aware, idempotent sweep entrypoint (§9).

Runs a shard of the (backbone × dataset × seed × forgotten-pair) sweep that produces the residue
measurement. Each **unit** computes, for one forgotten (A,X): the retrain **floor** (the
budget-driving from-scratch retrain — cached, never recomputed, §9 r4), the **naive-delete**
leak, the no-unlearn reference, and a placebo. Per-unit results are written atomically so a
killed/resumed job skips finished units and is numerically identical to an uninterrupted run.

Matches the §9 the scheduler contract:
    python -m experiments.run --run-id $RUN_ID --shard $i/$N \
        --ckpt-dir <SCRATCH>/... --keep-dir <DATA>/... --status runs/$RUN_ID/status.json \
        --max-runtime 46h --resume auto
Exit 0 = shard done; exit 64 = out-of-time, work remains -> wrapper resubmits a successor.

Backbones: `mf` is wired (validated). `fedncf` / sequential are P1.1+ — register a trainer in
BACKBONES; the whole sweep machinery is backbone-agnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from src.data.movielens import SOURCES, preprocess, summarize
from src.metrics.ranking import holdout_ndcg
from src.metrics.residue import aggregate
from src.models.mf import MFModel, evaluate_ranking, refit_user, train_mf
from src.probes.membership import ControlSampler, build_targets, probe_auc
from src.runtime import OUT_OF_TIME, RetrainCache, SweepRegistry, WalltimeBudget
from src.unlearning.methods import (
    fedshare_unlearn,
    finetune_unlearn,
    fru_unlearn,
    gradient_ascent_unlearn,
    ours_unlearn,
)

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CFG = dict(
    sweep_id="phase1_mf_ml1m", backbone="mf", dataset="ml1m",
    n_users=2000, data_seed=0, dim=32, epochs=30, ft_epochs=5,
    fed_rounds=20, fed_local_epochs=3, fed_lr=0.2,   # federated GMF (backbone=fedncf)
    seeds=[0], per_stratum=120, n_controls=50,
    # retrain = oracle/ground-truth & floor; the rest are practical baselines compared to it.
    # (fedncf adds FRU + ours, which need the federated contribution logs.)
    methods=["retrain", "naive_delete", "gradient_ascent", "finetune"],
)


# ---------- backbone registry ----------
def _mf_trainer(train, n_users, n_items, cfg, seed):
    return train_mf(train, n_users, n_items, dim=cfg["dim"], epochs=cfg["epochs"], seed=seed)


def _fedncf_trainer(train, n_users, n_items, cfg, seed):
    """Federated GMF (the GMF instantiation of FedNCF) via FedAvg, with contribution logs."""
    from src.federated.simulator import train_fedgmf
    return train_fedgmf(train, n_users, n_items, dim=cfg["dim"], rounds=cfg.get("fed_rounds", 20),
                        local_epochs=cfg.get("fed_local_epochs", 3), lr=cfg.get("fed_lr", 0.2),
                        seed=seed, log_contribs=True)


BACKBONES = {"mf": _mf_trainer, "fedncf": _fedncf_trainer}


# ---------- helpers ----------
def _pair_seed(seed, user, item):
    return int(np.random.SeedSequence([int(seed), int(user), int(item)]).generate_state(1)[0])


def _remove_pair(train, user, item, n_items):
    keys = train[:, 0].astype(np.int64) * n_items + train[:, 1]
    return train[keys != (int(user) * n_items + int(item))]


def _config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps({k: cfg[k] for k in sorted(cfg)}, default=str).encode()).hexdigest()[:12]


def _src_hash() -> str:
    h = hashlib.sha256()
    for f in sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "experiments").glob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def build_dataset(cfg):
    nu = None if cfg["n_users"] in (0, None) else cfg["n_users"]
    if cfg["dataset"] in SOURCES:
        return preprocess(n_users=nu, seed=cfg["data_seed"], dataset=cfg["dataset"])
    raise NotImplementedError(f"dataset {cfg['dataset']} not in SOURCES {list(SOURCES)}")


def build_units(cfg, ds) -> list[dict]:
    targets = build_targets(ds.user_items, ds.item_pop, per_stratum=cfg["per_stratum"],
                            seed=cfg["data_seed"])
    units = []
    for seed in cfg["seeds"]:
        for t in targets:
            uid = f"{cfg['backbone']}-{cfg['dataset']}-s{seed}-u{t.user}-i{t.item}"
            units.append(dict(unit_id=uid, seed=int(seed), user=int(t.user), item=int(t.item),
                              r=int(t.r), bin=int(t.bin)))
    return units


def shard_of(units: list[dict], k: int, n: int) -> list[dict]:
    return [u for i, u in enumerate(units) if i % n == (k - 1)]  # strided, balanced across strata


def _cached_train(cache, cfg, ds, seed, removed, mem):  # base (removed=[]) or retrain (removed=[pair])
    key = cache.key(cfg["backbone"], cfg["dataset"], seed, removed, cfg["config_hash"])
    if key in mem:
        return mem[key]
    m = cache.get(key)
    if m is None:
        train = ds.train
        for (u, i) in removed:
            train = _remove_pair(train, u, i, ds.n_items)
        m = BACKBONES[cfg["backbone"]](train, ds.n_users, ds.n_items, cfg, seed)
        # Only the BASE model (removed=[]) needs the per-user contribution logs (FRU/ours read
        # base.meta). Strip them from the ~3000 cached retrains — the logs are hundreds of MB
        # each on full data and would otherwise dominate scratch disk.
        if removed and m.meta is not None:
            m.meta = None
        cache.put(key, m)
    mem[key] = m
    return m


def compute_unit(cfg, ds, cs, unit, cache, mem, item_users) -> dict | None:
    """One forgotten (A,X): floor (oracle) + every applicable method on THREE axes —
      auc_*  : requesting-user privacy  (lower = better unlearning)
      util_* : the requesting user's own held-out utility
      oth_*  : OTHER users' utility for X under the method's GLOBAL deployment (higher = better)
    For fairness, every "rollback + local re-fit" method (floor, naive, FRU, ours) scores with a
    locally re-fit p_A against the model it deploys — so the privacy comparison isolates the
    SHARED-embedding change, not the p_A training procedure."""
    u, x, seed = unit["user"], unit["item"], unit["seed"]
    ps = _pair_seed(seed, u, x)
    ctrl = cs.sample(u, x, cfg["n_controls"], ps)
    if len(ctrl) < 3:
        return None
    base = _cached_train(cache, cfg, ds, seed, [], mem)                 # reference / substrate
    retrain = _cached_train(cache, cfg, ds, seed, [(u, x)], mem)        # FLOOR/ORACLE (cached)
    hist_minus = ds.user_items[u][ds.user_items[u] != x]
    t = int(ds.test[u, 1]); ex = cs.hist[u]

    def refit_model(q, b):                      # deploy shared (q,b); user re-fits p_A locally
        m = MFModel(p=base.p.copy(), q=q, b=b, dim=base.dim)
        m.p[u] = refit_user(q, b, hist_minus, ds.n_items, dim=cfg["dim"], seed=ps)
        return m

    def util(m):
        return holdout_ndcg(m, u, t, ex, ds.n_items, seed=ps)

    def collat(qx):
        return float(np.linalg.norm(qx - base.q[x]))

    others = [o for o in item_users[x] if o != u][:8]

    def oth(qx, bx):                            # other users' NDCG for X given X's deployed embedding
        if not others:
            return float("nan")
        r2 = np.random.default_rng(ps ^ 0x5151); vals = []
        for o in others:
            negs = []
            while len(negs) < 50:
                c = int(r2.integers(0, ds.n_items))
                if c != x and c not in cs.hist[o]:
                    negs.append(c)
            negs = np.array(negs)
            s_x = float(qx @ base.p[o] + bx)        # X scored for other user o via X's deployed embedding
            s_neg = base.q[negs] @ base.p[o] + base.b[negs]
            vals.append(1.0 / np.log2(int((s_neg > s_x).sum()) + 2))
        return float(np.mean(vals))

    oth_base = oth(base.q[x], base.b[x])        # X intact (ours / naive / no-unlearn deploy this)
    floor_m = refit_model(retrain.q, retrain.b)
    res = dict(unit_id=unit["unit_id"], seed=seed, user=u, item=x, r=unit["r"], bin=unit["bin"],
               auc_floor=probe_auc(floor_m, u, x, ctrl), auc_full=probe_auc(base, u, x, ctrl),
               util_floor=util(floor_m), util_full=util(base),
               collateral_floor=collat(retrain.q[x]), oth_floor=oth(retrain.q[x], retrain.b[x]),
               oth_full=oth_base)

    # naive local-delete: deploy base globally (X intact), user re-fits p_A
    naive_m = refit_model(base.q, base.b)
    res.update(auc_naive=probe_auc(naive_m, u, x, ctrl), util_naive=util(naive_m),
               collateral_naive=0.0, oth_naive=oth_base)

    # gradient-ascent (modifies the shared q_X)
    ga = gradient_ascent_unlearn(base, u, x, ds.n_items, seed=ps)
    res.update(auc_gradasc=probe_auc(ga, u, x, ctrl), util_gradasc=util(ga),
               collateral_gradasc=collat(ga.q[x]), oth_gradasc=oth(ga.q[x], ga.b[x]))

    # placebo (random-user control)
    rng = np.random.default_rng(ps ^ 0x9E3779B9); pa = -1
    for _ in range(30):
        a = int(rng.integers(0, ds.n_users))
        if x not in cs.hist[a] and len(cs.hist[a]) > 0:
            pa = a; break
    auc_placebo = float("nan")
    if pa >= 0:
        pc = cs.sample(pa, x, cfg["n_controls"], ps)
        if len(pc) >= 3:
            auc_placebo = probe_auc(base, pa, x, pc)
    res["auc_placebo"] = auc_placebo

    # fine-tune (centralized warm-start baseline) — only for the mf backbone
    if cfg["backbone"] == "mf":
        ft_key = cache.key("mf:ft", cfg["dataset"], seed, [(u, x)], cfg["config_hash"])
        ft = mem.get(ft_key) or cache.get(ft_key)
        if ft is None:
            ft = finetune_unlearn(base, _remove_pair(ds.train, u, x, ds.n_items),
                                  ds.n_users, ds.n_items, dim=cfg["dim"],
                                  epochs=cfg.get("ft_epochs", 5), seed=seed)
            cache.put(ft_key, ft)
        mem[ft_key] = ft
        res.update(auc_finetune=probe_auc(ft, u, x, ctrl), util_finetune=util(ft),
                   collateral_finetune=collat(ft.q[x]), oth_finetune=oth(ft.q[x], ft.b[x]))

    # FRU + FedShare + ours — need the federated contribution logs (fedncf backbone)
    if base.meta and base.meta.get("contribs_q") is not None:
        fru = fru_unlearn(base, u, x, hist_minus, ds.n_items, dim=cfg["dim"], seed=ps)
        ours = ours_unlearn(base, u, x, hist_minus, ds.n_items, dim=cfg["dim"], seed=ps)
        fs = fedshare_unlearn(base, u, x, hist_minus, ds.n_items, dim=cfg["dim"], seed=ps)
        res.update(
            auc_fru=probe_auc(fru, u, x, ctrl), util_fru=util(fru),
            collateral_fru=collat(fru.q[x]), oth_fru=oth(fru.q[x], fru.b[x]),  # global rollback hurts others
            auc_fedshare=probe_auc(fs, u, x, ctrl), util_fedshare=util(fs),    # closest-neighbor foil (§3)
            collateral_fedshare=collat(fs.q[x]), oth_fedshare=oth(fs.q[x], fs.b[x]),
            auc_ours=probe_auc(ours, u, x, ctrl), util_ours=util(ours),
            collateral_ours=0.0, oth_ours=oth_base)                              # local-only -> others intact
    return res


# ---------- modes ----------
def run_mode(cfg, args) -> int:
    t0 = time.time()
    results_dir = ROOT / "results"
    runs_dir = ROOT / "runs"
    keep = Path(args.keep_dir) if args.keep_dir else ROOT
    cache = RetrainCache(keep / "cache" / "retrain")
    reg = SweepRegistry(runs_dir, results_dir, cfg["sweep_id"])

    ds = build_dataset(cfg)
    cs = ControlSampler(ds.item_pop, ds.user_items)
    item_users = [[] for _ in range(ds.n_items)]        # for the OTHER-users' utility metric
    for uu, ii in ds.train:
        item_users[ii].append(int(uu))
    all_units = build_units(cfg, ds)
    k, n = (int(v) for v in args.shard.split("/"))
    units = shard_of(all_units, k, n)
    ids = [u["unit_id"] for u in units]

    reg.heartbeat(args.run_id, phase="P1", state="running", shard=args.shard,
                  total=len(units), done=len(units) - len(reg.pending(ids)),
                  config_hash=cfg["config_hash"], src_hash=_src_hash())
    print(f"[run] {summarize(ds)}")
    print(f"[run] sweep={cfg['sweep_id']} shard {k}/{n}: {len(units)} units "
          f"({len(reg.pending(ids))} pending) | cache={cache.dir}")

    wt = WalltimeBudget(args.max_runtime)
    mem: dict = {}
    # report the base recommender's ranking quality (standard recsys metrics, §12)
    base0 = _cached_train(cache, cfg, ds, cfg["seeds"][0], [], mem)
    u10 = evaluate_ranking(base0, ds.test, ds.user_items, ds.n_items, k=10)
    u20 = evaluate_ranking(base0, ds.test, ds.user_items, ds.n_items, k=20)
    util_str = (f"HR@10={u10['HR@10']:.3f} NDCG@10={u10['NDCG@10']:.3f} "
                f"HR@20={u20['HR@20']:.3f} NDCG@20={u20['NDCG@20']:.3f}")
    print(f"[util] base ({cfg['backbone']}) {util_str}")
    reg.heartbeat(args.run_id, phase="P1", state="running", shard=args.shard, base_utility=util_str)
    processed = 0
    for unit in units:
        if reg.is_done(unit["unit_id"]):
            continue
        if wt.should_stop() or (args.max_units and processed >= args.max_units):
            break
        reg.log_event(args.run_id, unit["unit_id"], "running")
        res = compute_unit(cfg, ds, cs, unit, cache, mem, item_users)
        if res is not None:
            reg.record_result(unit["unit_id"], res, run_id=args.run_id)
        processed += 1
        if processed % 10 == 0 or processed == 1:
            done = len(units) - len(reg.pending(ids))
            reg.heartbeat(args.run_id, phase="P1", state="running", shard=args.shard,
                          total=len(units), done=done, walltime_s=round(wt.elapsed(), 1),
                          reason=wt.reason())
            print(f"  [{wt.elapsed():.0f}s] processed {processed} | done {done}/{len(units)}")

    pending = reg.pending(ids)
    remaining = len(pending) > 0
    code = OUT_OF_TIME if (remaining and wt.should_stop()) else (OUT_OF_TIME if remaining else 0)
    state = "done" if not remaining else "out_of_time"
    reg.heartbeat(args.run_id, phase="P1", state=state, shard=args.shard, total=len(units),
                  done=len(units) - len(pending), walltime_s=round(wt.elapsed(), 1),
                  exit_code=code, reason=wt.reason())
    print(f"[run] shard {k}/{n} {state}: {len(units) - len(pending)}/{len(units)} units "
          f"in {time.time() - t0:.0f}s -> exit {code} ({wt.reason()})")
    return code


def aggregate_mode(cfg, args) -> int:
    reg = SweepRegistry(ROOT / "runs", ROOT / "results", cfg["sweep_id"])
    rows = reg.load_results()
    if not rows:
        print("[aggregate] no unit results yet"); return 1
    agg = aggregate(rows)
    out = ROOT / "results" / f"agg_{cfg['sweep_id']}.json"
    out.write_text(json.dumps(agg, indent=2))
    leaks = "  ".join(f"{m}={agg.get('mean_leak_' + m, float('nan')):+.3f}"
                      for m in agg.get("practical", []))
    print(f"[aggregate] {agg['n']} units | Spearman(logr,floor)={agg['spearman_logr_floor']:.3f} "
          f"| high-r floor={agg['high_redundancy_floor']:.3f}")
    print(f"            mean leak vs floor (ground truth):  {leaks}")
    methods = agg.get("methods", [])
    print(f"{'redundancy':>11} | {'n':>4} | " + " | ".join(f"{m:>8}" for m in methods))
    for b in agg["per_bin"]:
        if b.get("n", 0) == 0:
            continue
        cells = " | ".join(f"{b.get(m, float('nan')):>8.3f}" for m in methods)
        print(f"{b['bin']:>11} | {b['n']:>4} | {cells}")
    um = agg.get("util_methods", [])
    if um:
        util = "  ".join(f"{m}={agg.get('mean_util_' + m, float('nan')):.3f}" for m in um)
        print(f"utility (user's held-out NDCG, higher=better):  {util}")
    cm = agg.get("collateral_methods", [])
    if cm:
        coll = "  ".join(f"{m}={agg.get('mean_collateral_' + m, float('nan')):.3f}" for m in cm)
        print(f"collateral to OTHER users (||delta global q_X||, lower=better):  {coll}")
    om = agg.get("oth_methods", [])
    if om:
        oth = "  ".join(f"{m}={agg.get('mean_oth_' + m, float('nan')):.3f}" for m in om)
        print(f"OTHER users' utility for X (NDCG, higher=better):  {oth}")
    for fn in (_figure, _problem_figure, _pareto_figure):
        try:
            fn(agg, cfg["sweep_id"])
        except Exception as e:
            print(f"[{fn.__name__}] skipped:", e)
    return 0


def _pareto_figure(agg, sweep_id):
    """Two-axis 'ours advantage' plot: requesting-user privacy (x, lower=better) vs OTHER users'
    utility (y, higher=better). The method nearest the top-left wins both."""
    om = agg.get("oth_methods", [])
    if not om:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts = [(m, agg.get(f"mean_auc_{m}", np.nan), agg.get(f"mean_oth_{m}", np.nan))
           for m in agg["methods"] if m in om and m != "placebo"]
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for m, xj, yj in pts:
        col = {"ours": "tab:green", "floor": "tab:blue", "fru": "tab:red",
               "gradasc": "tab:orange", "naive": "tab:purple"}.get(m, "gray")
        big = m == "ours"
        ax.scatter([xj], [yj], s=200 if big else 90, c=col, edgecolor="k",
                   marker="*" if big else "o", zorder=3)
        ax.annotate(m, (xj, yj), textcoords="offset points", xytext=(7, 5),
                    fontweight="bold" if big else "normal")
    ax.set_xlabel("requesting-user inferability  (probe AUC, lower = more private)")
    ax.set_ylabel("OTHER users' utility for X  (NDCG, higher = better)")
    ax.set_title(f"{sweep_id}: privacy vs others' utility — ours wants top-left")
    ax.grid(alpha=.3); fig.tight_layout()
    (ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "results" / "figures" / f"pareto_{sweep_id}.png", dpi=130)
    print("[pareto] wrote", ROOT / "results" / "figures" / f"pareto_{sweep_id}.png")


def _bins_x(agg):
    pb = [b for b in agg["per_bin"] if b.get("n", 0) > 0]
    return pb, np.arange(len(pb))


def _col(pb, name):
    return np.array([b.get(name, np.nan) for b in pb], dtype=float)


def _figure(agg, sweep_id):
    """All methods vs the retrain floor, avg±std bands across seeds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pb, x = _bins_x(agg)
    if not len(pb):
        return

    def band(ax, name, color, **kw):
        y, s = _col(pb, name), np.nan_to_num(_col(pb, f"{name}_std"))
        ax.plot(x, y, color=color, **kw)
        ax.fill_between(x, y - s, y + s, color=color, alpha=0.15)

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    if "full" in agg["methods"]:
        ax.plot(x, _col(pb, "full"), "^:", color="tab:green", alpha=.6, label="no unlearning")
    marks = ["D-", "v-", "P-", "X-", "*-"]
    colors = ["tab:red", "tab:orange", "tab:purple", "tab:brown", "tab:pink"]
    for i, m in enumerate(agg.get("practical", [])):
        band(ax, m, colors[i % len(colors)], marker=marks[i % len(marks)][0], ls="-", lw=2, label=m)
    band(ax, "floor", "tab:blue", marker="o", ls="-", lw=2.4, label="retrain floor (oracle)")
    if "placebo" in agg["methods"]:
        ax.plot(x, _col(pb, "placebo"), "s--", color="gray", alpha=.7, label="placebo")
    ax.axhline(0.5, color="k", lw=1, alpha=.5)
    ax.set_xticks(x); ax.set_xticklabels([b["bin"] for b in pb], rotation=45)
    ax.set_xlabel("cross-user redundancy"); ax.set_ylabel("inference-probe AUC")
    ns = agg.get("n_seeds", 1)
    ax.set_title(f"{sweep_id}: methods vs retrain floor  (Spearman="
                 f"{agg['spearman_logr_floor']:.2f}±{agg.get('spearman_logr_floor_std', 0):.2f}, "
                 f"{ns} seed{'s' if ns != 1 else ''})")
    ax.legend(fontsize=8); ax.grid(alpha=.3); fig.tight_layout()
    (ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "results" / "figures" / f"agg_{sweep_id}.png", dpi=130)
    print("[figure] wrote", ROOT / "results" / "figures" / f"agg_{sweep_id}.png")


def _problem_figure(agg, sweep_id):
    """THE PROBLEM in one plot: after a provably-correct unlearning (retrain floor), the forgotten
    item stays inferable — above chance and rising with cross-user redundancy — i.e. the right to
    be forgotten leaks through collaboration. avg±std across seeds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pb, x = _bins_x(agg)
    if not len(pb):
        return
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    if "full" in agg["methods"]:
        ax.plot(x, _col(pb, "full"), "^:", color="tab:green", alpha=.55, label="before unlearning")
    y, s = _col(pb, "floor"), np.nan_to_num(_col(pb, "floor_std"))
    ax.plot(x, y, "o-", color="tab:blue", lw=2.6, label="after CORRECT unlearning (residue)")
    ax.fill_between(x, y - s, y + s, color="tab:blue", alpha=0.18)
    if "placebo" in agg["methods"]:
        ax.plot(x, _col(pb, "placebo"), "s--", color="gray", alpha=.7, label="random-user placebo")
    ax.axhline(0.5, color="k", lw=1.2, alpha=.6, label="chance (= forgotten)")
    ax.annotate("redundancy 0:\nforgotten cleanly", (x[0], y[0]), textcoords="offset points",
                xytext=(10, -30), fontsize=8, color="tab:blue",
                arrowprops=dict(arrowstyle="->", color="tab:blue"))
    ax.annotate("more co-users ->\nmore leakage", (x[-1], y[-1]), textcoords="offset points",
                xytext=(-92, 8), fontsize=8, color="tab:blue",
                arrowprops=dict(arrowstyle="->", color="tab:blue"))
    ax.set_xticks(x); ax.set_xticklabels([b["bin"] for b in pb], rotation=45)
    ax.set_xlabel("cross-user redundancy  (# OTHER users who interacted with the item)")
    ax.set_ylabel("probe AUC: is the forgotten item still inferable?")
    ns = agg.get("n_seeds", 1)
    ax.set_title("The collaborative-residue problem: correct unlearning still leaks the item\n"
                 f"({sweep_id}, mean±std over {ns} seed{'s' if ns != 1 else ''})")
    ax.set_ylim(0.35, 1.02)
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=.3); fig.tight_layout()
    (ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "results" / "figures" / f"problem_{sweep_id}.png", dpi=140)
    print("[problem] wrote", ROOT / "results" / "figures" / f"problem_{sweep_id}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--mode", choices=["run", "aggregate"], default="run")
    ap.add_argument("--run-id", default="local")
    ap.add_argument("--shard", default="1/1")
    ap.add_argument("--ckpt-dir"); ap.add_argument("--keep-dir"); ap.add_argument("--status")
    ap.add_argument("--max-runtime", default="46h")
    ap.add_argument("--resume", default="auto")
    ap.add_argument("--n-users", type=int); ap.add_argument("--per-stratum", type=int)
    ap.add_argument("--seeds"); ap.add_argument("--sweep-id"); ap.add_argument("--max-units", type=int)
    args = ap.parse_args()

    cfg = dict(DEFAULT_CFG)
    if args.config:
        cfg.update(yaml.safe_load(Path(args.config).read_text()))
    for key, val in (("n_users", args.n_users), ("per_stratum", args.per_stratum),
                     ("sweep_id", args.sweep_id)):
        if val is not None:
            cfg[key] = val
    if args.seeds:
        cfg["seeds"] = [int(s) for s in args.seeds.split(",")]
    cfg["config_hash"] = _config_hash(cfg)
    cfg["meta"] = dict(python=platform.python_version(), numpy=np.__version__, src_hash=_src_hash())

    sys.exit(aggregate_mode(cfg, args) if args.mode == "aggregate" else run_mode(cfg, args))


if __name__ == "__main__":
    main()
