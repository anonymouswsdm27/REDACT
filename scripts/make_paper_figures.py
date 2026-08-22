"""Publication-ready figures for the paper (§15), from the verified sweep + analysis
outputs in results/. Six figures kept deliberately small/readable (no giant multi-panel strips):
  fig0 motivation · fig1 problem · fig2 limit · fig3 method · fig4 robustness (1x3) · fig5 sequential.
Partial participation (F1) is a Table (scripts/make_participation_table.py), not a figure.

Design: Okabe-Ito colorblind-safe palette (teal = "ours", the hero); legends BELOW each panel;
serif-free clean rcParams; 300-dpi PNG + vector PDF for LaTeX. Reproducible:

    python scripts/make_paper_figures.py     # -> paper/figures/fig{0..5}_*.{png,pdf}
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # runnable as a plain script

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.metrics.limit import _model  # noqa: E402
from src.metrics.residue import aggregate  # noqa: E402
from src.probes.membership import BIN_LABELS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---- Okabe-Ito colorblind-safe palette (the reference's recommended style; teal is the hero) ----
OI = dict(blue="#0072B2", sky="#56B4E9", teal="#009E73", orange="#E69F00",
          verm="#D55E00", purple="#CC79A7", yellow="#F0E442", grey="#9AA0A6", ink="#222222")
DCOL = {"ml1m": OI["blue"], "ml100k": OI["sky"], "gowalla": OI["teal"],
        "lastfm": OI["orange"], "steam": OI["verm"], "yelp": OI["purple"]}
DNAME = {"ml1m": "ML-1M", "ml100k": "ML-100K", "gowalla": "Gowalla",
         "lastfm": "LastFM", "steam": "Steam", "yelp": "Yelp"}
DMARK = {"ml1m": "o", "ml100k": "s", "gowalla": "^", "lastfm": "D", "steam": "v", "yelp": "P"}
DS = list(DNAME)
# method colors: ours = teal (hero); naive = vermillion (bad); FRU/FedShare = orange/purple; floor = grey
MCOL = dict(naive=OI["verm"], fru=OI["orange"], fedshare=OI["purple"], ours=OI["teal"],
            floor=OI["grey"], gradasc=OI["sky"])

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "figure.dpi": 150, "savefig.dpi": 300,
    "legend.frameon": False, "lines.linewidth": 1.8, "lines.markersize": 5,
})


def load(sweep):
    rows = [json.loads(open(p).read()) for p in glob.glob(str(ROOT / "results" / "units" / sweep / "*.json"))]
    return aggregate(rows) if rows else None


def blabel(ax, letter, title):
    ax.set_title(f"({letter}) {title}", loc="left", fontsize=9.5, fontweight="bold")


def below(ax, ncol, y=-0.34):
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, columnspacing=1.0,
              handletextpad=0.5, borderaxespad=0.0)


def sparse_xticks(ax, labels, step=2):
    """Label every `step`-th tick HORIZONTALLY (no rotation) so the x-label stays at normal height
    and the bottom legend clears it — keeps compact figures readable."""
    idx = list(range(0, len(labels), step))
    ax.set_xticks(idx)
    ax.set_xticklabels([labels[i] for i in idx], fontsize=7)


def band(ax, x, y, yerr, color, *, marker=None, label=None, lw=1.3, ms=4, alpha=0.16, ls="-"):
    """Line + shaded ±std error band (cleaner than capped error bars for the paper figures)."""
    x = np.asarray(list(x), float); y = np.asarray(y, float); e = np.asarray(yerr, float)
    ax.plot(x, y, marker=marker, color=color, lw=lw, ms=ms, ls=ls, label=label)
    ax.fill_between(x, y - e, y + e, color=color, alpha=alpha, linewidth=0)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    print(f"  {OUT.relative_to(ROOT)}/{name}.{{png,pdf}}")


# =========================== FIGURE 0 — MOTIVATION (the teaser) ===========================
def fig_motivation():
    """The hook: correct unlearning removes a PRIVATE item but not a collaboratively-reinforced one."""
    a = load("mf_ml1m")

    def binv(lbl, m):
        p = [x for x in a["per_bin"] if x["bin"] == lbl][0]
        return p.get(m, np.nan)
    hi = [x for x in a["per_bin"] if x["bin"] in ("101-200", "201-500", "501+") and x.get("n", 0)]

    def hiv(m):
        return float(np.mean([x[m] for x in hi]))
    methods = [("full", "no unlearning", OI["grey"]),
               ("naive", "naive local delete", OI["verm"]),
               ("floor", "after unlearning", OI["blue"])]
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    x = np.arange(2); w = 0.26
    for i, (m, lab, col) in enumerate(methods):
        ax.bar(x + (i - 1) * w, [binv("0", m), hiv(m)], w, color=col, label=lab)
    ax.axhline(0.5, ls=":", c=OI["ink"], lw=1)
    ax.text(1.46, 0.52, "chance", fontsize=7, color=OI["ink"], style="italic")
    ax.annotate("chance-level\ninferability", xy=(0 + w, binv("0", "floor")),
                xytext=(0 + w, binv("0", "floor") + 0.20), ha="center", fontsize=8,
                color=OI["blue"], fontweight="bold", arrowprops=dict(arrowstyle="->", color=OI["blue"]))
    ax.annotate("residue remains ✗", xy=(1 + w, hiv("floor")),
                xytext=(1 + w - 0.05, hiv("floor") + 0.12), ha="center", fontsize=8,
                color=OI["verm"], fontweight="bold", arrowprops=dict(arrowstyle="->", color=OI["verm"]))
    ax.set_xticks(x)
    ax.set_xticklabels(["redundancy 0\n(only you touched it)", "high redundancy\n(many users touched it)"])
    ax.set_ylabel("probe AUC (still inferable?)")
    ax.set_ylim(0, 1.12)
    ax.set_title("After unlearning, a shared item stays inferable, a private one does not", loc="left",
                 fontsize=9.5, fontweight="bold")
    below(ax, 3, y=-0.24)
    fig.tight_layout(); save(fig, "fig0_motivation")


# =========================== FIGURE 1 — THE PROBLEM ===========================
def fig_problem():
    aggs = {k: load(f"mf_{k}") for k in DS}
    # Larger local fonts than the global defaults so the text reads bigger in the paper.
    with plt.rc_context({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11.5,
                         "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5}):
        fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.2))
        # (a) probe AUC vs redundancy, all datasets. Fixed axis over every bin any dataset populates;
        # plot each dataset at its TRUE bin index so a shorter range (e.g. Yelp) ends where its data ends.
        allbins = [bl for bl in BIN_LABELS
                   if any(p["bin"] == bl and p.get("n", 0) for k in DS for p in aggs[k]["per_bin"])]
        bidx = {bl: i for i, bl in enumerate(allbins)}
        for k in DS:
            a = aggs[k]; b = [x for x in a["per_bin"] if x.get("n", 0)]
            xs = [bidx[x["bin"]] for x in b]
            y = [x["floor"] for x in b]; e = [x.get("floor_std", 0) for x in b]
            band(ax[0], xs, y, e, DCOL[k], marker=DMARK[k], ms=4, lw=1.3, alpha=0.12,
                 label=f"{DNAME[k]} {a['spearman_logr_floor']:.2f}")
        ax[0].axhline(0.5, ls=":", c=OI["ink"], lw=1)
        _idx = list(range(0, len(allbins), 2))
        ax[0].set_xticks(_idx); ax[0].set_xticklabels([allbins[i] for i in _idx], fontsize=9)
        ax[0].set_xlabel("cross-user redundancy $r$"); ax[0].set_ylabel("probe AUC")
        ax[0].set_ylim(0.28, 1.0)
        ax[0].set_title("(a) Residue grows with redundancy", loc="left", fontsize=12, fontweight="bold")
        below(ax[0], 3, y=-0.30)
        # (b) controls: r0 & placebo near chance, high-r high (proves cross-user + user-specific)
        x = np.arange(len(DS)); w = 0.27
        r0 = [next(p["floor"] for p in aggs[k]["per_bin"] if p["bin"] == "0") for k in DS]
        placebo = [aggs[k].get("mean_auc_placebo", np.nan) for k in DS]
        hi = [aggs[k]["high_redundancy_floor"] for k in DS]
        ax[1].bar(x - w, r0, w, color=OI["grey"], label="$r$=0")
        ax[1].bar(x, placebo, w, color=OI["sky"], label="placebo")
        ax[1].bar(x + w, hi, w, color=OI["teal"], label="high $r$")
        ax[1].axhline(0.5, ls=":", c=OI["ink"], lw=1)
        ax[1].set_xticks(x); ax[1].set_xticklabels([DNAME[k] for k in DS], fontsize=8)
        ax[1].set_ylabel("probe AUC"); ax[1].set_ylim(0, 1.0)
        ax[1].set_title("(b) Cross-user & user-specific", loc="left", fontsize=12, fontweight="bold")
        below(ax[1], 3, y=-0.30)
        fig.tight_layout(); save(fig, "fig1_problem")


# =========================== FIGURE 2 — THE LIMIT ===========================
def fig_limit():
    from experiments.limit_theory import closed_form_auc, montecarlo_auc, real_mf_auc
    # Full-width figure (figure* / textwidth), panels side-by-side. Larger local fonts than the
    # global defaults so the text reads bigger in the paper.
    with plt.rc_context({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11.5,
                         "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5}):
        fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.2))
        # (a) empirical lower bound (ML-1M)
        fit = json.loads(open(ROOT / "results" / "limit_mf_ml1m.json").read())
        cv = fit["curve"]; xb = np.log1p(np.array(cv["r_med"]))
        grid = np.linspace(0, xb.max() * 1.02, 200)
        ax[0].axhline(0.5, ls=":", c=OI["ink"], lw=1, label="chance")
        ax[0].fill_between(grid, 0.5, _model(grid, fit["a_lo"], fit["c_lo"]), color=OI["teal"],
                           alpha=0.15, label="95% lower bound $L(r)$")
        ax[0].plot(grid, _model(grid, fit["a"], fit["c"]), color=OI["verm"], lw=2,
                   label=f"fit  ($R^2$={fit['r2']:.2f})")
        ax[0].errorbar(xb, cv["mean"], yerr=np.array(cv["sem"]) * 1.96, fmt="o", color=OI["blue"],
                       capsize=2, label="ML-1M")
        _idx = list(range(0, len(xb), 2))
        ax[0].set_xticks([xb[i] for i in _idx])
        ax[0].set_xticklabels([cv["label"][i] for i in _idx])
        ax[0].set_xlabel("cross-user redundancy $r$"); ax[0].set_ylabel("probe AUC")
        ax[0].set_ylim(0.45, 1.0)
        ax[0].set_title("(a) Empirical limit (per dataset)", loc="left", fontsize=12, fontweight="bold")
        below(ax[0], 2, y=-0.34)
        # (b) analytical: closed form + MC + real MF
        d, m, sig, tau = 16, 1.0, 1.0, 1.0
        rs = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]; rs_mf = [0, 2, 8, 32, 128]
        cf = [closed_form_auc(r, m, sig, tau, d) for r in rs]
        mc = [montecarlo_auc(r, m, sig, tau, d) for r in rs]
        mf = [np.mean([real_mf_auc(r, m, sig, d, seed=s) for s in range(3)]) for r in rs_mf]
        from math import erf, sqrt
        asym = 0.5 * (1 + erf(sqrt(m) / sig / sqrt(2)))
        x = np.log1p(rs); xm = np.log1p(rs_mf)
        ax[1].axhline(0.5, ls=":", c=OI["ink"], lw=1)
        ax[1].axhline(asym, ls="--", c=OI["teal"], lw=1.2,
                      label=f"asymptote $\\Phi(\\sqrt{{m}}/\\sigma)$={asym:.2f}")
        ax[1].plot(x, cf, "-", color=OI["verm"], lw=2, label="closed form $\\Phi(m/\\sqrt{Var})$")
        ax[1].plot(x, mc, "o", color=OI["blue"], label="Monte-Carlo")
        ax[1].plot(xm, mf, "s", color=OI["orange"], label="real BPR-MF")
        _j = list(range(0, len(rs), 2))
        ax[1].set_xticks([x[i] for i in _j]); ax[1].set_xticklabels([rs[i] for i in _j])
        ax[1].set_xlabel("redundancy $r$"); ax[1].set_ylabel("probe AUC")
        ax[1].set_ylim(0.45, 1.0)
        ax[1].set_title("(b) Analytical limit (closed form)", loc="left", fontsize=12, fontweight="bold")
        below(ax[1], 2, y=-0.34)
        fig.tight_layout(); save(fig, "fig2_limit")


# =========================== FIGURE 3 — THE METHOD ===========================
def fig_method():
    aggs = {k: load(f"fedncf_{k}") for k in DS}
    fig, ax = plt.subplots(1, 3, figsize=(10.8, 2.9))
    x = np.arange(len(DS))
    # (a) privacy: residual leak vs floor for the 4 practical methods
    ms = ["naive", "fru", "fedshare", "ours"]; nm = {"naive": "naive-delete", "fru": "FRU",
                                                     "fedshare": "FedShare", "ours": "REDACT"}
    w = 0.2
    for i, mth in enumerate(ms):
        vals = [aggs[k].get(f"mean_leak_{mth}", np.nan) for k in DS]
        ax[0].bar(x + (i - 1.5) * w, vals, w, color=MCOL[mth], label=nm[mth])
    ax[0].axhline(0, c=OI["ink"], lw=0.8)
    ax[0].set_xticks(x); ax[0].set_xticklabels([DNAME[k] for k in DS], rotation=45, ha="right")
    ax[0].set_ylabel("difference from retraining\n(0 = retraining level)")
    blabel(ax[0], "a", "Privacy relative to retraining"); below(ax[0], 4, y=-0.42)
    # (b) others' utility: ours vs FRU/FedShare (the P3 win)
    w = 0.38
    ax[1].bar(x - w / 2, [aggs[k].get("mean_oth_ours", np.nan) for k in DS], w, color=OI["teal"], label="REDACT")  # noqa: E501
    ax[1].bar(x + w / 2, [aggs[k].get("mean_oth_fru", np.nan) for k in DS], w, color=OI["orange"],
              label="FRU / FedShare")
    ax[1].set_xticks(x); ax[1].set_xticklabels([DNAME[k] for k in DS], rotation=45, ha="right")
    ax[1].set_ylabel("other users' utility (NDCG)"); ax[1].set_ylim(0, 1.05)
    blabel(ax[1], "b", "Others' utility: REDACT $\\geq$ FRU"); below(ax[1], 2, y=-0.42)
    # (c) the frontier (ML-1M): inferability vs others'-utility loss
    s = json.loads(open(ROOT / "results" / "frontier_ml1m.json").read())
    floor = s["floor_auc"]; naive = next(d["auc"] for d in s["ours"] if d["alpha"] == 0.0)
    nl = [d["oth_loss"] for d in s["neighbor"]]; na = [d["auc"] for d in s["neighbor"]]
    ax[2].plot(nl, na, "-s", color=MCOL["naive"], label="neighbor-deletion")
    ax[2].scatter([0], [naive], s=45, color=OI["grey"], zorder=5, label=f"naive ({naive:.2f})")
    ax[2].scatter([0], [floor], s=150, marker="*", color=OI["teal"], zorder=6, label="REDACT")
    ax[2].axhline(floor, ls="--", c=OI["teal"], lw=1, label="retraining reference")
    ax[2].axvspan(-0.015, 0.015, color=OI["teal"], alpha=0.08)
    ax[2].set_xlabel("other users' utility loss"); ax[2].set_ylabel("requesting user's inferability (AUC)")
    blabel(ax[2], "c", "Privacy-utility tradeoff"); below(ax[2], 2, y=-0.42)
    fig.tight_layout(); save(fig, "fig3_method")


# =========================== FIGURE 4 — ROBUSTNESS (3 checks, ML-1M) ===========================
def fig_robust():
    """Three "the residue is real & robust" checks in a clean horizontal triptych. Partial
    participation (F1) is reported in Table (make_participation_table.py), not here — 6 datasets ×
    4 levels reads better as a table than a cramped panel."""
    fig, ax = plt.subplots(1, 3, figsize=(10.8, 2.45))
    # (a) Gaussian-noise persists — no rotated ticks, short labels, legend one row just below the x-label
    dp = json.loads(open(ROOT / "results" / "ablation_dp.json").read())["rows"]
    xs = [str(r["knob"]) for r in dp]
    band(ax[0], range(len(dp)), [r["residue_floor"] for r in dp],
         [r.get("residue_floor_std", 0) for r in dp], OI["verm"], marker="o", alpha=0.2,
         label="residue")
    band(ax[0], range(len(dp)), [r["utility"] for r in dp],
         [r.get("utility_std", 0) for r in dp], OI["teal"], marker="^", alpha=0.2, label="utility")
    ax[0].axhline(0.5, ls=":", c=OI["ink"], lw=1)
    ax[0].set_xticks(range(len(dp))); ax[0].set_xticklabels(xs)
    ax[0].set_xlabel("Gaussian update noise $\\sigma$"); ax[0].set_ylabel("AUC / NDCG"); ax[0].set_ylim(0, 1)
    blabel(ax[0], "a", "Persists under update noise"); below(ax[0], 2, y=-0.30)
    # (b) scale-robust
    sc = json.loads(open(ROOT / "results" / "ablation_scale.json").read())["rows"]
    xs = [str(r["knob"]) for r in sc]
    band(ax[1], range(len(sc)), [r["residue_floor"] for r in sc],
         [r.get("residue_floor_std", 0) for r in sc], OI["verm"], marker="o", alpha=0.2,
         label="residue")
    ax[1].plot(range(len(sc)), [r["residue_floor"] + r["naive_leak"] for r in sc], "-s",
               color=OI["grey"], label="naive-delete")
    ax[1].axhline(0.5, ls=":", c=OI["ink"], lw=1)
    ax[1].set_xticks(range(len(sc))); ax[1].set_xticklabels(xs)
    ax[1].set_xlabel("number of clients"); ax[1].set_ylabel("AUC"); ax[1].set_ylim(0.4, 1)
    blabel(ax[1], "b", "Scale-robust residue"); below(ax[1], 2, y=-0.30)
    # (c) probe-architecture robust — rotated ticks push the x-label down, so its legend sits lower
    pr = json.loads(open(ROOT / "results" / "ablation_probe_ml1m.json").read())["per_bin"]
    xs = [r["bin"] for r in pr]
    for kind, col, mk, lab in [("score", OI["blue"], "o", "full score"),
                               ("nobias", OI["orange"], "s", "no bias"),
                               ("cosine", OI["teal"], "^", "cosine")]:
        ax[2].plot(range(len(pr)), [r[kind] for r in pr], marker=mk, color=col, label=lab)
    ax[2].axhline(0.5, ls=":", c=OI["ink"], lw=1)
    # label every other bin HORIZONTALLY (no rotation) so the x-label sits at the same height as
    # (a)/(b) and all three legends align at the same offset.
    tick_idx = list(range(0, len(pr), 2))
    ax[2].set_xticks(tick_idx); ax[2].set_xticklabels([xs[i] for i in tick_idx], fontsize=7)
    ax[2].set_xlabel("cross-user redundancy $r$"); ax[2].set_ylabel("probe AUC"); ax[2].set_ylim(0.3, 1)
    blabel(ax[2], "c", "Robust to probe choice"); below(ax[2], 3, y=-0.30)
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig4_robustness")


# =========================== FIGURE 5 — SEQUENTIAL RESIDUE (its own §2 claim) ===========================
def fig_sequential():
    """The sequential residue: in a GRU4Rec model BOTH channels appear — collaborative (AUC rises
    with redundancy) and within-user (A's real pre-X context beats a random context: the shaded gap)."""
    sq = json.loads(open(ROOT / "results" / "sequential_ml1m.json").read())["per_bin"]
    xs = [r["bin"] for r in sq]; xi = list(range(len(sq)))
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    ax.plot(xi, [r["auc"] for r in sq], "-o", color=OI["blue"], ms=4, label="A's real pre-$X$ context")
    ax.plot(xi, [r["auc_randctx"] for r in sq], "-s", color=OI["orange"], ms=4, label="random context")
    ax.fill_between(xi, [r["auc_randctx"] for r in sq], [r["auc"] for r in sq],
                    color=OI["teal"], alpha=0.25, label="within-user sequential component")
    ax.axhline(0.5, ls=":", c=OI["ink"], lw=1, label="chance")
    sparse_xticks(ax, xs, step=2)
    ax.set_xlabel("cross-user redundancy $r$"); ax.set_ylabel("sequential probe AUC")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Sequential recommendation", loc="left",
                 fontsize=9.5, fontweight="bold")
    below(ax, 2, y=-0.30)
    fig.tight_layout()
    save(fig, "fig5_sequential")


# =========================== FIGURE 6 — FL INVARIANCE (participation + training config) ===========
def fig_fl_invariance():
    """Two multi-dataset FL-robustness results as flat lines (=invariant): residue floor vs client
    participation (a) and vs FedAvg training config (b), one line per dataset."""
    part = {k: json.loads((ROOT / "results" / f"ablation_participation_{k}.json").read_text())
            for k in DS if (ROOT / "results" / f"ablation_participation_{k}.json").exists()}
    fed = {k: json.loads((ROOT / "results" / f"ablation_fedconfig_{k}.json").read_text())
           for k in DS if (ROOT / "results" / f"ablation_fedconfig_{k}.json").exists()}
    # SINGLE-COLUMN figure: 1x2 with a shared y-axis (fits one column), legend below.
    fig, ax = plt.subplots(1, 2, figsize=(5.6, 3.0), sharey=True)
    # (a) participation (shaded ±std across seeds)
    xs = None
    for k in part:
        rows = part[k]["rows"]; xs = [str(r["knob"]) for r in rows]
        band(ax[0], range(len(rows)), [r["residue_floor"] for r in rows],
             [r.get("residue_floor_std", 0) for r in rows], DCOL[k], marker=DMARK[k], ms=3.5,
             lw=1.2, alpha=0.12, label=DNAME[k])
    ax[0].axhline(0.5, ls=":", c=OI["ink"], lw=1)
    if xs:
        ax[0].set_xticks(range(len(xs))); ax[0].set_xticklabels(xs, fontsize=6.5)
    ax[0].set_xlabel("participation", fontsize=8); ax[0].set_ylabel("residue floor AUC", fontsize=8)
    ax[0].set_ylim(0.4, 0.85); blabel(ax[0], "a", "participation")
    # (b) training config — rounds R (5,15,30) then local epochs E (1,5); vertical divider between
    cfg = ["R5", "R15", "R30", "E1", "E5"]
    for k in fed:
        rf = {x["knob"]: x["residue_floor"] for x in fed[k]["rows"]}
        rs = {x["knob"]: x.get("residue_floor_std", 0) for x in fed[k]["rows"]}
        band(ax[1], range(len(cfg)), [rf.get(c, np.nan) for c in cfg],
             [rs.get(c, 0) for c in cfg], DCOL[k], marker=DMARK[k], ms=3.5, lw=1.2, alpha=0.12,
             label=DNAME[k])
    ax[1].axhline(0.5, ls=":", c=OI["ink"], lw=1)
    ax[1].axvline(2.5, ls="-", c=OI["grey"], lw=0.8, alpha=0.6)
    ax[1].set_xticks(range(len(cfg))); ax[1].set_xticklabels(["5", "15", "30", "1", "5"], fontsize=6.5)
    ax[1].set_xlabel("rounds $R$ $\\mid$ epochs $E$", fontsize=8)
    blabel(ax[1], "b", "training config")
    # shared legend below both panels (all 6 datasets), 3 columns x 2 rows, close under the panels
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0),
               frameon=False, columnspacing=1.0, handletextpad=0.4, fontsize=7)
    fig.tight_layout(rect=(0, 0.15, 1, 1), w_pad=0.6); save(fig, "fig6_fl_invariance")


if __name__ == "__main__":
    print("writing paper figures ->")
    fig_motivation(); fig_problem(); fig_limit(); fig_method(); fig_robust(); fig_sequential()
    fig_fl_invariance()
