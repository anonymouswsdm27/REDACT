"""Main comparison table vs existing systems (§11/§12): every method on the three axes —
requesting-user privacy, the user's own utility, OTHER users' utility, plus collateral and whether
it stays on-device. Averaged over the 6 federated datasets (mean +/- std across datasets).

    python scripts/make_comparison_table.py    # prints markdown + writes paper/tables/comparison.tex
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics.residue import aggregate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DS = ["ml1m", "ml100k", "gowalla", "lastfm", "steam", "yelp"]
# (key, display, citation, on-device?) — order = do-nothing -> generic FU -> SOTA -> ours; oracle last-ref
# FRU and FedShare coincide in the shared-MF setting (FedShare's snapshot-difference removal equals
# FRU's contribution rollback), so they produce identical numbers and are shown as a single row.
ROWS = [("floor", "Retrain (oracle)", "", False),
        ("naive", "Naive local delete", "", True),
        ("gradasc", "Gradient-ascent", r"~\cite{neggrad}", False),
        ("fru", r"FRU~\cite{fru} / FedShare~\cite{fedshare}", "", False),
        ("ours", "\\textbf{REDACT (ours)}", "", True)]


def main() -> None:
    raws = {}
    for k in DS:
        ps = glob.glob(str(ROOT / "results" / "units" / f"fedncf_{k}" / "*.json"))
        raws[k] = [json.loads(open(p).read()) for p in ps]
    aggs = {k: aggregate(raws[k]) for k in DS}

    def stat(prefix, m):
        v = np.array([aggs[k].get(f"{prefix}_{m}", np.nan) for k in DS], float)
        return float(np.nanmean(v)), float(np.nanstd(v))

    def auc_star(mkey):
        """Orientation-invariant AUC* = mean_u max(AUC,1-AUC) per dataset, then mean$\\pm$std across
        datasets — the attacker-agnostic view (>= 0.5 by construction)."""
        field = "auc_floor" if mkey == "floor" else f"auc_{mkey}"
        per_ds = []
        for k in DS:
            v = np.array([u[field] for u in raws[k] if field in u and u[field] == u[field]], float)
            if len(v):
                per_ds.append(float(np.maximum(v, 1.0 - v).mean()))
        return (float(np.mean(per_ds)), float(np.std(per_ds))) if per_ds else (float("nan"), 0.0)

    def row(mkey):
        leak = (0.0, 0.0) if mkey == "floor" else stat("mean_leak", mkey)
        return (stat("mean_auc", mkey), auc_star(mkey), leak, stat("mean_util", mkey),
                stat("mean_oth", mkey), stat("mean_collateral", mkey))

    # ---- console (markdown) ----
    hdr = (f"{'Method':20} | {'AUC↓':>6} | {'AUC*↓':>6} | {'leak↓':>12} | {'UserNDCG↑':>9} | "
           f"{'OthNDCG↑':>9} | {'Collat↓':>9} | Device")
    print(hdr); print("-" * len(hdr))
    for mkey, disp, _, dev in ROWS:
        au, aus, lk, ut, ot, co = row(mkey)
        name = disp.replace("\\textbf{", "").replace("}", "")
        lks = "0 (ref)" if mkey == "floor" else f"{lk[0]:+.3f}±{lk[1]:.2f}"
        dev_s = "on-dev" if dev else "global"
        print(f"{name:20} | {au[0]:>6.3f} | {aus[0]:>6.3f} | {lks:>12} | {ut[0]:>9.3f} | "
              f"{ot[0]:>9.3f} | {co[0]:>9.3f} | {dev_s}")

    # ---- LaTeX (booktabs) ----
    (ROOT / "paper" / "tables").mkdir(parents=True, exist_ok=True)
    L = [r"\begin{table*}[t]", r"\centering", r"\small",
         r"\caption{Unlearning methods on the federated collaborative-recommendation setting "
         r"(mean$\pm$std across 6 datasets). \emph{Probe AUC} is the requesting user's one-sided "
         r"inferability of the forgotten item, averaged over redundancy strata (same metric as "
         r"Fig.~\ref{fig:problem}; $0.5=$ forgotten); it is directional and can fall \emph{below} "
         r"$0.5$ when the item ranks below its controls, so the target is the retrain floor, not "
         r"zero. \emph{Priv.\ leak} is the excess over that (one-sided) floor ($0=$ reaches it). "
         r"$\mathrm{AUC}^\ast{=}\max(\mathrm{AUC},1{-}\mathrm{AUC})$ is the orientation-invariant "
         r"inferability to a direction-agnostic attacker ($\ge 0.5$ by construction); it exposes "
         r"gradient-ascent's low one-sided AUC as \emph{over-forgetting}---it corrupts the shared "
         r"embedding (collateral $1.06$, others' utility $0.54$) and is in fact the leakiest method "
         r"($\mathrm{AUC}^\ast{=}0.94$), while FRU/FedShare/\method{} sit just above the "
         r"$\mathrm{AUC}^\ast$ floor. \emph{User} and \emph{Others' NDCG} are "
         r"the requesting user's own and the other users' utility for the item. \emph{Collateral} is "
         r"$\lVert\Delta q_X\rVert$ on the shared embedding. FRU and FedShare coincide in the "
         r"shared-MF setting (FedShare's snapshot-difference removal equals FRU's contribution "
         r"rollback), so they share a row. All three of FRU/FedShare/\method{} remove only $A$'s "
         r"contribution and thus reach the same privacy floor; \method{} alone does so on-device, "
         r"leaving other users' utility intact at zero collateral.}",
         r"\label{tab:comparison}",
         r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccccc}", r"\toprule",
         r"Method & AUC (1-sided) $\downarrow$ & AUC$^\ast$ $\downarrow$ & Priv.\ leak $\downarrow$ "
         r"& User NDCG $\uparrow$ & Others' NDCG $\uparrow$ & Collateral $\downarrow$ & On-device \\",
         r"\midrule"]
    def pm(ms, signed=False):    # mean$\pm$std cell (std across datasets, \scriptsize)
        m = f"${ms[0]:+.3f}$" if signed else f"{ms[0]:.3f}"
        return f"{m}{{\\scriptsize$\\pm${ms[1]:.2f}}}"
    for mkey, disp, cite, dev in ROWS:
        au, aus, lk, ut, ot, co = row(mkey)
        b = (lambda s: f"\\textbf{{{s}}}") if mkey == "ours" else (lambda s: s)
        lks = "0 (ref)" if mkey == "floor" else pm(lk, signed=True)
        dmark = r"\cmark" if dev else r"\xmark"
        L.append(f"{disp}{cite} & {b(pm(au))} & {b(pm(aus))} & {b(lks)} & {b(pm(ut))} & "
                 f"{b(pm(ot))} & {b(pm(co))} & {dmark} \\\\")
    L += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
    out = ROOT / "paper" / "tables" / "comparison.tex"
    out.write_text("\n".join(L) + "\n")
    print(f"\n-> {out.relative_to(ROOT)}  (needs \\usepackage{{booktabs}} + \\cmark/\\xmark, e.g. pifont)")


if __name__ == "__main__":
    main()
