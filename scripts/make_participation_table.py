"""Partial-participation (non-IID FL, F1) table: residue floor AUC at each client-participation
level, per dataset, with the change Δ from full→10% participation. The story is INVARIANCE — the
residue does not decline as participation falls (only the model's overall utility does).

    python scripts/make_participation_table.py   # prints markdown + writes paper/tables/participation.tex
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
DS = [("ml1m", "ML-1M"), ("ml100k", "ML-100K"), ("gowalla", "Gowalla"),
      ("lastfm", "LastFM"), ("steam", "Steam"), ("yelp", "Yelp")]
LEVELS = [1.0, 0.5, 0.25, 0.1]


def rows():
    for key, name in DS:
        p = ROOT / "results" / f"ablation_participation_{key}.json"
        if not p.exists():
            continue
        d = {r["knob"]: r for r in json.loads(p.read_text())["rows"]}
        floors = [d[lv]["residue_floor"] for lv in LEVELS]
        util = [d[lv]["utility"] for lv in LEVELS]
        yield name, floors, util, floors[-1] - floors[0], util[-1] - util[0]


def main() -> None:
    print(f"{'Dataset':9} | " + " ".join(f"p={lv:<4}" for lv in LEVELS)
          + f" | {'Δresidue':>8} | {'Δutility':>8}")
    print("-" * 62)
    for name, fl, ut, df, du in rows():
        print(f"{name:9} | " + " ".join(f"{v:>5.3f} " for v in fl)
              + f" | {df:>+8.3f} | {du:>+8.3f}")

    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{Residue floor AUC (probe AUC after verified unlearning, high-redundancy items) "
         r"under partial client participation---the realistic non-IID federated setting where only a "
         r"fraction of clients is sampled each round. The residue is \emph{invariant} to participation "
         r"($|\Delta|\!\le\!0.09$ from full to 10\%), while recommender utility degrades. Partial "
         r"participation weakens the model but does not remove the collaborative residue. "
         r"($n{=}1200$ users/dataset; absolute level tracks dataset density, cf.\ Fig.~1.)}",
         r"\label{tab:participation}", r"\begin{tabular}{lcccccc}", r"\toprule",
         r"Dataset & $p{=}1.0$ & $p{=}0.5$ & $p{=}0.25$ & $p{=}0.1$ & $\Delta$residue & $\Delta$util \\",
         r"\midrule"]
    for name, fl, ut, df, du in rows():
        L.append(f"{name} & {fl[0]:.3f} & {fl[1]:.3f} & {fl[2]:.3f} & {fl[3]:.3f} "
                 f"& ${df:+.3f}$ & ${du:+.3f}$ \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (ROOT / "paper" / "tables").mkdir(parents=True, exist_ok=True)
    out = ROOT / "paper" / "tables" / "participation.tex"
    out.write_text("\n".join(L) + "\n")
    print(f"\n-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
