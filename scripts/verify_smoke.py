"""Verify a smoke sweep: do the oracle + baselines behave as the science predicts?

Reads results/agg_<sweep>.json and asserts the qualitative relationships each method should
satisfy, printing PASS/FAIL. NB: there is no "our" (P3) method yet — the four wired are the
retrain ORACLE + three baselines (naive-delete, gradient-ascent, fine-tune).

    python scripts/verify_smoke.py <sweep_id>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    sweep = sys.argv[1] if len(sys.argv) > 1 else "smoke_verify"
    agg = json.loads((ROOT / "results" / f"agg_{sweep}.json").read_text())
    pb = [b for b in agg["per_bin"] if b.get("n", 0) > 0]
    if not pb:
        print("no per-bin data"); return 1

    def col(name):
        return np.array([b.get(name, np.nan) for b in pb], dtype=float)

    r0 = next((b for b in pb if b["bin"] == "0"), None)
    floor_r0 = r0["floor"] if r0 else float("nan")
    naive_r0 = r0["naive"] if r0 else float("nan")
    mean_floor = float(np.nanmean(col("floor")))
    checks: list[tuple[str, bool, str]] = []

    def chk(name, cond, detail=""):
        checks.append((name, bool(cond), detail))
        print(f"[{'PASS' if cond else 'FAIL'}] {name:<52} {detail}")

    def obs(name, detail):                          # observation: reported, not asserted
        print(f"[obs ] {name:<52} {detail}")

    print(f"=== verifying sweep '{sweep}'  ({agg['n']} units, methods={agg['methods']}) ===")
    # --- the residue signature on the ORACLE (ground truth) ---
    chk("floor rises with redundancy (Spearman>0.2)",
        agg["spearman_logr_floor"] > 0.2, f"rho={agg['spearman_logr_floor']:.2f}")
    chk("floor ~chance at redundancy 0 (<=0.55)", floor_r0 <= 0.55, f"r0={floor_r0:.2f}")
    hi_floor = agg["high_redundancy_floor"]
    if np.isnan(hi_floor):
        obs("high-redundancy floor (no high-r items in this federation)", "n/a")
    else:
        chk("high-redundancy floor beats chance (>0.6)", hi_floor > 0.6, f"hi={hi_floor:.2f}")
    present = set(agg.get("practical", []))
    leak = {m: agg.get(f"mean_leak_{m}", float("nan")) for m in present}
    federated = "fru" in present or "ours" in present

    if "gradasc" in present:
        chk("gradient-ascent goes BELOW floor (over-forgets)", leak["gradasc"] < -0.05,
            f"mean leak={leak['gradasc']:+.2f}")
    if not federated:   # centralized MF: naive/fine-tune under-remove the collaborative signal
        chk("naive-delete LEAKS above floor", leak.get("naive", 0) > 0.05,
            f"mean leak={leak.get('naive', float('nan')):+.2f}")
        chk("naive-delete fails worst at r=0 (>floor by >0.2)", (naive_r0 - floor_r0) > 0.2,
            f"naive_r0={naive_r0:.2f} vs floor_r0={floor_r0:.2f}")
        if "finetune" in present:
            chk("fine-tune LEAKS above floor", leak["finetune"] > 0.0, f"mean leak={leak['finetune']:+.2f}")
    else:               # federated: FedAvg dilutes per-user contributions -> naive ~ floor already
        obs("naive leak vs floor (federated; ~0 = FedAvg dilution)",
            f"{leak.get('naive', float('nan')):+.2f}")
        if "ours" in present:
            chk("OURS reaches ~floor (|leak| < 0.20, no big leak / over-forget)",
                abs(leak["ours"]) < 0.20, f"leak={leak['ours']:+.2f}")
        if "fru" in present:
            chk("FRU reaches ~floor (|leak| < 0.20)", abs(leak["fru"]) < 0.20, f"leak={leak['fru']:+.2f}")
    # --- collateral to OTHER users: gradient-ascent wrecks the global q_X; ours/naive leave it intact ---
    cm = set(agg.get("collateral_methods", []))
    if cm:
        co = {m: agg.get(f"mean_collateral_{m}", float("nan")) for m in cm}
        if "ours" in cm:
            chk("OURS has zero collateral to others (local-only)", co["ours"] < 1e-6,
                f"||dq||={co['ours']:.4f}")
        if "gradasc" in cm and "ours" in cm:
            chk("gradient-ascent's collateral >> ours", co["gradasc"] > co["ours"] + 0.05,
                f"gradasc={co['gradasc']:.3f} vs ours={co['ours']:.3f}")
        if "fru" in cm and "ours" in cm:
            obs("FRU vs ours global perturbation (FRU>=ours by design)",
                f"fru={co['fru']:.4f}  ours={co['ours']:.4f}")
    # --- OURS' ADVANTAGE: same unlearning privacy, but OTHER users' utility preserved ---
    om = set(agg.get("oth_methods", []))
    if om:
        o = {m: agg.get(f"mean_oth_{m}", float("nan")) for m in om}
        if "ours" in om and "fru" in om:
            chk("OURS preserves OTHER users' utility >= FRU (the P3 win)", o["ours"] >= o["fru"] - 1e-9,
                f"ours={o['ours']:.3f} vs fru={o['fru']:.3f}  (gap {o['ours'] - o['fru']:+.3f})")
        if "ours" in om and "floor" in om:
            # retrain is an INDEPENDENT from-scratch run, so its q_X is a different stochastic draw
            # -> not a clean paired comparison (unlike ours-vs-FRU, both derived from base). Report,
            # don't assert; in expectation ours (=base) >= retrain, but it is noisy at smoke scale.
            obs("ours vs retrain-oracle on others' utility (noisy: retrain is a fresh draw)",
                f"ours={o['ours']:.3f} vs retrain={o['floor']:.3f}")
        if "ours" in om and "gradasc" in om:
            chk("OURS preserves others > gradient-ascent", o["ours"] > o["gradasc"],
                f"ours={o['ours']:.3f} vs gradasc={o['gradasc']:.3f}")
        if "ours" in present and "naive" in present:
            obs("ours vs naive privacy (ours <= naive is the privacy win)",
                f"ours leak={leak.get('ours', float('nan')):+.2f}  "
                f"naive leak={leak.get('naive', float('nan')):+.2f}")
    # --- references ---
    chk("no-unlearn (full) above floor", float(np.nanmean(col("full"))) > mean_floor,
        f"full={np.nanmean(col('full')):.2f} floor={mean_floor:.2f}")
    chk("placebo below no-unlearn (not a generic-popularity artifact)",
        float(np.nanmean(col("placebo"))) < float(np.nanmean(col("full"))),
        f"placebo={np.nanmean(col('placebo')):.2f} full={np.nanmean(col('full')):.2f}")
    # --- utility preserved (no method collapses the user's own recommendations) ---
    for m in agg.get("util_methods", []):
        u = agg.get(f"mean_util_{m}", float("nan"))
        chk(f"utility[{m}] not collapsed (NDCG>0.15)", u > 0.15, f"ndcg={u:.2f}")

    n_fail = sum(1 for _, c, _ in checks if not c)
    print(f"\n>>> {'ALL CHECKS PASS' if n_fail == 0 else f'{n_fail} CHECK(S) FAILED'} "
          f"({len(checks) - n_fail}/{len(checks)}) <<<")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
