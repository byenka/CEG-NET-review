#!/usr/bin/env python
"""step1_conditional_coverage_cohorts.py — conditional coverage on all four cohorts.

Asks the Step 0 question of every cohort, using stratification variables that exist in the
deployed evidence vector and therefore travel to the external cohorts, unlike the five
`detcmp` detectors of Step 0 whose caches cover OIA-DDR only.

Protocol locked before execution:
  _LGOVNET_NEW_SEP/PROTOCOL_LOCK_step1_conditional_coverage_cohorts.md

Reads the same CSVs, task and alpha as tools/eval_selective_conformal_baselines.py.
Nothing is refitted except the gate, on OIA-DDR validation, over the five fixed seeds.

Output: _LGOVNET_NEW_SEP/results/step1_conditional_coverage_cohorts.json
Run from the repo root:
  python _LGOVNET_NEW_SEP/scripts/step1_conditional_coverage_cohorts.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.eval_fp_reduction_benchmark import (  # noqa: E402
    EVIDENCE_FEATURES, apply_mlp, fit_mlp, load_csv,
)
from tools.eval_selective_conformal_baselines import (  # noqa: E402
    ALPHA, COHORTS, TASK, VALID, conformal_threshold,
)

OUT = ROOT / "_LGOVNET_NEW_SEP/results/step1_conditional_coverage_cohorts.json"
NOMINAL = 1.0 - ALPHA
SEEDS = (0, 1, 2, 3, 4)
MIN_POS = 200
DELTA = 0.30           # the tier threshold of the main article
PROCEED = 0.15         # C1 trigger, same value as Step 0


def pss_status(path: Path) -> np.ndarray:
    """ev_pss_status per row, in CSV order; load_csv keeps every row for this task."""
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    return np.array([(r.get("ev_pss_status") or "").strip().lower() for r in rows])


def strata_of(d: dict, status: np.ndarray) -> dict:
    """The three annotation-free taxonomies of the protocol lock."""
    measured = ~np.isin(status, ("", "placeholder", "unmatched", "missing", "none"))
    return {
        "A_corroboration": {
            "no secondary": d["ev_sds_nsrc"] < 1.0,
            "corroborated": d["ev_sds_nsrc"] >= 1.0,
        },
        "B_prior_conflict": {
            "low conflict": d["ev_sdca"] < DELTA,
            "high conflict": d["ev_sdca"] >= DELTA,
        },
        "C_pss_availability": {
            "PSS measured": measured,
            "PSS placeholder": ~measured,
        },
    }


def gate_inputs(d: dict, multisource: bool) -> np.ndarray:
    cols = [d[f] for f in EVIDENCE_FEATURES] + [d["pred_score"]]
    if multisource:
        cols = [d[f] for f in EVIDENCE_FEATURES] + [d["pred_score"], d["ev_sds"],
                                                    d["ev_sds_nsrc"]]
    return np.stack(cols, 1)


def coverage(mask: np.ndarray, score: np.ndarray, label: np.ndarray, thr: float):
    pos = mask & (label == 1)
    n = int(pos.sum())
    if n == 0:
        return None, 0
    return float((score[pos] >= thr).mean()), n


def main() -> None:
    valid = load_csv(VALID, positive=TASK)
    lab_v = valid["label"]

    # the four scores of the protocol lock, on validation and on every cohort
    models_kg, models_full = [], []
    for s in SEEDS:
        models_kg.append(fit_mlp(gate_inputs(valid, False), lab_v.astype(np.float64), seed=s))
        models_full.append(fit_mlp(gate_inputs(valid, True), lab_v.astype(np.float64), seed=s))

    def scores_of(d: dict) -> dict:
        return {
            "S (detector score)": [d["pred_score"]],
            "M (top1-top2 margin)": [d["unc_margin"]],
            "KG gate": [apply_mlp(gate_inputs(d, False), m) for m in models_kg],
            "S+E (full gate)": [apply_mlp(gate_inputs(d, True), m) for m in models_full],
        }

    sv = scores_of(valid)
    thresholds = {name: [conformal_threshold(s, lab_v, ALPHA) for s in lst]
                  for name, lst in sv.items()}

    rep = {
        "alpha": ALPHA, "nominal": NOMINAL, "task": TASK, "delta": DELTA,
        "min_positive": MIN_POS, "proceed_threshold": PROCEED, "seeds": list(SEEDS),
        "protocol": "_LGOVNET_NEW_SEP/PROTOCOL_LOCK_step1_conditional_coverage_cohorts.md",
        "cohorts": {}, "mondrian_preview": {},
    }

    for coh, path in COHORTS.items():
        d = load_csv(path, positive=TASK)
        lab = d["label"]
        st = strata_of(d, pss_status(path))
        sc = scores_of(d)
        block = {"n": int(lab.size), "n_positive": int(lab.sum()), "taxonomies": {}}
        print(f"\n=== {coh}  n={lab.size}  positives={int(lab.sum())} ===")

        for tax, groups in st.items():
            tblock = {"strata": {}, "methods": {}}
            for gname, mask in groups.items():
                npos = int((mask & (lab == 1)).sum())
                tblock["strata"][gname] = {
                    "n": int(mask.sum()), "n_positive": npos,
                    "positive_rate": float(lab[mask].mean()) if mask.sum() else None,
                    "eligible": npos >= MIN_POS}
            print(f"  [{tax}] " + "  ".join(
                f"{g}: n={v['n']} pos={v['n_positive']}"
                for g, v in tblock["strata"].items()))

            for name, slist in sc.items():
                per_group = {g: [] for g in groups}
                marg = []
                for r, s in enumerate(slist):
                    thr = thresholds[name][r]
                    c_all, _ = coverage(np.ones_like(lab, dtype=bool), s, lab, thr)
                    marg.append(c_all)
                    for g, mask in groups.items():
                        c, _ = coverage(mask, s, lab, thr)
                        if c is not None:
                            per_group[g].append(c - NOMINAL)
                rec = {"marginal_gap": float(np.mean(marg)) - NOMINAL, "per_stratum": {}}
                worst = 0.0
                for g in groups:
                    if not per_group[g]:
                        rec["per_stratum"][g] = None
                        continue
                    gap = float(np.mean(per_group[g]))
                    elig = tblock["strata"][g]["eligible"]
                    rec["per_stratum"][g] = {"gap": gap, "eligible": elig}
                    if elig:
                        worst = max(worst, abs(gap))
                rec["worst_eligible_abs_gap"] = worst
                tblock["methods"][name] = rec
                cells = "  ".join(
                    f"{g}={rec['per_stratum'][g]['gap']:+.3f}"
                    if rec["per_stratum"][g] else f"{g}=   -  " for g in groups)
                print(f"      {name:22s} marginal={rec['marginal_gap']:+.3f} | {cells}"
                      f" | worst={worst:.3f}")
            block["taxonomies"][tax] = tblock
        rep["cohorts"][coh] = block

    # ---------------- Mondrian preview, NOT part of any criterion
    print("\n=== Mondrian preview (per-stratum calibration), exploratory ===")
    st_v = strata_of(valid, pss_status(VALID))
    for coh, path in COHORTS.items():
        d = load_csv(path, positive=TASK)
        lab = d["label"]
        st = strata_of(d, pss_status(path))
        sc = scores_of(d)
        rep["mondrian_preview"][coh] = {}
        for tax in st:
            rep["mondrian_preview"][coh][tax] = {}
            for name, slist in sc.items():
                worst_seeds = []
                for r, s in enumerate(slist):
                    per = []
                    for g, mask in st[tax].items():
                        mv = st_v[tax][g]
                        if (mv & (lab_v == 1)).sum() < 20:
                            continue
                        thr = conformal_threshold(sv[name][r][mv], lab_v[mv], ALPHA)
                        c, _ = coverage(mask, s, lab, thr)
                        if c is not None and st[tax][g].sum() and \
                                int((mask & (lab == 1)).sum()) >= MIN_POS:
                            per.append(abs(c - NOMINAL))
                    if per:
                        worst_seeds.append(max(per))
                if worst_seeds:
                    rep["mondrian_preview"][coh][tax][name] = float(np.mean(worst_seeds))
        line = rep["mondrian_preview"][coh].get("A_corroboration", {})
        print(f"  {coh:14s} A: " + "  ".join(f"{k}={v:.3f}" for k, v in line.items()))

    # ---------------- locked criteria
    A = "A_corroboration"
    c1_hits = []
    for coh, blk in rep["cohorts"].items():
        for tax, tb in blk["taxonomies"].items():
            for name, rec in tb["methods"].items():
                if rec["worst_eligible_abs_gap"] > PROCEED:
                    c1_hits.append(f"{coh}/{tax}/{name}={rec['worst_eligible_abs_gap']:.3f}")
    c2 = sum(1 for coh, blk in rep["cohorts"].items()
             if blk["taxonomies"][A]["methods"]["S+E (full gate)"]["worst_eligible_abs_gap"]
             < blk["taxonomies"][A]["methods"]["S (detector score)"]["worst_eligible_abs_gap"])
    c3_cohorts = ["TJDR", "FGADR"]
    c3 = sum(1 for coh in c3_cohorts
             if rep["cohorts"][coh]["taxonomies"][A]["methods"]["M (top1-top2 margin)"]
             ["worst_eligible_abs_gap"]
             < rep["cohorts"][coh]["taxonomies"][A]["methods"]["S+E (full gate)"]
             ["worst_eligible_abs_gap"])
    rep["criteria"] = {
        "C1_diagnostic": {"met": bool(c1_hits), "hits": c1_hits},
        "C2_gate_beats_score_on_A": {"met": c2 >= 3, "cohorts_won": c2, "of": 4},
        "C3_margin_survives_conditioning": {"met": c3 == 2, "cohorts_won": c3,
                                            "of": len(c3_cohorts)},
    }
    print("\n=== locked criteria ===")
    print(f"  C1 diagnostic                  : {'MET' if c1_hits else 'NOT MET'}"
          f"  ({len(c1_hits)} method-strata above {PROCEED})")
    print(f"  C2 gate beats score on A       : {'MET' if c2 >= 3 else 'NOT MET'}  ({c2}/4)")
    print(f"  C3 margin survives conditioning: {'MET' if c3 == 2 else 'NOT MET'}  ({c3}/2)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
