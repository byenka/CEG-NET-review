#!/usr/bin/env python
"""step3_exact_tp_and_gap_ci.py - exact-TP sensitivity, paired coverage-gap intervals, pointwise risk dominance.

Protocol locked before the run: _LGOVNET_NEW_SEP/PROTOCOL_LOCK_step3_exact_tp_and_gap_ci.md
Output: _LGOVNET_NEW_SEP/results/step3_exact_tp_and_gap_ci.json
Run from the repository root with the project environment.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.eval_fp_reduction_benchmark import auprc, build_methods, load_csv  # noqa: E402
from tools.eval_selective_conformal_baselines import (  # noqa: E402
    COHORTS, VALID, aurc, conformal_threshold, coverage_at,
)

OUT = ROOT / "_LGOVNET_NEW_SEP/results/step3_exact_tp_and_gap_ci.json"
ALPHA = 0.20
NOMINAL = 1.0 - ALPHA
SEEDS = (0, 1, 2, 3, 4)
N_BOOT = 2000
BOOT_SEED = 0
M0, B4, B6 = "M0_score", "KG-gate-MLP (ours)", "KG-gate+SDS+corrob (B6)"
GRID = np.round(np.arange(1, 101) / 100.0, 2)


def ci(v):
    v = np.asarray(v, dtype=np.float64)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def w_curve_stats(order, label, w):
    """Expand the resample along a fixed score order; return (AURC, all-points trapezoid AUPRC)."""
    idx = np.repeat(order, w[order])
    y = label[idx].astype(np.float64)
    k = np.arange(1, y.size + 1, dtype=np.float64)
    neg = np.cumsum(1.0 - y)
    a = float((neg / k).mean())
    tp = np.cumsum(y)
    n_pos = max(float(y.sum()), 1.0)
    rec = np.concatenate(([0.0], tp / n_pos))
    prec = np.concatenate(([1.0], tp / np.maximum(k, 1.0)))
    return a, float(np.trapezoid(prec, rec))


def w_cov(score, label, thr, w):
    pos = label == 1
    den = float(w[pos].sum())
    return float(w[pos & (score >= thr)].sum()) / den if den > 0 else float("nan")


def risk_at_grid(score, label):
    order = np.argsort(-score, kind="mergesort")
    y = label[order].astype(np.float64)
    k = np.arange(1, y.size + 1, dtype=np.float64)
    risk = np.cumsum(1.0 - y) / k
    pos = np.clip(np.ceil(GRID * y.size).astype(int) - 1, 0, y.size - 1)
    return risk[pos]


def scores_for(valid, test):
    out = {M0: {"valid": None, "test": None}, B4: {"valid": [], "test": []}, B6: {"valid": [], "test": []}}
    for seed in SEEDS:
        for name, _fam, sv, st in build_methods(valid, test, mlp_seed=seed):
            if name == M0 and out[M0]["test"] is None:
                out[M0] = {"valid": sv, "test": st}
            elif name in (B4, B6):
                out[name]["valid"].append(sv)
                out[name]["test"].append(st)
    return out


def run_cohort(coh, path, task, valid, pointwise):
    test = load_csv(path, positive=task)
    lab = test["label"].astype(np.int64)
    sc = scores_for(valid, test)
    thr = {M0: conformal_threshold(sc[M0]["valid"], valid["label"], ALPHA)}
    for m in (B4, B6):
        thr[m] = [conformal_threshold(sv, valid["label"], ALPHA) for sv in sc[m]["valid"]]

    # point estimates, computed exactly as the published tables do
    rec = {"n": int(lab.size), "n_positive": int(lab.sum()), "prevalence": float(lab.mean())}
    cov_m0 = coverage_at(sc[M0]["test"], lab, thr[M0])["empirical_coverage"]
    rec["gap"] = {M0: cov_m0 - NOMINAL}
    rec["aurc"] = {M0: aurc(sc[M0]["test"], lab)}
    rec["auprc"] = {M0: auprc(sc[M0]["test"], lab)}
    for m in (B4, B6):
        covs = [coverage_at(st, lab, t)["empirical_coverage"] for st, t in zip(sc[m]["test"], thr[m])]
        rec["gap"][m] = float(np.mean(covs)) - NOMINAL
        rec["aurc"][m] = float(np.mean([aurc(st, lab) for st in sc[m]["test"]]))
        aps = [auprc(st, lab) for st in sc[m]["test"]]
        rec["auprc"][m] = float(np.mean(aps))
        rec.setdefault("auprc_per_seed", {})[m] = [float(x) for x in aps]

    # paired image-clustered bootstrap, thresholds and gates fixed
    _, img = np.unique(test["image_id"], return_inverse=True)
    n_img = int(img.max()) + 1
    orders = {M0: np.argsort(-sc[M0]["test"], kind="mergesort")}
    for m in (B4, B6):
        orders[m] = [np.argsort(-st, kind="mergesort") for st in sc[m]["test"]]
    rng = np.random.default_rng(BOOT_SEED)
    d_gap = {B4: [], B6: []}
    d_aurc = {B4: [], B6: []}
    d_ap_seed = {B4: [[] for _ in SEEDS], B6: [[] for _ in SEEDS]}
    t0 = time.time()
    for b in range(N_BOOT):
        w = np.bincount(rng.integers(0, n_img, n_img), minlength=n_img)[img]
        if w[lab == 1].sum() == 0 or w[lab == 0].sum() == 0:
            continue
        g0 = w_cov(sc[M0]["test"], lab, thr[M0], w) - NOMINAL
        a0, ap0 = w_curve_stats(orders[M0], lab, w)
        for m in (B4, B6):
            gs, as_, aps = [], [], []
            for i, st in enumerate(sc[m]["test"]):
                gs.append(w_cov(st, lab, thr[m][i], w))
                a_i, ap_i = w_curve_stats(orders[m][i], lab, w)
                as_.append(a_i)
                d_ap_seed[m][i].append(ap_i - ap0)
            d_gap[m].append(abs(float(np.mean(gs)) - NOMINAL) - abs(g0))
            d_aurc[m].append(float(np.mean(as_)) - a0)
        if (b + 1) % 250 == 0:
            print(f"    {coh} [{task}] bootstrap {b + 1}/{N_BOOT}  {time.time() - t0:.0f}s", flush=True)
    rec["abs_gap_advantage"] = {m: {"point": abs(rec["gap"][m]) - abs(rec["gap"][M0]), "ci95": ci(d_gap[m])} for m in (B4, B6)}
    rec["aurc_difference"] = {m: {"point": rec["aurc"][m] - rec["aurc"][M0], "ci95": ci(d_aurc[m])} for m in (B4, B6)}
    rec["auprc_gain_per_seed"] = {}
    for m in (B4, B6):
        rows = []
        for i, s in enumerate(SEEDS):
            lo, hi = ci(d_ap_seed[m][i])
            rows.append({"seed": s, "gain": rec["auprc_per_seed"][m][i] - rec["auprc"][M0], "ci95": [lo, hi]})
        gains = [r["gain"] for r in rows]
        rec["auprc_gain_per_seed"][m] = {
            "per_seed": rows, "mean": float(np.mean(gains)), "sd": float(np.std(gains, ddof=1)),
            "n_ci_above_0": sum(r["ci95"][0] > 0 for r in rows), "n_ci_below_0": sum(r["ci95"][1] < 0 for r in rows),
        }
    if pointwise:
        r0 = risk_at_grid(sc[M0]["test"], lab)
        rec["pointwise"] = {}
        for m in (B4, B6):
            rm = np.mean([risk_at_grid(st, lab) for st in sc[m]["test"]], axis=0)
            lower = rm < r0
            rec["pointwise"][m] = {
                "grid_points": int(GRID.size), "gate_lower": int(lower.sum()),
                "gate_higher_or_equal_at": [float(c) for c in GRID[~lower]],
                "dominates_on_grid": bool(lower.all()),
            }
    return rec


def main():
    report = {"protocol": "_LGOVNET_NEW_SEP/PROTOCOL_LOCK_step3_exact_tp_and_gap_ci.md", "alpha": ALPHA,
              "seeds": list(SEEDS), "n_boot": N_BOOT, "boot_seed": BOOT_SEED, "tasks": {}}
    for task, pointwise in (("governance_retain", True), ("tp_exact", False)):
        valid = load_csv(VALID, positive=task)
        report["tasks"][task] = {"n_valid": int(valid["label"].size), "n_valid_positive": int(valid["label"].sum()),
                                 "cohorts": {}}
        for coh, path in COHORTS.items():
            print(f"=== {task} {coh}", flush=True)
            report["tasks"][task]["cohorts"][coh] = run_cohort(coh, path, task, valid, pointwise)
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
