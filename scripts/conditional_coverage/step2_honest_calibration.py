#!/usr/bin/env python
"""step2_honest_calibration.py — re-run the operating-point result with the gate
fitting data and the threshold calibration data made disjoint.

Protocol locked before execution:
  _LGOVNET_NEW_SEP/PROTOCOL_LOCK_step2_honest_calibration.md

OIA-DDR validation is split BY IMAGE (never by detection). The gate is fitted on one
fold and frozen; the threshold is calibrated on the other fold; every method, fitted or
not, is calibrated on that same held-out fold. The frozen threshold is then applied
unchanged to OIA-DDR test, IDRiD, TJDR and FGADR.

This is a split by image with a finite-sample quantile correction. It is NOT an exact
conformal guarantee: detections inside one image are dependent, so the exchangeable unit
is closer to the image than to the detection.

Output: _LGOVNET_NEW_SEP/results/step2_honest_calibration.json
Run from the repo root:
  python _LGOVNET_NEW_SEP/scripts/step2_honest_calibration.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.eval_fp_reduction_benchmark import (  # noqa: E402
    EVIDENCE_FEATURES, apply_logreg, apply_mlp, auprc, fit_logreg, fit_mlp, load_csv,
)
from tools.eval_selective_conformal_baselines import (  # noqa: E402
    ALPHA, COHORTS, TASK, VALID, aurc, conformal_threshold,
)

OUT = ROOT / "_LGOVNET_NEW_SEP/results/step2_honest_calibration.json"
NOMINAL = 1.0 - ALPHA
SEEDS = (0, 1, 2, 3, 4)
N_BOOT = 2000
BOOT_SEED = 0
AURC_BOOT_MAX_N = 120_000      # same convention as the published tables


def fold_of(image_id: str) -> int:
    """Seed-free deterministic image-level split. Fold A = 0, fold B = 1."""
    return int(hashlib.sha256(image_id.encode()).hexdigest()[:8], 16) % 2


def inputs(d: dict, kind: str) -> np.ndarray:
    ev = [d[f] for f in EVIDENCE_FEATURES]
    if kind == "EL":                                     # evidence only, no score
        return np.stack(ev, 1)
    if kind == "KG":
        return np.stack(ev + [d["pred_score"]], 1)
    if kind == "FG":
        return np.stack(ev + [d["pred_score"], d["ev_sds"], d["ev_sds_nsrc"]], 1)
    raise ValueError(kind)


def sub(d: dict, mask: np.ndarray) -> dict:
    return {k: v[mask] for k, v in d.items()}


def coverage_stats(score, label, thr):
    keep = score >= thr
    pos = label == 1
    n_pos = int(pos.sum())
    cov = float((keep & pos).sum() / n_pos) if n_pos else float("nan")
    return cov, float(keep.mean()), n_pos


def image_clustered_boot(image_id, score, label, thr, n_boot=N_BOOT, seed=BOOT_SEED):
    """CI on the coverage gap by resampling images. Uses per-image sufficient
    statistics, so it is exact and cheap even on FGADR."""
    keep = (score >= thr) & (label == 1)
    pos = label == 1
    uniq, inv = np.unique(image_id, return_inverse=True)
    k = np.bincount(inv, weights=keep.astype(np.float64), minlength=uniq.size)
    p = np.bincount(inv, weights=pos.astype(np.float64), minlength=uniq.size)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, uniq.size, size=(n_boot, uniq.size))
    num = k[idx].sum(1)
    den = p[idx].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        gaps = num / den - NOMINAL
    gaps = gaps[np.isfinite(gaps)]
    return [float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))]


def aurc_boot(image_id, score, label, n_boot=500, seed=BOOT_SEED):
    uniq, inv = np.unique(image_id, return_inverse=True)
    order = [np.flatnonzero(inv == i) for i in range(uniq.size)]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, uniq.size, size=uniq.size)
        rows = np.concatenate([order[i] for i in pick])
        vals.append(aurc(score[rows], label[rows]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def build_scores(fit_d: dict, targets: dict, seed: int) -> dict:
    """Fit every parametric method on fit_d only, then score fit_d and each target."""
    y = fit_d["label"].astype(np.float64)
    out = {}
    out["S (detector score)"] = {k: v["pred_score"] for k, v in targets.items()}
    out["S (detector score)"]["__fit__"] = fit_d["pred_score"]
    out["M (top1-top2 margin)"] = {k: v["unc_margin"] for k, v in targets.items()}
    out["M (top1-top2 margin)"]["__fit__"] = fit_d["unc_margin"]
    for kind, name in [("EL", "EL (evidence, linear)"), ("KG", "KG gate"),
                       ("FG", "FG (full gate)")]:
        if kind == "EL":
            par = fit_logreg(inputs(fit_d, kind), y)
            f = lambda d, p=par, kk=kind: apply_logreg(inputs(d, kk), p)
        else:
            par = fit_mlp(inputs(fit_d, kind), y, seed=seed)
            f = lambda d, p=par, kk=kind: apply_mlp(inputs(d, kk), p)
        out[name] = {k: f(v) for k, v in targets.items()}
        out[name]["__fit__"] = f(fit_d)
    return out


def run_config(valid: dict, cohorts: dict, fit_mask: np.ndarray,
               cal_mask: np.ndarray, label: str, want_ci: bool) -> dict:
    fit_d, cal_d = sub(valid, fit_mask), sub(valid, cal_mask)
    methods = ["S (detector score)", "M (top1-top2 margin)", "EL (evidence, linear)",
               "KG gate", "FG (full gate)"]
    acc = {m: {c: {"cov": [], "gap": [], "ret": [], "aurc": [], "auprc": [], "thr": []}
               for c in cohorts} for m in methods}
    for seed in SEEDS:
        sc = build_scores(fit_d, dict(cohorts, **{"__cal__": cal_d}), seed)
        for m in methods:
            thr = conformal_threshold(sc[m]["__cal__"], cal_d["label"], ALPHA)
            for c, d in cohorts.items():
                cov, ret, _ = coverage_stats(sc[m][c], d["label"], thr)
                acc[m][c]["cov"].append(cov)
                acc[m][c]["gap"].append(cov - NOMINAL)
                acc[m][c]["ret"].append(ret)
                acc[m][c]["aurc"].append(aurc(sc[m][c], d["label"]))
                acc[m][c]["auprc"].append(auprc(sc[m][c], d["label"]))
                acc[m][c]["thr"].append(thr)
    block = {"label": label, "n_fit_images": int(np.unique(fit_d["image_id"]).size),
             "n_cal_images": int(np.unique(cal_d["image_id"]).size),
             "n_fit": int(fit_mask.sum()), "n_cal": int(cal_mask.sum()),
             "n_cal_positive": int(cal_d["label"].sum()), "methods": {}}
    for m in methods:
        block["methods"][m] = {}
        for c, d in cohorts.items():
            a = acc[m][c]
            rec = {"coverage": float(np.mean(a["cov"])),
                   "gap": float(np.mean(a["gap"])),
                   "abs_gap": float(abs(np.mean(a["gap"]))),
                   "gap_sd": float(np.std(a["gap"])),
                   "retained_fraction": float(np.mean(a["ret"])),
                   "aurc": float(np.mean(a["aurc"])),
                   "auprc": float(np.mean(a["auprc"])),
                   "threshold": float(np.mean(a["thr"]))}
            if want_ci:
                sc1 = build_scores(fit_d, {c: d}, SEEDS[0])[m][c]
                thr1 = a["thr"][0]
                rec["gap_ci95"] = image_clustered_boot(d["image_id"], sc1,
                                                       d["label"], thr1)
                if d["label"].size <= AURC_BOOT_MAX_N:
                    rec["aurc_ci95"] = aurc_boot(d["image_id"], sc1, d["label"])
            block["methods"][m][c] = rec
    return block


def main() -> None:
    valid = load_csv(VALID, positive=TASK)
    folds = np.array([fold_of(i) for i in valid["image_id"]])
    A, B = folds == 0, folds == 1
    cohorts = {c: load_csv(p, positive=TASK) for c, p in COHORTS.items()}

    rep = {"alpha": ALPHA, "nominal": NOMINAL, "task": TASK, "seeds": list(SEEDS),
           "n_boot": N_BOOT,
           "protocol": "_LGOVNET_NEW_SEP/PROTOCOL_LOCK_step2_honest_calibration.md",
           "split_rule": "sha256(image_id) parity; image-level, never detection-level",
           "guarantee_note": ("image-disjoint split with a finite-sample quantile "
                              "correction; NOT an exact conformal guarantee, because "
                              "detections within one image are dependent"),
           "configs": {}}

    for key, (fm, cm, lab, ci) in {
        "H1": (A, B, "gate on fold A, threshold on fold B (honest)", True),
        "H1p": (B, A, "gate on fold B, threshold on fold A (honest, swapped)", True),
        "H2": (A, A, "gate on fold A, threshold on fold A (in-sample control)", False),
    }.items():
        print(f"\n===== {key}: {lab}", flush=True)
        rep["configs"][key] = run_config(valid, cohorts, fm, cm, lab, ci)
        for m, per in rep["configs"][key]["methods"].items():
            cells = "  ".join(f"{c.split()[0][:7]}:{per[c]['gap']:+.3f}" for c in cohorts)
            print(f"   {m:24s} {cells}")

    # mean of the two honest directions = the primary reported number
    prim = {}
    for m in rep["configs"]["H1"]["methods"]:
        prim[m] = {}
        for c in cohorts:
            r1 = rep["configs"]["H1"]["methods"][m][c]
            r2 = rep["configs"]["H1p"]["methods"][m][c]
            prim[m][c] = {k: float(np.mean([r1[k], r2[k]]))
                          for k in ("coverage", "gap", "retained_fraction", "aurc",
                                    "auprc")}
            prim[m][c]["abs_gap"] = abs(prim[m][c]["gap"])
    rep["primary_mean_of_honest_directions"] = prim

    # ---- locked criteria
    gates = ["KG gate", "FG (full gate)"]
    others = ["M (top1-top2 margin)", "EL (evidence, linear)"]
    S = "S (detector score)"

    def d1(m):
        return [c for c in cohorts if prim[m][c]["abs_gap"] < prim[S][c]["abs_gap"]]

    def d2(m):
        return [c for c in cohorts if prim[m][c]["aurc"] < prim[S][c]["aurc"]]

    det = {m: {"D1_cohorts": d1(m), "D2_cohorts": d2(m)} for m in gates + others}
    D1 = all(len(det[m]["D1_cohorts"]) == 4 for m in gates)
    D2 = all(len(det[m]["D2_cohorts"]) == 4 for m in gates)
    D3 = not any(len(det[m]["D1_cohorts"]) == 4 and len(det[m]["D2_cohorts"]) == 4
                 for m in others)
    if D1 and D2 and D3:
        verdict = "A_CLAIM_SURVIVES"
    else:
        ok = [m for m in gates
              if len(det[m]["D2_cohorts"]) == 4 and len(det[m]["D1_cohorts"]) >= 2]
        both_d1_le1 = all(len(det[m]["D1_cohorts"]) <= 1 for m in gates)
        other_dom = any(len(det[m]["D1_cohorts"]) == 4 and len(det[m]["D2_cohorts"]) == 4
                        for m in others)
        verdict = "C_CLAIM_FAILS" if (both_d1_le1 or other_dom) else (
            "B_CLAIM_PARTIALLY_SURVIVES" if ok else "C_CLAIM_FAILS")
    rep["criteria"] = {"D1_both_gates_beat_S_on_abs_gap_all4": D1,
                       "D2_both_gates_beat_S_on_aurc_all4": D2,
                       "D3_no_unfitted_method_does_both_all4": D3,
                       "detail": det, "verdict": verdict}

    # ---- secondary: magnitude of the in-sample calibration effect, fit budget fixed
    sec = {}
    for m in rep["configs"]["H1"]["methods"]:
        sec[m] = {c: {"H2_in_sample_gap": rep["configs"]["H2"]["methods"][m][c]["gap"],
                      "H1_honest_gap": rep["configs"]["H1"]["methods"][m][c]["gap"],
                      "H2_minus_H1": rep["configs"]["H2"]["methods"][m][c]["gap"]
                      - rep["configs"]["H1"]["methods"][m][c]["gap"]}
                  for c in cohorts}
    rep["secondary_in_sample_effect"] = sec

    print("\n===== locked criteria")
    for k in ("D1_both_gates_beat_S_on_abs_gap_all4",
              "D2_both_gates_beat_S_on_aurc_all4",
              "D3_no_unfitted_method_does_both_all4"):
        print(f"   {k}: {rep['criteria'][k]}")
    for m in gates + others:
        print(f"   {m:24s} D1 in {len(det[m]['D1_cohorts'])}/4  "
              f"D2 in {len(det[m]['D2_cohorts'])}/4")
    print(f"   VERDICT: {verdict}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
