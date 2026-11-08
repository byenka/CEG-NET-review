#!/usr/bin/env python
"""step7_alpha_sensitivity.py - sensitivity of the coverage-gap comparison to the miscoverage level alpha.

Protocol locked before the run: _LGOVNET_NEW_SEP/PROTOCOL_LOCK_step7_alpha_sensitivity.md
Output: _LGOVNET_NEW_SEP/results/step7_alpha_sensitivity.json
Run from the repository root with the project environment.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import sklearn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.eval_fp_reduction_benchmark import load_csv  # noqa: E402
from tools.eval_selective_conformal_baselines import (  # noqa: E402
    COHORTS, VALID, conformal_threshold, coverage_at,
)

# Scores, the weighted coverage and the interval come from step 3 itself, so the
# alpha = 0.20 control compares like with like.
_spec = importlib.util.spec_from_file_location(
    "step3", ROOT / "_LGOVNET_NEW_SEP/scripts/step3_exact_tp_and_gap_ci.py")
step3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step3)

LOCK = ROOT / "_LGOVNET_NEW_SEP/PROTOCOL_LOCK_step7_alpha_sensitivity.md"
STEP3 = ROOT / "_LGOVNET_NEW_SEP/results/step3_exact_tp_and_gap_ci.json"
OUT = ROOT / "_LGOVNET_NEW_SEP/results/step7_alpha_sensitivity.json"
TASK = "governance_retain"
ALPHAS = (0.10, 0.20, 0.30)
REF_ALPHA = 0.20
SEEDS = step3.SEEDS
N_BOOT = step3.N_BOOT
BOOT_SEED = step3.BOOT_SEED
M0, B4, B6 = step3.M0, step3.B4, step3.B6
GATES = (B4, B6)
TOL = 1e-6


def key(a):
    return f"{a:.2f}"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(report):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


def run_cohort(coh, path, valid, ref):
    test = load_csv(path, positive=TASK)
    lab = test["label"].astype(np.int64)
    sc = step3.scores_for(valid, test)
    thr = {}
    for a in ALPHAS:
        thr[a] = {M0: conformal_threshold(sc[M0]["valid"], valid["label"], a)}
        for m in GATES:
            thr[a][m] = [conformal_threshold(sv, valid["label"], a) for sv in sc[m]["valid"]]

    rec = {"n": int(lab.size), "n_positive": int(lab.sum()), "alphas": {}}
    for a in ALPHAS:
        nom = 1.0 - a
        cov0 = coverage_at(sc[M0]["test"], lab, thr[a][M0])["empirical_coverage"]
        r = {"nominal": nom, "gap": {M0: cov0 - nom}, "coverage_per_seed": {}}
        for m in GATES:
            covs = [coverage_at(st, lab, t)["empirical_coverage"] for st, t in zip(sc[m]["test"], thr[a][m])]
            r["coverage_per_seed"][m] = [float(c) for c in covs]
            r["gap"][m] = float(np.mean(covs)) - nom
        r["closer_than_M0"] = {m: bool(abs(r["gap"][m]) < abs(r["gap"][M0])) for m in GATES}
        rec["alphas"][key(a)] = r

    # point control against step 3, before any bootstrap
    ctrl = {}
    for m in (M0,) + GATES:
        got, want = rec["alphas"][key(REF_ALPHA)]["gap"][m], ref["gap"][m]
        ctrl[m] = {"got": got, "published": want, "abs_diff": abs(got - want), "ok": abs(got - want) <= TOL}
    rec["control_point"] = ctrl
    if not all(c["ok"] for c in ctrl.values()):
        return rec, False

    # paired image-clustered bootstrap, same resampling sequence as step 3
    _, img = np.unique(test["image_id"], return_inverse=True)
    n_img = int(img.max()) + 1
    rng = np.random.default_rng(BOOT_SEED)
    d_gap = {a: {m: [] for m in GATES} for a in ALPHAS}
    t0 = time.time()
    for b in range(N_BOOT):
        w = np.bincount(rng.integers(0, n_img, n_img), minlength=n_img)[img]
        if w[lab == 1].sum() == 0 or w[lab == 0].sum() == 0:
            continue
        for a in ALPHAS:
            nom = 1.0 - a
            g0 = step3.w_cov(sc[M0]["test"], lab, thr[a][M0], w) - nom
            for m in GATES:
                gs = [step3.w_cov(st, lab, thr[a][m][i], w) for i, st in enumerate(sc[m]["test"])]
                d_gap[a][m].append(abs(float(np.mean(gs)) - nom) - abs(g0))
        if (b + 1) % 250 == 0:
            print(f"    {coh} bootstrap {b + 1}/{N_BOOT}  {time.time() - t0:.0f}s", flush=True)
    for a in ALPHAS:
        r = rec["alphas"][key(a)]
        r["abs_gap_advantage"] = {}
        for m in GATES:
            lo, hi = step3.ci(d_gap[a][m])
            r["abs_gap_advantage"][m] = {"point": abs(r["gap"][m]) - abs(r["gap"][M0]), "ci95": [lo, hi],
                                         "ci_below_0": bool(hi < 0)}

    ci_ctrl = {}
    for m in GATES:
        got = rec["alphas"][key(REF_ALPHA)]["abs_gap_advantage"][m]["ci95"]
        want = ref["abs_gap_advantage"][m]["ci95"]
        diff = max(abs(got[0] - want[0]), abs(got[1] - want[1]))
        ci_ctrl[m] = {"got": got, "published": want, "max_abs_diff": diff, "ok": diff <= TOL}
    rec["control_interval"] = ci_ctrl
    return rec, all(c["ok"] for c in ci_ctrl.values())


def summarize(report):
    cohorts = report["cohorts"]
    out = {}
    for m in GATES:
        cells = {}
        for a in ALPHAS:
            cells[key(a)] = {c: cohorts[c]["alphas"][key(a)]["closer_than_M0"][m] for c in cohorts}
        k = sum(v for a in ALPHAS if a != REF_ALPHA for v in cells[key(a)].values())
        n_ci = {key(a): sum(cohorts[c]["alphas"][key(a)]["abs_gap_advantage"][m]["ci_below_0"] for c in cohorts)
                for a in ALPHAS}
        out[m] = {"cells_closer": cells, "k_of_8_at_0.10_and_0.30": int(k), "n_ci_below_0": n_ci}
    report["P1_full_gate"] = {"criterion": "|g_B6| < |g_M0| in all 8 cells (alpha 0.10 and 0.30 x 4 datasets)",
                              "k": out[B6]["k_of_8_at_0.10_and_0.30"],
                              "met": out[B6]["k_of_8_at_0.10_and_0.30"] == 8}
    report["summary"] = out


def main():
    lock_sha = sha256(LOCK)
    ref_all = json.loads(STEP3.read_text(encoding="utf-8"))["tasks"][TASK]["cohorts"]
    report = {"protocol": "_LGOVNET_NEW_SEP/PROTOCOL_LOCK_step7_alpha_sensitivity.md", "protocol_sha256": lock_sha,
              "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "task": TASK, "alphas": list(ALPHAS),
              "seeds": list(SEEDS), "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "sklearn": sklearn.__version__, "executable": sys.executable},
              "cohorts": {}}
    print(f"lock sha256 {lock_sha}", flush=True)
    valid = load_csv(VALID, positive=TASK)
    valid_ok = True
    for coh, path in COHORTS.items():
        print(f"=== {coh}", flush=True)
        rec, ok = run_cohort(coh, path, valid, ref_all[coh])
        report["cohorts"][coh] = rec
        write(report)
        if "control_interval" not in rec:
            report["status"] = f"INVALID: point control failed on {coh}; run stopped before the bootstrap"
            write(report)
            print(report["status"], json.dumps(rec["control_point"], indent=1), flush=True)
            return 1
        valid_ok = valid_ok and ok
    report["status"] = "valid" if valid_ok else "INVALID: interval control failed; no finding is drawn"
    summarize(report)
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    write(report)
    print(f"status {report['status']}")
    print(f"P1 (full gate) k = {report['P1_full_gate']['k']}/8, met = {report['P1_full_gate']['met']}")
    print(f"written {OUT}")
    return 0 if valid_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
