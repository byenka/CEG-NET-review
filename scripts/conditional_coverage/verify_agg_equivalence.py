"""Proof that the memory-leaner aggregation changed no number.

IDRiD is recomputed with the ORIGINAL logic, which holds every seed's score array and reduces them only at the end,
and the result is compared with the values stored by the leaner run in step4_gate_calibration.json.
Any difference above 0 is printed. Read-only for the manuscript folders.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.eval_fp_reduction_benchmark import build_methods, load_csv  # noqa: E402
from tools.eval_selective_conformal_baselines import COHORTS, SEEDS, TASK, VALID, brier, ece  # noqa: E402

LEARNED = ("Evidence-LogReg", "Score-MLP", "KG-gate-MLP (ours)", "KG-gate+SDS (B5)", "KG-gate+SDS+corrob (B6)")
COH = "IDRiD"

valid = load_csv(VALID, positive=TASK)
test = load_csv(COHORTS[COH], positive=TASK)
lab = test["label"].astype(np.float64)

# ---- original logic: keep every array, reduce at the end
held = {}
for seed in SEEDS:
    for name, _family, _sv, st in build_methods(valid, test, mlp_seed=seed):
        if name in LEARNED:
            held.setdefault(name, []).append(np.asarray(st, dtype=np.float64))
old = {n: {"ece": [ece(p, lab) for p in a], "brier": [brier(p, lab) for p in a]} for n, a in held.items()}

new = json.loads((ROOT / "_LGOVNET_NEW_SEP/results/step4_gate_calibration.json").read_text(encoding="utf-8"))["cohorts"][COH]["methods"]
worst = 0.0
for n in LEARNED:
    for k in ("ece", "brier"):
        a = np.array(old[n][k], dtype=np.float64)
        b = np.array(new[n][f"per_seed_{k}"], dtype=np.float64)
        d_seed = float(np.max(np.abs(a - b)))
        d_mean = abs(float(a.mean()) - new[n][f"{k}_mean"])
        worst = max(worst, d_seed, d_mean)
        print(f"{n:26s} {k:5s} per-seed max|diff|={d_seed:.3e}  mean diff={d_mean:.3e}")
print(f"\nlargest difference over all methods, seeds and metrics: {worst:.3e}")
print("IDENTICAL" if worst == 0.0 else "DIFFERENT")
