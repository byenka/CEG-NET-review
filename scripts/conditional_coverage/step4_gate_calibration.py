"""Step 4 (exploratory): expected calibration error and Brier score of the gate outputs.

Locked design: _LGOVNET_NEW_SEP/PROTOCOL_LOCK_step4_gate_calibration.md. Descriptive only.

The three published variants of the detector score are recomputed first and compared with
outputs/0506/selective_conformal/baselines.json, the frozen source of Supplementary S16. Any difference above 1e-9
aborts the run, so the new numbers are only read once the published ones reproduce.

Nothing here is written to the manuscript folders.
"""
import gc
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.eval_fp_reduction_benchmark import (  # noqa: E402
    apply_temperature, build_methods, fit_temperature, load_csv,
)
from tools.eval_selective_conformal_baselines import (  # noqa: E402
    COHORTS, SEEDS, TASK, VALID, brier, ece, to_unit,
)

OUT = ROOT / "_LGOVNET_NEW_SEP/results/step4_gate_calibration.json"
PUBLISHED = ROOT / "outputs/0506/selective_conformal/baselines.json"
LEARNED = ("Evidence-LogReg", "Score-MLP", "KG-gate-MLP (ours)", "KG-gate+SDS (B5)", "KG-gate+SDS+corrob (B6)")
TOL = 1e-9


def stats(p: np.ndarray, lab: np.ndarray) -> dict:
    p = np.asarray(p, dtype=np.float64)
    return {
        "ece": ece(p, lab), "brier": brier(p, lab),
        "mean_predicted": float(p.mean()), "observed_rate": float(lab.mean()),
        "mean_minus_observed": float(p.mean() - lab.mean()),
        "outside_unit_interval": int(((p < 0.0) | (p > 1.0)).sum()),
    }


def agg(rows: list[dict]) -> dict:
    out = {"n_seeds": len(rows)}
    for k in ("ece", "brier", "mean_predicted", "mean_minus_observed"):
        v = np.array([r[k] for r in rows], dtype=np.float64)
        out[f"{k}_mean"] = float(v.mean())
        out[f"{k}_sd"] = float(v.std())
    out["observed_rate"] = rows[0]["observed_rate"]
    out["per_seed_ece"] = [r["ece"] for r in rows]
    out["per_seed_brier"] = [r["brier"] for r in rows]
    out["outside_unit_interval"] = int(max(r["outside_unit_interval"] for r in rows))
    return out


def main() -> None:
    published = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    valid = load_csv(VALID, positive=TASK)
    vlab = valid["label"].astype(np.float64)
    T = fit_temperature(valid["pred_score"], valid["label"])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(valid["pred_score"], vlab)
    tname = f"temperature (T={T:.2f})"
    print(f"validation n={vlab.size:,} positives={int(vlab.sum()):,} | temperature T={T:.4f}", flush=True)

    report = {
        "step": "step4_gate_calibration",
        "protocol": "_LGOVNET_NEW_SEP/PROTOCOL_LOCK_step4_gate_calibration.md",
        "task": TASK, "seeds": list(SEEDS), "ece_bins": 15,
        "note": ("Raw and temperature variants are min-max rescaled by to_unit, as published. Gate and other learned "
                 "outputs are read directly as probabilities. The raw score is not fitted to the RETAIN target."),
        "validation_in_sample": {}, "cohorts": {},
    }
    valid_captured = False

    for coh, path in COHORTS.items():
        if not path.exists():
            report["cohorts"][coh] = {"error": f"missing: {path}"}
            continue
        test = load_csv(path, positive=TASK)
        lab = test["label"].astype(np.float64)
        entry = {"n": int(lab.size), "n_positive": int(lab.sum()), "prevalence": float(lab.mean()),
                 "published_variants": {}, "methods": {}}
        print(f"\n=== {coh}  n={lab.size:,}  positives={int(lab.sum()):,}  prevalence={lab.mean():.4f}", flush=True)

        variants = {
            "raw score": to_unit(test["pred_score"]),
            tname: to_unit(apply_temperature(test["pred_score"], T)),
            "isotonic": np.clip(iso.predict(test["pred_score"]), 0.0, 1.0),
        }
        ref_all = published["cohorts"][coh]["calibration"]
        for vname, p in variants.items():
            s = stats(p, lab)
            ref = ref_all.get(vname)
            if ref is None:
                sys.exit(f"ABORT: {coh}: published calibration has no variant named {vname!r} ({list(ref_all)})")
            for k in ("ece", "brier"):
                if abs(s[k] - ref[k]) > TOL:
                    sys.exit(f"ABORT reproduction: {coh} {vname} {k}: recomputed {s[k]:.10f} vs published {ref[k]:.10f}")
            entry["published_variants"][vname] = s
            print(f"  [published] {vname:22s} ECE={s['ece']:.4f}  Brier={s['brier']:.4f}  (reproduced)", flush=True)

        per = {}
        for seed in SEEDS:
            # Per-seed scores are reduced to their summary immediately, so no score array is held across seeds.
            for name, family, sv, st in build_methods(valid, test, mlp_seed=seed):
                if name not in LEARNED:
                    continue
                d = per.setdefault(name, {"family": family, "test": [], "valid": []})
                d["test"].append(stats(st, lab))
                if not valid_captured:
                    d["valid"].append(stats(sv, vlab))
            gc.collect()
        iso_ece = entry["published_variants"]["isotonic"]["ece"]
        for name in LEARNED:
            d = per[name]
            a = agg(d["test"])
            a["family"] = d["family"]
            a["ece_minus_isotonic"] = a["ece_mean"] - iso_ece
            entry["methods"][name] = a
            print(f"  {name:26s} ECE={a['ece_mean']:.4f}+-{a['ece_sd']:.4f}  Brier={a['brier_mean']:.4f}"
                  f"  mean(p)-rate={a['mean_minus_observed_mean']:+.4f}  ECE-isotonic={a['ece_minus_isotonic']:+.4f}",
                  flush=True)
        if not valid_captured:
            vin = {"n": int(vlab.size), "n_positive": int(vlab.sum()), "prevalence": float(vlab.mean()),
                   "published_variants": {
                       "raw score": stats(to_unit(valid["pred_score"]), vlab),
                       tname: stats(to_unit(apply_temperature(valid["pred_score"], T)), vlab),
                       "isotonic": stats(np.clip(iso.predict(valid["pred_score"]), 0.0, 1.0), vlab),
                   },
                   "methods": {n: agg(per[n]["valid"]) for n in LEARNED}}
            report["validation_in_sample"] = vin
            valid_captured = True
        report["cohorts"][coh] = entry

    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwritten {OUT}")


if __name__ == "__main__":
    main()
