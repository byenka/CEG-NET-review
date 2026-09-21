#!/usr/bin/env python
"""bootstrap_paired_ci.py — REV-7D α-prime statistical attribution test.

Compute paired bootstrap confidence intervals for KG-on vs no-KG matched
ablation outcomes across multiple seeds. This is the architecture §C3 gate
(vi) attribution test under the α-prime reframed primary metric family
(ECE + AUROC + mAP_50 non-regression), per
``start29/1juni/CLAUDE/12_ALPHA_PRIME_STRATEGY_AND_SOTA_2026-06-02.md``.

Input:
  - --kg_dir   path containing per-seed clinical_metrics.json files of KG-on
                 (filename pattern: ``{kg_dir}/seed{seed}/audit_topk/clinical_metrics.json``)
  - --nokg_dir same for matched no-KG ablation
  - --seeds    list of seeds (e.g., 42 101 202)
  - --metric   metric key (e.g., overall_ece, macro_auroc, ma_he_confusion_rate, ma_auroc, mAP_50)
  - --direction "minimize" or "maximize" — lower is better (ECE, Brier) or higher is better (AUROC, mAP)

Output:
  - Paired delta per seed
  - Bootstrap percentile CI [2.5%, 97.5%]
  - Bootstrap mean
  - Probability that KG_on improves over no-KG (sign-agreement rate)
  - JSON summary

Statistical reasoning: paired bootstrap on seed-level deltas. Resample seeds
with replacement, compute mean delta per resample, build empirical
distribution. CI exclusion of 0 → claim attribution.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _load_metric(
    run_dir: Path,
    metric: str,
) -> Dict[str, float]:
    """Load metric value from a single run's clinical_metrics.json or pareto_summary.json.

    Falls back across multiple JSON paths and metric locations to handle
    different output conventions.

    REV-7D: metric names ending with "_posttemp" are looked up in
    ``posthoc_temp/calibrated_clinical_metrics.json``. The base name (strip
    suffix) is used as the macro lookup key. Example: 'overall_ece_posttemp'
    → look up 'overall_ece' in the posthoc-calibrated metrics.
    """
    # Posthoc temperature scaling lookup.
    if metric.endswith("_posttemp"):
        base_metric = metric[: -len("_posttemp")]
        posthoc_path = run_dir / "posthoc_temp" / "calibrated_clinical_metrics.json"
        if posthoc_path.exists():
            try:
                data = json.loads(posthoc_path.read_text(encoding="utf-8"))
                macro = (data.get("metrics") or {}).get("macro") or {}
                if base_metric in macro:
                    return {
                        "value": float(macro[base_metric]),
                        "source": str(posthoc_path),
                    }
            except Exception:
                pass
        # If posthoc not available, fail-clean.
        raise FileNotFoundError(
            f"Posthoc metric '{metric}' (base='{base_metric}') not found at "
            f"{posthoc_path}. Run tools/posthoc_temperature_scaling.py first."
        )

    candidates = [
        run_dir / "audit_topk" / "clinical_metrics.json",
        run_dir / "audit_topk" / "pareto_summary.json",
        run_dir / "clinical_metrics.json",
    ]
    for jp in candidates:
        if not jp.exists():
            continue
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        # Walk plausible metric keys.
        for parent_key in ("metrics", "winner_metrics", None):
            block = data.get(parent_key, data) if parent_key else data
            if isinstance(block, dict):
                # First, try macro.<metric>.
                macro = block.get("macro") or {}
                if metric in macro:
                    return {"value": float(macro[metric]), "source": str(jp)}
                # Direct lookup.
                if metric in block:
                    return {"value": float(block[metric]), "source": str(jp)}
                # Per-class auroc_auprc (e.g., MA.AUROC).
                per_class = block.get("per_class") or {}
                if "_" in metric:
                    head, tail = metric.split("_", 1)
                    if head in per_class:
                        sub = per_class[head].get("auroc_auprc") or {}
                        if tail in sub:
                            return {"value": float(sub[tail]), "source": str(jp)}
    raise FileNotFoundError(
        f"Metric '{metric}' not found in any of: {candidates}"
    )


def _bootstrap_ci(
    deltas: np.ndarray,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    rng_seed: int = 12345,
) -> Tuple[float, float, float, float]:
    """Percentile bootstrap CI on the mean of deltas.

    Returns (mean, lo, hi, p_positive) where p_positive is the fraction of
    bootstrap resamples whose mean delta is positive (≥ 0).
    """
    rng = np.random.default_rng(rng_seed)
    n = deltas.size
    if n == 0:
        return 0.0, 0.0, 0.0, 0.5
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    resampled = deltas[indices].mean(axis=1)
    mean = float(deltas.mean())
    lo = float(np.percentile(resampled, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(resampled, 100.0 * (1.0 - alpha / 2.0)))
    p_positive = float((resampled > 0).mean())
    return mean, lo, hi, p_positive


def main() -> int:
    p = argparse.ArgumentParser(
        description="REV-7D paired bootstrap CI for KG attribution test."
    )
    p.add_argument(
        "--kg_dir_template",
        required=True,
        help=(
            "Template path with {seed} placeholder for KG-on runs, e.g. "
            "'outputs/mode2_r4_multiseed_seed{seed}_REV7C'"
        ),
    )
    p.add_argument(
        "--nokg_dir_template",
        required=True,
        help="Template path with {seed} placeholder for no-KG matched runs.",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        required=True,
        help="List of seeds, e.g. 42 101 202",
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        required=True,
        help=(
            "List of metrics to test. Common: overall_ece overall_brier "
            "macro_auroc macro_auprc ma_he_confusion_rate ma_auroc"
        ),
    )
    p.add_argument(
        "--directions",
        nargs="+",
        required=True,
        choices=["minimize", "maximize"],
        help=(
            "Per-metric direction. Same length as --metrics. "
            "'minimize' means KG-on should be LOWER than no-KG (e.g., ECE). "
            "'maximize' means KG-on should be HIGHER (e.g., AUROC)."
        ),
    )
    p.add_argument(
        "--n_bootstrap",
        type=int,
        default=2000,
        help="Number of bootstrap resamples (default 2000).",
    )
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--output_json", required=True, type=Path)
    args = p.parse_args()

    if len(args.metrics) != len(args.directions):
        print("[bootstrap] ERROR: --metrics and --directions must have same length.")
        return 2

    results: Dict[str, dict] = {}
    summary_lines: List[str] = []
    for metric, direction in zip(args.metrics, args.directions):
        per_seed = []
        for seed in args.seeds:
            kg_dir = Path(args.kg_dir_template.format(seed=seed))
            nokg_dir = Path(args.nokg_dir_template.format(seed=seed))
            try:
                kg_val = _load_metric(kg_dir, metric)
                nokg_val = _load_metric(nokg_dir, metric)
            except FileNotFoundError as exc:
                print(f"[bootstrap] seed={seed} metric={metric}: {exc}")
                continue
            # Direction convention: positive delta = KG improves.
            if direction == "minimize":
                # Lower is better → KG improves if KG < no-KG → delta = no-KG - KG.
                delta = nokg_val["value"] - kg_val["value"]
            else:
                # Higher is better → KG improves if KG > no-KG → delta = KG - no-KG.
                delta = kg_val["value"] - nokg_val["value"]
            per_seed.append({
                "seed": int(seed),
                "kg_value": float(kg_val["value"]),
                "nokg_value": float(nokg_val["value"]),
                "delta_kg_improves": float(delta),
                "kg_source": str(kg_val.get("source", "")),
                "nokg_source": str(nokg_val.get("source", "")),
            })
        if not per_seed:
            results[metric] = {"error": "no data loaded"}
            summary_lines.append(f"  {metric}: no data")
            continue
        deltas = np.array([s["delta_kg_improves"] for s in per_seed], dtype=np.float64)
        mean, lo, hi, p_pos = _bootstrap_ci(
            deltas, n_bootstrap=args.n_bootstrap, alpha=args.alpha
        )
        ci_excludes_zero = bool((lo > 0.0) or (hi < 0.0))
        verdict = (
            "PASS_ATTRIBUTION_KG_HELPS" if (lo > 0.0)
            else "FAIL_KG_DOES_NOT_HELP" if (hi < 0.0)
            else "MARGINAL_CI_INCLUDES_ZERO"
        )
        results[metric] = {
            "direction": direction,
            "per_seed": per_seed,
            "delta_mean": float(mean),
            "delta_ci_lo": float(lo),
            "delta_ci_hi": float(hi),
            "ci_excludes_zero": ci_excludes_zero,
            "p_positive": float(p_pos),
            "verdict": verdict,
            "n_seeds": int(deltas.size),
            "n_bootstrap": int(args.n_bootstrap),
            "alpha": float(args.alpha),
        }
        summary_lines.append(
            f"  {metric:30s} ({direction:8s}): "
            f"delta_mean={mean:+.4f}  CI=[{lo:+.4f}, {hi:+.4f}]  "
            f"p_pos={p_pos:.3f}  verdict={verdict}"
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps({
            "tool": "bootstrap_paired_ci",
            "seeds": list(args.seeds),
            "metrics": list(args.metrics),
            "directions": list(args.directions),
            "n_bootstrap": int(args.n_bootstrap),
            "alpha": float(args.alpha),
            "results": results,
        }, indent=2),
        encoding="utf-8",
    )
    print()
    print("[bootstrap] === Paired Bootstrap CI Summary ===")
    print(f"[bootstrap] seeds: {args.seeds}")
    print(f"[bootstrap] n_bootstrap: {args.n_bootstrap}  alpha: {args.alpha}")
    print()
    for line in summary_lines:
        print(line)
    print()
    print(f"[bootstrap] wrote: {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
