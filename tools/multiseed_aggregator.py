#!/usr/bin/env python
"""multiseed_aggregator.py — REV-7D α-prime multi-seed result aggregator.

Aggregate clinical_metrics.json + pareto_summary.json across multiple seed
runs (e.g., 42, 101, 202) into a single summary table per run group.

Output:
  - CSV with seed-level metrics (one row per seed per group)
  - JSON with mean/std/min/max per metric per group
  - Cross-group comparison row (e.g., R4 KG-on mean vs R4.1 no-KG mean)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


_DEFAULT_METRICS = [
    "mAP_50",
    "false_positive_pressure",
    "MA_AP50",
    "macro_auroc",
    "macro_auprc",
    "macro_sens_at_spec_0p95",
    "macro_recall_at_fp_1",
    "overall_ece",
    "overall_brier",
    "ma_he_confusion_rate",
    "ma_auroc",
    "ma_auprc",
]


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _gather_metrics_one_seed(run_dir: Path) -> Dict[str, float]:
    """Return metric_key → value dict for a single seed run."""
    out: Dict[str, float] = {}

    # Pareto summary holds winner metrics (mAP_50, MA_AP50, FPpr, mAP_medium).
    ps = _read_json(run_dir / "audit_topk" / "pareto_summary.json") or {}
    winner = ps.get("winner_metrics") or {}
    out["mAP_50"] = float(winner.get("mAP_50", 0.0))
    out["mAP_medium"] = float(winner.get("mAP_medium", 0.0))
    out["false_positive_pressure"] = float(winner.get("false_positive_pressure", 0.0))
    out["MA_AP50"] = float(winner.get("MA_AP50", 0.0))
    if "mode2a_nonreg_pass" in ps:
        out["mode2a_nonreg_pass"] = bool(ps["mode2a_nonreg_pass"])

    # Clinical metrics hold macro + per-class.
    cm = _read_json(run_dir / "audit_topk" / "clinical_metrics.json") or {}
    macro = (cm.get("metrics") or {}).get("macro") or {}
    for k in [
        "macro_auroc",
        "macro_auprc",
        "macro_sens_at_spec_0p95",
        "macro_recall_at_fp_1",
        "overall_ece",
        "overall_brier",
        "ma_he_confusion_rate",
        "ma_he_forward_rate",
        "ma_he_reverse_rate",
        "ma_auroc",
        "ma_auprc",
    ]:
        if k in macro:
            out[k] = float(macro[k])

    # Post-hoc temperature scaled metrics (optional).
    pt = _read_json(run_dir / "posthoc_temp" / "calibrated_clinical_metrics.json") or {}
    pt_macro = (pt.get("metrics") or {}).get("macro") or {}
    if pt_macro:
        out["overall_ece_posttemp"] = float(pt_macro.get("overall_ece", 0.0))
        out["overall_brier_posttemp"] = float(pt_macro.get("overall_brier", 0.0))
        out["macro_auroc_posttemp"] = float(pt_macro.get("macro_auroc", 0.0))
    return out


def _mean_std(xs: List[float]) -> Tuple[float, float, float, float]:
    if not xs:
        return 0.0, 0.0, 0.0, 0.0
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / n
    sd = var ** 0.5
    return float(m), float(sd), float(min(xs)), float(max(xs))


def _aggregate_group(
    template: str,
    seeds: List[int],
    metrics: List[str],
) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, object]]]:
    """Aggregate across seeds. Returns (per_metric_stats, per_seed_rows)."""
    per_seed_rows: List[Dict[str, object]] = []
    collected: Dict[str, List[float]] = {m: [] for m in metrics}
    for seed in seeds:
        run_dir = Path(template.format(seed=seed))
        row = {"seed": int(seed), "run_dir": str(run_dir)}
        gm = _gather_metrics_one_seed(run_dir)
        for m in metrics:
            if m in gm:
                row[m] = float(gm[m])
                collected[m].append(float(gm[m]))
            else:
                row[m] = None
        per_seed_rows.append(row)
    per_metric_stats: Dict[str, Dict[str, float]] = {}
    for m, vals in collected.items():
        mean, sd, mn, mx = _mean_std(vals)
        per_metric_stats[m] = {
            "n": int(len(vals)),
            "mean": mean,
            "std": sd,
            "min": mn,
            "max": mx,
        }
    return per_metric_stats, per_seed_rows


def main() -> int:
    p = argparse.ArgumentParser(
        description="REV-7D multi-seed aggregator for α-prime claim."
    )
    p.add_argument(
        "--groups",
        nargs="+",
        required=True,
        help=(
            "Group definitions in the form 'name:template'. Template uses "
            "{seed} placeholder. Example: "
            "kg_on:outputs/mode2_r4_multiseed_seed{seed}_REV7C "
            "no_kg:outputs/mode2_r4_multiseed_noKG_seed{seed}_REV7C"
        ),
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        required=True,
        help="List of seeds (e.g., 42 101 202).",
    )
    p.add_argument(
        "--metrics",
        nargs="*",
        default=_DEFAULT_METRICS,
        help=f"Metrics to aggregate. Default: {_DEFAULT_METRICS}",
    )
    p.add_argument("--output_dir", required=True, type=Path)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    parsed_groups: List[Tuple[str, str]] = []
    for g in args.groups:
        if ":" not in g:
            print(f"[multiseed] ERROR: group spec '{g}' missing ':' separator.")
            return 2
        name, template = g.split(":", 1)
        parsed_groups.append((name, template))

    all_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    all_rows: Dict[str, List[Dict[str, object]]] = {}
    for name, template in parsed_groups:
        stats, rows = _aggregate_group(template, args.seeds, args.metrics)
        all_stats[name] = stats
        all_rows[name] = rows

    # Write per-seed CSV.
    csv_path = args.output_dir / "per_seed.csv"
    fieldnames = ["group", "seed", "run_dir"] + list(args.metrics)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, rows in all_rows.items():
            for r in rows:
                rec = {"group": name}
                rec.update({k: r.get(k) for k in fieldnames if k != "group"})
                writer.writerow(rec)

    # Write summary JSON.
    summary = {
        "tool": "multiseed_aggregator",
        "seeds": list(args.seeds),
        "metrics": list(args.metrics),
        "groups": {name: {"template": tmpl} for name, tmpl in parsed_groups},
        "per_seed_rows": all_rows,
        "per_metric_stats_per_group": all_stats,
    }
    # Cross-group delta if exactly 2 groups (e.g., KG-on vs no-KG).
    if len(parsed_groups) == 2:
        g_a, g_b = parsed_groups[0][0], parsed_groups[1][0]
        delta_block: Dict[str, Dict[str, float]] = {}
        for m in args.metrics:
            a = all_stats[g_a].get(m, {})
            b = all_stats[g_b].get(m, {})
            delta_block[m] = {
                f"{g_a}_mean": float(a.get("mean", 0.0)),
                f"{g_b}_mean": float(b.get("mean", 0.0)),
                f"delta_{g_a}_minus_{g_b}": float(a.get("mean", 0.0))
                - float(b.get("mean", 0.0)),
                f"{g_a}_std": float(a.get("std", 0.0)),
                f"{g_b}_std": float(b.get("std", 0.0)),
            }
        summary["cross_group_delta"] = delta_block

    json_path = args.output_dir / "multiseed_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Stdout summary.
    print()
    print("[multiseed] === Multi-Seed Aggregation Summary ===")
    print(f"[multiseed] seeds: {args.seeds}")
    print(f"[multiseed] groups: {[n for n, _ in parsed_groups]}")
    print()
    for name in [n for n, _ in parsed_groups]:
        print(f"[multiseed] Group: {name}")
        for m in args.metrics:
            s = all_stats[name].get(m, {})
            if s and s.get("n", 0) > 0:
                print(
                    f"  {m:32s} n={s['n']}  mean={s['mean']:+.4f}  "
                    f"std={s['std']:.4f}  min={s['min']:+.4f}  max={s['max']:+.4f}"
                )
            else:
                print(f"  {m:32s} no data")
        print()
    if "cross_group_delta" in summary:
        print(f"[multiseed] Cross-group delta ({g_a} - {g_b}):")
        for m, b in summary["cross_group_delta"].items():
            print(
                f"  {m:32s} delta_mean={b[f'delta_{g_a}_minus_{g_b}']:+.4f}"
            )
        print()
    print(f"[multiseed] wrote CSV : {csv_path}")
    print(f"[multiseed] wrote JSON: {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
