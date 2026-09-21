"""
experiment_runner.py
Minimal experiment orchestrator for LGov-Net — Phase 17 / 17R.

Provides a structured multi-step experiment runner that wraps the existing
ablation infrastructure (src/ablation.py) and integration step
(src/train_step.py) without duplicating any of their logic.

Design principles:
  - Does NOT implement a Trainer class, callback system, or scheduler.
  - Multi-step looping means: call run_single_ablation() N times with
    different detector seeds (different random batches), then aggregate.
  - Coverage is evaluated honestly: partial/non_informative runs are
    NOT inflated to "informative".
  - All heavy lifting is delegated to existing modules.

Phase 17R additions (real-data support):
  - detector_mode awareness (stub / single_head_real / multi_head_sequential /
    cached_predictions) via src/detector_backend.py
  - dry_run_full_dataset flag — report what would run without running it
  - per_split processing (train | valid | test | all)
  - max_images_per_split cap for safe incremental real-data runs
  - detector_mode is reported in every ExperimentResult for audit traceability

Public API:
  load_experiment_config(yaml_path)  -> dict
  evaluate_tier_coverage(tier_counts_sum, coverage_rules) -> dict
  run_single_experiment(case, ...)   -> dict  (one ExperimentResult)
  run_repeated_experiment(case, ...) -> List[dict]
  summarize_experiment_results(results) -> dict
  save_experiment_results(results, summary, out_dir, label) -> None
  # Phase 17R additions:
  run_realdata_dry_run(cfg, splits, max_images_per_split) -> dict
  report_realdata_plan(plan) -> None

Supported experiment modes (MODE constants):
  MODE_BASELINE = "baseline_full"
  MODE_SINGLE   = "single_ablation"
  MODE_SUBSET   = "ablation_suite_subset"

ExperimentResult schema (per run / repeat):
  experiment_label, mode, ablation_name, num_steps, repeat_index,
  status ("OK" | "PARTIAL_OK" | "SKIPPED" | "ERROR"),
  loss_total_mean, loss_sdgl_mean, loss_gate_mean, loss_kg_mean,
  tier_distribution_mean: {R1: float, R2: float, R3: float, R4: float},
  tier_coverage: {R1: bool, R2: bool, R3: bool, R4: bool},
  coverage_status: "informative" | "partial" | "non_informative",
  fallback_sas_count_mean, mean_sdca, mean_aps,
  num_steps_completed, caveats: List[str]
"""

from __future__ import annotations

import os
import sys
import csv
import json
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from typing import Dict, List, Optional

# ── Project root insertion ─────────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, "..")
if _root not in sys.path:
    sys.path.insert(0, _root)

# yaml with fallback guard
try:
    import yaml as _yaml

    _YAML_OK = True
except ImportError:
    _YAML_OK = False

import torch

# ── Module imports — delegate all heavy lifting to existing modules ────────────
from lgovnet.src.ablation import run_single_ablation, load_ablation_cases
from lgovnet.src.train_step import detector_stub, build_gat_encoder
from lgovnet.src.governance.gate import LGG
from lgovnet.src.losses.sdgl import SDGLLoss
from lgovnet.src.losses.gate_loss import GateLoss

# ── Constants ─────────────────────────────────────────────────────────────────
MODE_BASELINE = "baseline_full"
MODE_SINGLE = "single_ablation"
MODE_SUBSET = "ablation_suite_subset"

_TIERS = ["R1", "R2", "R3", "R4"]
_PLACEHOLDER_GT = {
    "img_0001": [{"bbox": [10.0, 10.0, 60.0, 60.0], "class_label": "MA"}],
    "img_0002": [{"bbox": [15.0, 15.0, 65.0, 65.0], "class_label": "Hard_EX"}],
    "img_0003": [{"bbox": [20.0, 20.0, 70.0, 70.0], "class_label": "HE"}],
}


# ══════════════════════════════════════════════════════════════════════════════
# Config loader
# ══════════════════════════════════════════════════════════════════════════════


def load_experiment_config(yaml_path: str) -> dict:
    """Load configs/experiment.yaml.

    Args:
        yaml_path: Path to experiment.yaml (absolute or relative).

    Returns:
        Parsed config dict.

    Raises:
        FileNotFoundError: if yaml_path does not exist.
        ImportError:       if PyYAML is not installed.
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"experiment.yaml not found: {yaml_path}")
    if not _YAML_OK:
        raise ImportError("PyYAML is required. Install via: pip install pyyaml")
    with open(yaml_path, "r", encoding="utf-8") as f:
        return _yaml.safe_load(f)


# ══════════════════════════════════════════════════════════════════════════════
# Coverage evaluator
# ══════════════════════════════════════════════════════════════════════════════


def evaluate_tier_coverage(
    tier_counts_sum: Dict[str, int],
    coverage_rules: Optional[dict] = None,
) -> dict:
    """Classify a run by tier coverage across all its steps.

    Evaluation logic (priority order):
    1. **informative** — all tiers listed in coverage_rules["informative"]["require_tiers"]
       appear (default: R2 AND R3 must both be present).
       Rationale: R2 exercises L_soft^R2 (C3/C1 tests); R3 exercises R3 weighting (C3)
       and refinement (C2). A run without both cannot isolate these contributions.
    2. **partial** — at least 2 distinct tiers seen; the R2+R3 pair is absent
       (e.g. only R3+R4 — common with the synthetic smoke-test batch).
    3. **non_informative** — only 1 distinct tier across all steps.
       baseline_no_governance (all R1 forced) will always be non_informative.
       This is honest: it cannot observe tier distribution effects.

    Args:
        tier_counts_sum: Accumulated bbox counts per tier across all steps.
                         e.g. {"R1": 0, "R2": 0, "R3": 15, "R4": 45}
        coverage_rules:  Section from experiment.yaml["coverage_rules"].
                         If None, uses defaults (require R2+R3 for informative).

    Returns:
        {
          "tier_coverage":  {"R1": bool, "R2": bool, "R3": bool, "R4": bool},
          "coverage_status": "informative" | "partial" | "non_informative",
          "num_tiers_seen": int,
          "r2_present": bool,
          "r3_present": bool,
        }
    """
    # Which tiers actually appeared
    tier_coverage = {t: int(tier_counts_sum.get(t, 0)) > 0 for t in _TIERS}
    num_tiers_seen = sum(1 for v in tier_coverage.values() if v)

    # Required tiers for "informative" (default: R2 and R3)
    require_tiers: List[str] = ["R2", "R3"]
    if coverage_rules and "informative" in coverage_rules:
        req = coverage_rules["informative"].get("require_tiers", ["R2", "R3"])
        if req:
            require_tiers = list(req)

    all_required_present = all(tier_coverage.get(t, False) for t in require_tiers)

    if all_required_present:
        coverage_status = "informative"
    elif num_tiers_seen >= 2:
        coverage_status = "partial"
    else:
        coverage_status = "non_informative"

    return {
        "tier_coverage": tier_coverage,
        "coverage_status": coverage_status,
        "num_tiers_seen": num_tiers_seen,
        "r2_present": bool(tier_coverage.get("R2", False)),
        "r3_present": bool(tier_coverage.get("R3", False)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Core single-experiment runner
# ══════════════════════════════════════════════════════════════════════════════


def run_single_experiment(
    case: dict,
    experiment_label: str,
    mode: str,
    num_steps: int,
    repeat_index: int,
    image_ids: List[str],
    gt_boxes: Dict[str, List[dict]],
    kg_artifacts: dict,
    encoder,
    gate,
    sdgl_fn: SDGLLoss,
    gate_loss_fn: GateLoss,
    base_cfg: Optional[dict] = None,
    coverage_rules: Optional[dict] = None,
    base_seed: int = 42,
    bboxes_per_image: int = 4,
) -> dict:
    """Execute one (case, repeat) experiment — looping num_steps with seed variation.

    Each step uses a fresh synthetic batch from detector_stub() with a
    deterministic but varied seed:
        seed_i = base_seed + step_i + repeat_index * 1000

    Losses and tier distributions are averaged across all steps.

    NOTE on model state: encoder/gate/sdgl_fn/gate_loss_fn are shared across
    all step iterations of this experiment. For ablations that need a fresh
    model (e.g. without_kg_embedding), run_single_ablation() handles this
    internally by creating a fresh LGG instance per call. This means each step
    in that ablation uses a *different* random init. This is expected behaviour
    for a smoke-test / infrastructure test.

    Args:
        case:             One ablation case dict from load_ablation_cases().
        experiment_label: Experiment identifier string.
        mode:             One of MODE_BASELINE, MODE_SINGLE, MODE_SUBSET.
        num_steps:        Number of step iterations.
        repeat_index:     Repeat index (0-based) for seed offset.
        image_ids:        Image IDs to pass to detector_stub.
        gt_boxes:         Fixed GT boxes for routing.
        kg_artifacts:     KG graph artifact dict (static for all steps).
        encoder:          LGovNetEncoder (eval mode).
        gate:             LGG gate (eval mode).
        sdgl_fn:          SDGLLoss instance.
        gate_loss_fn:     GateLoss instance.
        base_cfg:         Base cfg dict merged with ablation cfg_overrides.
        coverage_rules:   Section from experiment.yaml. If None, uses defaults.
        base_seed:        Base random seed.
        bboxes_per_image: Bboxes per image in synthetic batch.

    Returns:
        ExperimentResult dict.  See module docstring for field schema.
    """
    abl_name: str = case.get("name", "unnamed")
    contribution: Optional[str] = case.get("contribution")

    # ── Unsupported / consolidated cases ─────────────────────────────────────
    if not case.get("supported", True):
        return _make_result(
            experiment_label=experiment_label,
            mode=mode,
            abl_name=abl_name,
            contribution=contribution,
            num_steps=0,
            repeat_index=repeat_index,
            status="SKIPPED",
            reason=case.get("note", "supported=false in ablation.yaml"),
        )

    # ── Per-step accumulators ─────────────────────────────────────────────────
    loss_totals: List[float] = []
    loss_sdgls: List[float] = []
    loss_gates: List[float] = []
    loss_kgs: List[float] = []
    tier_counts_sum: Dict[str, int] = {t: 0 for t in _TIERS}
    fallback_sas_counts: List[float] = []
    mean_sdcas: List[float] = []
    mean_apss: List[float] = []
    caveats: List[str] = []

    if case.get("caveat"):
        caveats.append(case["caveat"].strip())

    # ── Multi-step loop (varied seeds → different random batches) ─────────────
    for step_i in range(num_steps):
        seed = base_seed + step_i + repeat_index * 1000
        det_outputs = detector_stub(
            image_ids, bboxes_per_image=bboxes_per_image, seed=seed
        )

        try:
            step_result = run_single_ablation(
                case=case,
                image_ids=image_ids,
                detector_outputs=det_outputs,
                gt_boxes=gt_boxes,
                kg_artifacts=kg_artifacts,
                encoder=encoder,
                gate=gate,
                sdgl_fn=sdgl_fn,
                gate_loss_fn=gate_loss_fn,
                base_cfg=base_cfg or {},
                step=step_i,
            )
        except Exception as exc:
            return _make_result(
                experiment_label=experiment_label,
                mode=mode,
                abl_name=abl_name,
                contribution=contribution,
                num_steps=step_i,
                repeat_index=repeat_index,
                status="ERROR",
                error=f"{type(exc).__name__}: {exc}",
                error_tb=traceback.format_exc(),
            )

        if step_result.get("status") == "ERROR":
            return _make_result(
                experiment_label=experiment_label,
                mode=mode,
                abl_name=abl_name,
                contribution=contribution,
                num_steps=step_i,
                repeat_index=repeat_index,
                status="ERROR",
                error=step_result.get("error", "unknown error in run_single_ablation"),
            )

        if step_result.get("status") == "SKIPPED":
            return _make_result(
                experiment_label=experiment_label,
                mode=mode,
                abl_name=abl_name,
                contribution=contribution,
                num_steps=0,
                repeat_index=repeat_index,
                status="SKIPPED",
                reason=step_result.get("reason", ""),
            )

        # Accumulate step outputs
        loss_totals.append(step_result["loss_total"])
        loss_sdgls.append(step_result["loss_sdgl"])
        loss_gates.append(step_result["loss_gate"])
        loss_kgs.append(step_result["loss_kg"])
        fallback_sas_counts.append(float(step_result.get("fallback_sas_count", 0)))
        mean_sdcas.append(float(step_result.get("mean_sdca", 0.0)))
        mean_apss.append(float(step_result.get("mean_aps", 0.0)))

        # Accumulate tier distribution (per-bbox counts per step)
        td: dict = step_result.get("tier_distribution", {})
        for t in _TIERS:
            tier_counts_sum[t] += int(td.get(t, 0))

        # Capture step-level caveats
        sc = step_result.get("caveat")
        if sc and sc not in caveats:
            caveats.append(sc.strip())

    # ── Aggregate across steps ────────────────────────────────────────────────
    n = max(len(loss_totals), 1)
    loss_total_mean = round(sum(loss_totals) / n, 6)
    loss_sdgl_mean = round(sum(loss_sdgls) / n, 6) if loss_sdgls else None
    loss_gate_mean = round(sum(loss_gates) / n, 6) if loss_gates else None
    loss_kg_mean = round(sum(loss_kgs) / n, 6) if loss_kgs else None
    fallback_mean = (
        round(sum(fallback_sas_counts) / n, 4) if fallback_sas_counts else 0.0
    )
    mean_sdca = round(sum(mean_sdcas) / n, 6) if mean_sdcas else 0.0
    mean_aps = round(sum(mean_apss) / n, 6) if mean_apss else 0.0

    # Average bbox counts per step (fractional is OK — represents mean per batch_step)
    tier_distribution_mean = {t: round(tier_counts_sum[t] / n, 2) for t in _TIERS}

    # ── Coverage assessment ───────────────────────────────────────────────────
    coverage = evaluate_tier_coverage(tier_counts_sum, coverage_rules)

    result = {
        "experiment_label": experiment_label,
        "mode": mode,
        "ablation_name": abl_name,
        "contribution": contribution,
        "num_steps": num_steps,
        "num_steps_completed": len(loss_totals),
        "repeat_index": repeat_index,
        "status": "OK",
        # Loss means
        "loss_total_mean": loss_total_mean,
        "loss_sdgl_mean": loss_sdgl_mean,
        "loss_gate_mean": loss_gate_mean,
        "loss_kg_mean": loss_kg_mean,
        # Tier means and coverage
        "tier_distribution_mean": tier_distribution_mean,
        "tier_coverage": coverage["tier_coverage"],
        "coverage_status": coverage["coverage_status"],
        "num_tiers_seen": coverage["num_tiers_seen"],
        "r2_present": coverage["r2_present"],
        "r3_present": coverage["r3_present"],
        # Evidence quality
        "fallback_sas_count_mean": fallback_mean,
        "mean_sdca": mean_sdca,
        "mean_aps": mean_aps,
        # Caveats from ablation definitions
        "caveats": caveats,
    }
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Repeated experiment runner
# ══════════════════════════════════════════════════════════════════════════════


def run_repeated_experiment(
    case: dict,
    experiment_label: str,
    mode: str,
    num_steps: int,
    num_repeats: int,
    image_ids: List[str],
    gt_boxes: Dict[str, List[dict]],
    kg_artifacts: dict,
    encoder,
    gate,
    sdgl_fn: SDGLLoss,
    gate_loss_fn: GateLoss,
    base_cfg: Optional[dict] = None,
    coverage_rules: Optional[dict] = None,
    base_seed: int = 42,
    bboxes_per_image: int = 4,
    verbose: bool = True,
) -> List[dict]:
    """Run one experiment case num_repeats times.

    Each repeat uses a different seed offset (repeat_index * 1000) so the
    step-level seeds do not overlap between repeats.

    Args:
        (see run_single_experiment for shared args)
        num_repeats: Number of independent repetitions.
        verbose:     Print progress to stdout.

    Returns:
        List of ExperimentResult dicts, one per repeat.
    """
    results: List[dict] = []
    abl_name = case.get("name", "unnamed")

    for r in range(num_repeats):
        if verbose:
            print(
                f"    [repeat {r + 1}/{num_repeats}] {abl_name}"
                f"  steps={num_steps}  repeat_idx={r}"
            )
        result = run_single_experiment(
            case=case,
            experiment_label=experiment_label,
            mode=mode,
            num_steps=num_steps,
            repeat_index=r,
            image_ids=image_ids,
            gt_boxes=gt_boxes,
            kg_artifacts=kg_artifacts,
            encoder=encoder,
            gate=gate,
            sdgl_fn=sdgl_fn,
            gate_loss_fn=gate_loss_fn,
            base_cfg=base_cfg,
            coverage_rules=coverage_rules,
            base_seed=base_seed,
            bboxes_per_image=bboxes_per_image,
        )
        if verbose:
            _print_result_line(result)
        results.append(result)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Summary aggregator
# ══════════════════════════════════════════════════════════════════════════════


def summarize_experiment_results(results: List[dict]) -> dict:
    """Compute aggregate summary over all ExperimentResult dicts.

    Handles mixed modes and ablation names gracefully.

    Returns:
        {
          "total_runs": int,
          "num_steps_run": int,                 # total step calls across all runs
          "ok_count": int,
          "skipped_count": int,
          "error_count": int,
          "num_informative_runs": int,
          "num_partial_runs": int,
          "num_non_informative_runs": int,
          "r2_presence_rate": float,             # fraction of runs with R2 > 0
          "r3_presence_rate": float,
          "loss_total_mean_overall": float,
          "by_ablation": {name: {"mean_loss": float, "coverage": str, ...}},
        }
    """
    total = len(results)
    ok_results = [r for r in results if r.get("status") == "OK"]
    skipped = sum(1 for r in results if r.get("status") == "SKIPPED")
    errors = sum(1 for r in results if r.get("status") == "ERROR")

    informative = sum(
        1 for r in ok_results if r.get("coverage_status") == "informative"
    )
    partial = sum(1 for r in ok_results if r.get("coverage_status") == "partial")
    non_info = sum(
        1 for r in ok_results if r.get("coverage_status") == "non_informative"
    )

    num_steps_run = sum(
        r.get("num_steps_completed", r.get("num_steps", 0)) for r in ok_results
    )

    r2_rate = round(
        sum(1 for r in ok_results if r.get("r2_present", False))
        / max(len(ok_results), 1),
        4,
    )
    r3_rate = round(
        sum(1 for r in ok_results if r.get("r3_present", False))
        / max(len(ok_results), 1),
        4,
    )

    losses = [
        r["loss_total_mean"] for r in ok_results if r.get("loss_total_mean") is not None
    ]
    loss_overall = round(sum(losses) / len(losses), 6) if losses else None

    # Per-ablation breakdown
    by_ablation: dict = {}
    for r in ok_results:
        name = r.get("ablation_name", "unknown")
        if name not in by_ablation:
            by_ablation[name] = {
                "runs": 0,
                "loss_total_values": [],
                "coverage_statuses": [],
                "contribution": r.get("contribution"),
            }
        by_ablation[name]["runs"] += 1
        by_ablation[name]["loss_total_values"].append(r.get("loss_total_mean", 0.0))
        by_ablation[name]["coverage_statuses"].append(r.get("coverage_status"))

    # Collapse per-ablation lists to scalar summaries
    by_ablation_summary: dict = {}
    for name, d in by_ablation.items():
        vals = d["loss_total_values"]
        mean_l = round(sum(vals) / len(vals), 6) if vals else None
        dominant_cov = max(
            set(d["coverage_statuses"]), key=d["coverage_statuses"].count
        )
        by_ablation_summary[name] = {
            "contribution": d["contribution"],
            "runs": d["runs"],
            "loss_total_mean": mean_l,
            "dominant_coverage_status": dominant_cov,
        }

    return {
        "total_runs": total,
        "num_steps_run": num_steps_run,
        "ok_count": len(ok_results),
        "skipped_count": skipped,
        "error_count": errors,
        "num_informative_runs": informative,
        "num_partial_runs": partial,
        "num_non_informative_runs": non_info,
        "r2_presence_rate": r2_rate,
        "r3_presence_rate": r3_rate,
        "loss_total_mean_overall": loss_overall,
        "by_ablation": by_ablation_summary,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Output serialiser
# ══════════════════════════════════════════════════════════════════════════════

#: CSV column order for the experiment summary table.
_EXP_CSV_COLUMNS = [
    "experiment_label",
    "mode",
    "ablation_name",
    "contribution",
    "repeat_index",
    "num_steps",
    "status",
    "coverage_status",
    "tier_R1",
    "tier_R2",
    "tier_R3",
    "tier_R4",
    "r2_present",
    "r3_present",
    "loss_total_mean",
    "loss_sdgl_mean",
    "loss_gate_mean",
    "loss_kg_mean",
    "fallback_sas_count_mean",
    "mean_sdca",
    "mean_aps",
    "caveats",
    "error",
]


def save_experiment_results(
    results: List[dict],
    summary: dict,
    out_dir: str,
    label: str,
) -> None:
    """Persist experiment results to JSON + CSV files.

    Output files:
        <out_dir>/experiment_results.json   — full result list + summary + meta
        <out_dir>/experiment_summary.csv    — flat CSV for tabular inspection

    Args:
        results:  List of ExperimentResult dicts.
        summary:  Output of summarize_experiment_results().
        out_dir:  Directory to write to (created if absent).
        label:    Experiment label string embedded in filenames and JSON meta.
    """
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ── JSON ──────────────────────────────────────────────────────────────────
    json_path = os.path.join(out_dir, "experiment_results.json")
    payload = {
        "meta": {
            "experiment_label": label,
            "phase": "Phase 17",
            "timestamp": ts,
        },
        "summary": summary,
        "results": results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, "experiment_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_EXP_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row: dict = {col: r.get(col, "") for col in _EXP_CSV_COLUMNS}
            td = r.get("tier_distribution_mean") or {}
            row["tier_R1"] = td.get("R1", "")
            row["tier_R2"] = td.get("R2", "")
            row["tier_R3"] = td.get("R3", "")
            row["tier_R4"] = td.get("R4", "")
            cav = r.get("caveats", [])
            row["caveats"] = "; ".join(c[:80] for c in cav) if cav else ""
            writer.writerow(row)

    print(f"  [EXP] Saved JSON -> {json_path}")
    print(f"  [EXP] Saved CSV  -> {csv_path}")


# ══════════════════════════════════════════════════════════════════════════════
# KG artifact bootstrap (local copy — shared pattern with run_ablation_suite)
# ══════════════════════════════════════════════════════════════════════════════


def build_synthetic_kg_artifacts() -> dict:
    """Return a minimal synthetic KG artifact dict with N=9 nodes.

    Identical structure to the demo in run_ablation_suite.py and
    train_step.__main__ smoke test.
    """
    node_names = [
        "MA",
        "HE",
        "Hard_EX",
        "Soft_EX",
        "Foveal",
        "Parafoveal",
        "Peripapillary",
        "Vascular_Arcades",
        "Peripheral",
    ]
    node_features: dict = {}
    for i, nm in enumerate(node_names):
        feat_dim = 17 if i < 4 else 9
        feat = [0.0] * feat_dim
        feat[i % feat_dim] = 1.0
        node_features[nm] = feat

    nodes = [{"name": nm, "idx": i} for i, nm in enumerate(node_names)]
    _edge_pairs = [(0, 1), (0, 2), (1, 2), (2, 3), (0, 3), (1, 3)]
    edges = {
        "morphological_confusability": [
            {"src": s, "dst": d, "weight": 0.5, "directed": False}
            for s, d in _edge_pairs
        ],
        "severity_transition": [
            {"src": 0, "dst": 1, "weight": 0.1, "directed": False},
            {"src": 0, "dst": 3, "weight": 2.0, "directed": False},
        ],
        "spatial_distribution": [
            {"src": 0, "dst": 4, "weight": 0.3, "directed": False},
            {"src": 2, "dst": 4, "weight": 0.8, "directed": False},
        ],
        "pathophysiological_mechanism": [
            {"src": 0, "dst": 1, "weight": 0.4, "directed": True},
        ],
        "co_occurrence_pattern": [
            {"src": 1, "dst": 2, "weight": 0.2, "directed": False},
        ],
    }
    return {"nodes": nodes, "edges": edges, "node_features": node_features, "stats": {}}


# ══════════════════════════════════════════════════════════════════════════════
# Private helpers
# ══════════════════════════════════════════════════════════════════════════════


def _make_result(
    experiment_label: str,
    mode: str,
    abl_name: str,
    contribution: Optional[str],
    num_steps: int,
    repeat_index: int,
    status: str,
    reason: str = "",
    error: str = "",
    error_tb: str = "",
) -> dict:
    """Build a minimal ExperimentResult for non-OK outcomes."""
    result = {
        "experiment_label": experiment_label,
        "mode": mode,
        "ablation_name": abl_name,
        "contribution": contribution,
        "num_steps": num_steps,
        "num_steps_completed": 0,
        "repeat_index": repeat_index,
        "status": status,
        "loss_total_mean": None,
        "loss_sdgl_mean": None,
        "loss_gate_mean": None,
        "loss_kg_mean": None,
        "tier_distribution_mean": None,
        "tier_coverage": None,
        "coverage_status": None,
        "num_tiers_seen": None,
        "r2_present": False,
        "r3_present": False,
        "fallback_sas_count_mean": None,
        "mean_sdca": None,
        "mean_aps": None,
        "caveats": [],
    }
    if reason:
        result["reason"] = reason
    if error:
        result["error"] = error
    if error_tb:
        result["traceback"] = error_tb
    return result


def _print_result_line(result: dict) -> None:
    """Print a one-line summary of an ExperimentResult."""
    status = result.get("status", "?")
    name = result.get("ablation_name", "?")
    if status == "OK":
        lt = result.get("loss_total_mean")
        cov = result.get("coverage_status", "?")
        r2 = "R2+" if result.get("r2_present") else "R2-"
        r3 = "R3+" if result.get("r3_present") else "R3-"
        print(
            f"      -> OK  loss={lt:.6f}  coverage={cov} [{r2} {r3}]"
            + (f"  [caveat]" if result.get("caveats") else "")
        )
    elif status == "SKIPPED":
        print(f"      -> SKIPPED: {result.get('reason', '')[:60]}")
    else:
        print(f"      -> {status}: {result.get('error', '')[:80]}")


def _json_default(obj):
    """JSON serialiser fallback for non-standard types."""
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 17.5 — Scenario-based experiment modes
# ══════════════════════════════════════════════════════════════════════════════

#: New mode constants for targeted scenario validation (Phase 17.5).
MODE_SMOKE = "smoke_synthetic"  # alias — same as existing detector_stub path
MODE_TARGETED = "targeted_scenarios"  # run each ScenarioBundle individually
MODE_MIXED = "mixed_tier_validation"  # run only the mixed_batch scenario


def run_scenario_experiment(
    scenario,  # ScenarioBundle (not typed to avoid circular; imported lazily)
    baseline_case: dict,
    experiment_label: str,
    mode: str,
    kg_artifacts: dict,
    encoder,
    gate,
    sdgl_fn: SDGLLoss,
    gate_loss_fn: GateLoss,
    base_cfg: Optional[dict] = None,
    coverage_rules: Optional[dict] = None,
) -> dict:
    """Run one ScenarioBundle through the pipeline → ExperimentResult dict.

    Unlike run_single_experiment (which loops over multiple detector_stub seeds),
    this function runs a single deterministic pass:
      1. Merges base_cfg with scenario.cfg_override (for routing_thresholds etc.)
      2. Calls run_single_ablation with scenario's detector_outputs + gt_boxes
      3. Evaluates tier coverage and builds an ExperimentResult

    Args:
        scenario:          ScenarioBundle from scenario_bank.py
        baseline_case:     Ablation case dict to use (typically baseline_full).
        experiment_label:  Label string.
        mode:              One of MODE_TARGETED, MODE_MIXED.
        kg_artifacts:      Synthetic KG artifacts dict.
        encoder:           LGovNetEncoder (eval mode).
        gate:              LGG gate (eval mode).
        sdgl_fn:           SDGLLoss instance.
        gate_loss_fn:      GateLoss instance.
        base_cfg:          Base cfg from experiment.yaml.
        coverage_rules:    Coverage evaluation rules.

    Returns:
        ExperimentResult dict with the same schema as run_single_experiment.
        Additional keys:
          - scenario_expected_tiers: List[str]
          - scenario_notes: List[str]
          - tier_match: bool  (True if all expected_tiers appeared in distribution)
    """
    scn_name: str = getattr(scenario, "scenario_name", "unnamed_scenario")
    expected_tiers: List[str] = getattr(scenario, "expected_tiers", [])

    # Merge base_cfg with scenario's routing/cfg override
    merged_cfg: dict = {**(base_cfg or {}), **getattr(scenario, "cfg_override", {})}

    try:
        step_result = run_single_ablation(
            case=baseline_case,
            image_ids=list(scenario.image_ids),
            detector_outputs=dict(scenario.detector_outputs),
            gt_boxes=dict(scenario.gt_boxes),
            kg_artifacts=kg_artifacts,
            encoder=encoder,
            gate=gate,
            sdgl_fn=sdgl_fn,
            gate_loss_fn=gate_loss_fn,
            base_cfg=merged_cfg,
            step=0,
        )
    except Exception as exc:
        return _make_result(
            experiment_label=experiment_label,
            mode=mode,
            abl_name=scn_name,
            contribution="scenario",
            num_steps=0,
            repeat_index=0,
            status="ERROR",
            error=f"{type(exc).__name__}: {exc}",
            error_tb=traceback.format_exc(),
        )

    if step_result.get("status") == "SKIPPED":
        return _make_result(
            experiment_label=experiment_label,
            mode=mode,
            abl_name=scn_name,
            contribution="scenario",
            num_steps=0,
            repeat_index=0,
            status="SKIPPED",
            reason=step_result.get("reason", ""),
        )

    if step_result.get("status") == "ERROR":
        return _make_result(
            experiment_label=experiment_label,
            mode=mode,
            abl_name=scn_name,
            contribution="scenario",
            num_steps=0,
            repeat_index=0,
            status="ERROR",
            error=step_result.get("error", "unknown"),
        )

    # Unpack tier distribution from the single ablation step
    tier_dist: Dict[str, int] = step_result.get("tier_distribution") or {}
    tier_counts_sum: Dict[str, int] = {t: int(tier_dist.get(t, 0)) for t in _TIERS}

    # Coverage evaluation
    coverage = evaluate_tier_coverage(tier_counts_sum, coverage_rules)

    # Tier match: did all expected tiers actually appear?
    tier_match = all(tier_counts_sum.get(t, 0) > 0 for t in expected_tiers)

    tier_distribution_mean = {t: float(tier_counts_sum[t]) for t in _TIERS}

    caveats: List[str] = []
    if baseline_case.get("caveat"):
        caveats.append(str(baseline_case["caveat"]).strip())
    if not tier_match:
        missing = [t for t in expected_tiers if tier_counts_sum.get(t, 0) == 0]
        caveats.append(f"TIER_MISMATCH: expected {expected_tiers}, missing {missing}")

    result = {
        "experiment_label": experiment_label,
        "mode": mode,
        "ablation_name": scn_name,
        "contribution": "scenario",
        "num_steps": 1,
        "num_steps_completed": 1,
        "repeat_index": 0,
        "status": "OK",
        # Loss scalars (single step, no averaging)
        "loss_total_mean": round(float(step_result.get("loss_total") or 0), 6),
        "loss_sdgl_mean": round(float(step_result.get("loss_sdgl") or 0), 6),
        "loss_gate_mean": round(float(step_result.get("loss_gate") or 0), 6),
        "loss_kg_mean": round(float(step_result.get("loss_kg") or 0), 6),
        # Tier metrics
        "tier_distribution_mean": tier_distribution_mean,
        "tier_coverage": coverage["tier_coverage"],
        "coverage_status": coverage["coverage_status"],
        "num_tiers_seen": coverage["num_tiers_seen"],
        "r2_present": coverage["r2_present"],
        "r3_present": coverage["r3_present"],
        # Evidence quality
        "fallback_sas_count_mean": float(step_result.get("fallback_sas_count") or 0),
        "mean_sdca": round(float(step_result.get("mean_sdca") or 0), 6),
        "mean_aps": round(float(step_result.get("mean_aps") or 0), 6),
        # Scenario-specific metadata
        "scenario_expected_tiers": expected_tiers,
        "scenario_notes": list(getattr(scenario, "notes", [])),
        "tier_match": tier_match,
        "caveats": caveats,
    }
    return result


def run_scenario_suite(
    scenarios: List,  # List[ScenarioBundle]
    baseline_case: dict,
    experiment_label: str,
    mode: str,
    kg_artifacts: dict,
    encoder,
    gate,
    sdgl_fn: SDGLLoss,
    gate_loss_fn: GateLoss,
    base_cfg: Optional[dict] = None,
    coverage_rules: Optional[dict] = None,
    verbose: bool = True,
) -> List[dict]:
    """Run a list of ScenarioBundles through run_scenario_experiment.

    Args:
        scenarios:  Ordered list of ScenarioBundle objects.
        (remaining args: same as run_scenario_experiment)
        verbose:    Print per-scenario progress and tier distribution.

    Returns:
        List of ExperimentResult dicts, one per scenario.
    """
    results: List[dict] = []
    for scn in scenarios:
        scn_name = getattr(scn, "scenario_name", "unnamed")
        exp_tiers = getattr(scn, "expected_tiers", [])
        if verbose:
            override_note = "+" if getattr(scn, "cfg_override", {}) else " "
            print(f"    [{override_note}] {scn_name:<46}  expected={exp_tiers}")

        with torch.no_grad():
            result = run_scenario_experiment(
                scenario=scn,
                baseline_case=baseline_case,
                experiment_label=experiment_label,
                mode=mode,
                kg_artifacts=kg_artifacts,
                encoder=encoder,
                gate=gate,
                sdgl_fn=sdgl_fn,
                gate_loss_fn=gate_loss_fn,
                base_cfg=base_cfg,
                coverage_rules=coverage_rules,
            )

        if verbose:
            _print_scenario_result_line(result)
        results.append(result)

    return results


def _print_scenario_result_line(result: dict) -> None:
    """Print a detailed one-line summary for a scenario result."""
    status = result.get("status", "?")
    if status != "OK":
        print(
            f"      -> {status}: {result.get('error', result.get('reason', ''))[:80]}"
        )
        return

    lt = result.get("loss_total_mean")
    cov = result.get("coverage_status", "?")
    td = result.get("tier_distribution_mean") or {}
    match = "PASS" if result.get("tier_match") else "FAIL"
    r1 = int(td.get("R1", 0))
    r2 = int(td.get("R2", 0))
    r3 = int(td.get("R3", 0))
    r4 = int(td.get("R4", 0))
    caveats = result.get("caveats", [])
    caveat_flag = "  [!] " + caveats[0][:50] if caveats else ""

    print(
        f"      -> OK  loss={lt:.4f}  coverage={cov:<16}"
        f"  tiers=[R1:{r1} R2:{r2} R3:{r3} R4:{r4}]"
        f"  tier_match={match}{caveat_flag}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Phase 17R — Real-data dry-run and planning utilities
# ══════════════════════════════════════════════════════════════════════════════

_SPLIT_NAMES = ("train", "valid", "test")

_DATA_IMAGE_ROOT = os.path.join(_root, "data", "images")
_DATA_XML_ROOT = os.path.join(_root, "data", "xml")


def run_realdata_dry_run(
    cfg: Optional[dict] = None,
    splits: Optional[List[str]] = None,
    max_images_per_split: Optional[int] = None,
) -> dict:
    """Compute what a real-data run would process, without running any model.

    Scans the data directories, counts paired image/XML files per split,
    applies the max_images_per_split cap, and reports the execution plan.
    No model, encoder, or detector is instantiated.

    Args:
        cfg:                   Experiment config dict.  Used to resolve
                               ``max_images_per_split`` and ``detector_mode``
                               when not overridden by direct arguments.
        splits:                List of split names to include.
                               If None, uses ["train", "valid", "test"].
        max_images_per_split:  Cap on number of images per split.
                               Overrides cfg["max_images_per_split"] if given.

    Returns:
        {
          "detector_mode":        str,
          "splits_planned":       {split: {"total_paired": int,
                                           "will_process": int,
                                           "capped": bool}},
          "total_images_planned": int,
          "dry_run":              True,
          "warnings":             List[str],
        }
    """
    from lgovnet.src.detector_backend import get_detector_mode, describe_detector_mode

    cfg = cfg or {}
    resolved_splits: List[str] = list(splits or _SPLIT_NAMES)
    detector_mode = get_detector_mode(cfg)
    mode_info = describe_detector_mode(detector_mode)

    # Resolve max_images_per_split: direct arg > cfg > None (no cap)
    cap: Optional[int] = max_images_per_split
    if cap is None:
        cfg_cap = cfg.get("max_images_per_split")
        if cfg_cap is not None:
            cap = int(cfg_cap)

    warnings_list: List[str] = []
    splits_planned: dict = {}

    for split in resolved_splits:
        img_dir = os.path.join(_DATA_IMAGE_ROOT, split)
        xml_dir = os.path.join(_DATA_XML_ROOT, split)

        if not os.path.isdir(img_dir):
            warnings_list.append(f"[MISSING] data/images/{split}/ not found")
            splits_planned[split] = {
                "total_paired": 0,
                "will_process": 0,
                "capped": False,
                "missing": True,
            }
            continue
        if not os.path.isdir(xml_dir):
            warnings_list.append(f"[MISSING] data/xml/{split}/ not found")
            splits_planned[split] = {
                "total_paired": 0,
                "will_process": 0,
                "capped": False,
                "missing": True,
            }
            continue

        img_stems = {
            os.path.splitext(f)[0]
            for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        }
        xml_stems = {
            os.path.splitext(f)[0]
            for f in os.listdir(xml_dir)
            if f.lower().endswith(".xml")
        }
        paired = sorted(img_stems & xml_stems)
        total_paired = len(paired)

        if cap is not None and total_paired > cap:
            will_process = cap
            capped = True
        else:
            will_process = total_paired
            capped = False

        splits_planned[split] = {
            "total_paired": total_paired,
            "will_process": will_process,
            "capped": capped,
            "first_n_sample": paired[:3],  # representative sample stems
        }

    if not mode_info.get("implemented", False):
        warnings_list.append(
            f"[WARN] detector_mode={detector_mode!r} is NOT implemented. "
            f"Real-data run will raise NotImplementedError unless mode is changed to "
            f"'stub' or 'cached_predictions'."
        )

    total_images_planned = sum(
        s.get("will_process", 0) for s in splits_planned.values()
    )

    return {
        "detector_mode": detector_mode,
        "detector_implemented": mode_info.get("implemented", False),
        "detector_gpu_required": mode_info.get("gpu_required", False),
        "detector_description": mode_info.get("description", ""),
        "splits_planned": splits_planned,
        "total_images_planned": total_images_planned,
        "max_images_per_split": cap,
        "dry_run": True,
        "warnings": warnings_list,
    }


def report_realdata_plan(plan: dict, verbose: bool = True) -> None:
    """Print a human-readable summary of a dry-run plan.

    Args:
        plan:    Output dict from run_realdata_dry_run().
        verbose: If False, only prints one-line summary.
    """
    mode = plan.get("detector_mode", "?")
    implemented = plan.get("detector_implemented", False)
    impl_tag = "IMPLEMENTED" if implemented else "NOT IMPLEMENTED"
    total = plan.get("total_images_planned", 0)
    cap = plan.get("max_images_per_split")

    print("=" * 60)
    print("  Phase 17R — Real-Data Execution Plan (DRY RUN)")
    print("=" * 60)
    print(f"  detector_mode  : {mode}  [{impl_tag}]")
    print(f"  description    : {plan.get('detector_description', '')[:72]}")
    print(f"  total planned  : {total} images")
    if cap:
        print(f"  cap per split  : {cap}")
    print()

    if verbose:
        for split, info in plan.get("splits_planned", {}).items():
            if info.get("missing"):
                print(f"  {split:<8}  [MISSING directory]")
                continue
            capped = "  (capped)" if info.get("capped") else ""
            sample = info.get("first_n_sample", [])
            sample_str = ", ".join(sample[:3]) + ("..." if len(sample) >= 3 else "")
            print(
                f"  {split:<8}  {info.get('will_process', 0):>4} / "
                f"{info.get('total_paired', 0):>4} images{capped}"
                f"  sample=[{sample_str}]"
            )
        print()

    for w in plan.get("warnings", []):
        print(f"  [!] {w}")

    print("=" * 60)
