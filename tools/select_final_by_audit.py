#!/usr/bin/env python
"""select_final_by_audit.py — Mode 2 v2.1 Pareto-front final selector.

Reads the top-k candidate checkpoints written by ``TopKCheckpointManager``,
runs ``audit_detector_stage1a.py`` on each, then selects the final winner
using either a Pareto-front criterion (default) or a priority-ordered tiebreak.

Usage::

    python tools/select_final_by_audit.py \\
        --run_dir outputs/mode2_v2_1_<RUN_TAG>/v2_standard_seed42 \\
        --data_root data/oiaddr --split valid --score_thresh 0.02 \\
        --baseline_audit outputs/detector_eval/r5b_baseline/stage1a_raw.json \\
        --strategy pareto_then_rank \\
        --output_dir <run_dir>/audit_topk \\
        [--prune_losers]

Result (under ``--output_dir``)::

    audit_<criterion>/stage1a_raw.json         (one per candidate)
    pareto_summary.json                        (Pareto front + selection)
    selected_best.pt                           (copy of winner)
    selected_best.json                         (winner metadata)

Selection strategies::

    pareto_then_rank   : keep only non-dominated checkpoints on
                          {mAP_50, mAP_medium, MA_AP50, -FPpr}; among them
                          pick the one with lowest sum-of-ranks across the
                          same axes. (default)
    map50_only         : pick highest audit mAP_50.
    composite_v2       : pick lowest baseline-centered composite score
                          (mirror of training-time selection but on audit).
    mode2a_primary     : (REV-7) Mode 2A primary-metric-family selection per
                          architecture §C3 gate (v)+(vi). Filter candidates by
                          non-regression bounds vs baseline:
                            mAP_50 Δ >= -0.020,  FPpr Δ <= +0.20.
                          Among survivors, rank by MA_AP50 descending; tiebreak
                          by FPpr ascending. If no candidate passes the non-
                          regression filter, the run is INELIGIBLE for a
                          positive Mode 2A claim and the strategy falls back to
                          ``pareto_then_rank`` with the verdict flag
                          ``mode2a_nonreg_pass=false`` so downstream auditors
                          can reject the run automatically.

Exit code 0 on success; 1 if no candidate completed audit.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_here = Path(__file__).resolve().parent
_root = _here.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


CANDIDATE_NAMES = ("composite", "map50", "map_medium", "ma_ap50")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit-based Pareto-front Mode 2 v2.1 final selector.")
    p.add_argument("--run_dir", required=True, type=Path)
    p.add_argument("--data_root", required=True, type=Path)
    p.add_argument("--split", default="valid")
    p.add_argument("--score_thresh", type=float, default=0.02)
    p.add_argument("--detections_per_img", type=int, default=200)
    p.add_argument(
        "--baseline_audit",
        type=Path,
        default=Path("outputs/detector_eval/r5b_baseline/stage1a_raw.json"),
    )
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument(
        "--strategy",
        choices=[
            "pareto_then_rank",
            "map50_only",
            "composite_v2",
            "mode2a_primary",
        ],
        default="pareto_then_rank",
    )
    # REV-7 (Mode 2A primary): non-regression bounds.
    p.add_argument(
        "--mode2a_map50_min_delta",
        type=float,
        default=-0.020,
        help="Mode 2A non-regression: mAP_50 candidate >= baseline + this delta.",
    )
    p.add_argument(
        "--mode2a_fppr_max_delta",
        type=float,
        default=0.20,
        help="Mode 2A non-regression: FPpr candidate <= baseline + this delta.",
    )
    p.add_argument("--candidates", nargs="*", default=list(CANDIDATE_NAMES))
    p.add_argument("--prune_losers", action="store_true",
                   help="Delete non-winning candidate checkpoints after selection.")
    p.add_argument("--require_cuda", action="store_true",
                   help="Hard-fail audit if CUDA is not available (recommended on server).")
    return p.parse_args()


def _run_audit(
    checkpoint: Path,
    out_dir: Path,
    data_root: Path,
    split: str,
    score_thresh: float,
    detections_per_img: int,
    require_cuda: bool,
) -> Optional[Dict]:
    """Invoke audit_detector_stage1a.py and return parsed stage1a_raw.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(_root / "tools" / "audit_detector_stage1a.py"),
        "--checkpoint", str(checkpoint),
        "--model_type", "fasterrcnn",
        "--data_root", str(data_root),
        "--split", split,
        "--score_thresh", str(score_thresh),
        "--output_dir", str(out_dir),
    ]
    log_path = out_dir / "audit_stdout.log"
    with open(log_path, "w", encoding="utf-8") as log_f:
        ret = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT)
    raw = out_dir / "stage1a_raw.json"
    if not raw.exists():
        print(f"[select] WARN: audit failed for {checkpoint} (no stage1a_raw.json); see {log_path}")
        return None
    data = json.loads(raw.read_text(encoding="utf-8"))
    if require_cuda:
        # Best-effort: try to assert the audit ran on CUDA by looking at the log.
        try:
            text = log_path.read_text(encoding="utf-8")
            if "device=cuda" not in text and "device=cuda:0" not in text:
                print(f"[select] WARN: audit log does not show device=cuda for {checkpoint}")
        except OSError:
            pass
    return data


def _extract_metrics(audit: Dict) -> Dict[str, float]:
    m = audit.get("map_metrics", {}) or {}
    per_class = audit.get("per_class", {}) or {}
    ap_per_class = audit.get("ap_per_class", {}) or {}
    ma_ap50 = 0.0
    # the audit JSON typically encodes per_class entries with AP50 fields.
    if isinstance(per_class, dict):
        ma = per_class.get("MA") or per_class.get("1") or {}
        if isinstance(ma, dict):
            ma_ap50 = float(ma.get("AP50") or ma.get("ap50") or 0.0)
    if ma_ap50 == 0.0 and isinstance(ap_per_class, dict):
        ma_ap50 = float(ap_per_class.get("MA") or ap_per_class.get("1") or 0.0)
    return {
        "mAP_50": float(m.get("mAP_50", 0.0) or 0.0),
        "mAP_75": float(m.get("mAP_75", 0.0) or 0.0),
        "mAP_small": float(m.get("mAP_small", 0.0) or 0.0),
        "mAP_medium": float(m.get("mAP_medium", 0.0) or 0.0),
        "mAP_large": float(m.get("mAP_large", 0.0) or 0.0),
        "false_positive_pressure": float(audit.get("false_positive_pressure", 0.0) or 0.0),
        "MA_AP50": ma_ap50,
    }


# ─── Pareto-front + ranking ──────────────────────────────────────────────────


def _pareto_front(
    candidates: List[Dict[str, object]],
    axes: List[Tuple[str, str]],
) -> List[int]:
    """Return indices of non-dominated candidates on ``axes``.

    Each axis is ``(metric_name, direction)`` where direction in {"higher","lower"}.
    """
    keep: List[int] = []
    for i, ci in enumerate(candidates):
        mi = ci["metrics"]
        dominated = False
        for j, cj in enumerate(candidates):
            if i == j:
                continue
            mj = cj["metrics"]
            strictly_better_any = False
            at_least_eq_all = True
            for name, direction in axes:
                vi = float(mi.get(name, 0.0))
                vj = float(mj.get(name, 0.0))
                if direction == "higher":
                    if vj < vi:
                        at_least_eq_all = False; break
                    if vj > vi:
                        strictly_better_any = True
                else:  # lower
                    if vj > vi:
                        at_least_eq_all = False; break
                    if vj < vi:
                        strictly_better_any = True
            if at_least_eq_all and strictly_better_any:
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep


def _rank_sum(
    candidates: List[Dict[str, object]],
    axes: List[Tuple[str, str]],
) -> List[float]:
    """Compute sum-of-ranks per candidate (lower = better)."""
    n = len(candidates)
    rank_sums = [0.0] * n
    for name, direction in axes:
        values = [float(c["metrics"].get(name, 0.0)) for c in candidates]
        # Rank: 1 = best.
        order = sorted(range(n), key=lambda k: -values[k] if direction == "higher" else values[k])
        for rank, idx in enumerate(order, start=1):
            rank_sums[idx] += rank
    return rank_sums


# ─── composite score (mirror of training-time) ────────────────────────────────


def _composite_score(metrics: Dict[str, float], baseline: Dict[str, float]) -> float:
    """Baseline-centered composite: LOWER is better.

    score = -mAP_50 + 0.5 * (FPpr / baseline_FPpr) + 0.3 * (-mAP_medium)
    """
    fp = float(metrics.get("false_positive_pressure", 0.0))
    fp_base = float(baseline.get("false_positive_pressure", 1.0) or 1.0)
    return (
        -float(metrics.get("mAP_50", 0.0))
        + 0.5 * (fp / max(fp_base, 1e-6))
        + 0.3 * (-float(metrics.get("mAP_medium", 0.0)))
    )


# ─── main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    args = _parse_args()
    out_dir = args.output_dir or (args.run_dir / "audit_topk")
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = {"mAP_50": 0.0, "false_positive_pressure": 1.0, "mAP_medium": 0.0}
    if args.baseline_audit.exists():
        baseline = _extract_metrics(json.loads(args.baseline_audit.read_text(encoding="utf-8")))
        print(f"[select] baseline ({args.baseline_audit}): mAP_50={baseline['mAP_50']:.4f}, "
              f"FPpr={baseline['false_positive_pressure']:.4f}, "
              f"mAP_medium={baseline['mAP_medium']:.4f}")

    candidates: List[Dict[str, object]] = []
    for name in args.candidates:
        ckpt = args.run_dir / f"best_by_{name}.pt"
        if not ckpt.exists():
            print(f"[select] SKIP missing candidate: {ckpt}")
            continue
        cand_out = out_dir / f"audit_{name}"
        print(f"[select] auditing candidate '{name}' -> {cand_out}")
        audit = _run_audit(
            ckpt, cand_out,
            args.data_root, args.split,
            args.score_thresh, args.detections_per_img,
            args.require_cuda,
        )
        if audit is None:
            continue
        metrics = _extract_metrics(audit)
        metrics["composite"] = _composite_score(metrics, baseline)
        candidates.append({
            "name": name,
            "checkpoint": str(ckpt),
            "audit_dir": str(cand_out),
            "metrics": metrics,
        })

    if not candidates:
        print("[select] ERROR: no candidate audited successfully.")
        return 1

    # Selection
    axes = [
        ("mAP_50", "higher"),
        ("mAP_medium", "higher"),
        ("MA_AP50", "higher"),
        ("false_positive_pressure", "lower"),
    ]
    # REV-7 (Mode 2A primary): apply non-regression filter when requested.
    mode2a_nonreg_pass: Optional[bool] = None
    mode2a_filter_log: List[str] = []
    if args.strategy == "mode2a_primary":
        bl_map50 = float(baseline.get("mAP_50", 0.0))
        bl_fppr = float(baseline.get("false_positive_pressure", 0.0) or 0.0)
        map50_floor = bl_map50 + float(args.mode2a_map50_min_delta)
        fppr_ceiling = bl_fppr + float(args.mode2a_fppr_max_delta)
        survivors: List[Dict[str, object]] = []
        for cand in candidates:
            m = cand["metrics"]
            map50 = float(m.get("mAP_50", 0.0))
            fppr = float(m.get("false_positive_pressure", 0.0))
            if map50 < map50_floor:
                mode2a_filter_log.append(
                    f"{cand['name']} REJECT mAP_50={map50:.4f} < floor={map50_floor:.4f}"
                )
                continue
            if fppr > fppr_ceiling:
                mode2a_filter_log.append(
                    f"{cand['name']} REJECT FPpr={fppr:.4f} > ceiling={fppr_ceiling:.4f}"
                )
                continue
            survivors.append(cand)
        if survivors:
            # Rank survivors by MA_AP50 descending, tiebreak by FPpr ascending.
            winner = max(
                survivors,
                key=lambda c: (
                    float(c["metrics"].get("MA_AP50", 0.0)),
                    -float(c["metrics"].get("false_positive_pressure", 0.0)),
                ),
            )
            front = survivors
            mode2a_nonreg_pass = True
            print(
                f"[select] Mode 2A non-regression: {len(survivors)}/"
                f"{len(candidates)} survivors; winner = '{winner['name']}'"
            )
            for line in mode2a_filter_log:
                print(f"[select] {line}")
        else:
            # No survivor → INELIGIBLE for positive Mode 2A claim. Fall back
            # to pareto_then_rank for an honest detector-side winner record.
            mode2a_nonreg_pass = False
            print(
                "[select] Mode 2A non-regression: NO candidate passed. "
                "Falling back to pareto_then_rank for record only; verdict "
                "flag mode2a_nonreg_pass=false will be written into "
                "pareto_summary.json."
            )
            for line in mode2a_filter_log:
                print(f"[select] {line}")
            front_idx = _pareto_front(candidates, axes)
            front = [candidates[i] for i in front_idx]
            if len(front) > 1:
                ranks = _rank_sum(front, axes)
                winner_local = int(min(range(len(front)), key=lambda i: ranks[i]))
            else:
                winner_local = 0
            winner = front[winner_local]
    elif args.strategy == "pareto_then_rank":
        front_idx = _pareto_front(candidates, axes)
        front = [candidates[i] for i in front_idx]
        if len(front) > 1:
            ranks = _rank_sum(front, axes)
            winner_local = int(min(range(len(front)), key=lambda i: ranks[i]))
        else:
            winner_local = 0
        winner = front[winner_local]
    elif args.strategy == "map50_only":
        winner = max(candidates, key=lambda c: float(c["metrics"]["mAP_50"]))
        front = [winner]
    else:  # composite_v2
        winner = min(candidates, key=lambda c: float(c["metrics"]["composite"]))
        front = [winner]

    summary = {
        "run_dir": str(args.run_dir),
        "strategy": args.strategy,
        "baseline": baseline,
        "candidates": candidates,
        "pareto_front_names": [c["name"] for c in front],
        "winner_name": winner["name"],
        "winner_metrics": winner["metrics"],
        "winner_checkpoint": winner["checkpoint"],
        # REV-7 (Mode 2A non-regression audit trail).
        "mode2a_nonreg_pass": mode2a_nonreg_pass,
        "mode2a_filter_log": list(mode2a_filter_log),
        "mode2a_map50_floor": (
            float(baseline.get("mAP_50", 0.0))
            + float(args.mode2a_map50_min_delta)
        )
        if args.strategy == "mode2a_primary"
        else None,
        "mode2a_fppr_ceiling": (
            float(baseline.get("false_positive_pressure", 0.0) or 0.0)
            + float(args.mode2a_fppr_max_delta)
        )
        if args.strategy == "mode2a_primary"
        else None,
    }
    (out_dir / "pareto_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    selected_path = out_dir / "selected_best.pt"
    shutil.copy2(winner["checkpoint"], selected_path)
    (out_dir / "selected_best.json").write_text(json.dumps(
        {"winner_name": winner["name"], "winner_metrics": winner["metrics"]},
        indent=2,
    ), encoding="utf-8")

    print(f"[select] WINNER: {winner['name']}  mAP_50={winner['metrics']['mAP_50']:.4f}  "
          f"mAP_medium={winner['metrics']['mAP_medium']:.4f}  "
          f"MA_AP50={winner['metrics']['MA_AP50']:.4f}  "
          f"FPpr={winner['metrics']['false_positive_pressure']:.4f}")
    print(f"[select] Selected checkpoint copied to: {selected_path}")

    if args.prune_losers:
        keep = {Path(winner["checkpoint"]).resolve()}
        pruned: List[str] = []
        for cand in candidates:
            p = Path(cand["checkpoint"])
            if p.resolve() in keep:
                continue
            try:
                p.unlink()
                pruned.append(str(p))
            except OSError:
                continue
        print(f"[select] pruned {len(pruned)} losing candidate(s).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
