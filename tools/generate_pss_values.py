#!/usr/bin/env python
"""Generate pss_values.json from base + perturbation-view detector caches.

This utility builds PSSResult records that can be consumed directly by:
  - tools/audit_stage3b_evidence.py --pss_dir <output_dir>
  - tools/audit_stage3c_governance_targets.py --pss_dir <output_dir>

PSS proxy rule used here:
  1) For each base bbox, match a bbox from each view cache by IoU.
  2) Collect predicted lesion labels across [base + matched views].
  3) pss = majority_vote_count / n_votes.
  4) status = "implemented" when at least one view is matched,
     otherwise status = "placeholder" with FALLBACK_PSS.

Notes:
  - `n_augmentations` follows contract semantics and counts matched views only.
  - Matching is per image stem and bbox geometry; bbox_id does not need to match
    across views.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, "..")
if _root not in sys.path:
    sys.path.insert(0, _root)

try:
    from src.detector_backend import load_cached_predictions
    from src.governance.contracts import FALLBACK_PSS
except ModuleNotFoundError:
    from lgovnet.src.detector_backend import load_cached_predictions
    from lgovnet.src.governance.contracts import FALLBACK_PSS

_LESION_CLASSES = ("MA", "HE", "Hard_EX", "Soft_EX")


def _safe_float(x: object, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _iou(a: List[float], b: List[float]) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _extract_bbox(box: dict) -> Optional[List[float]]:
    raw = box.get("detector_raw") or box.get("r2a_raw") or {}
    bbox = raw.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        return [float(v) for v in bbox]

    heads = box.get("head_predictions") or []
    if isinstance(heads, list) and heads:
        hb = heads[0].get("bbox")
        if isinstance(hb, list) and len(hb) == 4:
            return [float(v) for v in hb]

    bb = box.get("bbox")
    if isinstance(bb, list) and len(bb) == 4:
        return [float(v) for v in bb]

    return None


def _extract_score(box: dict) -> float:
    raw = box.get("detector_raw") or box.get("r2a_raw") or {}
    score = raw.get("score")
    if score is not None:
        return _safe_float(score, 0.0)

    dcp = box.get("detector_class_probs") or {}
    if isinstance(dcp, dict) and dcp:
        return max(_safe_float(v, 0.0) for v in dcp.values())

    ep = box.get("ensemble_probs") or {}
    if isinstance(ep, dict) and ep:
        return max(_safe_float(v, 0.0) for v in ep.values())

    return 0.0


def _extract_label(box: dict) -> str:
    dcp = box.get("detector_class_probs") or {}
    if isinstance(dcp, dict) and dcp:
        return max(_LESION_CLASSES, key=lambda c: _safe_float(dcp.get(c), 0.0))

    ep = box.get("ensemble_probs") or {}
    if isinstance(ep, dict) and ep:
        return max(_LESION_CLASSES, key=lambda c: _safe_float(ep.get(c), 0.0))

    raw = box.get("detector_raw") or box.get("r2a_raw") or {}
    label_name = raw.get("label_name")
    if isinstance(label_name, str) and label_name:
        return label_name

    return "UNK"


def _match_best_box(
    base_bbox: List[float],
    base_label: str,
    view_boxes: List[dict],
    used_indices: Optional[set],
    match_iou: float,
    class_aware: bool,
) -> Optional[Tuple[int, float]]:
    best_idx = None
    best_iou = match_iou

    for idx, cand in enumerate(view_boxes):
        if used_indices is not None and idx in used_indices:
            continue

        if class_aware:
            cand_label = _extract_label(cand)
            if cand_label != base_label:
                continue

        cb = _extract_bbox(cand)
        if cb is None:
            continue

        iou_val = _iou(base_bbox, cb)
        if iou_val >= best_iou:
            best_iou = iou_val
            best_idx = idx

    if best_idx is None:
        return None
    return best_idx, best_iou


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate pss_values.json from detector cache views (end-to-end supporting pipeline)."
    )
    parser.add_argument(
        "--base_cache_dir",
        required=True,
        help="Base detector cache dir (e.g., outputs/detector_cache/primary/oiaddr/valid).",
    )
    parser.add_argument(
        "--view_cache_dirs",
        nargs="+",
        required=True,
        help="One or more perturbation/secondary cache dirs to compare against base.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output dir where pss_values.json and pss_summary.json will be written.",
    )
    parser.add_argument(
        "--require_clean_output_dir",
        action="store_true",
        help="Fail when output_dir already contains files (prevents old-artifact mixing).",
    )
    parser.add_argument(
        "--match_iou",
        type=float,
        default=0.30,
        help="IoU threshold for base-to-view bbox matching (default: 0.30).",
    )
    parser.add_argument(
        "--score_min",
        type=float,
        default=0.0,
        help="Minimum base bbox score to include in PSS generation (default: 0.0).",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="Max number of images to process (0 = all).",
    )
    parser.add_argument(
        "--class_aware_match",
        action="store_true",
        help="Require matched view box to have the same predicted class as the base box.",
    )
    parser.add_argument(
        "--allow_many_to_one",
        action="store_true",
        help="Allow multiple base boxes to match the same view box (default: disabled).",
    )
    parser.add_argument(
        "--require_min_implemented_coverage",
        type=float,
        default=0.0,
        help="Optional gate in [0,1]. Exit code 2 when implemented coverage is below this value.",
    )
    args = parser.parse_args()

    if not (0.0 <= args.match_iou <= 1.0):
        print("[PSSGen] ERROR: --match_iou must be in [0,1]")
        sys.exit(2)
    if not (0.0 <= args.score_min <= 1.0):
        print("[PSSGen] ERROR: --score_min must be in [0,1]")
        sys.exit(2)
    if args.max_images < 0:
        print("[PSSGen] ERROR: --max_images must be >= 0")
        sys.exit(2)
    if not (0.0 <= args.require_min_implemented_coverage <= 1.0):
        print("[PSSGen] ERROR: --require_min_implemented_coverage must be in [0,1]")
        sys.exit(2)

    base_dir = Path(args.base_cache_dir)
    if not base_dir.exists():
        print(f"[PSSGen] ERROR: base_cache_dir not found: {base_dir}")
        sys.exit(1)

    view_dirs = [Path(p) for p in args.view_cache_dirs]
    for vd in view_dirs:
        if not vd.exists():
            print(f"[PSSGen] ERROR: view cache dir not found: {vd}")
            sys.exit(1)

    stems = sorted(p.stem for p in base_dir.glob("*.json"))
    if args.max_images > 0:
        stems = stems[: args.max_images]

    if not stems:
        print(f"[PSSGen] ERROR: no cache json found in {base_dir}")
        sys.exit(1)

    print(
        f"[PSSGen] Loading base cache ({len(stems)} images) + "
        f"{len(view_dirs)} view cache(s)..."
    )
    base_preds = load_cached_predictions(stems, cache_dir=str(base_dir))
    view_preds_all = [
        load_cached_predictions(stems, cache_dir=str(vd)) for vd in view_dirs
    ]

    pss_map: Dict[str, dict] = {}
    per_image_stats: Dict[str, dict] = {}

    for stem in stems:
        base_boxes = base_preds.get(stem, [])
        if not base_boxes:
            per_image_stats[stem] = {
                "n_base": 0,
                "n_total": 0,
                "n_implemented": 0,
                "implemented_coverage": 0.0,
            }
            continue

        view_boxes_per_view = [vp.get(stem, []) for vp in view_preds_all]
        used_per_view = [set() for _ in view_boxes_per_view]

        order = sorted(
            range(len(base_boxes)),
            key=lambda i: _extract_score(base_boxes[i]),
            reverse=True,
        )

        n_total = 0
        n_implemented = 0

        for idx in order:
            base_box = base_boxes[idx]
            base_score = _extract_score(base_box)
            if base_score < args.score_min:
                continue

            bbox_id = str(base_box.get("bbox_id") or f"{stem}_{idx}")
            base_bbox = _extract_bbox(base_box)
            if base_bbox is None:
                pss_map[bbox_id] = {
                    "bbox_id": bbox_id,
                    "value": float(FALLBACK_PSS),
                    "status": "placeholder",
                    "n_augmentations": 0,
                }
                n_total += 1
                continue

            base_label = _extract_label(base_box)
            votes = [base_label]
            n_aug = 0

            for view_idx, view_boxes in enumerate(view_boxes_per_view):
                used = None if args.allow_many_to_one else used_per_view[view_idx]
                match = _match_best_box(
                    base_bbox=base_bbox,
                    base_label=base_label,
                    view_boxes=view_boxes,
                    used_indices=used,
                    match_iou=args.match_iou,
                    class_aware=args.class_aware_match,
                )
                if match is None:
                    continue

                matched_idx, _ = match
                if used is not None:
                    used.add(matched_idx)

                matched_box = view_boxes[matched_idx]
                votes.append(_extract_label(matched_box))
                n_aug += 1

            if n_aug > 0:
                maj_count = Counter(votes).most_common(1)[0][1]
                pss_val = float(maj_count) / float(len(votes))
                status = "implemented"
                n_implemented += 1
            else:
                pss_val = float(FALLBACK_PSS)
                status = "placeholder"

            pss_map[bbox_id] = {
                "bbox_id": bbox_id,
                "value": round(float(pss_val), 6),
                "status": status,
                "n_augmentations": int(n_aug),
            }
            n_total += 1

        coverage = (n_implemented / n_total) if n_total > 0 else 0.0
        per_image_stats[stem] = {
            "n_base": len(base_boxes),
            "n_total": n_total,
            "n_implemented": n_implemented,
            "implemented_coverage": round(float(coverage), 4),
        }

    total = len(pss_map)
    n_impl = sum(1 for v in pss_map.values() if v.get("status") == "implemented")
    n_placeholder = total - n_impl
    impl_cov = (n_impl / total) if total > 0 else 0.0
    avg_pss = (
        sum(float(v.get("value", 0.0)) for v in pss_map.values()) / total
        if total > 0
        else 0.0
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.require_clean_output_dir and any(out_dir.iterdir()):
        print(
            f"[PSSGen] ERROR: output_dir is not empty while "
            f"--require_clean_output_dir is set: {out_dir}"
        )
        sys.exit(2)

    pss_path = out_dir / "pss_values.json"
    with open(pss_path, "w", encoding="utf-8") as f:
        json.dump(pss_map, f, indent=2, ensure_ascii=False)

    summary = {
        "base_cache_dir": str(base_dir),
        "view_cache_dirs": [str(v) for v in view_dirs],
        "match_iou": args.match_iou,
        "score_min": args.score_min,
        "class_aware_match": bool(args.class_aware_match),
        "allow_many_to_one": bool(args.allow_many_to_one),
        "n_images": len(stems),
        "n_total_boxes": total,
        "n_implemented": n_impl,
        "n_placeholder": n_placeholder,
        "implemented_coverage": round(float(impl_cov), 4),
        "avg_pss": round(float(avg_pss), 4),
        "output_pss_values": str(pss_path),
        "per_image": per_image_stats,
    }
    summary_path = out_dir / "pss_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[PSSGen] pss_values written: {pss_path}")
    print(f"[PSSGen] summary written:    {summary_path}")
    print(
        "[PSSGen] implemented coverage: " f"{n_impl}/{total} ({100.0 * impl_cov:.1f}%)"
    )

    if impl_cov < args.require_min_implemented_coverage:
        print(
            "[PSSGen] GATE FAIL: implemented coverage "
            f"{impl_cov:.4f} < required {args.require_min_implemented_coverage:.4f}"
        )
        sys.exit(2)

    print("[PSSGen] GATE PASS")


if __name__ == "__main__":
    main()
