#!/usr/bin/env python
"""
build_native_detector_bridge.py
Phase 19N — conservative native bridge for dual-source detector caches.

Why this tool exists:
  Stage 3B/3C/3D currently consume one DetectorBox cache directory at a time.
  A secondary detector cache can already be exported, but without a bridge it
  remains auxiliary only and does not change SAS / SDCA behavior downstream.

Safe design principle:
  - Keep the existing one-cache Stage 3 contract unchanged.
  - Anchor on the primary cache.
  - Only fuse a secondary detection when it spatially matches a primary box.
  - Do NOT inject unmatched secondary boxes by default.
  - Preserve unmatched primary boxes exactly as degraded single-source records.

Effect:
  - Matched boxes become true multi-source DetectorBox records with
    non-empty head_predictions and merged ensemble_probs.
  - Unmatched boxes remain primary-only, so the bridge is conservative.

Important scope note:
    This does not retrain any detector. It only changes how exported detector
    cache entries are represented before Stage 3. The bridge can also be chained
    conservatively across more than two detector caches, as long as the existing
    head_predictions carried by the primary cache are preserved.

Cache hygiene note:
    Stage 3 consumers glob all `*.json` files inside one split cache directory.
    Therefore the bridge summary MUST live outside the split directory; otherwise
    it is misread as an image cache file.

Usage:
  python tools/build_native_detector_bridge.py \
    --primary_dir outputs/detector_cache/primary/oiaddr/valid \
    --secondary_dir outputs/detector_cache/secondary/oiaddr/valid \
    --output_dir outputs/detector_cache/native_bridge/oiaddr/valid

  python tools/build_native_detector_bridge.py \
    --primary_dir outputs/detector_cache/primary/oiaddr/test \
    --secondary_dir outputs/detector_cache/secondary/oiaddr/test \
    --output_dir outputs/detector_cache/native_bridge/oiaddr/test \
    --iou_match 0.25 \
    --include_secondary_only
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, "..")
if _root not in sys.path:
    sys.path.insert(0, _root)

from lgovnet.src.governance.contracts import LESION_CLASSES, ZONE_NAMES

_CLASS_TO_LABEL_ID: Dict[str, int] = {
    "MA": 1,
    "HE": 2,
    "Hard_EX": 3,
    "Soft_EX": 4,
}


def _load_boxes(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data)}")
    return data


def _save_boxes(path: Path, boxes: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(boxes, f, indent=2)


def _round_bbox(bbox: List[float]) -> List[float]:
    return [round(float(v), 2) for v in bbox]


def _is_valid_bbox(bbox: object) -> bool:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return False
    return x1 < x2 and y1 < y2


def _mean_head_bbox(box: dict) -> Optional[List[float]]:
    heads = box.get("head_predictions") or []
    bboxes = [h.get("bbox") for h in heads if _is_valid_bbox(h.get("bbox"))]
    if not bboxes:
        return None
    n = float(len(bboxes))
    return [sum(float(b[i]) for b in bboxes) / n for i in range(4)]


def _get_detector_raw(box: dict) -> dict:
    return box.get("detector_raw") or box.get("r2a_raw") or {}


def _get_box_bbox(box: dict) -> Optional[List[float]]:
    raw = _get_detector_raw(box)
    bbox = raw.get("bbox")
    if _is_valid_bbox(bbox):
        return [float(v) for v in bbox]
    mean_bbox = _mean_head_bbox(box)
    if mean_bbox is not None:
        return mean_bbox
    return None


def _get_box_probs(box: dict) -> Dict[str, float]:
    probs = box.get("ensemble_probs") or {}
    clean: Dict[str, float] = {}
    for cls in LESION_CLASSES:
        try:
            clean[cls] = max(0.0, float(probs.get(cls, 0.0)))
        except (TypeError, ValueError):
            clean[cls] = 0.0

    total = sum(clean.values())
    if total <= 0.0:
        raw = _get_detector_raw(box)
        label_name = raw.get("label_name")
        score = raw.get("score")
        if label_name in LESION_CLASSES:
            try:
                score_f = min(max(float(score), 0.0), 1.0)
            except (TypeError, ValueError):
                score_f = 0.25
            rest = (1.0 - score_f) / 3.0
            return {
                cls: (score_f if cls == label_name else rest) for cls in LESION_CLASSES
            }
        return {cls: 0.25 for cls in LESION_CLASSES}

    return {cls: clean[cls] / total for cls in LESION_CLASSES}


def _get_box_label(box: dict) -> Optional[str]:
    raw = _get_detector_raw(box)
    label_name = raw.get("label_name")
    if label_name in LESION_CLASSES:
        return label_name
    probs = _get_box_probs(box)
    if not probs:
        return None
    return max(probs, key=lambda cls: probs[cls])


def _get_box_score(box: dict) -> float:
    raw = _get_detector_raw(box)
    score = raw.get("score")
    try:
        score_f = float(score)
        if math.isfinite(score_f):
            return min(max(score_f, 0.0), 1.0)
    except (TypeError, ValueError):
        pass
    probs = _get_box_probs(box)
    return max(probs.values()) if probs else 0.0


def _choose_zone(primary_box: dict, secondary_box: Optional[dict] = None) -> str:
    pz = primary_box.get("assigned_zone")
    if pz in ZONE_NAMES:
        return pz
    if secondary_box is not None:
        sz = secondary_box.get("assigned_zone")
        if sz in ZONE_NAMES:
            return sz
    return "Peripheral"


def _iou(a: List[float], b: List[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _source_head(box: dict, source_name: str) -> dict:
    bbox = _get_box_bbox(box) or [0.0, 0.0, 1.0, 1.0]
    probs = _get_box_probs(box)
    label_name = _get_box_label(box) or max(probs, key=lambda cls: probs[cls])
    score = _get_box_score(box)
    return {
        "head": label_name,
        "source": source_name,
        "bbox": _round_bbox(bbox),
        "class_probs": {cls: round(float(probs[cls]), 6) for cls in LESION_CLASSES},
        "score": round(float(score), 6),
    }


def _source_count(box: dict) -> int:
    heads = box.get("head_predictions") or []
    count = sum(1 for head in heads if isinstance(head, dict))
    return count if count > 0 else 1


def _collect_source_heads(box: dict, default_source_name: str) -> List[dict]:
    heads = box.get("head_predictions") or []
    normalized: List[dict] = []

    fallback_bbox = _get_box_bbox(box) or [0.0, 0.0, 1.0, 1.0]
    fallback_probs = _get_box_probs(box)
    fallback_label = _get_box_label(box) or max(
        fallback_probs, key=lambda cls: fallback_probs[cls]
    )
    fallback_score = _get_box_score(box)

    for head in heads:
        if not isinstance(head, dict):
            continue

        probs = head.get("class_probs") or {}
        clean_probs: Dict[str, float] = {}
        for cls in LESION_CLASSES:
            try:
                clean_probs[cls] = max(0.0, float(probs.get(cls, 0.0)))
            except (TypeError, ValueError):
                clean_probs[cls] = 0.0
        total = sum(clean_probs.values())
        if total <= 0.0:
            clean_probs = dict(fallback_probs)
        else:
            clean_probs = {cls: clean_probs[cls] / total for cls in LESION_CLASSES}

        bbox = head.get("bbox")
        if not _is_valid_bbox(bbox):
            bbox = fallback_bbox

        label_name = head.get("head")
        if label_name not in LESION_CLASSES:
            label_name = max(clean_probs, key=lambda cls: clean_probs[cls])

        try:
            score = float(head.get("score"))
            if not math.isfinite(score):
                raise ValueError
        except (TypeError, ValueError):
            score = max(clean_probs.values()) if clean_probs else fallback_score

        source_name = (
            str(head.get("source") or default_source_name).strip()
            or default_source_name
        )
        normalized.append(
            {
                "head": label_name,
                "source": source_name,
                "bbox": _round_bbox(bbox),
                "class_probs": {
                    cls: round(float(clean_probs[cls]), 6) for cls in LESION_CLASSES
                },
                "score": round(float(min(max(score, 0.0), 1.0)), 6),
            }
        )

    if normalized:
        return normalized
    return [_source_head(box, default_source_name)]


def _weights(
    primary_box: dict,
    secondary_box: dict,
    mode: str,
    primary_weight: float,
    secondary_weight: float,
) -> Tuple[float, float]:
    primary_sources = _source_count(primary_box)
    secondary_sources = _source_count(secondary_box)
    if mode == "uniform":
        return (
            max(primary_weight * primary_sources, 1e-6),
            max(secondary_weight * secondary_sources, 1e-6),
        )
    pw = max(
        primary_weight * primary_sources * max(_get_box_score(primary_box), 1e-3),
        1e-6,
    )
    sw = max(
        secondary_weight * secondary_sources * max(_get_box_score(secondary_box), 1e-3),
        1e-6,
    )
    return pw, sw


def _merge_probs(
    primary_box: dict,
    secondary_box: dict,
    mode: str,
    primary_weight: float,
    secondary_weight: float,
) -> Dict[str, float]:
    p_probs = _get_box_probs(primary_box)
    s_probs = _get_box_probs(secondary_box)
    pw, sw = _weights(
        primary_box, secondary_box, mode, primary_weight, secondary_weight
    )
    total = pw + sw
    merged = {
        cls: ((pw * p_probs[cls]) + (sw * s_probs[cls])) / total
        for cls in LESION_CLASSES
    }
    merged_sum = sum(merged.values())
    if merged_sum <= 0.0:
        return {cls: 0.25 for cls in LESION_CLASSES}
    return {cls: merged[cls] / merged_sum for cls in LESION_CLASSES}


def _merge_bbox(
    primary_box: dict,
    secondary_box: dict,
    mode: str,
    primary_weight: float,
    secondary_weight: float,
) -> List[float]:
    pb = _get_box_bbox(primary_box) or [0.0, 0.0, 1.0, 1.0]
    sb = _get_box_bbox(secondary_box) or pb
    pw, sw = _weights(
        primary_box, secondary_box, mode, primary_weight, secondary_weight
    )
    total = pw + sw
    return [((pw * pb[i]) + (sw * sb[i])) / total for i in range(4)]


def _make_raw_meta(
    label_name: str,
    score: float,
    bbox: List[float],
    status: str,
    sources: List[dict],
    match_iou: Optional[float],
) -> dict:
    payload = {
        "label": _CLASS_TO_LABEL_ID.get(label_name),
        "label_name": label_name,
        "score": round(float(score), 6),
        "bbox": _round_bbox(bbox),
        "bridge_status": status,
        "sources": sources,
    }
    if match_iou is not None:
        payload["match_iou"] = round(float(match_iou), 4)
    return payload


def _copy_with_bridge_status(box: dict, status: str) -> dict:
    new_box = copy.deepcopy(box)
    raw = _get_detector_raw(new_box)
    if raw:
        raw = copy.deepcopy(raw)
        raw["bridge_status"] = status
        new_box["detector_raw"] = raw
        new_box["r2a_raw"] = copy.deepcopy(raw)
    new_box["native_bridge"] = {
        "status": status,
        "head_prediction_mode": (
            "native_multi_source"
            if new_box.get("head_predictions")
            else "degraded_single_source"
        ),
    }
    return new_box


def _pair_candidates(
    primary_boxes: List[dict],
    secondary_boxes: List[dict],
    iou_match: float,
) -> List[Tuple[float, int, int, int, float]]:
    pairs: List[Tuple[float, int, int, int, float]] = []
    for i, p_box in enumerate(primary_boxes):
        pb = _get_box_bbox(p_box)
        if pb is None:
            continue
        p_label = _get_box_label(p_box)
        p_score = _get_box_score(p_box)
        for j, s_box in enumerate(secondary_boxes):
            sb = _get_box_bbox(s_box)
            if sb is None:
                continue
            iou_val = _iou(pb, sb)
            if iou_val < iou_match:
                continue
            same_label = 1 if _get_box_label(s_box) == p_label else 0
            score_sum = p_score + _get_box_score(s_box)
            pairs.append((iou_val, same_label, i, j, score_sum))
    pairs.sort(reverse=True)
    return pairs


def _match_indices(
    primary_boxes: List[dict],
    secondary_boxes: List[dict],
    iou_match: float,
) -> Dict[int, Tuple[int, float]]:
    matched: Dict[int, Tuple[int, float]] = {}
    used_secondary: set = set()
    for iou_val, _same_label, i, j, _score_sum in _pair_candidates(
        primary_boxes, secondary_boxes, iou_match
    ):
        if i in matched or j in used_secondary:
            continue
        matched[i] = (j, iou_val)
        used_secondary.add(j)
    return matched


def _merge_primary_anchor(
    primary_box: dict,
    secondary_box: dict,
    match_iou: float,
    merge_mode: str,
    primary_weight: float,
    secondary_weight: float,
    primary_source_name: str,
    secondary_source_name: str,
) -> dict:
    """Evidence-enriching primary-anchored merge.

    FIX (2026-05-21): the matched output box IS the primary box. Its
    `detector_raw` (score, bbox, label), `detector_class_probs`,
    `ensemble_probs` and `bbox_id` are kept UNCHANGED. This keeps
    `pred_score` and the Stage 3C governance labels in exactly the same
    regime as the single-primary cache — so B0 is no longer confounded by
    a bridge-rewritten score, and the real ROI-softmax class probabilities
    survive the bridge (no synthetic fallback).

    Only `head_predictions` is enriched with the secondary source's head(s)
    so that SAS becomes a real multi-source agreement signal. `merge_mode`
    / `primary_weight` / `secondary_weight` no longer rewrite the box score
    or bbox; they are kept only for CLI backward compatibility.
    """
    new_box = copy.deepcopy(primary_box)

    primary_heads = _collect_source_heads(primary_box, primary_source_name)
    secondary_heads = _collect_source_heads(secondary_box, secondary_source_name)
    merged_heads: List[dict] = []
    sources: List[dict] = []
    seen_sources: Dict[str, int] = {}
    for origin_box, head_list in (
        (primary_box, primary_heads),
        (secondary_box, secondary_heads),
    ):
        for head in head_list:
            merged_head = copy.deepcopy(head)
            source_name = str(merged_head.get("source") or "source")
            seen = seen_sources.get(source_name, 0)
            seen_sources[source_name] = seen + 1
            if seen > 0:
                source_name = f"{source_name}_{seen + 1}"
                merged_head["source"] = source_name
            merged_heads.append(merged_head)
            sources.append(
                {
                    "source": source_name,
                    "bbox_id": origin_box.get("bbox_id"),
                    "label_name": merged_head.get("head"),
                    "score": round(float(merged_head.get("score", 0.0)), 6),
                    "bbox": _round_bbox(
                        merged_head.get("bbox") or [0.0, 0.0, 1.0, 1.0]
                    ),
                }
            )
    bridge_status = (
        "matched_multi_source" if len(merged_heads) > 2 else "matched_dual_source"
    )

    # SAS enrichment: head_predictions becomes the multi-source union.
    new_box["head_predictions"] = merged_heads
    new_box["assigned_zone"] = _choose_zone(primary_box, secondary_box)

    # Keep the PRIMARY detector_raw exactly (score/bbox/label unchanged);
    # only annotate bridge provenance. No merged_score, no merged_bbox.
    raw = copy.deepcopy(_get_detector_raw(primary_box))
    raw["bridge_status"] = bridge_status
    raw["sources"] = sources
    raw["match_iou"] = round(float(match_iou), 4)
    new_box["detector_raw"] = raw
    new_box["r2a_raw"] = copy.deepcopy(raw)

    new_box["native_bridge"] = {
        "status": bridge_status,
        "match_iou": round(match_iou, 4),
        "merge_mode": merge_mode,
        "primary_bbox_id": primary_box.get("bbox_id"),
        "secondary_bbox_id": secondary_box.get("bbox_id"),
        "n_sources": len(merged_heads),
        "source_names": [h.get("source") for h in merged_heads],
        "zone_conflict": primary_box.get("assigned_zone")
        != secondary_box.get("assigned_zone"),
        "evidence_enriching_only": True,
    }
    return new_box


def build_native_bridge(
    primary_dir: Path,
    secondary_dir: Path,
    output_dir: Path,
    iou_match: float,
    merge_mode: str,
    primary_weight: float,
    secondary_weight: float,
    primary_source_name: str,
    secondary_source_name: str,
    include_secondary_only: bool,
    max_images: int,
) -> dict:
    primary_files = {p.stem: p for p in sorted(primary_dir.glob("*.json"))}
    secondary_files = {p.stem: p for p in sorted(secondary_dir.glob("*.json"))}

    stems = sorted(primary_files.keys())
    if max_images > 0:
        stems = stems[:max_images]

    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "primary_dir": str(primary_dir),
        "secondary_dir": str(secondary_dir),
        "output_dir": str(output_dir),
        "iou_match": iou_match,
        "merge_mode": merge_mode,
        "primary_weight": primary_weight,
        "secondary_weight": secondary_weight,
        "primary_source_name": primary_source_name,
        "secondary_source_name": secondary_source_name,
        "include_secondary_only": include_secondary_only,
        "images_processed": 0,
        "images_with_secondary": 0,
        "primary_boxes_total": 0,
        "secondary_boxes_total": 0,
        "matched_boxes": 0,
        "primary_only_boxes": 0,
        "secondary_only_boxes": 0,
        "class_agreement_on_matches": 0,
        "zone_conflicts_on_matches": 0,
        "avg_match_iou": 0.0,
        "output_boxes_total": 0,
    }

    match_ious: List[float] = []

    for stem in stems:
        primary_boxes = _load_boxes(primary_files[stem])
        secondary_boxes = _load_boxes(secondary_files.get(stem, Path()))

        summary["images_processed"] += 1
        summary["primary_boxes_total"] += len(primary_boxes)
        summary["secondary_boxes_total"] += len(secondary_boxes)
        if secondary_boxes:
            summary["images_with_secondary"] += 1

        matched = _match_indices(primary_boxes, secondary_boxes, iou_match=iou_match)
        used_secondary = {j for j, _ in matched.values()}

        merged_boxes: List[dict] = []
        for i, p_box in enumerate(primary_boxes):
            if i not in matched:
                merged_boxes.append(
                    _copy_with_bridge_status(p_box, "primary_only_unmatched")
                )
                summary["primary_only_boxes"] += 1
                continue

            j, match_iou = matched[i]
            s_box = secondary_boxes[j]
            merged_boxes.append(
                _merge_primary_anchor(
                    p_box,
                    s_box,
                    match_iou=match_iou,
                    merge_mode=merge_mode,
                    primary_weight=primary_weight,
                    secondary_weight=secondary_weight,
                    primary_source_name=primary_source_name,
                    secondary_source_name=secondary_source_name,
                )
            )
            summary["matched_boxes"] += 1
            match_ious.append(match_iou)
            if _get_box_label(p_box) == _get_box_label(s_box):
                summary["class_agreement_on_matches"] += 1
            if p_box.get("assigned_zone") != s_box.get("assigned_zone"):
                summary["zone_conflicts_on_matches"] += 1

        if include_secondary_only:
            for j, s_box in enumerate(secondary_boxes):
                if j in used_secondary:
                    continue
                raw = _get_detector_raw(s_box)
                new_box = _copy_with_bridge_status(s_box, "secondary_only_unmatched")
                new_bbox_id = (
                    raw.get("bbox_id") or s_box.get("bbox_id") or f"{stem}_sec_{j:04d}"
                )
                new_box["bbox_id"] = f"{new_bbox_id}_sec"
                merged_boxes.append(new_box)
                summary["secondary_only_boxes"] += 1

        _save_boxes(output_dir / f"{stem}.json", merged_boxes)
        summary["output_boxes_total"] += len(merged_boxes)

    if match_ious:
        summary["avg_match_iou"] = round(sum(match_ious) / len(match_ious), 4)
        summary["match_iou_distribution"] = {
            "min": round(min(match_ious), 4),
            "max": round(max(match_ious), 4),
        }
    else:
        summary["match_iou_distribution"] = {"min": None, "max": None}

    total_matches = max(summary["matched_boxes"], 1)
    summary["match_class_agreement_rate"] = (
        round(summary["class_agreement_on_matches"] / total_matches, 4)
        if summary["matched_boxes"] > 0
        else None
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a conservative native multi-source detector bridge cache."
    )
    parser.add_argument(
        "--primary_dir",
        required=True,
        help="Primary detector cache dir for one split (per-image JSON files).",
    )
    parser.add_argument(
        "--secondary_dir",
        required=True,
        help="Secondary detector cache dir for the same split (per-image JSON files).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output dir for the merged native bridge cache.",
    )
    parser.add_argument(
        "--iou_match",
        type=float,
        default=0.30,
        help="Minimum IoU to fuse a secondary box into a primary anchor (default: 0.30).",
    )
    parser.add_argument(
        "--merge_mode",
        choices=["score", "uniform"],
        default="score",
        help="How to merge class probabilities and bbox coordinates for matched pairs (default: score).",
    )
    parser.add_argument(
        "--primary_weight",
        type=float,
        default=1.0,
        help="Global multiplier for the primary source during merging (default: 1.0).",
    )
    parser.add_argument(
        "--secondary_weight",
        type=float,
        default=1.0,
        help="Global multiplier for the secondary source during merging (default: 1.0).",
    )
    parser.add_argument(
        "--primary_source_name",
        default="primary",
        help="Source label for primary boxes that do not already carry head_predictions.",
    )
    parser.add_argument(
        "--secondary_source_name",
        default="secondary",
        help="Source label for secondary boxes that do not already carry head_predictions.",
    )
    parser.add_argument(
        "--include_secondary_only",
        action="store_true",
        help="Also emit unmatched secondary boxes. Off by default for conservative behavior.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="Optional limit for smoke tests. 0 = all images.",
    )
    args = parser.parse_args()

    primary_dir = Path(args.primary_dir)
    secondary_dir = Path(args.secondary_dir)
    output_dir = Path(args.output_dir)

    if not primary_dir.exists():
        raise FileNotFoundError(f"Primary cache dir not found: {primary_dir}")
    if not secondary_dir.exists():
        raise FileNotFoundError(f"Secondary cache dir not found: {secondary_dir}")
    if not (0.0 < args.iou_match <= 1.0):
        raise ValueError("--iou_match must be in (0, 1].")
    if args.primary_weight <= 0.0 or args.secondary_weight <= 0.0:
        raise ValueError("--primary_weight and --secondary_weight must be > 0.")

    summary = build_native_bridge(
        primary_dir=primary_dir,
        secondary_dir=secondary_dir,
        output_dir=output_dir,
        iou_match=args.iou_match,
        merge_mode=args.merge_mode,
        primary_weight=args.primary_weight,
        secondary_weight=args.secondary_weight,
        primary_source_name=args.primary_source_name,
        secondary_source_name=args.secondary_source_name,
        include_secondary_only=args.include_secondary_only,
        max_images=args.max_images,
    )

    legacy_summary_path = output_dir / "native_bridge_summary.json"
    if legacy_summary_path.exists():
        legacy_summary_path.unlink()

    summary_path = output_dir.parent / f"{output_dir.name}_native_bridge_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[NativeBridge] Done")
    print(f"[NativeBridge] images_processed={summary['images_processed']}")
    print(f"[NativeBridge] matched_boxes={summary['matched_boxes']}")
    print(f"[NativeBridge] primary_only_boxes={summary['primary_only_boxes']}")
    print(f"[NativeBridge] secondary_only_boxes={summary['secondary_only_boxes']}")
    print(f"[NativeBridge] avg_match_iou={summary['avg_match_iou']}")
    print(f"[NativeBridge] summary_json={summary_path}")


if __name__ == "__main__":
    main()
