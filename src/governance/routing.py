

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from lgovnet.src.governance.contracts import FALLBACK_PSS, LESION_CLASSES
from lgovnet.src.governance.evidence import compute_aps, compute_sdca

# ── Tier index mapping ────────────────────────────────────────────────────────

TIER_NAMES: List[str] = ["R1", "R2", "R3", "R4"]
_TIER_INDEX: Dict[str, int] = {t: i for i, t in enumerate(TIER_NAMES)}

# ── Default thresholds ────────────────────────────────────────────────────────

#: Configurable thresholds for tier target derivation.
#: All keys may be overridden by passing a thresholds dict to the public
#: functions.  Chosen to match the architecture spec defaults.
DEFAULT_ROUTING_THRESHOLDS: Dict[str, float] = {
    # IoU ≥ iou_match → detection is considered matched to a GT bbox.
    "iou_match": 0.5,
    # Minimum number of heads (out of 4) predicting the GT class for R1.
    "min_heads_r1": 3,
    # SDCA_norm ≤ clinical_sim_max → minority class is clinically similar
    # to the majority class.  Used in the R2 rule.
    "clinical_sim_max": 0.4,
    # δ1: exclusive lower bound for R3 structured disagreement range.
    "sdca_r3_lo": 0.3,
    # δ2: inclusive upper bound for R3 structured disagreement range.
    "sdca_r3_hi": 0.7,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Private helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_iou(bbox_a: List[float], bbox_b: List[float]) -> float:

    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    if inter == 0.0:
        return 0.0

    area_a = max(0.0, bbox_a[2] - bbox_a[0]) * max(0.0, bbox_a[3] - bbox_a[1])
    area_b = max(0.0, bbox_b[2] - bbox_b[0]) * max(0.0, bbox_b[3] - bbox_b[1])
    union = area_a + area_b - inter

    if union <= 0.0:
        return 0.0

    return inter / union


def _mean_bbox(box: dict) -> Optional[List[float]]:

    heads = box.get("head_predictions") or []
    bboxes = [
        h["bbox"]
        for h in heads
        if isinstance(h.get("bbox"), (list, tuple)) and len(h["bbox"]) == 4
    ]
    if not bboxes:
        return None
    n = float(len(bboxes))
    return [sum(b[i] for b in bboxes) / n for i in range(4)]


def _find_best_gt_match(
    pred_bbox: Optional[List[float]],
    gt_boxes: List[dict],
    iou_threshold: float,
) -> Tuple[Optional[dict], float]:

    if pred_bbox is None or not gt_boxes:
        return None, 0.0

    best_gt: Optional[dict] = None
    best_iou: float = 0.0

    for gt in gt_boxes:
        gt_bbox = gt.get("bbox")
        if not gt_bbox or len(gt_bbox) != 4:
            continue
        iou = _compute_iou(pred_bbox, gt_bbox)
        if iou > best_iou:
            best_iou = iou
            best_gt = gt

    return best_gt, best_iou


def _count_heads_correct(box: dict, gt_class: str) -> int:

    heads = box.get("head_predictions") or []
    count = 0
    for h in heads:
        cp = h.get("class_probs")
        if not cp:
            continue
        if max(cp, key=lambda c: cp[c]) == gt_class:
            count += 1
    return count


def _get_minority_classes(box: dict, majority_class: str) -> List[str]:

    heads = box.get("head_predictions") or []
    minority: set = set()
    for h in heads:
        cp = h.get("class_probs")
        if not cp:
            continue
        pred = max(cp, key=lambda c: cp[c])
        if pred != majority_class:
            minority.add(pred)
    # Sort by LESION_CLASSES index for deterministic output
    _idx = {c: i for i, c in enumerate(LESION_CLASSES)}
    return sorted(minority, key=lambda c: _idx.get(c, 99))


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════


def is_clinically_similar(
    class_a: str,
    class_b: str,
    sdm_norm: Dict[Tuple[str, str], Dict[str, float]],
    threshold: float,
) -> bool:

    if class_a == class_b:
        return True

    _idx = {c: i for i, c in enumerate(LESION_CLASSES)}

    # Canonical key: lower index first
    if _idx.get(class_a, 99) <= _idx.get(class_b, 99):
        key = (class_a, class_b)
    else:
        key = (class_b, class_a)

    entry = sdm_norm.get(key)
    if entry is None:
        # Pair not in SDM — no information; conservatively not similar
        return False

    return float(entry["norm"]) <= threshold


def derive_gate_target(
    box: dict,
    gt_boxes: List[dict],
    sdm_norm: Dict[Tuple[str, str], Dict[str, float]],
    compat_scores: Dict[Tuple[str, str], float],
    thresholds: Optional[Dict[str, float]] = None,
    pss: float = FALLBACK_PSS,
) -> dict:

    thr: Dict[str, float] = {**DEFAULT_ROUTING_THRESHOLDS, **(thresholds or {})}

    ensemble_probs: dict = box["ensemble_probs"]
    assigned_zone: str = box["assigned_zone"]
    bbox_id: str = box["bbox_id"]

    # Ensemble majority class
    majority_class: Optional[str] = (
        max(ensemble_probs, key=lambda c: ensemble_probs[c]) if ensemble_probs else None
    )

    # SDCA between majority and runner-up
    sdca_raw: float = 0.0
    sdca_norm_val: float = 0.0
    if majority_class is not None:
        sdca_raw, sdca_norm_val = compute_sdca(majority_class, ensemble_probs, sdm_norm)

    # APS for (majority_class, assigned_zone)
    aps: float = compute_aps(majority_class or "", assigned_zone, compat_scores)

    # Representative bbox for IoU matching (mean of head bboxes)
    pred_bbox = _mean_bbox(box)

    # GT matching
    best_gt, best_iou = _find_best_gt_match(
        pred_bbox, gt_boxes, float(thr["iou_match"])
    )
    matched_gt: bool = best_iou >= float(thr["iou_match"])
    iou_to_gt: Optional[float] = best_iou if gt_boxes else None

    # ── Tier decision ─────────────────────────────────────────────────────

    if not matched_gt:
        # Rule 1: no GT match → R4
        tier = "R4"
        num_heads_correct = 0
        minority_classes: List[str] = (
            _get_minority_classes(box, majority_class) if majority_class else []
        )
        reason = (
            f"No GT match (best IoU={best_iou:.3f} < iou_match={thr['iou_match']:.2f})"
        )

    else:
        gt_class: str = best_gt["class_label"]  # type: ignore[index]
        num_heads_correct = _count_heads_correct(box, gt_class)
        minority_classes = (
            _get_minority_classes(box, majority_class) if majority_class else []
        )

        min_heads = int(thr["min_heads_r1"])
        clin_max = float(thr["clinical_sim_max"])
        r3_lo = float(thr["sdca_r3_lo"])
        r3_hi = float(thr["sdca_r3_hi"])

        if num_heads_correct >= min_heads and majority_class == gt_class:
            # Rule 2: R1 — well-matched, high multi-head agreement
            tier = "R1"
            reason = (
                f"GT matched (IoU={best_iou:.3f}), "
                f"{num_heads_correct}/4 heads correct, "
                f"majority=={gt_class}"
            )

        elif majority_class == gt_class and all(
            is_clinically_similar(mc, majority_class, sdm_norm, clin_max)
            for mc in minority_classes
        ):
            # Rule 3: R2 — majority correct, minority clinically close
            tier = "R2"
            reason = (
                f"GT matched (IoU={best_iou:.3f}), majority=={gt_class}, "
                f"minority {minority_classes} clinically similar "
                f"(SDCA_norm≤{clin_max:.2f})"
            )

        elif r3_lo < sdca_norm_val <= r3_hi:
            # Rule 4: R3 — structured disagreement
            tier = "R3"
            reason = (
                f"GT matched (IoU={best_iou:.3f}), structured disagreement: "
                f"δ1={r3_lo:.2f} < SDCA_norm={sdca_norm_val:.3f} ≤ δ2={r3_hi:.2f}"
            )

        else:
            # Rule 5: fallback R3 — GT matched but ambiguous disagreement
            tier = "R3"
            reason = (
                f"GT matched (IoU={best_iou:.3f}), fallback R3: "
                f"majority={majority_class} (GT={gt_class}), "
                f"SDCA_norm={sdca_norm_val:.3f} outside structured range "
                f"(δ1={r3_lo:.2f}, δ2={r3_hi:.2f})"
            )

    return {
        "bbox_id": bbox_id,
        "target_tier": tier,
        "target_tier_index": _TIER_INDEX[tier],
        "matched_gt": matched_gt,
        "iou_to_gt": iou_to_gt,
        "num_heads_correct": num_heads_correct,
        "majority_class": majority_class,
        "minority_classes": minority_classes,
        "sdca_norm": sdca_norm_val,
        "aps": aps,
        "pss": float(pss),
        "reason": reason,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }


def route_bbox_batch(
    boxes: List[dict],
    gt_boxes: List[dict],
    sdm_norm: Dict[Tuple[str, str], Dict[str, float]],
    compat_scores: Dict[Tuple[str, str], float],
    thresholds: Optional[Dict[str, float]] = None,
    pss_values: Optional[Dict[str, float]] = None,
) -> List[dict]:

    pss_map: Dict[str, float] = pss_values or {}
    return [
        derive_gate_target(
            box=box,
            gt_boxes=gt_boxes,
            sdm_norm=sdm_norm,
            compat_scores=compat_scores,
            thresholds=thresholds,
            pss=pss_map.get(box.get("bbox_id", ""), FALLBACK_PSS),
        )
        for box in boxes
    ]
