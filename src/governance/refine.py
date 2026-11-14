

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from lgovnet.src.governance.contracts import LESION_CLASSES
from lgovnet.src.governance.evidence import compute_aps, compute_sdca

# ── Lesion class index lookup (lower index = canonical first in SDM key) ──────
_LESION_IDX: Dict[str, int] = {c: i for i, c in enumerate(LESION_CLASSES)}

# ── Default thresholds ────────────────────────────────────────────────────────

#: Configurable thresholds for R3 refinement.
#: All keys may be overridden by passing a ``thresholds`` dict to public functions.
DEFAULT_REFINE_THRESHOLDS: Dict[str, float] = {
    # Step 2 — KG class arbitration
    # SDM_norm ≤ delta_merge → majority and runner-up are clinically close
    # enough to commit to majority class and upgrade to R2.
    "delta_merge": 0.4,
    # Step 3 — Anatomical plausibility check
    # APS < aps_min → downgrade to R4.  0.5 is the neutral compat midpoint.
    "aps_min": 0.5,
    # Step 4 — SDM-modulated NMS: standard IoU suppression threshold.
    "nms_iou_thresh": 0.5,
    # Step 4 — Lower IoU threshold used when both boxes are clinically similar.
    # A lower value = more aggressive merging for similar classes.
    "nms_iou_similar": 0.3,
    # Step 4 — Classes with SDM_norm ≤ this cap are treated as "similar" for NMS.
    "sdm_nms_similar_cap": 0.4,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Step helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _iou(a: List[float], b: List[float]) -> float:

    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _sdm_norm_for_pair(
    class_a: str,
    class_b: str,
    sdm_norm: Dict[Tuple[str, str], Dict[str, float]],
) -> Optional[float]:

    if class_a == class_b:
        return 0.0
    if _LESION_IDX.get(class_a, 99) <= _LESION_IDX.get(class_b, 99):
        key = (class_a, class_b)
    else:
        key = (class_b, class_a)
    entry = sdm_norm.get(key)
    return float(entry["norm"]) if entry is not None else None


# ═══════════════════════════════════════════════════════════════════════════════
# Public step functions
# ═══════════════════════════════════════════════════════════════════════════════


def aggregate_head_bboxes(box: dict) -> Optional[List[float]]:

    heads = box.get("head_predictions") or []
    valid = [
        h
        for h in heads
        if isinstance(h.get("bbox"), (list, tuple))
        and len(h["bbox"]) == 4
        and float(h.get("score", 0.0)) > 0.0
    ]
    if not valid:
        # Fallback: try unweighted mean of any head with a bbox
        unweighted = [
            h
            for h in heads
            if isinstance(h.get("bbox"), (list, tuple)) and len(h["bbox"]) == 4
        ]
        if not unweighted:
            return None
        n = float(len(unweighted))
        return [sum(h["bbox"][i] for h in unweighted) / n for i in range(4)]

    total_score = sum(float(h["score"]) for h in valid)
    if total_score <= 0.0:
        return None

    result = [0.0, 0.0, 0.0, 0.0]
    for h in valid:
        w = float(h["score"]) / total_score
        for i in range(4):
            result[i] += w * float(h["bbox"][i])
    return result


def kg_class_arbitration(
    box: dict,
    sdm_norm: Dict[Tuple[str, str], Dict[str, float]],
    thresholds: Dict[str, float],
) -> Tuple[Optional[str], bool]:

    ep: dict = box.get("ensemble_probs") or {}
    if not ep:
        return None, False

    delta_merge = float(thresholds["delta_merge"])

    # Majority and runner-up by ensemble probability
    sorted_classes = sorted(ep, key=lambda c: ep[c], reverse=True)
    majority = sorted_classes[0]
    if len(sorted_classes) < 2:
        # Single class — trivially commit, treat as upgradeable
        return majority, True

    runner_up = next((c for c in sorted_classes if c != majority), None)
    if runner_up is None:
        return majority, True

    sdm_val = _sdm_norm_for_pair(majority, runner_up, sdm_norm)

    if sdm_val is None:
        # SDM not available (e.g. before training): conservative — keep R3
        return majority, False

    upgraded = sdm_val <= delta_merge
    return majority, upgraded


def anatomical_check(
    refined_class: str,
    assigned_zone: str,
    compat_scores: Dict[Tuple[str, str], float],
    aps_min: float,
) -> bool:

    aps = compute_aps(refined_class, assigned_zone, compat_scores)
    return aps >= aps_min


def sdm_modulated_nms(
    candidates: List[dict],
    sdm_norm: Dict[Tuple[str, str], Dict[str, float]],
    thresholds: Optional[Dict[str, float]] = None,
) -> List[dict]:

    thr = {**DEFAULT_REFINE_THRESHOLDS, **(thresholds or {})}
    iou_standard = float(thr["nms_iou_thresh"])
    iou_similar = float(thr["nms_iou_similar"])
    sim_cap = float(thr["sdm_nms_similar_cap"])

    # Sort by score descending; stable sort preserves insertion order on ties
    order = sorted(
        range(len(candidates)),
        key=lambda i: float(candidates[i].get("score", 0.0)),
        reverse=True,
    )

    suppressed = [False] * len(candidates)
    surviving: List[dict] = []

    for i_pos, i in enumerate(order):
        if suppressed[i]:
            continue
        surviving.append(candidates[i])

        bbox_i = candidates[i]["bbox"]
        cls_i = candidates[i].get("majority_class", "")

        for j_pos in range(i_pos + 1, len(order)):
            j = order[j_pos]
            if suppressed[j]:
                continue

            bbox_j = candidates[j]["bbox"]
            cls_j = candidates[j].get("majority_class", "")

            iou_val = _iou(bbox_i, bbox_j)
            if iou_val <= 0.0:
                continue

            # Choose threshold based on clinical similarity
            sdm_val = _sdm_norm_for_pair(cls_i, cls_j, sdm_norm)
            if sdm_val is not None and sdm_val <= sim_cap:
                threshold = iou_similar
            else:
                threshold = iou_standard

            if iou_val >= threshold:
                suppressed[j] = True

    return surviving


# ═══════════════════════════════════════════════════════════════════════════════
# Full single-box refinement pipeline
# ═══════════════════════════════════════════════════════════════════════════════


def refine_r3_bbox(
    box: dict,
    sdm_norm: Dict[Tuple[str, str], Dict[str, float]],
    compat_scores: Dict[Tuple[str, str], float],
    thresholds: Optional[Dict[str, float]] = None,
) -> dict:

    thr = {**DEFAULT_REFINE_THRESHOLDS, **(thresholds or {})}

    bbox_id: str = box["bbox_id"]
    assigned_zone: str = box["assigned_zone"]
    reasons: List[str] = []

    # ── Step 1: Cross-head box aggregation ───────────────────────────────────
    refined_bbox = aggregate_head_bboxes(box)
    if refined_bbox is not None:
        reasons.append("Step1: bbox aggregated (confidence-weighted head mean)")
    else:
        reasons.append("Step1: no valid head bboxes — refined_bbox=None")

    # ── Step 2: KG class arbitration ─────────────────────────────────────────
    refined_class, upgraded = kg_class_arbitration(box, sdm_norm, thr)

    if refined_class is None:
        # Degenerate box — no ensemble probs; reject
        return {
            "bbox_id": bbox_id,
            "refined_bbox": refined_bbox,
            "refined_class": None,
            "upgraded_to": None,
            "downgraded_to": "R4",
            "kept_as": None,
            "refinement_reason": " | ".join(reasons)
            + " | Step2: no ensemble_probs — downgraded to R4",
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }

    ep = box.get("ensemble_probs") or {}
    sorted_ep = sorted(ep, key=lambda c: ep[c], reverse=True)
    runner_up = next((c for c in sorted_ep if c != refined_class), "—")
    sdm_val = _sdm_norm_for_pair(refined_class, runner_up, sdm_norm)
    sdm_str = f"{sdm_val:.3f}" if sdm_val is not None else "N/A"

    if upgraded:
        reasons.append(
            f"Step2: SDM_norm({refined_class},{runner_up})={sdm_str} "
            f"<= delta_merge={thr['delta_merge']:.2f} -> upgrade to R2"
        )
    else:
        reasons.append(
            f"Step2: SDM_norm({refined_class},{runner_up})={sdm_str} "
            f"> delta_merge={thr['delta_merge']:.2f} -> keep R3"
        )

    # ── Step 3: Anatomical plausibility check ─────────────────────────────────
    aps = compute_aps(refined_class, assigned_zone, compat_scores)
    aps_min = float(thr["aps_min"])
    passes_anat = aps >= aps_min

    if not passes_anat:
        reasons.append(
            f"Step3: APS({refined_class},{assigned_zone})={aps:.3f} "
            f"< aps_min={aps_min:.2f} -> downgrade to R4"
        )
        return {
            "bbox_id": bbox_id,
            "refined_bbox": refined_bbox,
            "refined_class": refined_class,
            "upgraded_to": None,
            "downgraded_to": "R4",
            "kept_as": None,
            "refinement_reason": " | ".join(reasons),
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }

    reasons.append(
        f"Step3: APS({refined_class},{assigned_zone})={aps:.3f} "
        f">= aps_min={aps_min:.2f} -> passes"
    )

    # ── Final tier assignment ─────────────────────────────────────────────────
    if upgraded:
        return {
            "bbox_id": bbox_id,
            "refined_bbox": refined_bbox,
            "refined_class": refined_class,
            "upgraded_to": "R2",
            "downgraded_to": None,
            "kept_as": None,
            "refinement_reason": " | ".join(reasons),
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
    else:
        return {
            "bbox_id": bbox_id,
            "refined_bbox": refined_bbox,
            "refined_class": refined_class,
            "upgraded_to": None,
            "downgraded_to": None,
            "kept_as": "R3",
            "refinement_reason": " | ".join(reasons),
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Batch refinement with SDM-modulated NMS
# ═══════════════════════════════════════════════════════════════════════════════


def refine_r3_batch(
    boxes: List[dict],
    sdm_norm: Dict[Tuple[str, str], Dict[str, float]],
    compat_scores: Dict[Tuple[str, str], float],
    thresholds: Optional[Dict[str, float]] = None,
) -> List[dict]:

    thr = {**DEFAULT_REFINE_THRESHOLDS, **(thresholds or {})}

    # Step 1–3: per-box pipeline
    results: List[dict] = [
        refine_r3_bbox(box, sdm_norm, compat_scores, thr) for box in boxes
    ]

    # Build NMS candidates from boxes that survived Steps 1–3
    # (i.e., not yet downgraded to R4)
    nms_candidates: List[dict] = []
    for i, (box, res) in enumerate(zip(boxes, results)):
        if res["downgraded_to"] == "R4":
            continue
        refined_bbox = res["refined_bbox"]
        if refined_bbox is None:
            continue  # no bbox to compare

        ep = box.get("ensemble_probs") or {}
        majority = max(ep, key=lambda c: ep[c]) if ep else ""
        score = float(ep.get(majority, 0.0)) if majority else 0.0

        nms_candidates.append(
            {
                "_result_index": i,
                "bbox": refined_bbox,
                "score": score,
                "majority_class": majority,
            }
        )

    if len(nms_candidates) > 1:
        # Step 4: SDM-modulated NMS
        surviving_nms = sdm_modulated_nms(nms_candidates, sdm_norm, thr)
        surviving_ids = {c["_result_index"] for c in surviving_nms}

        # Mark suppressed boxes as downgraded
        for cand in nms_candidates:
            idx = cand["_result_index"]
            if idx not in surviving_ids:
                old = results[idx]
                results[idx] = {
                    **old,
                    "upgraded_to": None,
                    "downgraded_to": "R4",
                    "kept_as": None,
                    "refinement_reason": old["refinement_reason"]
                    + " | Step4: suppressed by SDM-modulated NMS -> R4",
                }

    return results
