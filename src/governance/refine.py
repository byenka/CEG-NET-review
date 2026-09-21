"""
refine.py
KG-guided bounding-box refinement for tier R3 detections — LGov-Net Contribution 2.

Applies a four-step pipeline to each R3 bbox:

  Step 1 — Cross-head box aggregation
    Confidence-weighted average of all head_predictions bboxes.  Heads with
    higher detection scores influence the aggregated box more.  If no valid
    head bboxes are present, falls back to None.

  Step 2 — KG class arbitration
    Checked: SDM_norm(majority_class, runner_up_class) ≤ delta_merge?
      YES → the two classes are clinically close in the KG embedding space.
            Commit to majority_class as the refined class, upgrade to R2.
      NO  → disagreement spans a meaningful KG gap. Keep as R3 (reduced
            confidence; addressed by downstream radiologist / training weight).

  Step 3 — Anatomical plausibility check
    APS(refined_class, assigned_zone) is looked up. If APS < aps_min (default
    0.5) the detection is anatomically implausible → downgrade to R4,
    regardless of the Step 2 outcome.

  Step 4 — SDM-modulated NMS
    A local NMS over a batch of R3 candidates.  The IoU suppression threshold
    is modulated by clinical similarity:
      SDM_norm(class_A, class_B) ≤ sdm_nms_similar_cap
          → use nms_iou_similar (lower, e.g. 0.3) — more aggressive merging
      SDM_norm(class_A, class_B) >  sdm_nms_similar_cap
          → use nms_iou_thresh (standard, e.g. 0.5) — preserve distinct classes

    This step is applied to a batch (via refine_r3_batch) and reduces the
    candidate list before returning per-box refinement records.

Output contract per bbox:
  {
    "bbox_id":          str,
    "refined_bbox":     [x1, y1, x2, y2] | None,
    "refined_class":    str | None,
    "upgraded_to":      "R2" | None,   # set when Step 2 upgrades
    "downgraded_to":    "R4" | None,   # set when Step 3 fails; overrides upgrade
    "kept_as":          "R3" | None,   # set when neither upgraded nor downgraded
    "refinement_reason": str,
    "timestamp":        str,            # ISO-8601 UTC string — audit trail
  }
  Exactly one of {upgraded_to, downgraded_to, kept_as} is non-None per record.

Public API:
  aggregate_head_bboxes(box)                 -> Optional[List[float]]
  kg_class_arbitration(box, sdm_norm, thr)   -> Tuple[Optional[str], bool]
  anatomical_check(cls, zone, compat, min_a) -> bool
  sdm_modulated_nms(candidates, sdm_norm, thr) -> List[dict]
  refine_r3_bbox(box, sdm_norm, compat, thr) -> dict
  refine_r3_batch(boxes, sdm_norm, compat, thr) -> List[dict]

Dependencies:
  - src/governance/contracts.py  (LESION_CLASSES)
  - src/governance/evidence.py   (compute_aps, compute_sdca)
  NO dependency on gate.py, routing.py, training loop, or loss functions.
"""

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
    """Axis-aligned IoU between two [x1, y1, x2, y2] boxes."""
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
    """Return SDM_norm between class_a and class_b, or None if not found."""
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
    """Step 1 — Confidence-weighted average of detection head bboxes.

    Each head in ``head_predictions`` contributes its bbox weighted by its
    ``score``.  Heads with missing or zero score are excluded.  If all heads
    have equal score the result is the unweighted mean.

    Args:
        box: DetectorBox dict.  Must have a ``head_predictions`` list where
             each entry has ``bbox`` ([x1, y1, x2, y2]) and ``score`` (float).

    Returns:
        [x1, y1, x2, y2] weighted-average bbox, or None if no valid heads.
    """
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
    """Step 2 — KG-guided class arbitration for R3 box.

    Looks up the SDM distance between the ensemble majority class and the
    runner-up class.  If the distance is small (≤ delta_merge), the two
    classes are clinically close enough in the KG space to commit to the
    majority class and upgrade the tier to R2.

    Args:
        box:          DetectorBox dict with ``ensemble_probs``.
        sdm_norm:     Output of ``encoder.normalize_sdca()``.
                      Keys: (ci, cj) with lower LESION_CLASSES index first.
        thresholds:   Must have key ``"delta_merge"``.

    Returns:
        (refined_class, upgraded_to_r2)
        refined_class:   The committed class after arbitration (majority class
                         in both cases).  None if ensemble_probs is empty.
        upgraded_to_r2:  True when SDM_norm ≤ delta_merge.
    """
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
    """Step 3 — Anatomical plausibility check.

    Returns True (passes) when APS ≥ aps_min, False (fails) when APS < aps_min.

    Args:
        refined_class:  The class committed to in Step 2.
        assigned_zone:  ETDRS zone for this bbox (from DetectorBox).
        compat_scores:  Output of ``encoder.compute_compat()``.
        aps_min:        APS threshold below which the box is deemed implausible
                        (architecture default: 0.5).

    Returns:
        bool — True → plausible (keep or upgrade); False → downgrade to R4.
    """
    aps = compute_aps(refined_class, assigned_zone, compat_scores)
    return aps >= aps_min


def sdm_modulated_nms(
    candidates: List[dict],
    sdm_norm: Dict[Tuple[str, str], Dict[str, float]],
    thresholds: Optional[Dict[str, float]] = None,
) -> List[dict]:
    """Step 4 — SDM-modulated non-maximum suppression for R3 candidates.

    Standard NMS suppresses boxes with high IoU.  This variant modulates the
    IoU threshold based on the KG-derived clinical similarity of the two boxes'
    predicted classes:

      SDM_norm(class_A, class_B) ≤ sdm_nms_similar_cap
          → use nms_iou_similar (lower = more aggressive merging)
      SDM_norm(class_A, class_B) >  sdm_nms_similar_cap (or unknown)
          → use nms_iou_thresh (standard merging)

    Rationale: two R3 boxes predicting clinically similar classes (e.g. MA and
    HE, SDM ≈ 0.1) are likely duplicates of the same lesion — merge them
    readily.  Two boxes predicting distinct classes (e.g. MA and Soft_EX,
    SDM ≈ 0.8) might represent genuinely different co-located lesions — keep
    them unless IoU is very high.

    Each candidate dict must have:
        ``bbox``         : [x1, y1, x2, y2]
        ``score``        : float — used for sort order (higher kept)
        ``majority_class``: str

    All other fields are passed through unchanged.

    Args:
        candidates: List of dicts with the fields above.
        sdm_norm:   Output of ``encoder.normalize_sdca()``.
        thresholds: Optional overrides for keys:
                    ``nms_iou_thresh``, ``nms_iou_similar``,
                    ``sdm_nms_similar_cap``.

    Returns:
        Subset of ``candidates`` that survived suppression, in descending
        score order.
    """
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
    """Full R3 refinement pipeline for one DetectorBox.

    Applies Steps 1–3 in order.  Step 4 (SDM-modulated NMS) is a batch
    operation; use ``refine_r3_batch()`` to apply it across a set of boxes.

    Decision logic:
      - Step 2 can set upgraded_to = "R2".
      - Step 3 can override that with downgraded_to = "R4" (aps_min check
        applies regardless of Step 2 outcome).
      - If neither condition fires, kept_as = "R3".

    Args:
        box:           DetectorBox dict.  Must have:
                       ``bbox_id``, ``head_predictions``,
                       ``ensemble_probs``, ``assigned_zone``.
        sdm_norm:      Output of ``encoder.normalize_sdca()``.
        compat_scores: Output of ``encoder.compute_compat()``.
        thresholds:    Optional overrides for DEFAULT_REFINE_THRESHOLDS.
                       Keys: ``delta_merge``, ``aps_min``.

    Returns:
        dict with output contract fields (see module docstring).
    """
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
    """Batch R3 refinement: Steps 1–3 per box, then Step 4 (SDM-NMS) across batch.

    Workflow:
      1. Run ``refine_r3_bbox`` on every input box.
      2. Collect boxes not downgraded to R4 as NMS candidates.
      3. Apply ``sdm_modulated_nms`` to the surviving candidates.
      4. Re-mark boxes suppressed by NMS as downgraded_to="R4".
      5. Return the full list (same order as input), with updated outcomes.

    Args:
        boxes:         List of R3 DetectorBox dicts (same image).
        sdm_norm:      Output of ``encoder.normalize_sdca()``.
        compat_scores: Output of ``encoder.compute_compat()``.
        thresholds:    Optional overrides for ALL DEFAULT_REFINE_THRESHOLDS keys.

    Returns:
        List of refinement result dicts (one per input box, input order preserved).
    """
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
