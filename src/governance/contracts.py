"""
contracts.py
Formal data contracts for LGov-Net Governance Gate inputs (Contribution 2).

Two contracts are defined here:

  A. DetectorBox — the canonical output unit from the multi-head detector.
     Every bbox produced by the detector backbone must conform to this schema
     before it enters evidence computation or the governance gate.

  B. PSSSpec — formal specification of the Perturbation Stability Score.
     PSS quantifies how stable a single bbox prediction is across N augmented
     views of the same image patch.  Full computation requires a perturbation
     loop that runs the detector N times per image — that loop is NOT in scope
     for this module.  This module only defines:
       - what PSS means
       - its valid range and interpretation
       - how callers must declare their PSS status
       - a safe fallback value for phases before the loop exists

Usage:
  from src.governance.contracts import (
      HeadPrediction,
      DetectorBox,
      PSSResult,
      PSSStatus,
      validate_detector_box,
      validate_pss_result,
      make_placeholder_pss,
  )

Governance Gate, SDGL loss, and evidence computation are NOT in scope here.
"""

from __future__ import annotations

from typing import Dict, List, Optional

try:
    from typing import TypedDict, Literal
except ImportError:  # Python < 3.8
    from typing_extensions import TypedDict, Literal


# ── Constants ─────────────────────────────────────────────────────────────────

# Canonical lesion class names — must match ontology.LESION_NODES keys.
LESION_CLASSES: tuple = ("MA", "HE", "Hard_EX", "Soft_EX")

# Canonical zone names — must match ontology.ZONE_NODES keys.
ZONE_NAMES: tuple = (
    "Foveal",
    "Parafoveal",
    "Peripapillary",
    "Vascular_Arcades",
    "Peripheral",
)


# ═══════════════════════════════════════════════════════════════════════════════
# CONTRACT A — Detector Output
# ═══════════════════════════════════════════════════════════════════════════════


class HeadPrediction(TypedDict):
    """Single-head prediction for one bounding box.

    The detector backbone produces one HeadPrediction per head × per bbox.
    LGov-Net uses 4 independent detection heads, one per lesion class.

    Fields:
        head:         Canonical lesion class this head specialises in.
                      Must be one of LESION_CLASSES: "MA", "HE", "Hard_EX",
                      "Soft_EX".
        bbox:         Detection coordinates [x1, y1, x2, y2] in pixel space
                      (absolute, not normalised).  x1 < x2, y1 < y2.
        class_probs:  Softmax probability distribution over all 4 lesion
                      classes.  Keys = LESION_CLASSES.  Values sum to 1.0
                      (± floating-point epsilon).
        score:        Head-specific confidence score in [0.0, 1.0].
                      Typically = max(class_probs.values()) or IoU-weighted.
    """

    head: str
    bbox: List[float]  # [x1, y1, x2, y2]
    class_probs: Dict[str, float]
    score: float


class DetectorBox(TypedDict):
    """Canonical output unit from the multi-head detector for one bounding box.

    This is the single source of truth for bbox data that flows into evidence
    computation (evidence.py) and the governance gate (governance.py).
    Nothing downstream should access raw detector tensors directly — they must
    always go through this schema.

    Fields:
        bbox_id:          Unique identifier for this bbox within an image.
                          Format: "<image_id>_<sequential_int>", e.g. "img0042_3".
        head_predictions: List of HeadPrediction, one per detection head.
                          For LGov-Net: length == 4 (one per lesion class).
                          Order must match LESION_CLASSES ordering.
        ensemble_probs:   Per-class probability averaged across all heads.
                          Keys = LESION_CLASSES.  Values sum to 1.0.
                          Computed as: ensemble_probs[c] = mean over heads of
                          head.class_probs[c].
        assigned_zone:    ETDRS zone assigned to this bbox's centre point by
                          src/data/zoning.assign_bbox_to_zone().
                          Zone assignment is mandatory — boxes must be zoned
                          before entering evidence computation or the gate.

    Invariants enforced by validate_detector_box():
        1. bbox_id is non-empty.
        2. All 4 LESION_CLASSES present in ensemble_probs.
        3. ensemble_probs values sum to 1.0 ± 1e-4.
        4. Each HeadPrediction has a bbox with exactly 4 coordinates,
           x1 < x2, y1 < y2.
        5. Each HeadPrediction.class_probs has exactly the 4 LESION_CLASSES.
        6. Each HeadPrediction.score is in [0.0, 1.0].
        7. assigned_zone must be one of ZONE_NAMES (None is NOT allowed).
    """

    bbox_id: str
    head_predictions: List[HeadPrediction]
    ensemble_probs: Dict[str, float]
    assigned_zone: str


# ═══════════════════════════════════════════════════════════════════════════════
# CONTRACT B — Perturbation Stability Score (PSS)
# ═══════════════════════════════════════════════════════════════════════════════

# Literal type for PSS implementation status.
PSSStatus = Literal["implemented", "placeholder", "missing"]


class PSSResult(TypedDict):
    """Formal result record for the Perturbation Stability Score of one bbox.

    PSS Definition:
        PSS(b_j) measures how consistent the detector's class prediction is
        across N spatially/photometrically perturbed views of the same image
        patch containing bbox b_j.

        Full formula (for implemented status):
            1. Run detector on N augmented copies of the image patch:
               {pred_1, pred_2, ..., pred_N}
            2. For each augmented pred_k, take argmax class.
            3. PSS(b_j) = (count of majority class across N preds) / N

        Range: [1/N, 1.0]
            1/N  — prediction completely unstable (always different class)
            1.0  — prediction perfectly stable (same class every augmentation)

        Threshold for governance:
            PSS >= 0.7 → stable, contributes positive evidence
            PSS <  0.5 → unstable, flags bbox for escalation or demotion
            PSS in [0.5, 0.7) → ambiguous, governance uses other signals

    Fields:
        bbox_id:         Identifies which bbox this PSS belongs to.
                         Must match a DetectorBox.bbox_id.
        value:           PSS score in [0.0, 1.0], or None if status != "implemented".
                         Callers must check status before using value.
        status:          "implemented" — PSS was computed from real perturbation loop.
                         "placeholder" — value is a fallback constant (see FALLBACK_PSS).
                         "missing"     — PSS was not computed; value is None.
        n_augmentations: Number of augmented views used to compute PSS.
                         0 if status is "placeholder" or "missing".
    """

    bbox_id: str
    value: Optional[float]
    status: PSSStatus
    n_augmentations: int


# Safe fallback constant used when PSS is not yet implemented.
# 0.5 is the neutral midpoint — it neither rewards nor penalizes a bbox,
# preventing the governance gate from being distorted by a missing signal.
# Must be replaced by real PSS before publication.
FALLBACK_PSS: float = 0.5

# Threshold constants (from architecture spec, Contribution 2).
PSS_STABLE_THRESHOLD: float = 0.7
PSS_UNSTABLE_THRESHOLD: float = 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# Validation helpers
# ═══════════════════════════════════════════════════════════════════════════════


def validate_detector_box(box: dict) -> list:
    """Validate a DetectorBox dict against the formal contract.

    Args:
        box: A dict intended to conform to DetectorBox.

    Returns:
        List of violation strings.  Empty list means the box is valid.
        Callers decide how to handle violations (raise, warn, or log).

    Examples:
        >>> errs = validate_detector_box(my_box)
        >>> if errs:
        ...     raise ValueError(errs)
    """
    errors: list = []

    # bbox_id
    if not box.get("bbox_id"):
        errors.append("bbox_id must be a non-empty string")

    # ensemble_probs
    ep = box.get("ensemble_probs", {})
    missing_classes = [c for c in LESION_CLASSES if c not in ep]
    if missing_classes:
        errors.append(f"ensemble_probs missing classes: {missing_classes}")
    else:
        prob_sum = sum(ep.values())
        if abs(prob_sum - 1.0) > 1e-4:
            errors.append(f"ensemble_probs sum = {prob_sum:.6f}, expected 1.0 ± 1e-4")

    # assigned_zone — mandatory, must be a valid zone
    az = box.get("assigned_zone")
    if not az:
        errors.append(
            "assigned_zone must be a non-empty string (zone assignment is mandatory)"
        )
    elif az not in ZONE_NAMES:
        errors.append(f"assigned_zone '{az}' is not in ZONE_NAMES {ZONE_NAMES}")

    # head_predictions
    heads = box.get("head_predictions", [])
    if not isinstance(heads, list) or len(heads) == 0:
        errors.append("head_predictions must be a non-empty list")
    else:
        for i, h in enumerate(heads):
            prefix = f"head_predictions[{i}]"

            # head
            if h.get("head") not in LESION_CLASSES:
                errors.append(f"{prefix}.head '{h.get('head')}' not in LESION_CLASSES")

            # bbox
            bbox = h.get("bbox", [])
            if len(bbox) != 4:
                errors.append(f"{prefix}.bbox must have 4 elements [x1,y1,x2,y2]")
            elif not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
                errors.append(
                    f"{prefix}.bbox invalid: x1<x2 and y1<y2 required, got {bbox}"
                )

            # class_probs
            cp = h.get("class_probs", {})
            missing_cp = [c for c in LESION_CLASSES if c not in cp]
            if missing_cp:
                errors.append(f"{prefix}.class_probs missing classes: {missing_cp}")

            # score
            score = h.get("score", -1)
            if not (0.0 <= score <= 1.0):
                errors.append(f"{prefix}.score = {score}, must be in [0.0, 1.0]")

    return errors


def validate_pss_result(pss: dict) -> list:
    """Validate a PSSResult dict against the formal contract.

    Args:
        pss: A dict intended to conform to PSSResult.

    Returns:
        List of violation strings.
    """
    errors: list = []
    valid_statuses = ("implemented", "placeholder", "missing")

    if not pss.get("bbox_id"):
        errors.append("bbox_id must be a non-empty string")

    status = pss.get("status")
    if status not in valid_statuses:
        errors.append(f"status '{status}' must be one of {valid_statuses}")

    value = pss.get("value")
    if status == "implemented":
        if value is None:
            errors.append("value must not be None when status='implemented'")
        elif not (0.0 <= value <= 1.0):
            errors.append(f"value = {value}, must be in [0.0, 1.0]")
    elif status == "missing":
        if value is not None:
            errors.append("value must be None when status='missing'")

    n_aug = pss.get("n_augmentations", -1)
    if not isinstance(n_aug, int) or n_aug < 0:
        errors.append("n_augmentations must be a non-negative integer")
    if status == "implemented" and n_aug == 0:
        errors.append("n_augmentations must be > 0 when status='implemented'")

    return errors


def make_placeholder_pss(bbox_id: str) -> dict:
    """Return a valid PSSResult with placeholder status.

    Use this wherever PSS is required by the governance gate contract but the
    full perturbation loop has not yet been implemented.

    The returned value (FALLBACK_PSS = 0.5) is the neutral midpoint — it does
    not artificially inflate or deflate the governance decision.

    Args:
        bbox_id: Must match the DetectorBox.bbox_id this PSS belongs to.

    Returns:
        PSSResult dict with status="placeholder" and value=FALLBACK_PSS.
    """
    return {
        "bbox_id": bbox_id,
        "value": FALLBACK_PSS,
        "status": "placeholder",
        "n_augmentations": 0,
    }
