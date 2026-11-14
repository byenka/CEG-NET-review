

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


    head: str
    bbox: List[float]  # [x1, y1, x2, y2]
    class_probs: Dict[str, float]
    score: float


class DetectorBox(TypedDict):


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

    return {
        "bbox_id": bbox_id,
        "value": FALLBACK_PSS,
        "status": "placeholder",
        "n_augmentations": 0,
    }
