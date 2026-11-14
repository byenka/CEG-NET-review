

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

try:
    from src.governance.contracts import (
        FALLBACK_PSS,
        LESION_CLASSES,
        make_placeholder_pss,
    )
except ModuleNotFoundError:
    from lgovnet.src.governance.contracts import (
        FALLBACK_PSS,
        LESION_CLASSES,
        make_placeholder_pss,
    )

# ── Constants ─────────────────────────────────────────────────────────────────

# Canonical orderings — must match contracts.py and encoder.py
_LESION_CLASSES: Tuple[str, ...] = ("MA", "HE", "Hard_EX", "Soft_EX")
_ZONE_NAMES: Tuple[str, ...] = (
    "Foveal",
    "Parafoveal",
    "Peripapillary",
    "Vascular_Arcades",
    "Peripheral",
)

# SAS floor: with 4 classes the minimum possible max-prob under uniform is 0.25.
# We use this as the theoretical lower bound for documentation only.
_SAS_MIN_THEORETICAL: float = 1.0 / len(_LESION_CLASSES)

# SDCA fallback when the two top classes are not found in the SDM dict.
# 0.0 → "no KG separation information available" — conservative.
_SDCA_FALLBACK: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Signal computation functions
# ═══════════════════════════════════════════════════════════════════════════════


def compute_sas(
    ensemble_probs: Dict[str, float],
    head_predictions: Optional[List[dict]] = None,
    mode: str = "unweighted",
) -> Tuple[float, str]:

    if not ensemble_probs:
        raise ValueError("ensemble_probs is empty — cannot compute SAS")

    majority_class = max(ensemble_probs, key=lambda c: ensemble_probs[c])

    # ── Explicit max_ensemble mode (no head data needed) ─────────────────
    if mode == "max_ensemble":
        return float(ensemble_probs[majority_class]), majority_class, False

    # ── Fallback: no per-head data available ──────────────────────────────
    # Returns max(ensemble_probs) as a degraded proxy; used_fallback=True
    # signals to callers that this is NOT the canonical head-agreement SAS.
    if not head_predictions:
        return float(ensemble_probs[majority_class]), majority_class, True

    # ── Cross-head agreement ──────────────────────────────────────────────
    valid_heads = [
        h
        for h in head_predictions
        if isinstance(h.get("class_probs"), dict) and h["class_probs"]
    ]
    if not valid_heads:
        return float(ensemble_probs[majority_class]), majority_class, True

    if mode == "confidence_weighted":
        total_weight = sum(float(h.get("score", 1.0)) for h in valid_heads)
        if total_weight < 1e-9:
            # degenerate: all scores zero → unweighted fallback
            agree_weight = sum(
                1.0
                for h in valid_heads
                if max(h["class_probs"], key=lambda c: h["class_probs"][c])
                == majority_class
            )
            sas_value = agree_weight / len(valid_heads)
        else:
            agree_weight = sum(
                float(h.get("score", 1.0))
                for h in valid_heads
                if max(h["class_probs"], key=lambda c: h["class_probs"][c])
                == majority_class
            )
            sas_value = agree_weight / total_weight
    else:  # unweighted (default, canonical)
        agree_count = sum(
            1
            for h in valid_heads
            if max(h["class_probs"], key=lambda c: h["class_probs"][c])
            == majority_class
        )
        sas_value = agree_count / len(valid_heads)

    return float(sas_value), majority_class, False


# ──────────────────────────────────────────────────────────────────────────────
# Multi-source predictive DISAGREEMENT (epistemic uncertainty)  [2026-06-05]
#
# Motivation (root-cause audit): the canonical SAS = argmax head-agreement
# saturates near 1.0 (mean 0.98, std 0.10 on the locked run) because heterogeneous
# heads usually agree on the *argmax* even when their full class DISTRIBUTIONS
# diverge. Argmax-agreement therefore throws away the distributional disagreement
# that is the strongest epistemic-uncertainty / false-positive signal.
#
# compute_source_disagreement() replaces the saturating agreement signal with the
# mean pairwise Jensen–Shannon divergence (log base 2, range [0,1]) across the
# per-source class distributions (e.g. Faster R-CNN primary + RetinaNet-r1a
# secondary + TTA views). High JSD = sources distributionally disagree = likely
# FP; it has full dynamic range and is largely orthogonal to the detector score.
# This is the new evidence pillar `ev_sds` (Source Disagreement Score).
# ──────────────────────────────────────────────────────────────────────────────

def _kl_div(p: List[float], q: List[float], eps: float = 1e-12) -> float:
    return sum(pi * math.log2((pi + eps) / (qi + eps)) for pi, qi in zip(p, q) if pi > 0.0)


def _js_div(p: List[float], q: List[float]) -> float:

    m = [0.5 * (pi + qi) for pi, qi in zip(p, q)]
    return 0.5 * _kl_div(p, m) + 0.5 * _kl_div(q, m)


def compute_source_disagreement(
    source_dists: List[Dict[str, float]],
    classes: Tuple[str, ...] = ("MA", "HE", "Hard_EX", "Soft_EX"),
) -> Tuple[float, int]:

    vecs: List[List[float]] = []
    for d in source_dists or []:
        if not isinstance(d, dict) or not d:
            continue
        v = [max(float(d.get(c, 0.0)), 0.0) for c in classes]
        s = sum(v)
        if s <= 0.0:
            continue
        vecs.append([x / s for x in v])
    n = len(vecs)
    if n < 2:
        return 0.0, n
    total, pairs = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += _js_div(vecs[i], vecs[j])
            pairs += 1
    sds = total / pairs if pairs else 0.0
    # numeric guard: JSD(log2) ∈ [0,1]
    return float(min(max(sds, 0.0), 1.0)), n


def compute_sdca(
    majority_class: str,
    ensemble_probs: Dict[str, float],
    sdm_norm: Dict[Tuple[str, str], Dict[str, float]],
) -> Tuple[float, float]:

    # Find runner-up (second highest probability class)
    sorted_classes = sorted(
        ensemble_probs, key=lambda c: ensemble_probs[c], reverse=True
    )
    if len(sorted_classes) < 2:
        return _SDCA_FALLBACK, _SDCA_FALLBACK

    runner_up = next((c for c in sorted_classes if c != majority_class), None)
    if runner_up is None:
        return _SDCA_FALLBACK, _SDCA_FALLBACK

    # Build canonical key (lower lesion index first)
    idx = {c: i for i, c in enumerate(_LESION_CLASSES)}
    if majority_class not in idx or runner_up not in idx:
        return _SDCA_FALLBACK, _SDCA_FALLBACK

    a, b = (
        (majority_class, runner_up)
        if idx[majority_class] < idx[runner_up]
        else (runner_up, majority_class)
    )
    entry = sdm_norm.get((a, b))

    if entry is None:
        return _SDCA_FALLBACK, _SDCA_FALLBACK

    return float(entry["raw"]), float(entry["norm"])


def compute_aps(
    majority_class: str,
    assigned_zone: str,
    compat_scores: Dict[Tuple[str, str], float],
) -> float:

    key = (majority_class, assigned_zone)
    return float(compat_scores.get(key, 0.5))


def compute_pss(pss_result: dict) -> Tuple[float, str]:

    status = pss_result.get("status", "missing")
    value = pss_result.get("value")

    if status == "implemented" and value is not None:
        return float(value), "implemented"
    # placeholder or missing → use neutral fallback
    return float(FALLBACK_PSS), status


# ═══════════════════════════════════════════════════════════════════════════════
# Evidence record
# ═══════════════════════════════════════════════════════════════════════════════


def build_evidence_for_bbox(
    box: dict,
    pss_result: dict,
    sdm_norm: Dict[Tuple[str, str], Dict[str, float]],
    compat_scores: Dict[Tuple[str, str], float],
    anatomy_context: Optional[dict] = None,
) -> dict:

    warnings: List[str] = []

    bbox_id: str = box.get("bbox_id", "")
    if not bbox_id:
        raise KeyError("box missing required field: bbox_id")

    ensemble_probs: dict = box.get("ensemble_probs")
    if not ensemble_probs:
        raise ValueError(f"[{bbox_id}] ensemble_probs is empty or missing")

    # Validate class keys
    missing_classes = [c for c in _LESION_CLASSES if c not in ensemble_probs]
    if missing_classes:
        warnings.append(f"ensemble_probs missing classes: {missing_classes}")

    assigned_zone: Optional[str] = box.get("assigned_zone")
    if not assigned_zone:
        warnings.append("assigned_zone is missing — APS will return neutral 0.5")
        assigned_zone = None

    if assigned_zone and assigned_zone not in _ZONE_NAMES:
        warnings.append(
            f"assigned_zone '{assigned_zone}' not in ZONE_NAMES — APS neutral"
        )
        assigned_zone = None

    # ── SAS ─────────────────────────────────────────────────────────────────
    head_predictions: list = box.get("head_predictions") or []
    sas, majority_class, sas_used_fallback = compute_sas(
        ensemble_probs, head_predictions=head_predictions, mode="unweighted"
    )
    if sas_used_fallback:
        warnings.append(
            "SAS fell back to max(ensemble_probs) — head_predictions absent or invalid. "
            "This is a DEGRADED PROXY, not the canonical head-agreement SAS. "
            "Populate head_predictions for accurate SAS computation."
        )

    if majority_class not in _LESION_CLASSES:
        warnings.append(
            f"majority_class '{majority_class}' not in LESION_CLASSES — proceeding"
        )

    # ── SDCA ────────────────────────────────────────────────────────────────
    # Pilar 1 (gap G1): prefer the REAL multinomial probabilities from the
    # detector ROI head (`detector_class_probs`) over the synthetic
    # `ensemble_probs` for runner-up selection. This turns SDCA from a
    # 4-valued quantized lookup into a continuous-resolution signal.
    _real_probs = box.get("detector_class_probs")
    if isinstance(_real_probs, dict) and all(c in _real_probs for c in _LESION_CLASSES):
        sdca_raw, sdca_norm = compute_sdca(majority_class, _real_probs, sdm_norm)
    else:
        sdca_raw, sdca_norm = compute_sdca(majority_class, ensemble_probs, sdm_norm)
    if sdca_raw == _SDCA_FALLBACK and sdca_norm == _SDCA_FALLBACK:
        warnings.append(
            f"SDCA pair ({majority_class}, runner-up) not found in sdm_norm — using fallback 0.0"
        )

    # ── APS ─────────────────────────────────────────────────────────────────
    # Architecture v3 fail-safe: when AnatomyProvider returns od_confidence
    # == "low" or "n/a", APS must be 0.5 (neutral) and anatomy_uncertain=True.
    _ac_confidence = (anatomy_context or {}).get("od_confidence", "n/a")
    _ac_provider = (anatomy_context or {}).get("provider_name")
    _APS_LOW_CONFIDENCE = frozenset({"low", "n/a"})

    # Pilar 3 (gap G3): when RadialZoneAnatomyProvider attaches a `radial_zone`
    # to the box AND od_confidence == 'moderate', compute APS from the radial
    # ring × class compatibility prior (merged into compat_scores upstream).
    radial_zone = box.get("radial_zone")

    anatomy_uncertain: bool
    if _ac_confidence in _APS_LOW_CONFIDENCE:
        aps = 0.5  # neutral — low-confidence anatomy cannot bias APS
        anatomy_uncertain = True
        if anatomy_context is not None:
            warnings.append(
                f"APS forced to 0.5 (neutral) — anatomy od_confidence='{_ac_confidence}' "
                f"(provider: {_ac_provider!r}). anatomy_uncertain=True in certificate."
            )
        else:
            warnings.append(
                "APS set to 0.5 (neutral) — no anatomy_context provided. "
                "anatomy_uncertain=True. Pass anatomy_context for APS computation."
            )
    elif radial_zone is not None and _ac_confidence == "moderate":
        aps = compute_aps(majority_class, radial_zone, compat_scores)
        anatomy_uncertain = False
    elif assigned_zone is not None:
        aps = compute_aps(majority_class, assigned_zone, compat_scores)
        anatomy_uncertain = False
    else:
        aps = 0.5  # neutral when zone unknown despite having moderate/high confidence
        anatomy_uncertain = True
        warnings.append("APS set to 0.5 (neutral) — assigned_zone unavailable")

    # ── PSS ─────────────────────────────────────────────────────────────────
    pss_id = pss_result.get("bbox_id", "")
    if pss_id and pss_id != bbox_id:
        warnings.append(
            f"pss_result.bbox_id '{pss_id}' does not match box.bbox_id '{bbox_id}'"
        )

    pss_value, pss_status = compute_pss(pss_result)

    # ── Evidence vector ──────────────────────────────────────────────────────
    # Clamp all values to [0, 1] defensively before handing to gate MLP.
    def _clamp(v: float) -> float:
        return max(0.0, min(1.0, v))

    evidence_vector = [
        _clamp(sas),
        _clamp(sdca_norm),
        _clamp(aps),
        _clamp(pss_value),
    ]

    return {
        "bbox_id": bbox_id,
        "majority_class": majority_class,
        "zone": assigned_zone,
        "sas": _clamp(sas),
        "sdca_raw": sdca_raw,
        "sdca_norm": _clamp(sdca_norm),
        "aps": _clamp(aps),
        "pss": _clamp(pss_value),
        "pss_status": pss_status,
        "anatomy_uncertain": anatomy_uncertain,
        "anatomy_provider": _ac_provider,
        "evidence_vector": evidence_vector,
        "warnings": warnings,
    }


def build_evidence_batch(
    boxes: List[dict],
    pss_results: Dict[str, dict],
    sdm_norm: Dict[Tuple[str, str], Dict[str, float]],
    compat_scores: Dict[Tuple[str, str], float],
    anatomy_context: Optional[dict] = None,
) -> List[dict]:

    results: List[dict] = []

    for box in boxes:
        bbox_id = box.get("bbox_id", f"unknown_{len(results)}")
        pss = pss_results.get(bbox_id)

        if pss is None:
            pss = make_placeholder_pss(bbox_id)

        try:
            ev = build_evidence_for_bbox(
                box=box,
                pss_result=pss,
                sdm_norm=sdm_norm,
                compat_scores=compat_scores,
                anatomy_context=anatomy_context,
            )
        except Exception as exc:  # noqa: BLE001
            ev = {
                "bbox_id": bbox_id,
                "error": str(exc),
                "evidence_vector": None,
                "warnings": [],
            }

        results.append(ev)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Minimal dummy usage demo (runs when executed directly)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json as _json

    # ── Fake SDM_norm (as would come from normalize_sdca()) ──────────────────
    _fake_sdm_norm = {
        ("MA", "HE"): {"raw": 0.11, "norm": 0.85},
        ("MA", "Hard_EX"): {"raw": 0.13, "norm": 1.00},
        ("MA", "Soft_EX"): {"raw": 0.10, "norm": 0.77},
        ("HE", "Hard_EX"): {"raw": 0.04, "norm": 0.31},
        ("HE", "Soft_EX"): {"raw": 0.06, "norm": 0.46},
        ("Hard_EX", "Soft_EX"): {"raw": 0.08, "norm": 0.62},
    }

    # ── Fake Compat scores (as would come from compute_compat()) ─────────────
    _fake_compat = {
        ("MA", "Foveal"): 0.51,
        ("MA", "Parafoveal"): 0.51,
        ("MA", "Peripapillary"): 0.51,
        ("MA", "Vascular_Arcades"): 0.51,
        ("MA", "Peripheral"): 0.51,
        ("HE", "Foveal"): 0.50,
        ("HE", "Parafoveal"): 0.50,
        ("HE", "Peripapillary"): 0.50,
        ("HE", "Vascular_Arcades"): 0.50,
        ("HE", "Peripheral"): 0.50,
        ("Hard_EX", "Foveal"): 0.72,
        ("Hard_EX", "Parafoveal"): 0.72,
        ("Hard_EX", "Peripapillary"): 0.51,
        ("Hard_EX", "Vascular_Arcades"): 0.51,
        ("Hard_EX", "Peripheral"): 0.51,
        ("Soft_EX", "Foveal"): 0.51,
        ("Soft_EX", "Parafoveal"): 0.51,
        ("Soft_EX", "Peripapillary"): 0.74,
        ("Soft_EX", "Vascular_Arcades"): 0.51,
        ("Soft_EX", "Peripheral"): 0.51,
    }

    # ── Dummy DetectorBox ────────────────────────────────────────────────────
    _box = {
        "bbox_id": "img0042_0",
        "head_predictions": [
            {
                "head": cls,
                "bbox": [100.0, 200.0, 150.0, 250.0],
                "class_probs": {
                    c: (0.7 if c == "Hard_EX" else 0.1) for c in LESION_CLASSES
                },
                "score": 0.85,
            }
            for cls in LESION_CLASSES
        ],
        "ensemble_probs": {"MA": 0.05, "HE": 0.10, "Hard_EX": 0.75, "Soft_EX": 0.10},
        "assigned_zone": "Foveal",
    }

    _pss = make_placeholder_pss("img0042_0")

    ev = build_evidence_for_bbox(
        box=_box,
        pss_result=_pss,
        sdm_norm=_fake_sdm_norm,
        compat_scores=_fake_compat,
    )

    print("=== evidence.py demo ===")
    print(_json.dumps(ev, indent=2))
    print()
    print(f"evidence_vector = {ev['evidence_vector']}")
    print(f"  SAS       = {ev['sas']:.4f}  (head agreement)")
    print(f"  SDCA_norm = {ev['sdca_norm']:.4f}  (KG semantic separation)")
    print(f"  APS       = {ev['aps']:.4f}  (anatomical plausibility)")
    print(f"  PSS       = {ev['pss']:.4f}  [{ev['pss_status']}]")
    if ev["warnings"]:
        print(f"  warnings  = {ev['warnings']}")
