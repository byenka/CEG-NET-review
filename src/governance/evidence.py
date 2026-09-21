"""
evidence.py
Per-bbox evidence computation for LGov-Net Governance Gate (Contribution 2).

Computes an evidence vector  g(b_j) = [SAS, SDCA_norm, APS, PSS]
for every candidate bounding box before it enters the governance gate.

Four signals:
  SAS  — Source Agreement Score
         Measures how consistently the 4 detection heads agree on the
         predicted class for this bbox.
         SAS = max(ensemble_probs)  i.e. the winning class probability
         averaged over all heads.
         Range: (0.25, 1.0].  SAS = 1.0 → all heads agree perfectly.
         SAS = 0.25 → maximum disagreement (uniform over 4 classes).

  SDCA — Semantic Distance Conflict Assessment
         Quantifies how semantically "distant" the predicted class is from
         the runner-up class in the KG embedding space.
         A large L2 distance between predicted and second-most-probable class
         embeddings means the KG representation strongly separates them
         (the prediction is unambiguous w.r.t. KG knowledge).
         Uses normalized SDM (log_scaled) so range is [0, 1].
         sdca_raw is preserved for debugging.

  APS  — Anatomical Plausibility Score
         Compat(predicted_class, assigned_zone) from the calibrated bilinear
         compatibility scorer in encoder.py.
         APS = σ( e_c^T W_compat e_a + prior[c, a] )
         Range: (0, 1).  APS > 0.5 → plausible; < 0.5 → suspicious.
         Requires that assigned_zone is a valid ETDRS zone.

  PSS  — Perturbation Stability Score  (Phase 9.3 formal spec)
         count(majority_class across N augmented predictions) / N
         In present phase: placeholder value FALLBACK_PSS = 0.5 is accepted
         and passed through transparently.  When status="implemented" the
         real value is used.

Evidence vector contract:
  evidence_vector = [SAS, SDCA_norm, APS, PSS_or_fallback]
  All 4 values must be in [0.0, 1.0].

Usage:
  from src.governance.evidence import build_evidence_for_bbox, build_evidence_batch

  ev = build_evidence_for_bbox(
      box=detector_box,
      pss_result=make_placeholder_pss(box["bbox_id"]),
      sdm_norm=sdca_dict,       # from normalize_sdca()
      compat_scores=compat,     # from compute_compat()
  )
  # ev["evidence_vector"] → [SAS, SDCA_norm, APS, PSS]

Dependencies:
  - src/governance/contracts.py  (DetectorBox, PSSResult, FALLBACK_PSS, …)
  - src/kg/encoder.py             (normalize_sdca, compute_sdm, compute_compat)
  NO dependency on governance gate, detector backbone, or training loop.
"""

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
    """Compute Source Agreement Score from cross-head agreement.

    OFFICIAL DEFINITION (architecture spec, Contribution 2):
        SAS = (# detection heads whose argmax == majority_class)
              ─────────────────────────────────────────────────────
              (# total valid heads)

        majority_class = argmax(ensemble_probs)
        Range: (0, 1].  SAS=1.0 → all heads agree; SAS=0.25 → maximal split.

    Three modes are supported:

    ``unweighted`` (default, CANONICAL):
        SAS = agree_count / total_valid_heads  (as per official definition above).
        Requires head_predictions to be populated.

    ``confidence_weighted``:
        SAS = Σ_{h agrees} head.score  /  Σ_{h all} head.score
        Weights each head vote by detection confidence.  Same range as unweighted.
        Use when head quality is known to vary significantly.

    ``max_ensemble`` (explicit opt-in):
        SAS = max(ensemble_probs)  — the probability mass on the majority class.
        NOT the same as the official head-agreement formula; this is a rougher
        proxy, appropriate when per-head predictions are unavailable at inference.
        Range: (0.25, 1.0].

    Fallback (backward-compatible):
        When head_predictions is None or empty and mode is not ``max_ensemble``,
        the function automatically returns max(ensemble_probs) and sets
        used_fallback=True.  This is a DEGRADED PROXY for the true SAS —
        it uses ensemble output instead of per-head agreement.
        Callers should treat used_fallback=True as a data-quality warning.

    Args:
        ensemble_probs:   {class_name: probability} summing to ~1.0.
                          Must contain at least one entry.
        head_predictions: List of HeadPrediction dicts (DetectorBox contract).
                          Each must have ``class_probs`` and optionally
                          ``score``.  May be None or empty — triggers fallback
                          unless mode="max_ensemble".
        mode:             ``"unweighted"`` (default) | ``"confidence_weighted"``
                          | ``"max_ensemble"``.

    Returns:
        (sas_value, majority_class, used_fallback)
        sas_value:      float in (0, 1].
        majority_class: argmax of ensemble_probs.
        used_fallback:  bool — True when per-head data was absent and
                        max(ensemble_probs) was used as a proxy instead.
                        Always False for mode="max_ensemble".

    Raises:
        ValueError: if ensemble_probs is empty.
    """
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
    """Jensen–Shannon divergence (log2 -> range [0,1]) between two distributions."""
    m = [0.5 * (pi + qi) for pi, qi in zip(p, q)]
    return 0.5 * _kl_div(p, m) + 0.5 * _kl_div(q, m)


def compute_source_disagreement(
    source_dists: List[Dict[str, float]],
    classes: Tuple[str, ...] = ("MA", "HE", "Hard_EX", "Soft_EX"),
) -> Tuple[float, int]:
    """Mean pairwise Jensen–Shannon divergence across heterogeneous source heads.

    Args:
        source_dists: one class-probability dict per source (primary detector,
                      secondary detector, TTA views, ...). Each maps class->prob.
        classes:      canonical class order used to build the probability vectors.

    Returns:
        (sds, n_sources):
          sds       — mean pairwise JSD in [0,1]; 0.0 when < 2 valid sources
                      (no disagreement is measurable). Higher = more disagreement.
          n_sources — number of valid (non-empty, positive-mass) source dists used.
    """
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
    """Compute Semantic Distance Conflict Assessment (normalized).

    Looks up the KG semantic distance between predicted class and the
    runner-up class (second-highest probability).  Uses normalized SDM.

    Rationale: if the bbox is ambiguous (high runner-up prob), we want to
    know whether KG separates the two candidates.  A large SDM_norm between
    them means the KG treats them as distinct → evidence supports a decision.
    A small SDM_norm (e.g. MA vs HE) means the KG knows they are similar →
    the ambiguity is clinically understandable.

    Formula:
        runner_up  = argmax among all classes except majority_class
        sdca_raw   = sdm_norm[(majority, runner_up)]["raw"]
        sdca_norm  = sdm_norm[(majority, runner_up)]["norm"]

    The key in sdm_norm is always the pair with the lower lesion index first
    (as produced by encoder.compute_sdm / normalize_sdca).

    Args:
        majority_class: Predicted class name (argmax of ensemble_probs).
        ensemble_probs: {class_name: probability}.
        sdm_norm:       Output of normalize_sdca() — keys are (ci, cj) tuples
                        with ci < cj by LESION_CLASSES index order.
                        Values are {"raw": float, "norm": float}.

    Returns:
        (sdca_raw, sdca_norm)  Both floats in [0, ∞) and [0, 1] respectively.
        Returns (_SDCA_FALLBACK, _SDCA_FALLBACK) if pair not found.
    """
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
    """Compute Anatomical Plausibility Score.

    APS = Compat(majority_class, assigned_zone)
        = σ( e_c^T W_compat e_a + prior[c, a] )

    Looks up the pre-computed calibrated Compat score for this
    (predicted_class, zone) pair.

    Args:
        majority_class:  Predicted class name.
        assigned_zone:   ETDRS zone the bbox was assigned to.
        compat_scores:   Output of compute_compat() — keys are
                         (lesion_name, zone_name) tuples, values are floats
                         in (0, 1).

    Returns:
        float in (0, 1).  Returns 0.5 (neutral) if pair not in compat_scores.
    """
    key = (majority_class, assigned_zone)
    return float(compat_scores.get(key, 0.5))


def compute_pss(pss_result: dict) -> Tuple[float, str]:
    """Extract PSS value from a PSSResult dict (Phase 9.3 contract).

    Transparently passes through either the real PSS value (status=
    "implemented") or the placeholder fallback (status="placeholder").
    When status="missing" returns FALLBACK_PSS with a warning flag.

    Args:
        pss_result: A dict conforming to PSSResult from contracts.py.
                    Must have fields: bbox_id, value, status, n_augmentations.

    Returns:
        (pss_value, status_string)
        pss_value:     float in [0, 1].  FALLBACK_PSS for non-implemented.
        status_string: "implemented" | "placeholder" | "missing".
    """
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
    """Compute the full evidence record for one candidate bounding box.

    Assembles all four signals into a single evidence dict that the
    governance gate will consume.

    Args:
        box:             DetectorBox dict (Phase 9.3 contract).
                         Must have: bbox_id, ensemble_probs, assigned_zone.
        pss_result:      PSSResult dict for this bbox (Phase 9.3 contract).
                         May be placeholder — status is preserved.
        sdm_norm:        Output of encoder.normalize_sdca() for the current
                         image/KG state.  Keys: (ci, cj) with ci < cj.
                         Values: {"raw": float, "norm": float}.
        compat_scores:   Output of encoder.compute_compat() with compat_prior.
                         Keys: (lesion_name, zone_name), Values: float in (0,1).
        anatomy_context: Optional AnatomyContext.to_dict() for this image.
                         When provided and od_confidence is "low" or "n/a",
                         APS is forced to 0.5 (neutral) and anatomy_uncertain=True
                         is set in the returned record (architecture v3 APS fail-safe).
                         When None, APS is computed from assigned_zone as usual
                         but anatomy_uncertain is True (unknown provider).

    Returns:
        EvidenceRecord dict:
        {
            "bbox_id":             str,
            "majority_class":      str | None,
            "zone":                str | None,
            "sas":                 float,         # Source Agreement Score
            "sdca_raw":            float,         # raw L2 distance (debug)
            "sdca_norm":           float,         # normalized [0,1]
            "aps":                 float,         # Anatomical Plausibility
            "pss":                 float,         # Perturbation Stability
            "pss_status":          str,           # "implemented"/"placeholder"/"missing"
            "anatomy_uncertain":   bool,          # True when APS fail-safe was applied
            "anatomy_provider":    str | None,    # provider_name from AnatomyContext
            "evidence_vector":     [float]*4,     # [SAS, SDCA_norm, APS, PSS]
            "warnings":            list[str],     # non-fatal issues found
        }

    Raises:
        KeyError:   if bbox_id or ensemble_probs missing from box.
        ValueError: if ensemble_probs contains no entries.
    """
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
    """Compute evidence records for a batch of bounding boxes.

    A missing PSS entry for a bbox is handled gracefully by generating a
    placeholder PSS automatically — the absence is recorded in the returned
    warning field.

    Args:
        boxes:           List of DetectorBox dicts for one image.
        pss_results:     {bbox_id: PSSResult dict}.  May be empty or partial.
        sdm_norm:        Output of normalize_sdca() — shared across all boxes
                         in the same forward pass (KG state is image-independent).
        compat_scores:   Output of compute_compat() — shared across all boxes.
        anatomy_context: AnatomyContext.to_dict() for this image (shared by all
                         boxes in the image). When od_confidence is "low"/"n/a",
                         APS is forced to 0.5 for all boxes in the batch.
                         Pass None to use zone-based APS without provider metadata
                         (anatomy_uncertain=True will be set for all boxes).

    Returns:
        List of EvidenceRecord dicts, one per input box.  Same order as input.
        Failed boxes are included with an "error" key instead of signals.
    """
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
