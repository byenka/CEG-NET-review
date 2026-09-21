"""
certificate.py
Per-detection governance certificate for LGov-Net (Contribution 2 output).

LEGACY MODULE, NOT USED FOR THE REPORTED RESULTS.
--------------------------------------------------
`build_certificate` and `build_certificate_batch` below are not called by any script in
this repository. They describe an earlier schema in which the tier came from a four-way
gate argmax (`predicted_tier`, `tier_confidence`). The manuscript instead defines tiers by
the two-way rule over (retain decision, SDCA) in Eq. (tier), and `target_tier` here is an
annotation-derived pseudo-label that the current routing never uses.

The certificates reported in the paper are produced by:
  tools/emit_certificates.py            emits one record per detection, four cohorts
  tools/verify_certificates.py          determinism, decision reconstruction, prior
                                        consistency, ranker independence
Their schema is recorded in outputs/0506/certificates/manifest.json.

This file is retained because the field definitions below document the original design.
Read it as documentation, not as the code path behind any reported number.

A governance certificate is a structured audit record produced after the
integration step completes.  It captures all signals used to make a
routing/tier decision plus the final outcome, enabling:
  - Clinical audit trail (who decided what and why)
  - Paper ablation analysis (field-by-field)
  - Debugging mis-routed detections
  - JSON/CSV export for downstream evaluation

Certificate fields (all required):
  bbox_id             str    — unique bbox identifier
  image_id            str    — parent image identifier
  head_votes          dict   — {class_name: vote_count} from head_predictions
  majority_vote       str    — class with most head votes
  sas                 float  — Source Agreement Score [0,1]
  sas_status          str    — "head_agreement" (canonical) or
                               "fallback_from_ensemble" (degraded proxy)
  sdca                float  — Semantic Distance Conflict Assessment [0,1]
  aps                 float  — Anatomical Plausibility Score [0,1]
  pss                 float  — Perturbation Stability Score [0,1]
  assigned_zone       str|None — ETDRS zone label
  target_tier         str    — routing pseudo-label: R1/R2/R3/R4
  predicted_tier      str|None — gate argmax tier name (None if no gate_output)
  predicted_tier_index int|None — gate argmax index 0–3 (None if no gate_output)
  tier_confidence     float|None — gate probability for predicted_tier
  refinement_outcome  str|None — "upgraded_to_R2"|"downgraded_to_R4"|"kept_as_R3"|None
  refined_bbox        list|None — [x1,y1,x2,y2] after R3 refinement or None
  timestamp           str    — ISO-8601 UTC from routing record

Public API:
  build_certificate(image_id, box, ev_record, routing_record,
                    gate_output, refine_map) -> dict
  build_certificate_batch(image_ids_per_box, boxes, evidence_batch,
                          routing_batch, gate_output, refinement_records) -> list[dict]

Dependencies:
  - src/governance/contracts.py  (LESION_CLASSES)
  NO dependency on training loop, losses, or encoder.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple


# ── Helpers ───────────────────────────────────────────────────────────────────


def _count_head_votes(box: dict) -> Tuple[Dict[str, int], str]:
    """Count how many heads voted for each class.

    Each head_prediction casts one vote for its argmax class (highest class_prob).

    Args:
        box: DetectorBox dict with "head_predictions" list.

    Returns:
        (votes_dict, majority_class)
        votes_dict: {class_name: count}  — zero counts omitted
        majority_class: class with the highest vote count; ties broken by
                        first occurrence in class_probs key order.
    """
    head_predictions: List[dict] = box.get("head_predictions", [])
    votes: Dict[str, int] = {}
    for hp in head_predictions:
        class_probs: Dict[str, float] = hp.get("class_probs", {})
        if not class_probs:
            continue
        argmax_class = max(class_probs, key=lambda c: class_probs[c])
        votes[argmax_class] = votes.get(argmax_class, 0) + 1

    if not votes:
        # Fallback: use ensemble_probs argmax
        ens = box.get("ensemble_probs", {})
        if ens:
            argmax_class = max(ens, key=lambda c: ens[c])
            votes = {argmax_class: 1}
            majority = argmax_class
        else:
            majority = "MA"  # last-resort default
    else:
        majority = max(votes, key=lambda c: votes[c])

    return votes, majority


def _refinement_outcome_from_record(refine_record: Optional[dict]) -> Optional[str]:
    """Extract a human-readable refinement outcome from a refine.py record.

    Args:
        refine_record: One record from refine_r3_batch(), or None if not R3.

    Returns:
        "upgraded_to_R2" | "downgraded_to_R4" | "kept_as_R3" | None
    """
    if refine_record is None:
        return None
    if refine_record.get("upgraded_to") == "R2":
        return "upgraded_to_R2"
    if refine_record.get("downgraded_to") == "R4":
        return "downgraded_to_R4"
    if refine_record.get("kept_as") == "R3":
        return "kept_as_R3"
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Certificate builder
# ══════════════════════════════════════════════════════════════════════════════


def build_certificate(
    image_id: str,
    box: dict,
    ev_record: dict,
    routing_record: dict,
    gate_output: Optional[dict],
    refine_record: Optional[dict],
) -> dict:
    """Build one governance certificate for a single bbox.

    Args:
        image_id:
            Identifier of the parent image.
        box:
            DetectorBox dict (must contain bbox_id, ensemble_probs,
            head_predictions, assigned_zone).
        ev_record:
            One record from build_evidence_batch() for this bbox.
            Expected keys: sas, sdca_norm, aps, pss (or pss_value).
            Optional key: sas_status (str) — "head_agreement" or
            "fallback_from_ensemble".  Defaults to "head_agreement" when absent.
        routing_record:
            One record from route_bbox_batch() for this bbox.
            Expected keys: target_tier, timestamp.
        gate_output:
            Full gate_output dict from gumbel_forward() for the batch,
            OR None.  If provided, "tier_hard" ([B] tensor) is used to
            extract the per-bbox predicted tier.  The index into this
            tensor is *not* passed here — pass None and set
            predicted_tier_str directly via the batch helper.
        refine_record:
            One record from refine_r3_batch() for this bbox (only
            present if target_tier == "R3"), or None.

    Returns:
        Governance certificate dict with all required fields.
    """
    bbox_id: str = box.get("bbox_id", "unknown")
    assigned_zone: Optional[str] = box.get("assigned_zone", None)

    # Head votes
    head_votes, majority_vote = _count_head_votes(box)

    # Evidence signals — prefer explicit keys, fall back to evidence_vector
    ev_vec = ev_record.get("evidence_vector", [0.0, 0.0, 0.0, 0.0])
    sas: float = float(ev_record.get("sas", ev_vec[0] if len(ev_vec) > 0 else 0.0))
    sdca: float = float(
        ev_record.get("sdca_norm", ev_vec[1] if len(ev_vec) > 1 else 0.0)
    )
    aps: float = float(ev_record.get("aps", ev_vec[2] if len(ev_vec) > 2 else 0.0))
    # PSS: check pss key first, then pss_value (contracts.py uses both)
    pss: float = float(
        ev_record.get(
            "pss",
            ev_record.get("pss_value", ev_vec[3] if len(ev_vec) > 3 else 0.5),
        )
    )

    # SAS status: explicit from ev_record, or default to head_agreement
    sas_status: str = ev_record.get("sas_status", "head_agreement")

    # Tier from routing pseudo-label
    target_tier: str = routing_record.get("target_tier", "R4")
    timestamp: str = routing_record.get(
        "timestamp", datetime.now(timezone.utc).isoformat()
    )

    # Predicted tier / confidence — populated by build_certificate_batch().
    predicted_tier: Optional[str] = None
    predicted_tier_index: Optional[int] = None
    tier_confidence: Optional[float] = None

    # Refinement outcome
    refinement_outcome: Optional[str] = _refinement_outcome_from_record(refine_record)
    refined_bbox: Optional[list] = (
        refine_record.get("refined_bbox") if refine_record is not None else None
    )

    return {
        "bbox_id": bbox_id,
        "image_id": image_id,
        "head_votes": head_votes,
        "majority_vote": majority_vote,
        "sas": sas,
        "sas_status": sas_status,
        "sdca": sdca,
        "aps": aps,
        "pss": pss,
        "assigned_zone": assigned_zone,
        "target_tier": target_tier,
        "predicted_tier": predicted_tier,  # filled in by batch helper
        "predicted_tier_index": predicted_tier_index,  # filled in by batch helper
        "tier_confidence": tier_confidence,  # filled in by batch helper
        "refinement_outcome": refinement_outcome,
        "refined_bbox": refined_bbox,
        "timestamp": timestamp,
    }


# ── Tier name lookup ──────────────────────────────────────────────────────────

_TIER_NAMES = ["R1", "R2", "R3", "R4"]


def build_certificate_batch(
    image_ids_per_box: List[str],
    boxes: List[dict],
    evidence_batch: List[dict],
    routing_batch: List[dict],
    gate_output: Optional[dict],
    refinement_records: List[dict],
) -> List[dict]:
    """Build governance certificates for a full bbox batch.

    Args:
        image_ids_per_box:
            image_id string for each bbox in the batch (len == B).
        boxes:
            Flat list of DetectorBox dicts — same order as evidence_batch /
            routing_batch.
        evidence_batch:
            Output of build_evidence_batch() — one record per bbox.
        routing_batch:
            Output of route_bbox_batch() — one record per bbox.
        gate_output:
            Dict from LGG.gumbel_forward().  Must contain "tier_hard"
            ([B] int64 tensor) to populate predicted_tier.  Pass None
            to skip predicted_tier population.
        refinement_records:
            Output of refine_r3_batch() — records ONLY for R3 bboxes.
            These are matched by bbox_id back to the batch.

    Returns:
        List of certificate dicts, one per bbox in the same order as boxes.
    """
    B = len(boxes)
    assert (
        len(evidence_batch) == B
    ), f"evidence_batch length {len(evidence_batch)} != boxes length {B}"
    assert (
        len(routing_batch) == B
    ), f"routing_batch length {len(routing_batch)} != boxes length {B}"
    assert (
        len(image_ids_per_box) == B
    ), f"image_ids_per_box length {len(image_ids_per_box)} != boxes length {B}"

    # Build bbox_id → refinement_record map for O(1) lookup
    refine_map: Dict[str, dict] = {
        r["bbox_id"]: r for r in refinement_records if "bbox_id" in r
    }

    # Extract predicted tier indices + probabilities from gate output
    tier_hard_list: Optional[List[int]] = None
    tier_probs_list: Optional[List[List[float]]] = None
    if gate_output is not None:
        tier_hard = gate_output.get("tier_hard")
        if tier_hard is not None:
            try:
                tier_hard_list = tier_hard.tolist()
            except AttributeError:
                tier_hard_list = list(tier_hard)

        tier_probs = gate_output.get("tier_probs")
        if tier_probs is not None:
            try:
                tier_probs_list = tier_probs.tolist()
            except AttributeError:
                tier_probs_list = list(tier_probs)

    certificates: List[dict] = []
    for i, (img_id, box, ev_rec, route_rec) in enumerate(
        zip(image_ids_per_box, boxes, evidence_batch, routing_batch)
    ):
        bbox_id = box.get("bbox_id", f"unknown_{i}")
        refine_rec = refine_map.get(bbox_id, None)

        cert = build_certificate(
            image_id=img_id,
            box=box,
            ev_record=ev_rec,
            routing_record=route_rec,
            gate_output=None,  # handled below
            refine_record=refine_rec,
        )

        # P1 (PATCH-15.5): Populate predicted_tier, predicted_tier_index,
        # and tier_confidence from gate output tensor slices.
        if tier_hard_list is not None and i < len(tier_hard_list):
            tier_idx = int(tier_hard_list[i])
            cert["predicted_tier"] = (
                _TIER_NAMES[tier_idx] if 0 <= tier_idx < len(_TIER_NAMES) else None
            )
            cert["predicted_tier_index"] = (
                tier_idx if 0 <= tier_idx < len(_TIER_NAMES) else None
            )
            # tier_confidence: gate probability mass on the predicted tier
            if tier_probs_list is not None and i < len(tier_probs_list):
                probs_row = tier_probs_list[i]
                cert["tier_confidence"] = (
                    float(probs_row[tier_idx])
                    if 0 <= tier_idx < len(probs_row)
                    else None
                )

        certificates.append(cert)

    return certificates
