"""Pipeline debug metrics for LGov-Net batch outputs.

``compute_pipeline_metrics()`` aggregates four debug counters from a single
inference batch.  Call it after ``route_bbox_batch()`` and ``refine_r3_batch()``
have both completed for the same batch of bboxes.

Debug metrics
-------------
fallback_sas_count
    Number of evidence records that used the degraded-proxy SAS path
    (i.e. ``head_predictions=None`` was passed to ``compute_sas()``).
    Tracked via the ``"DEGRADED PROXY"`` sentinel in the warnings list.
tier_distribution
    Dict mapping each governance tier (int 0-3) to the number of bboxes
    routed to that tier.  R4 = 3, R3 = 2, R2 = 1, R1 = 0.
r3_refined_to_r2_count
    Number of R3 bboxes that were upgraded to R2 after the four-step
    refinement pipeline (Step 2: SDM class arbitration below delta_merge).
r3_downgraded_to_r4_count
    Number of R3 bboxes that were downgraded to R4 after the four-step
    refinement pipeline (Step 3: anatomical plausibility failure).
"""

from collections import Counter
from typing import Dict, List


def compute_pipeline_metrics(
    evidence_batch: List[dict],
    routing_batch: List[dict],
    refine_batch: List[dict],
) -> Dict[str, object]:
    """Compute four debug metrics from LGov-Net pipeline batch outputs.

    Parameters
    ----------
    evidence_batch : list[dict]
        One dict per bbox from ``build_evidence_for_bbox()``.
        Each dict may contain a ``"warnings"`` key (list[str]).
    routing_batch : list[dict]
        One dict per bbox from ``route_bbox_batch()``.
        Each dict must contain a ``"target_tier"`` key (int 0–3).
    refine_batch : list[dict]
        One dict per R3 bbox from ``refine_r3_batch()``.
        May contain ``"upgraded_to"`` (str) and/or ``"downgraded_to"`` (str).

    Returns
    -------
    dict
        Keys: ``fallback_sas_count`` (int), ``tier_distribution`` (dict),
        ``r3_refined_to_r2_count`` (int), ``r3_downgraded_to_r4_count`` (int),
        ``total_evidence`` (int), ``total_routing`` (int), ``total_refine`` (int).
    """
    fallback_sas_count: int = sum(
        1
        for ev in evidence_batch
        if any("DEGRADED PROXY" in w for w in ev.get("warnings", []))
    )

    tier_distribution: Dict[int, int] = dict(
        Counter(r["target_tier"] for r in routing_batch)
    )

    r3_refined_to_r2_count: int = sum(
        1 for r in refine_batch if r.get("upgraded_to") == "R2"
    )

    r3_downgraded_to_r4_count: int = sum(
        1 for r in refine_batch if r.get("downgraded_to") == "R4"
    )

    return {
        "fallback_sas_count": fallback_sas_count,
        "tier_distribution": tier_distribution,
        "r3_refined_to_r2_count": r3_refined_to_r2_count,
        "r3_downgraded_to_r4_count": r3_downgraded_to_r4_count,
        "total_evidence": len(evidence_batch),
        "total_routing": len(routing_batch),
        "total_refine": len(refine_batch),
    }
