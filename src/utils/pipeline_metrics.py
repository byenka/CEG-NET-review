

from collections import Counter
from typing import Dict, List


def compute_pipeline_metrics(
    evidence_batch: List[dict],
    routing_batch: List[dict],
    refine_batch: List[dict],
) -> Dict[str, object]:

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
