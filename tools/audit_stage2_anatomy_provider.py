#!/usr/bin/env python
"""
audit_stage2_anatomy_provider.py
Phase 19N — Stage 2 Audit: AnatomyProvider + Evidence Pipeline Validation

Validates:
  1. AnatomyProvider interface works (ApproximateAnatomyProvider)
  2. DetectorBox cache (from export_detector_predictions.py) loads correctly
  3. Evidence computation (SAS, SDCA, APS, PSS) runs end-to-end
  4. APS fail-safe fires correctly when od_confidence='low'
  5. Evidence vector is well-formed ([SAS, SDCA_norm, APS, PSS] ∈ [0,1])

This script does NOT evaluate governance gate (LGG) — that is Stage 3.
Its purpose is to confirm that the Stage 1→Stage 2 bridge is working before
building C1 KG + C2 LGG training infrastructure.

Expected outputs:
  - Per-image evidence summary (n_boxes, avg SAS, avg APS, anatomy_uncertain %)
  - Aggregate stats across split
  - PASS/FAIL for evidence vector integrity

Usage:
    python tools/audit_stage2_anatomy_provider.py \\
    --cache_dir outputs/detector_cache/primary/oiaddr/valid \
      --split valid \\
      --max_images 20 \\
      --output_dir outputs/stage2_audit/anatomy_provider_valid
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, "..")
if _root not in sys.path:
    sys.path.insert(0, _root)

from lgovnet.src.detector_backend import load_cached_predictions
from lgovnet.src.governance.anatomy_provider import ApproximateAnatomyProvider
from lgovnet.src.governance.contracts import make_placeholder_pss
from lgovnet.src.governance.evidence import build_evidence_batch

_DATA_ROOT = os.path.join(_root, "data")


# ── Dummy KG data (until real KG is built in Stage 3) ──────────────────────────

# Approximate SDM_norm from architecture v3 (rough values for audit only)
_STUB_SDM_NORM: Dict = {
    ("MA", "HE"): {"raw": 0.11, "norm": 0.85},
    ("MA", "Hard_EX"): {"raw": 0.13, "norm": 1.00},
    ("MA", "Soft_EX"): {"raw": 0.10, "norm": 0.77},
    ("HE", "Hard_EX"): {"raw": 0.04, "norm": 0.31},
    ("HE", "Soft_EX"): {"raw": 0.06, "norm": 0.46},
    ("Hard_EX", "Soft_EX"): {"raw": 0.08, "norm": 0.62},
}

# Approximate Compat scores from clinical priors
# Format: {(lesion, zone): plausibility_score ∈ (0,1)}
# Values based on ETDRS + clinical literature (approximate for audit)
_STUB_COMPAT: Dict = {
    ("MA", "Foveal"): 0.72,
    ("MA", "Parafoveal"): 0.65,
    ("MA", "Peripapillary"): 0.40,
    ("MA", "Vascular_Arcades"): 0.58,
    ("MA", "Peripheral"): 0.30,
    ("HE", "Foveal"): 0.35,
    ("HE", "Parafoveal"): 0.55,
    ("HE", "Peripapillary"): 0.68,
    ("HE", "Vascular_Arcades"): 0.62,
    ("HE", "Peripheral"): 0.40,
    ("Hard_EX", "Foveal"): 0.52,
    ("Hard_EX", "Parafoveal"): 0.70,
    ("Hard_EX", "Peripapillary"): 0.48,
    ("Hard_EX", "Vascular_Arcades"): 0.65,
    ("Hard_EX", "Peripheral"): 0.45,
    ("Soft_EX", "Foveal"): 0.20,
    ("Soft_EX", "Parafoveal"): 0.30,
    ("Soft_EX", "Peripapillary"): 0.80,
    ("Soft_EX", "Vascular_Arcades"): 0.40,
    ("Soft_EX", "Peripheral"): 0.25,
}


def _audit_image(
    image_id: str,
    boxes: List[dict],
    provider: ApproximateAnatomyProvider,
    image_path: Optional[str] = None,
) -> dict:
    """Run evidence computation for one image and return stats."""
    if not boxes:
        return {
            "image_id": image_id,
            "n_boxes": 0,
            "anatomy_provider": provider.provider_name,
            "od_confidence": "n/a",
            "anatomy_uncertain_pct": 100.0,
            "avg_sas": None,
            "avg_sdca": None,
            "avg_aps": None,
            "avg_pss": None,
            "ev_vector_ok": True,
            "warnings_total": 0,
        }

    # Approximate anatomy context (from provider)
    # We pass a dummy image object — provider uses it only for dimensions
    # When image_path is available, load real dims; otherwise use 512×512 default
    if image_path and os.path.exists(image_path):
        try:
            from PIL import Image as PILImage

            img = PILImage.open(image_path)
            ctx = provider(img, image_id=image_id)
        except Exception:
            ctx = provider(None, image_id=image_id)
    else:
        ctx = provider(None, image_id=image_id)

    anatomy_ctx_dict = ctx.to_dict()

    # Placeholder PSS for all boxes (real PSS = Stage 3+ scope)
    pss_map = {b["bbox_id"]: make_placeholder_pss(b["bbox_id"]) for b in boxes}

    # Run evidence batch
    ev_records = build_evidence_batch(
        boxes=boxes,
        pss_results=pss_map,
        sdm_norm=_STUB_SDM_NORM,
        compat_scores=_STUB_COMPAT,
        anatomy_context=anatomy_ctx_dict,
    )

    # Aggregate stats
    valid = [ev for ev in ev_records if "error" not in ev]
    errors = [ev for ev in ev_records if "error" in ev]
    warnings_total = sum(len(ev.get("warnings", [])) for ev in valid)

    def _avg(key: str) -> Optional[float]:
        vals = [ev[key] for ev in valid if ev.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    anatomy_uncertain_pct = (
        100.0
        * sum(1 for ev in valid if ev.get("anatomy_uncertain", True))
        / max(len(valid), 1)
    )

    ev_vector_ok = all(
        ev.get("evidence_vector") is not None
        and len(ev["evidence_vector"]) == 4
        and all(0.0 <= v <= 1.0 for v in ev["evidence_vector"])
        for ev in valid
    )

    return {
        "image_id": image_id,
        "n_boxes": len(boxes),
        "n_valid_ev": len(valid),
        "n_errors": len(errors),
        "anatomy_provider": ctx.provider_name,
        "od_confidence": ctx.od_confidence,
        "anatomy_uncertain_pct": round(anatomy_uncertain_pct, 1),
        "avg_sas": _avg("sas"),
        "avg_sdca": _avg("sdca_norm"),
        "avg_aps": _avg("aps"),
        "avg_pss": _avg("pss"),
        "ev_vector_ok": ev_vector_ok,
        "warnings_total": warnings_total,
        "error_msgs": [ev.get("error") for ev in errors][:3],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2 audit: AnatomyProvider + evidence pipeline validation."
    )
    parser.add_argument(
        "--cache_dir",
        required=True,
        help="Per-split DetectorBox JSON cache dir (from export_detector_predictions.py).",
    )
    parser.add_argument(
        "--split",
        default="valid",
        choices=["train", "valid", "test"],
        help="Split name (for image path lookup). Default: valid.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=20,
        help="Max images to audit (default: 20; use 0 for all).",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/stage2_audit/anatomy_provider",
        help="Directory to write audit outputs.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover image stems from cache dir
    cache_path = Path(args.cache_dir)
    if not cache_path.exists():
        print(f"[Stage2Audit] ERROR: cache_dir not found: {cache_path}")
        sys.exit(1)

    stems = sorted([f.stem for f in cache_path.glob("*.json")])
    if args.max_images and args.max_images > 0:
        stems = stems[: args.max_images]

    print(f"[Stage2Audit] === Phase 19N Stage 2 Audit ===")
    print(f"[Stage2Audit] cache_dir={args.cache_dir}  split={args.split}")
    print(f"[Stage2Audit] images to audit: {len(stems)}")

    # Load cached predictions
    all_preds = load_cached_predictions(stems, cache_dir=str(cache_path))

    # AnatomyProvider
    provider = ApproximateAnatomyProvider(auto_scale_dd=True)
    print(f"[Stage2Audit] provider={provider.provider_name}")

    # Per-image audit
    img_dir = os.path.join(_DATA_ROOT, "images", args.split)
    all_stats = []

    for stem in stems:
        boxes = all_preds.get(stem, [])
        # Try to find image for accurate dimension extraction
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            p = os.path.join(img_dir, stem + ext)
            if os.path.exists(p):
                img_path = p
                break

        stats = _audit_image(stem, boxes, provider, img_path)
        all_stats.append(stats)

    # Aggregate
    n_total = sum(s["n_boxes"] for s in all_stats)
    n_valid = sum(s.get("n_valid_ev", 0) for s in all_stats)
    n_uncertain = sum(1 for s in all_stats if s["anatomy_uncertain_pct"] == 100.0)
    ev_ok = all(s["ev_vector_ok"] for s in all_stats)

    avg_sas_vals = [s["avg_sas"] for s in all_stats if s["avg_sas"] is not None]
    avg_aps_vals = [s["avg_aps"] for s in all_stats if s["avg_aps"] is not None]

    grand_avg_sas = round(sum(avg_sas_vals) / max(len(avg_sas_vals), 1), 4)
    grand_avg_aps = round(sum(avg_aps_vals) / max(len(avg_aps_vals), 1), 4)

    print(f"\n[Stage2Audit] ══════════════ SUMMARY ══════════════")
    print(f"  Images audited:         {len(all_stats)}")
    print(f"  Total candidate boxes:  {n_total}")
    print(f"  Evidence records valid: {n_valid}")
    print(
        f"  anatomy_uncertain=True: {n_uncertain}/{len(all_stats)} images (expected: all)"
    )
    print(f"  Evidence vector [0,1]:  {'PASS' if ev_ok else 'FAIL'}")
    print(
        f"  Grand avg SAS:          {grand_avg_sas}  (max_ensemble fallback — detection confidence)"
    )
    print(
        f"  Grand avg APS:          {grand_avg_aps}  (expected ~0.5 — od_confidence=low fail-safe)"
    )
    print(f"[Stage2Audit] ════════════════════════════════════════")

    gate_pass = ev_ok and grand_avg_aps < 0.55  # APS ~0.5 confirms fail-safe fired

    print(f"\n[Stage2Audit] GATE: {'PASS' if gate_pass else 'FAIL'}")
    if not gate_pass:
        print("  APS is not near 0.5 — fail-safe may not have fired correctly.")
        print(
            "  Check that anatomy_context od_confidence='low' → APS=0.5 in evidence.py"
        )

    # Write outputs
    summary = {
        "split": args.split,
        "cache_dir": str(cache_path),
        "provider": provider.provider_name,
        "n_images": len(all_stats),
        "n_total_boxes": n_total,
        "n_valid_evidence": n_valid,
        "anatomy_uncertain_all_images": n_uncertain == len(all_stats),
        "evidence_vector_ok": ev_ok,
        "grand_avg_sas": grand_avg_sas,
        "grand_avg_aps": grand_avg_aps,
        "gate": "PASS" if gate_pass else "FAIL",
        "per_image": all_stats,
    }

    out_json = out_dir / f"stage2_anatomy_audit_{args.split}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[Stage2Audit] Report: {out_json}")


if __name__ == "__main__":
    main()
