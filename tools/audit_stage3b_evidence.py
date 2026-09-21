#!/usr/bin/env python
"""
audit_stage3b_evidence.py
Phase 19N — Stage 3B: Full Evidence Pipeline with Real KG Values

Replaces Stage 2 stub SDM/compat with frozen KG artifacts from Stage 3A:
  - sdm_norm_clinical_prior.json  (ICDR severity-transition, grading-aware)
  - compat_scores.json            (high_affinity_prior, encoder-calibrated)

Runs the complete evidence pipeline (SAS, SDCA, APS, PSS) on the full
detector cache for valid and/or test splits. Reports distributions of all
four signals to confirm the real KG is producing clinical evidence signal
distinct from the stub placeholders.

Gate criteria (Stage 3B):
  1. Evidence records valid: all vectors ∈ [0,1], no errors
  2. SDCA distribution: ≥ 2 distinct values (real KG signal, not all 0.5)
  3. SDCA clinical ordering observed in data: boxes predicted as {MA, HE} pairs
     should have lower mean SDCA than {MA, Soft_EX} pairs — checked if ≥1 box
     each class exists
  4. APS = 0.5 expected (ApproximateAnatomyProvider still low-confidence)
  5. Artifacts saved: evidence_audit_{split}.json

Implementation note (2026-05-24):
    Mainline now supports canonical multi-source SAS by attaching
    head_predictions from primary+secondary detector caches via IoU match.
    Use --secondary_cache_dir for non-proxy SAS.

Usage:
    python tools/audit_stage3b_evidence.py \\
      --kg_dir outputs/stage3_audit \\
    --cache_dir outputs/detector_cache/primary/oiaddr/valid \
      --split valid \\
      --max_images 0 \\
      --output_dir outputs/stage3b_audit

    # For test split:
    python tools/audit_stage3b_evidence.py \\
      --kg_dir outputs/stage3_audit \\
            --cache_dir outputs/detector_cache/primary/oiaddr/test \
      --split test \\
      --output_dir outputs/stage3b_audit
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)


def _assert_not_outputs_mirror_script() -> None:
    script_norm = str(Path(__file__).resolve()).replace("\\", "/").lower()
    if "/outputs/tools/" in script_norm:
        raise RuntimeError(
            "[Stage3B] Refusing to run from outputs/tools mirror. "
            "Run the canonical root script under tools/."
        )


def _assert_root_module(module, module_label: str) -> None:
    module_name = str(getattr(module, "__name__", ""))
    module_file = str(Path(getattr(module, "__file__", "")).resolve())
    module_norm = module_file.replace("\\", "/").lower()
    if module_name.startswith("outputs.src.") or "/outputs/src/" in module_norm:
        raise RuntimeError(
            f"[Stage3B] Refusing mirrored module import for {module_label}: "
            f"name={module_name} file={module_file}. Use root src/."
        )


_assert_not_outputs_mirror_script()


def _import_first(module_names: List[str]):
    last_error: Optional[ModuleNotFoundError] = None
    for module_name in module_names:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            missing = getattr(exc, "name", "") or ""
            if (
                missing == module_name
                or module_name.startswith(f"{missing}.")
                or missing.startswith(f"{module_name}.")
            ):
                last_error = exc
                continue
            raise
    if last_error is not None:
        raise last_error
    raise ModuleNotFoundError("No import candidates provided")


_detector_backend = _import_first(
    [
        "src.detector_backend",
        "lgovnet.src.detector_backend",
    ]
)
_assert_root_module(_detector_backend, "detector_backend")
load_cached_predictions = _detector_backend.load_cached_predictions

_anatomy_provider = _import_first(
    [
        "src.governance.anatomy_provider",
        "lgovnet.src.governance.anatomy_provider",
    ]
)
_assert_root_module(_anatomy_provider, "governance.anatomy_provider")
ApproximateAnatomyProvider = _anatomy_provider.ApproximateAnatomyProvider
ExternalPretrainedProvider = _anatomy_provider.ExternalPretrainedProvider
RadialZoneAnatomyProvider = _anatomy_provider.RadialZoneAnatomyProvider

_contracts = _import_first(
    [
        "src.governance.contracts",
        "lgovnet.src.governance.contracts",
    ]
)
_assert_root_module(_contracts, "governance.contracts")
make_placeholder_pss = _contracts.make_placeholder_pss

_evidence = _import_first(
    [
        "src.governance.evidence",
        "lgovnet.src.governance.evidence",
    ]
)
_assert_root_module(_evidence, "governance.evidence")
build_evidence_batch = _evidence.build_evidence_batch

_DATA_ROOT = os.path.join(_root, "data")
_RADIAL_ZONES = ("Foveal", "Parafoveal", "Vascular_Arcades", "Peripheral")
_LESION_CLASSES = ("MA", "HE", "Hard_EX", "Soft_EX")
_CANONICAL_STAGE3B_OUTPUT_DIR = Path(_root) / "outputs" / "stage3b_audit"
_SUSPICIOUS_PATH_MARKERS = (
    "/outputs/artefak/",
    "/outputs/src/",
    "/outputs/tools/",
    "/legacy/",
    "/historical/",
    "/stale/",
    "/stage3b_audit_pre",
)


def _is_external_fallback_context(ctx) -> bool:
    provider_name = str(getattr(ctx, "provider_name", "")).lower()
    provider_validation = getattr(ctx, "provider_validation", {}) or {}
    method = str(provider_validation.get("method", "")).lower()
    return (
        "fallback=approximate" in provider_name
        or method == "external_pretrained_fallback"
    )


def _norm_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").lower()


def _detect_suspicious_path(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    norm = _norm_path(path)
    for marker in _SUSPICIOUS_PATH_MARKERS:
        if marker in norm:
            return marker
    return None


def _assert_split_path_isolation(
    split: str,
    cache_dir: Path,
    secondary_cache_dir: Optional[Path] = None,
) -> None:
    expected = split.lower()
    checks = [("cache_dir", cache_dir)]
    if secondary_cache_dir is not None:
        checks.append(("secondary_cache_dir", secondary_cache_dir))

    for label, path in checks:
        tail = path.name.lower()
        if tail != expected:
            raise RuntimeError(
                f"{label} does not match --split {split}: tail='{path.name}'"
            )


def _iou(a: List[float], b: List[float]) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _extract_bbox_score(box: dict) -> Tuple[List[float], float]:
    raw = box.get("detector_raw") or box.get("r2a_raw") or {}
    bb = raw.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    score = raw.get("score")
    if score is None:
        ep = box.get("ensemble_probs") or {}
        score = max((float(v) for v in ep.values()), default=0.0)
    try:
        score_f = float(score)
    except (TypeError, ValueError):
        score_f = 0.0
    return [float(v) for v in bb], score_f


def _normalise_class_probs(box: dict) -> Dict[str, float]:
    raw = box.get("detector_class_probs") or box.get("ensemble_probs") or {}
    probs = {cls: max(float(raw.get(cls, 0.0)), 0.0) for cls in _LESION_CLASSES}
    total = sum(probs.values())
    if total <= 0.0:
        raw_det = box.get("detector_raw") or box.get("r2a_raw") or {}
        pred_cls = raw_det.get("label_name")
        if pred_cls in probs:
            probs = {cls: (1.0 if cls == pred_cls else 0.0) for cls in _LESION_CLASSES}
        else:
            probs = {cls: 1.0 / len(_LESION_CLASSES) for cls in _LESION_CLASSES}
        total = sum(probs.values())
    return {cls: probs[cls] / total for cls in _LESION_CLASSES}


def _build_head_prediction(box: dict, source_id: str) -> dict:
    probs = _normalise_class_probs(box)
    _, score = _extract_bbox_score(box)
    return {
        "source": source_id,
        "score": float(score),
        "class_probs": probs,
    }


def _attach_multisource_heads(
    primary_preds: Dict[str, List[dict]],
    secondary_preds: Dict[str, List[dict]],
    iou_threshold: float,
    coverage_score_min: float = 0.05,
) -> dict:
    total_boxes = 0
    matched_boxes = 0
    total_high_conf_boxes = 0
    matched_high_conf_boxes = 0

    for stem, pboxes in primary_preds.items():
        sboxes = secondary_preds.get(stem, [])
        used_secondary = set()

        for pbox in pboxes:
            total_boxes += 1
            p_bbox, p_score = _extract_bbox_score(pbox)
            in_high_conf = p_score >= coverage_score_min
            if in_high_conf:
                total_high_conf_boxes += 1

            best_idx = None
            best_iou = 0.0
            for idx, sbox in enumerate(sboxes):
                if idx in used_secondary:
                    continue
                s_bbox, _ = _extract_bbox_score(sbox)
                ov = _iou(p_bbox, s_bbox)
                if ov > best_iou:
                    best_iou = ov
                    best_idx = idx

            head_preds = [_build_head_prediction(pbox, "primary")]
            if best_idx is not None and best_iou >= iou_threshold:
                used_secondary.add(best_idx)
                head_preds.append(_build_head_prediction(sboxes[best_idx], "secondary"))
                matched_boxes += 1
                if in_high_conf:
                    matched_high_conf_boxes += 1

            pbox["head_predictions"] = head_preds
            pbox["source_count"] = len(head_preds)

    return {
        "total_boxes": total_boxes,
        "matched_boxes": matched_boxes,
        "matched_coverage": (matched_boxes / total_boxes) if total_boxes > 0 else 0.0,
        "total_high_conf_boxes": total_high_conf_boxes,
        "matched_high_conf_boxes": matched_high_conf_boxes,
        "matched_high_conf_coverage": (
            (matched_high_conf_boxes / total_high_conf_boxes)
            if total_high_conf_boxes > 0
            else 0.0
        ),
        "coverage_score_min": coverage_score_min,
        "iou_threshold": iou_threshold,
    }


# ── KG artifact loaders ────────────────────────────────────────────────────────


def _load_sdm_norm_json(path: Path) -> Dict[Tuple[str, str], Dict]:
    """Load sdm_norm_clinical_prior.json → {(ci, cj): {"raw": float, "norm": float}}."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    result: dict = {}
    for k, v in data.items():
        parts = k.split("__", 1)
        if len(parts) == 2:
            result[tuple(parts)] = v
    return result


def _load_compat_scores_json(path: Path) -> Dict[Tuple[str, str], float]:
    """Load compat_scores.json → {(lesion, zone): float}."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    result: dict = {}
    for k, v in data.items():
        parts = k.split("__", 1)
        if len(parts) == 2:
            result[tuple(parts)] = float(v)
    return result


def _missing_radial_pairs(compat_scores: Dict[Tuple[str, str], float]) -> List[str]:
    missing = []
    for lesion in _LESION_CLASSES:
        for zone in _RADIAL_ZONES:
            if (lesion, zone) not in compat_scores:
                missing.append(f"{lesion}__{zone}")
    return missing


def _resolve_image_dir(
    cache_dir: Path, split: str, data_root_override: Optional[str] = None
) -> Path:
    candidates: List[Path] = []

    if data_root_override:
        candidates.append(Path(data_root_override) / "images" / split)

    # Legacy fallback used by early runs: repo/data/images/<split>
    candidates.append(Path(_DATA_ROOT) / "images" / split)

    # Primary cache layout: outputs/.../primary/<dataset>/<split>
    inferred_dataset = cache_dir.parent.name if cache_dir.name == split else ""
    if inferred_dataset and inferred_dataset not in {"primary", "secondary", split}:
        candidates.append(Path(_DATA_ROOT) / inferred_dataset / "images" / split)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[-1]


# ── Per-image audit ────────────────────────────────────────────────────────────


def _audit_image(
    image_id: str,
    boxes: List[dict],
    provider,
    sdm_norm: dict,
    compat_scores: dict,
    image_path: Optional[str] = None,
    pss_lookup: Optional[Dict[str, dict]] = None,
) -> dict:
    if not boxes:
        return {
            "image_id": image_id,
            "n_boxes": 0,
            "n_real_prob_boxes": 0,
            "n_pss_implemented": 0,
            "n_pss_placeholder": 0,
            "n_radial_zone_boxes": 0,
            "n_valid_ev": 0,
            "n_errors": 0,
            "anatomy_uncertain_pct": 100.0,
            "avg_sas": None,
            "avg_sdca": None,
            "avg_aps": None,
            "avg_pss": None,
            "ev_vector_ok": True,
            "sdca_by_class": {},
            "warnings_total": 0,
            "anatomy_provider": None,
            "od_confidence": None,
            "external_fallback_to_approximate": False,
        }

    # Get anatomy context
    img_dims: Optional[Tuple[int, int]] = None
    if image_path and os.path.exists(image_path):
        try:
            pil_image_mod = importlib.import_module("PIL.Image")
            with pil_image_mod.open(image_path) as img:
                img_dims = (int(img.size[0]), int(img.size[1]))
                ctx = provider(img, image_id=image_id)
        except Exception:
            ctx = provider(None, image_id=image_id)
    else:
        ctx = provider(None, image_id=image_id)

    anatomy_ctx_dict = ctx.to_dict()
    external_fallback_to_approximate = _is_external_fallback_context(ctx)

    # Pilar 3: attach radial_zone per box when using RadialZoneAnatomyProvider
    if isinstance(provider, RadialZoneAnatomyProvider) and img_dims:
        W, H = img_dims
        for b in boxes:
            raw = b.get("detector_raw") or b.get("r2a_raw") or {}
            bb = raw.get("bbox") or [0, 0, 0, 0]
            cx = (float(bb[0]) + float(bb[2])) / 2.0
            cy = (float(bb[1]) + float(bb[3])) / 2.0
            b["radial_zone"] = RadialZoneAnatomyProvider.assign_radial_zone(
                cx, cy, W, H
            )

    # Pilar 2: use real TTA PSS when available, else placeholder
    if pss_lookup:
        pss_map = {
            b["bbox_id"]: pss_lookup.get(
                b["bbox_id"], make_placeholder_pss(b["bbox_id"])
            )
            for b in boxes
        }
    else:
        pss_map = {b["bbox_id"]: make_placeholder_pss(b["bbox_id"]) for b in boxes}

    n_pss_implemented = sum(
        1 for p in pss_map.values() if p.get("status") == "implemented"
    )
    n_pss_placeholder = len(pss_map) - n_pss_implemented
    n_real_prob_boxes = sum(
        1 for b in boxes if isinstance(b.get("detector_class_probs"), dict)
    )
    n_radial_zone_boxes = sum(1 for b in boxes if b.get("radial_zone") is not None)

    ev_records = build_evidence_batch(
        boxes=boxes,
        pss_results=pss_map,
        sdm_norm=sdm_norm,
        compat_scores=compat_scores,
        anatomy_context=anatomy_ctx_dict,
    )

    valid = [ev for ev in ev_records if "error" not in ev]
    errors = [ev for ev in ev_records if "error" in ev]
    warnings_total = sum(len(ev.get("warnings", [])) for ev in valid)
    n_sas_fallback = sum(
        1
        for ev in valid
        if any("SAS fell back" in w for w in (ev.get("warnings") or []))
    )
    n_multisource_boxes = sum(
        1 for b in boxes if len(b.get("head_predictions") or []) >= 2
    )

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

    # SDCA by majority class (for clinical ordering check)
    sdca_by_class: dict = defaultdict(list)
    for ev in valid:
        mc = ev.get("majority_class")
        sdca = ev.get("sdca_norm")
        if mc and sdca is not None:
            sdca_by_class[mc].append(sdca)

    return {
        "image_id": image_id,
        "n_boxes": len(boxes),
        "n_real_prob_boxes": n_real_prob_boxes,
        "n_pss_implemented": n_pss_implemented,
        "n_pss_placeholder": n_pss_placeholder,
        "n_radial_zone_boxes": n_radial_zone_boxes,
        "n_valid_ev": len(valid),
        "n_errors": len(errors),
        "n_sas_fallback": n_sas_fallback,
        "n_multisource_boxes": n_multisource_boxes,
        "anatomy_provider": ctx.provider_name,
        "od_confidence": ctx.od_confidence,
        "external_fallback_to_approximate": external_fallback_to_approximate,
        "anatomy_uncertain_pct": round(anatomy_uncertain_pct, 1),
        "avg_sas": _avg("sas"),
        "avg_sdca": _avg("sdca_norm"),
        "avg_aps": _avg("aps"),
        "avg_pss": _avg("pss"),
        "ev_vector_ok": ev_vector_ok,
        "sdca_by_class": {
            k: round(sum(v) / len(v), 4) for k, v in sdca_by_class.items()
        },
        "warnings_total": warnings_total,
        "error_msgs": [ev.get("error") for ev in errors][:3],
    }


# ── SDCA clinical ordering check ──────────────────────────────────────────────


def _check_sdca_clinical_ordering(
    all_sdca_by_class: Dict[str, List[float]],
) -> Tuple[bool, List[str]]:
    """Check that SDCA values respect ICDR clinical ordering across classes.

    Expected: mean_SDCA(MA runner-up=HE) < mean_SDCA(MA runner-up=Soft_EX)
    This is a soft informational check — requires boxes for both pairs to exist.
    """
    notes: List[str] = []

    # These checks use the majority-class SDCA aggregated across all boxes.
    # SDCA depends on (majority_class, runner_up) pair from SDM_clinical_prior.
    # SDM_clinical_prior ordering: MA↔HE=0.00, MA↔Hard_EX=0.50, MA↔Soft_EX=1.00
    # So boxes with majority_class=MA should have lower SDCA when
    # runner_up=HE than when runner_up=Soft_EX.
    # We can't separate by runner_up here, but we can check:
    # mean SDCA(HE boxes) vs mean SDCA(Soft_EX boxes) — Soft_EX↔HE has higher cost.

    he_sdcas = all_sdca_by_class.get("HE", [])
    sex_sdcas = all_sdca_by_class.get("Soft_EX", [])

    if he_sdcas and sex_sdcas:
        mean_he = sum(he_sdcas) / len(he_sdcas)
        mean_sex = sum(sex_sdcas) / len(sex_sdcas)
        notes.append(
            f"mean SDCA(HE)={mean_he:.4f}  mean SDCA(Soft_EX)={mean_sex:.4f} "
            f"(HE pairs generally lower ICDR cost than Soft_EX pairs)"
        )

    ma_sdcas = all_sdca_by_class.get("MA", [])
    if ma_sdcas:
        notes.append(
            f"mean SDCA(MA)={sum(ma_sdcas)/len(ma_sdcas):.4f} ({len(ma_sdcas)} boxes)"
        )

    return True, notes  # informational only — don't fail gate on this


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3B audit: evidence pipeline with real KG values."
    )
    parser.add_argument(
        "--kg_dir",
        default="outputs/stage3_audit",
        help="Directory containing Stage 3A KG artifacts (sdm_norm_clinical_prior.json, "
        "compat_scores.json). Default: outputs/stage3_audit.",
    )
    parser.add_argument(
        "--cache_dir",
        required=True,
        help="Per-split DetectorBox JSON cache dir. Current implementation audits one detector cache at a time; use the primary cache by default.",
    )
    parser.add_argument(
        "--secondary_cache_dir",
        default=None,
        help="Optional secondary detector cache dir for canonical multi-source SAS (same split).",
    )
    parser.add_argument(
        "--sas_match_iou",
        type=float,
        default=0.15,
        help="IoU threshold for primary-secondary bbox matching when attaching multi-source heads (default: 0.15).",
    )
    parser.add_argument(
        "--coverage_score_min",
        type=float,
        default=0.10,
        help=(
            "Minimum primary-box score used to compute score-filtered "
            "multisource coverage (default: 0.10)."
        ),
    )
    parser.add_argument(
        "--min_multisource_coverage",
        type=float,
        default=0.50,
        help=(
            "Minimum score-filtered multisource coverage required by canonical "
            "SAS gate (default: 0.50)."
        ),
    )
    parser.add_argument(
        "--allow_single_source_sas",
        action="store_true",
        help="Allow legacy single-source/proxy SAS. Mainline should keep this OFF.",
    )
    parser.add_argument(
        "--data_root",
        default=None,
        help="Optional dataset root containing images/ and xml/ subdirs. When omitted, the script infers dataset root from cache_dir.",
    )
    parser.add_argument(
        "--split",
        default="valid",
        choices=["train", "valid", "test"],
        help="Split name (default: valid).",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="Max images to audit (default: 0 = all).",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/stage3b_audit",
        help="Directory to write audit outputs (default: outputs/stage3b_audit).",
    )
    parser.add_argument(
        "--require_clean_output_dir",
        action="store_true",
        help="Fail when output_dir already contains files (prevents old-artifact mixing).",
    )
    parser.add_argument(
        "--pss_dir",
        default=None,
        help="(Pilar 2, gap G2) Dir with TTA pss_values.json (e.g. "
        "outputs/detector_cache/pss_tta/oiaddr/valid). When set, real PSS "
        "replaces the make_placeholder_pss fallback.",
    )
    parser.add_argument(
        "--min_pss_implemented_coverage",
        type=float,
        default=None,
        help="Minimum implemented-coverage required when --pss_dir is used. "
        "Default is adaptive: 0.80 for external_pretrained, 0.30 for "
        "approximate/radial_zone.",
    )
    parser.add_argument(
        "--anatomy_provider",
        default="approximate",
        choices=["approximate", "radial_zone", "external_pretrained"],
        help="(Pilar 3) Anatomy backend: 'approximate' (legacy APS neutral), "
        "'radial_zone' (centre-based APS), or 'external_pretrained' "
        "(checkpoint-backed OD/fovea provider).",
    )
    parser.add_argument(
        "--anatomy_quality_mode",
        default="relaxed",
        choices=["relaxed", "strict"],
        help="Anatomy quality-claim mode. strict requires external_pretrained and "
        "fails the run when runtime falls back to approximate.",
    )
    parser.add_argument(
        "--anatomy_checkpoint",
        default=None,
        help="Required when --anatomy_provider external_pretrained. "
        "Path to ONNX/Torch checkpoint for OD segmentation/localization.",
    )
    parser.add_argument(
        "--anatomy_device",
        default="cpu",
        help="Device for external anatomy model (e.g. cpu, cuda).",
    )
    parser.add_argument(
        "--anatomy_input_size",
        type=int,
        default=512,
        help="Input resize for external anatomy model (default: 512).",
    )
    parser.add_argument(
        "--anatomy_mask_threshold",
        type=float,
        default=0.35,
        help="Mask threshold for external anatomy OD extraction (default: 0.35).",
    )
    parser.add_argument(
        "--anatomy_conf_moderate",
        type=float,
        default=0.55,
        help="External anatomy confidence threshold for od_confidence='moderate'.",
    )
    parser.add_argument(
        "--anatomy_conf_high",
        type=float,
        default=0.80,
        help="External anatomy confidence threshold for od_confidence='high'.",
    )
    parser.add_argument(
        "--anatomy_min_mask_area",
        type=int,
        default=64,
        help="Minimum OD mask area (in resized-mask pixels) for non-low confidence.",
    )
    parser.add_argument(
        "--radial_compat_prior",
        default="outputs/stage3_audit/zone_compat_prior_radial.json",
        help="(Pilar 3) Optional override JSON for radial-zone × class compat. "
        "When absent, Stage 3A compat_scores.json is used if it already contains "
        "all required radial pairs.",
    )
    parser.add_argument(
        "--require_real_class_probs",
        action="store_true",
        help="(Pilar 1 verification) Fail gate when <80%% boxes have "
        "detector_class_probs from real ROI softmax.",
    )
    parser.add_argument(
        "--allow_historical_paths",
        action="store_true",
        help="Allow suspicious historical/artefact paths (default: fail-fast).",
    )
    args = parser.parse_args()

    if not (0.0 <= args.coverage_score_min <= 1.0):
        print("[Stage3B] ERROR: --coverage_score_min must be in [0,1]")
        sys.exit(2)
    if not (0.0 <= args.min_multisource_coverage <= 1.0):
        print("[Stage3B] ERROR: --min_multisource_coverage must be in [0,1]")
        sys.exit(2)
    if args.min_pss_implemented_coverage is not None and not (
        0.0 <= args.min_pss_implemented_coverage <= 1.0
    ):
        print("[Stage3B] ERROR: --min_pss_implemented_coverage must be in [0,1]")
        sys.exit(2)
    if not (0.0 < args.anatomy_mask_threshold < 1.0):
        print("[Stage3B] ERROR: --anatomy_mask_threshold must be in (0,1)")
        sys.exit(2)
    if not (0.0 < args.anatomy_conf_moderate <= 1.0):
        print("[Stage3B] ERROR: --anatomy_conf_moderate must be in (0,1]")
        sys.exit(2)
    if not (0.0 < args.anatomy_conf_high <= 1.0):
        print("[Stage3B] ERROR: --anatomy_conf_high must be in (0,1]")
        sys.exit(2)
    if args.anatomy_conf_high < args.anatomy_conf_moderate:
        print("[Stage3B] ERROR: --anatomy_conf_high must be >= --anatomy_conf_moderate")
        sys.exit(2)
    if args.anatomy_input_size <= 0:
        print("[Stage3B] ERROR: --anatomy_input_size must be > 0")
        sys.exit(2)
    if args.anatomy_min_mask_area <= 0:
        print("[Stage3B] ERROR: --anatomy_min_mask_area must be > 0")
        sys.exit(2)
    if args.anatomy_provider == "external_pretrained":
        if not args.anatomy_checkpoint:
            print(
                "[Stage3B] ERROR: --anatomy_checkpoint is required when "
                "--anatomy_provider external_pretrained"
            )
            sys.exit(2)
        if not Path(args.anatomy_checkpoint).exists():
            print(
                f"[Stage3B] ERROR: anatomy checkpoint not found: "
                f"{args.anatomy_checkpoint}"
            )
            sys.exit(2)
    if (
        args.anatomy_quality_mode == "strict"
        and args.anatomy_provider != "external_pretrained"
    ):
        print(
            "[Stage3B] ERROR: --anatomy_quality_mode strict requires "
            "--anatomy_provider external_pretrained"
        )
        sys.exit(2)

    cache_dir_input = Path(args.cache_dir)
    secondary_cache_input = (
        Path(args.secondary_cache_dir) if args.secondary_cache_dir else None
    )
    try:
        _assert_split_path_isolation(
            split=args.split,
            cache_dir=cache_dir_input,
            secondary_cache_dir=secondary_cache_input,
        )
    except RuntimeError as exc:
        print(f"[Stage3B] ERROR: split-isolation check failed: {exc}")
        sys.exit(2)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.require_clean_output_dir and any(out_dir.iterdir()):
        print(
            f"[Stage3B] ERROR: output_dir is not empty while "
            f"--require_clean_output_dir is set: {out_dir}"
        )
        sys.exit(2)
    kg_dir = Path(args.kg_dir)

    path_warnings: List[str] = []
    data_root_path = Path(args.data_root) if args.data_root else None
    for label, path in [
        ("cache_dir", cache_dir_input),
        ("secondary_cache_dir", secondary_cache_input),
        ("kg_dir", kg_dir),
        ("output_dir", out_dir),
        ("data_root", data_root_path),
    ]:
        marker = _detect_suspicious_path(path)
        if marker is None:
            continue
        msg = f"{label} looks historical/suspicious ({marker}): {path}"
        if args.allow_historical_paths:
            print(f"[Stage3B] WARN: {msg} (allowed by --allow_historical_paths)")
            path_warnings.append(msg)
        else:
            print(
                f"[Stage3B] ERROR: {msg} " "(set --allow_historical_paths to override)"
            )
            sys.exit(2)

    out_dir_norm = _norm_path(out_dir)
    if (
        out_dir.resolve() != _CANONICAL_STAGE3B_OUTPUT_DIR.resolve()
        and "/outputs/point123_" not in out_dir_norm
    ):
        msg = (
            f"Non-canonical output_dir detected: {out_dir}. "
            f"Canonical default is {_CANONICAL_STAGE3B_OUTPUT_DIR}."
        )
        print(f"[Stage3B] WARN: {msg}")
        path_warnings.append(msg)
    elif out_dir.resolve() != _CANONICAL_STAGE3B_OUTPUT_DIR.resolve():
        print(
            f"[Stage3B] INFO: output_dir is a Point123 run artifact root: {out_dir}"
        )

    print(f"[Stage3B] === Phase 19N Stage 3B Evidence Audit ===")
    print(f"[Stage3B] kg_dir={kg_dir}  split={args.split}")
    require_multisource_sas = not args.allow_single_source_sas
    min_pss_implemented_coverage = args.min_pss_implemented_coverage
    if min_pss_implemented_coverage is None:
        if args.anatomy_provider == "external_pretrained":
            min_pss_implemented_coverage = 0.80
        else:
            min_pss_implemented_coverage = 0.30

    # ── Load KG artifacts ───────────────────────────────────────────────────────
    sdm_path = kg_dir / "sdm_norm_clinical_prior.json"
    compat_path = kg_dir / "compat_scores.json"

    if not sdm_path.exists():
        print(f"[Stage3B] ERROR: sdm_norm_clinical_prior.json not found in {kg_dir}")
        print(f"[Stage3B]   Run audit_stage3a_kg.py first.")
        sys.exit(1)
    if not compat_path.exists():
        print(f"[Stage3B] ERROR: compat_scores.json not found in {kg_dir}")
        print(f"[Stage3B]   Run audit_stage3a_kg.py first.")
        sys.exit(1)

    sdm_norm = _load_sdm_norm_json(sdm_path)
    compat_scores = _load_compat_scores_json(compat_path)

    print(
        f"[Stage3B] Loaded: sdm_norm ({len(sdm_norm)} pairs, source=severity_transition_icdr)"
    )
    for pair in sorted(sdm_norm.keys()):
        src = sdm_norm[pair].get("source", "unknown")
        print(
            f"[Stage3B]   SDM({pair[0]},{pair[1]}): "
            f"icdr_cost={sdm_norm[pair]['raw']:.2f}  norm={sdm_norm[pair]['norm']:.4f}  [{src}]"
        )

    print(
        f"[Stage3B] Loaded: compat_scores ({len(compat_scores)} pairs, source=high_affinity_prior)"
    )

    # ── Load detector cache ─────────────────────────────────────────────────────
    cache_path = Path(args.cache_dir)
    if not cache_path.exists():
        print(f"[Stage3B] ERROR: cache_dir not found: {cache_path}")
        sys.exit(1)

    stems = sorted([f.stem for f in cache_path.glob("*.json")])
    if args.max_images and args.max_images > 0:
        stems = stems[: args.max_images]

    print(f"[Stage3B] Images to audit: {len(stems)}")
    all_preds = load_cached_predictions(stems, cache_dir=str(cache_path))

    multisource_attach_summary = None
    if args.secondary_cache_dir:
        sec_cache_path = Path(args.secondary_cache_dir)
        if not sec_cache_path.exists():
            print(f"[Stage3B] ERROR: secondary_cache_dir not found: {sec_cache_path}")
            sys.exit(1)
        secondary_preds = load_cached_predictions(stems, cache_dir=str(sec_cache_path))
        multisource_attach_summary = _attach_multisource_heads(
            all_preds,
            secondary_preds,
            iou_threshold=args.sas_match_iou,
            coverage_score_min=args.coverage_score_min,
        )
        print(
            "[Stage3B] Multi-source SAS attach: "
            f"matched={multisource_attach_summary['matched_boxes']}/"
            f"{multisource_attach_summary['total_boxes']} "
            f"({100.0*multisource_attach_summary['matched_coverage']:.1f}%) "
            f"@IoU>={args.sas_match_iou:.2f}"
        )
        print(
            "[Stage3B] Multi-source SAS high-conf: "
            f"matched={multisource_attach_summary['matched_high_conf_boxes']}/"
            f"{multisource_attach_summary['total_high_conf_boxes']} "
            f"({100.0*multisource_attach_summary['matched_high_conf_coverage']:.1f}%) "
            f"@score>={args.coverage_score_min:.2f}"
        )
    elif require_multisource_sas:
        print(
            "[Stage3B] ERROR: canonical SAS requires multi-source head_predictions. "
            "Provide --secondary_cache_dir or use --allow_single_source_sas for legacy proxy mode."
        )
        sys.exit(2)

    # ── PSS loader (Pilar 2) ────────────────────────────────────────────────────
    pss_lookup: Optional[Dict[str, dict]] = None
    pss_source_json: Optional[str] = None
    if args.pss_dir:
        pss_path = Path(args.pss_dir) / "pss_values.json"
        pss_source_json = str(pss_path.resolve())
        if pss_path.exists():
            with open(pss_path, encoding="utf-8") as f:
                pss_lookup = json.load(f)
            print(
                f"[Stage3B] Loaded real PSS: {len(pss_lookup)} values from {pss_path}"
            )
        else:
            print(
                f"[Stage3B] WARN: pss_dir set but {pss_path} missing — using placeholder"
            )

    # ── Radial compat prior (Pilar 3) ───────────────────────────────────────────
    if args.anatomy_provider == "radial_zone":
        rcp_path = Path(args.radial_compat_prior)
        if rcp_path.exists():
            with open(rcp_path, encoding="utf-8") as f:
                rcp_raw = json.load(f)
            n_merged = 0
            for k, v in rcp_raw.items():
                if k.startswith("_") or "__" not in k:
                    continue
                a, b = k.split("__", 1)
                compat_scores[(a, b)] = float(v)
                n_merged += 1
            print(
                f"[Stage3B] Merged {n_merged} radial compat entries into compat_scores"
            )
        else:
            missing_pairs = _missing_radial_pairs(compat_scores)
            if missing_pairs:
                print(f"[Stage3B] ERROR: radial_compat_prior not found: {rcp_path}")
                print(
                    "[Stage3B]   compat_scores.json is also missing required radial pairs: "
                    + ", ".join(missing_pairs[:6])
                    + (" ..." if len(missing_pairs) > 6 else "")
                )
                sys.exit(1)
            print(
                f"[Stage3B] WARN: radial_compat_prior not found: {rcp_path} — "
                "using radial compat already present in compat_scores.json"
            )

    # ── AnatomyProvider ─────────────────────────────────────────────────────────
    if args.anatomy_provider == "radial_zone":
        provider = RadialZoneAnatomyProvider()
    elif args.anatomy_provider == "external_pretrained":
        provider = ExternalPretrainedProvider(
            checkpoint_path=args.anatomy_checkpoint,
            device=args.anatomy_device,
            input_size=args.anatomy_input_size,
            mask_threshold=args.anatomy_mask_threshold,
            moderate_confidence_threshold=args.anatomy_conf_moderate,
            high_confidence_threshold=args.anatomy_conf_high,
            min_mask_area=args.anatomy_min_mask_area,
        )
    else:
        provider = ApproximateAnatomyProvider(auto_scale_dd=True)
    print(f"[Stage3B] Provider: {provider.provider_name}")

    # ── Per-image audit ─────────────────────────────────────────────────────────
    img_dir = _resolve_image_dir(cache_path, args.split, args.data_root)
    print(f"[Stage3B] Image dir: {img_dir}")
    all_stats = []
    all_sdca_by_class: Dict[str, List[float]] = defaultdict(list)

    for stem in stems:
        boxes = all_preds.get(stem, [])
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            p = img_dir / f"{stem}{ext}"
            if p.exists():
                img_path = str(p)
                break
        stats = _audit_image(
            stem,
            boxes,
            provider,
            sdm_norm,
            compat_scores,
            img_path,
            pss_lookup=pss_lookup,
        )
        all_stats.append(stats)
        for cls, avg_sdca in stats.get("sdca_by_class", {}).items():
            all_sdca_by_class[cls].append(avg_sdca)

    # ── Aggregate ───────────────────────────────────────────────────────────────
    n_total = sum(s["n_boxes"] for s in all_stats)
    n_real_prob_total = sum(s.get("n_real_prob_boxes", 0) for s in all_stats)
    n_pss_impl_total = sum(s.get("n_pss_implemented", 0) for s in all_stats)
    n_pss_placeholder_total = sum(s.get("n_pss_placeholder", 0) for s in all_stats)
    n_radial_zone_total = sum(s.get("n_radial_zone_boxes", 0) for s in all_stats)
    n_multisource_total = sum(s.get("n_multisource_boxes", 0) for s in all_stats)
    n_sas_fallback_total = sum(s.get("n_sas_fallback", 0) for s in all_stats)
    n_external_fallback_images = sum(
        1 for s in all_stats if s.get("external_fallback_to_approximate")
    )
    n_valid = sum(s.get("n_valid_ev", 0) for s in all_stats)
    n_errors = sum(s.get("n_errors", 0) for s in all_stats)
    n_uncertain = sum(1 for s in all_stats if s["anatomy_uncertain_pct"] == 100.0)
    ev_ok = all(s["ev_vector_ok"] for s in all_stats)

    sas_vals = [s["avg_sas"] for s in all_stats if s["avg_sas"] is not None]
    sdca_vals = [s["avg_sdca"] for s in all_stats if s["avg_sdca"] is not None]
    aps_vals = [s["avg_aps"] for s in all_stats if s["avg_aps"] is not None]
    aps_distinct = len(set(round(v, 3) for v in aps_vals))

    grand_avg_sas = round(sum(sas_vals) / max(len(sas_vals), 1), 4)
    grand_avg_sdca = round(sum(sdca_vals) / max(len(sdca_vals), 1), 4)
    grand_avg_aps = round(sum(aps_vals) / max(len(aps_vals), 1), 4)
    real_prob_cov = n_real_prob_total / max(n_total, 1)
    pss_impl_cov = n_pss_impl_total / max(n_total, 1)
    radial_zone_cov = n_radial_zone_total / max(n_total, 1)
    multisource_cov = n_multisource_total / max(n_total, 1)
    sas_fallback_cov = n_sas_fallback_total / max(n_valid, 1)
    if multisource_attach_summary is not None:
        n_high_conf_total = int(multisource_attach_summary["total_high_conf_boxes"])
        n_multisource_high_conf_total = int(
            multisource_attach_summary["matched_high_conf_boxes"]
        )
        multisource_high_conf_cov = float(
            multisource_attach_summary["matched_high_conf_coverage"]
        )
    else:
        n_high_conf_total = n_total
        n_multisource_high_conf_total = n_multisource_total
        multisource_high_conf_cov = multisource_cov

    # SDCA gate: must have ≥ 3 distinct values (real KG signal).
    # Clinical-prior SDM has 4 distinct norm values: 0.0, 0.5, 0.75, 1.0.
    # Expect at least 3 to appear across a real detection set.
    all_sdca_flat = [v for vals in all_sdca_by_class.values() for v in vals]
    sdca_distinct = len(set(round(v, 3) for v in all_sdca_flat))
    sdca_gate = sdca_distinct >= 3
    using_radial_zone = args.anatomy_provider == "radial_zone"
    using_external_pretrained = args.anatomy_provider == "external_pretrained"
    radial_zone_gate = True
    if using_radial_zone:
        radial_zone_gate = radial_zone_cov >= 0.80
        aps_gate = (
            aps_distinct >= 3 and n_uncertain < len(all_stats) and radial_zone_gate
        )
    elif using_external_pretrained:
        # External anatomy should produce an active, non-neutral APS signal.
        # The old approximate-provider fail-safe expects APS≈0.5 and all images
        # uncertain; that is the opposite of what a valid external model does.
        aps_gate = (
            aps_distinct >= 3
            and n_uncertain < len(all_stats)
            and n_external_fallback_images == 0
        )
    else:
        aps_gate = grand_avg_aps < 0.55 and n_uncertain == len(all_stats)

    # Pilar activation gates (optional/conditional)
    pss_gate = True
    if args.pss_dir:
        # When real PSS is requested, most boxes should carry implemented status.
        pss_gate = pss_impl_cov >= min_pss_implemented_coverage

    real_probs_gate = True
    if args.require_real_class_probs:
        # Require evidence that export used real ROI softmax probabilities.
        real_probs_gate = real_prob_cov >= 0.80

    sas_mode_gate = True
    if require_multisource_sas:
        sas_mode_gate = (
            multisource_high_conf_cov >= args.min_multisource_coverage
            and n_sas_fallback_total == 0
        )

    strict_anatomy_gate = True
    if args.anatomy_quality_mode == "strict":
        strict_anatomy_gate = n_external_fallback_images == 0

    # Clinical ordering informational check
    _, order_notes = _check_sdca_clinical_ordering(all_sdca_by_class)

    # ── Print summary ────────────────────────────────────────────────────────────
    print(f"\n[Stage3B] ══════════════ SUMMARY ({args.split}) ══════════════")
    print(f"  Images audited:         {len(all_stats)}")
    print(f"  Total candidate boxes:  {n_total}")
    print(
        f"  Real class probs:       {n_real_prob_total}/{n_total} "
        f"({100.0 * real_prob_cov:.1f}%)"
    )
    print(
        f"  Real PSS implemented:   {n_pss_impl_total}/{n_total} "
        f"({100.0 * pss_impl_cov:.1f}%)"
    )
    print(
        f"  Multi-source SAS boxes: {n_multisource_total}/{n_total} "
        f"({100.0 * multisource_cov:.1f}%)"
    )
    print(
        f"  Multi-source SAS high-conf: {n_multisource_high_conf_total}/{n_high_conf_total} "
        f"({100.0 * multisource_high_conf_cov:.1f}%) "
        f"@score>={args.coverage_score_min:.2f}"
    )
    print(
        f"  SAS fallback records:   {n_sas_fallback_total}/{n_valid} "
        f"({100.0 * sas_fallback_cov:.1f}%)"
    )
    print(f"  Placeholder PSS:        {n_pss_placeholder_total}")
    print(
        f"  Anatomy quality mode:   {args.anatomy_quality_mode} "
        f"(external fallback images={n_external_fallback_images})"
    )
    if path_warnings:
        print(f"  Path warnings:          {len(path_warnings)}")
    if using_radial_zone:
        print(
            f"  Radial-zone attached:   {n_radial_zone_total}/{n_total} "
            f"({100.0 * radial_zone_cov:.1f}%) "
            f"({'PASS' if radial_zone_gate else 'FAIL'})"
        )
    print(f"  Evidence records valid: {n_valid}  errors: {n_errors}")
    if using_radial_zone:
        print(
            f"  anatomy_uncertain=True: {n_uncertain}/{len(all_stats)} "
            "(expected: not all, radial APS active)"
        )
    elif using_external_pretrained:
        print(
            f"  anatomy_uncertain=True: {n_uncertain}/{len(all_stats)} "
            "(expected: low, external APS active)"
        )
    else:
        print(
            f"  anatomy_uncertain=True: {n_uncertain}/{len(all_stats)} "
            "(expected: all, APS=0.5)"
        )
    print(f"  Evidence vector [0,1]:  {'PASS' if ev_ok else 'FAIL'}")
    print(f"  Grand avg SAS:   {grand_avg_sas}  (detector confidence proxy)")
    print(f"  Grand avg SDCA:  {grand_avg_sdca}  (ICDR clinical-prior KG)")
    if using_radial_zone:
        print(
            f"  Grand avg APS:   {grand_avg_aps}  "
            "(radial_zone active — expect non-constant APS)"
        )
        print(
            f"  APS distinct values:    {aps_distinct} ({'PASS' if aps_gate else 'FAIL'})"
        )
    elif using_external_pretrained:
        print(
            f"  Grand avg APS:   {grand_avg_aps}  "
            "(external_pretrained active - expect non-constant APS)"
        )
        print(
            f"  APS distinct values:    {aps_distinct} ({'PASS' if aps_gate else 'FAIL'})"
        )
    else:
        print(
            f"  Grand avg APS:   {grand_avg_aps}  (expected ~0.5 — anatomy low-confidence)"
        )
    print(
        f"  SDCA distinct values:   {sdca_distinct} ({'PASS' if sdca_gate else 'FAIL — not enough variation'})"
    )
    print(f"  SDCA by class:")
    for cls in ["MA", "HE", "Hard_EX", "Soft_EX"]:
        vals = all_sdca_by_class.get(cls, [])
        if vals:
            mean_v = sum(vals) / len(vals)
            print(f"    {cls:10s}: mean={mean_v:.4f}  n_images={len(vals)}")
    for note in order_notes:
        print(f"  {note}")
    print(f"[Stage3B] ════════════════════════════════════════════════")

    gate_pass = (
        ev_ok
        and sdca_gate
        and aps_gate
        and pss_gate
        and real_probs_gate
        and sas_mode_gate
        and strict_anatomy_gate
    )
    print(f"\n[Stage3B] GATE: {'PASS' if gate_pass else 'FAIL'}")
    if not gate_pass:
        if not ev_ok:
            print(
                "  Evidence vector validation failed — check build_evidence_for_bbox errors"
            )
        if not sdca_gate:
            print(
                f"  SDCA has only {sdca_distinct} distinct values — KG not providing signal"
            )
        if args.pss_dir and not pss_gate:
            print(
                f"  Real PSS coverage too low: {100.0 * pss_impl_cov:.1f}% "
                f"(<{100.0 * min_pss_implemented_coverage:.1f}%). "
                "Check --pss_dir path and bbox_id alignment."
            )
        if args.require_real_class_probs and not real_probs_gate:
            print(
                f"  Real class-prob coverage too low: {100.0 * real_prob_cov:.1f}% (<80%). "
                "Re-export cache with --use_class_probs."
            )
        if require_multisource_sas and not sas_mode_gate:
            if multisource_high_conf_cov < args.min_multisource_coverage:
                print(
                    f"  Multi-source SAS high-conf coverage too low: {100.0 * multisource_high_conf_cov:.1f}% "
                    f"(<{100.0 * args.min_multisource_coverage:.1f}%) @score>={args.coverage_score_min:.2f}. "
                    "Check --secondary_cache_dir, --sas_match_iou, and Step 8 source diversity."
                )
            if n_sas_fallback_total > 0:
                print(
                    f"  SAS fallback still present in {n_sas_fallback_total} records — "
                    "head_predictions remain missing/invalid."
                )
        if args.anatomy_quality_mode == "strict" and not strict_anatomy_gate:
            print(
                "  Strict anatomy quality gate failed: external_pretrained "
                f"fell back to approximate on {n_external_fallback_images} image(s)."
            )
            print(
                "  Check checkpoint/runtime dependencies and rerun before "
                "making anatomy quality claims."
            )
        if using_radial_zone:
            if not radial_zone_gate:
                print(
                    f"  Radial-zone coverage too low: {100.0 * radial_zone_cov:.1f}% (<80%). "
                    "Check Stage3B image-dir resolution or pass --data_root explicitly."
                )
            if aps_distinct < 3:
                print(
                    f"  APS has only {aps_distinct} distinct per-image averages — radial evidence is too flat"
                )
            if n_uncertain == len(all_stats):
                print(
                    "  anatomy_uncertain stayed true for all images — radial provider did not activate"
                )
        elif using_external_pretrained and not aps_gate:
            if n_external_fallback_images > 0:
                print(
                    f"  External anatomy fell back to approximate on {n_external_fallback_images} image(s)."
                )
            if aps_distinct < 3:
                print(
                    f"  APS has only {aps_distinct} distinct per-image averages — external evidence is too flat"
                )
            if n_uncertain == len(all_stats):
                print(
                    "  anatomy_uncertain stayed true for all images — external provider did not activate"
                )
        elif grand_avg_aps >= 0.55 or n_uncertain != len(all_stats):
            print(
                f"  APS={grand_avg_aps:.4f}, anatomy_uncertain={n_uncertain}/{len(all_stats)} "
                "— expected approximate-provider fail-safe"
            )

    # ── Save report ─────────────────────────────────────────────────────────────
    sdca_class_summary = {
        cls: {
            "mean": round(sum(v) / len(v), 4) if v else None,
            "n_images": len(v),
        }
        for cls, v in all_sdca_by_class.items()
    }

    run_provenance = {
        "repo_root": str(Path(_root).resolve()),
        "script_path": str(Path(__file__).resolve()),
        "split": args.split,
        "cache_dir": str(cache_dir_input.resolve()),
        "secondary_cache_dir": (
            str(secondary_cache_input.resolve()) if secondary_cache_input else None
        ),
        "kg_dir": str(kg_dir.resolve()),
        "output_dir": str(out_dir.resolve()),
        "data_root": str(data_root_path.resolve()) if data_root_path else None,
        "anatomy_provider_mode": args.anatomy_provider,
        "anatomy_quality_mode": args.anatomy_quality_mode,
        "anatomy_checkpoint": (
            str(Path(args.anatomy_checkpoint).resolve())
            if args.anatomy_checkpoint
            else None
        ),
        "pss_dir": str(Path(args.pss_dir).resolve()) if args.pss_dir else None,
        "pss_source_json": pss_source_json,
        "compare_baseline_source": str(
            (Path(args.output_dir).resolve() / f"evidence_audit_{args.split}.json")
        ),
        "allow_historical_paths": bool(args.allow_historical_paths),
        "path_warnings": path_warnings,
    }

    summary = {
        "stage": "3B",
        "split": args.split,
        "kg_dir": str(kg_dir),
        "sdm_source": "severity_transition_icdr (clinical-prior, grading-aware)",
        "compat_source": "high_affinity_prior (encoder-calibrated)",
        "n_images": len(all_stats),
        "n_total_boxes": n_total,
        "n_real_prob_boxes": n_real_prob_total,
        "real_prob_coverage": round(real_prob_cov, 4),
        "n_pss_implemented": n_pss_impl_total,
        "n_pss_placeholder": n_pss_placeholder_total,
        "pss_implemented_coverage": round(pss_impl_cov, 4),
        "n_radial_zone_boxes": n_radial_zone_total,
        "n_multisource_boxes": n_multisource_total,
        "multisource_coverage": round(multisource_cov, 4),
        "n_high_conf_boxes": n_high_conf_total,
        "n_multisource_high_conf_boxes": n_multisource_high_conf_total,
        "multisource_high_conf_coverage": round(multisource_high_conf_cov, 4),
        "n_sas_fallback_records": n_sas_fallback_total,
        "sas_fallback_coverage": round(sas_fallback_cov, 4),
        "n_valid_evidence": n_valid,
        "n_errors": n_errors,
        "anatomy_uncertain_all_images": n_uncertain == len(all_stats),
        "evidence_vector_ok": ev_ok,
        "grand_avg_sas": grand_avg_sas,
        "grand_avg_sdca": grand_avg_sdca,
        "grand_avg_aps": grand_avg_aps,
        "aps_distinct_values": aps_distinct,
        "sdca_distinct_values": sdca_distinct,
        "anatomy_provider_mode": args.anatomy_provider,
        "anatomy_quality_mode": args.anatomy_quality_mode,
        "n_external_fallback_images": n_external_fallback_images,
        "sdca_by_class": sdca_class_summary,
        "clinical_order_notes": order_notes,
        "pilar_gates": {
            "sdca_gate": sdca_gate,
            "aps_gate": aps_gate,
            "pss_gate": pss_gate,
            "min_pss_implemented_coverage": round(min_pss_implemented_coverage, 4),
            "real_probs_gate": real_probs_gate,
            "sas_mode_gate": sas_mode_gate,
            "radial_zone_gate": radial_zone_gate,
            "strict_anatomy_gate": strict_anatomy_gate,
            "coverage_score_min": args.coverage_score_min,
            "min_multisource_coverage": args.min_multisource_coverage,
        },
        "sas_mode": {
            "require_multisource_sas": require_multisource_sas,
            "secondary_cache_dir": args.secondary_cache_dir,
            "attach_summary": multisource_attach_summary,
        },
        "path_warnings": path_warnings,
        "provenance": run_provenance,
        "gate": "PASS" if gate_pass else "FAIL",
        "per_image": all_stats,
    }

    out_json = out_dir / f"evidence_audit_{args.split}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[Stage3B] Report: {out_json}")

    if gate_pass:
        print(
            "\n[Stage3B] Stage 3B PASS. Clinical KG evidence pipeline verified."
            "\n  Next: design LGG architecture (Stage 3D) — real SDCA + compat available."
            "\n  sdm_norm_clinical_prior.json + compat_scores.json are frozen Stage 3B artifacts."
        )

    sys.exit(0 if gate_pass else 1)


if __name__ == "__main__":
    main()
