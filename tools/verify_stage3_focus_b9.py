#!/usr/bin/env python
"""Focused local verification for B9 (lightweight, no heavy compute).

This verifier is intentionally read-only for Stage3 hardening checks before
server execution. It performs:
1) static validation (py_compile on critical scripts),
2) import/path guard validation (required guard tokens still present),
3) schema + provenance validation for governance_features/audit files,
4) dry-run split isolation checks (no training/eval workload).

Usage:
    python tools/verify_stage3_focus_b9.py
    python tools/verify_stage3_focus_b9.py --stage3c_dir outputs/stage3c_audit
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import py_compile
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

_SUSPICIOUS_MARKERS = (
    "/outputs/artefak/",
    "/outputs/src/",
    "/outputs/tools/",
    "/legacy/",
    "/historical/",
    "/stale/",
    "/stage3c_audit_pre",
)

_REQUIRED_COLUMNS = sorted(
    set(
        [
            "bbox_id",
            "image_id",
            "split",
            "category",
            "label",
            "pred_score",
            "unc_msp",
            "unc_margin",
            "unc_entconf",
            "ev_sas",
            "ev_sdca",
            "ev_aps",
            "ev_pss",
            "source_output_dir",
            "source_cache_dir",
            "source_annotations_dir",
            "anatomy_provider_mode",
            "anatomy_quality_mode",
            "pss_source_json",
            "compare_baseline_source",
        ]
    )
)

_REQUIRED_FINITE_COLUMNS = ["pred_score", "ev_sas", "ev_sdca", "ev_pss", "ev_aps"]
_REQUIRED_PROVENANCE_KEYS = [
    "repo_root",
    "cache_dir",
    "annotations_dir",
    "output_dir",
    "compare_baseline_source",
]

_SCRIPT_TARGETS = [
    "tools/audit_stage3b_evidence.py",
    "tools/audit_stage3c_governance_targets.py",
    "tools/train_stage3d_lgg.py",
    "tools/eval_stage3d4_4tier_routing.py",
]

_GUARD_TOKENS = {
    "tools/audit_stage3b_evidence.py": [
        "_assert_not_outputs_mirror_script",
        "_assert_root_module",
        "Refusing to run from outputs/tools mirror",
        "Refusing mirrored module import",
    ],
    "tools/audit_stage3c_governance_targets.py": [
        "_assert_not_outputs_mirror_script",
        "_assert_root_module",
        "Refusing to run from outputs/tools mirror",
        "Refusing mirrored module import",
    ],
    "tools/train_stage3d_lgg.py": [
        "_assert_not_outputs_mirror_script",
        "_assert_root_module",
        "Refusing to run from outputs/tools mirror",
        "Refusing mirrored module import",
    ],
    "tools/eval_stage3d4_4tier_routing.py": [
        "_assert_not_outputs_mirror_script",
        "Refusing to run from outputs/tools mirror",
    ],
}


def _norm_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").lower()


def _detect_suspicious_path(path: Path) -> Optional[str]:
    norm = _norm_path(path)
    for marker in _SUSPICIOUS_MARKERS:
        if marker in norm:
            return marker
    return None


def _read_rows(path: Path) -> Tuple[List[dict], List[str]]:
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = list(reader.fieldnames or [])
    return rows, header


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _has_any_finite(rows: List[dict], col: str) -> bool:
    for row in rows:
        raw = str(row.get(col, "")).strip()
        if raw == "":
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if math.isfinite(value):
            return True
    return False


def _compile_and_guard_validate(
    repo_root: Path,
) -> Tuple[List[dict], List[str], List[str]]:
    compile_results: List[dict] = []
    errors: List[str] = []
    warnings: List[str] = []

    for rel in _SCRIPT_TARGETS:
        path = repo_root / rel
        if not path.exists():
            errors.append(f"Missing target script: {rel}")
            continue

        try:
            py_compile.compile(str(path), doraise=True)
            compile_results.append({"script": rel, "status": "PASS"})
        except py_compile.PyCompileError as exc:
            errors.append(f"Static compile failed for {rel}: {exc.msg}")
            compile_results.append({"script": rel, "status": "FAIL"})
            continue

        required_tokens = _GUARD_TOKENS.get(rel, [])
        text = path.read_text(encoding="utf-8", errors="replace")
        missing_tokens = [token for token in required_tokens if token not in text]
        if missing_tokens:
            errors.append(
                f"Import/path guard token missing in {rel}: {', '.join(missing_tokens)}"
            )
        else:
            compile_results[-1]["guard_tokens"] = "PASS"

        marker = _detect_suspicious_path(path)
        if marker is not None:
            warnings.append(f"Target script path looks suspicious ({marker}): {path}")

    return compile_results, errors, warnings


def _validate_split_csv(
    split: str,
    csv_path: Path,
    allow_historical_inputs: bool,
) -> Tuple[dict, List[dict], Set[str], Set[str], Optional[str]]:
    result = {
        "split": split,
        "csv_path": str(csv_path),
        "n_rows": 0,
        "status": "PASS",
        "errors": [],
        "warnings": [],
    }

    if not csv_path.exists():
        result["status"] = "FAIL"
        result["errors"].append(f"Missing CSV: {csv_path}")
        return result, [], set(), set(), None

    rows, header = _read_rows(csv_path)
    result["n_rows"] = len(rows)

    if not rows:
        result["status"] = "FAIL"
        result["errors"].append(f"CSV is empty: {csv_path}")
        return result, [], set(), set(), None

    missing_cols = [col for col in _REQUIRED_COLUMNS if col not in header]
    if missing_cols:
        result["status"] = "FAIL"
        result["errors"].append(
            f"Missing required columns in {csv_path.name}: {', '.join(missing_cols)}"
        )

    has_source_output = "source_output_dir" in header
    has_source_cache = "source_cache_dir" in header
    has_source_ann = "source_annotations_dir" in header

    for col in _REQUIRED_FINITE_COLUMNS:
        if not _has_any_finite(rows, col):
            result["status"] = "FAIL"
            result["errors"].append(
                f"No finite values found for required column '{col}' in {csv_path.name}"
            )

    split_values = {
        str(row.get("split", "")).strip().lower()
        for row in rows
        if str(row.get("split", "")).strip()
    }
    if split_values and split_values != {split}:
        result["status"] = "FAIL"
        result["errors"].append(
            f"Split column mismatch in {csv_path.name}: {sorted(split_values)}"
        )

    image_ids = {
        str(row.get("image_id", "")).strip()
        for row in rows
        if str(row.get("image_id", "")).strip()
    }
    bbox_ids = {
        str(row.get("bbox_id", "")).strip()
        for row in rows
        if str(row.get("bbox_id", "")).strip()
    }

    source_output_dirs = set()
    source_cache_dirs = set()
    source_ann_dirs = set()

    if has_source_output:
        source_output_dirs = {
            str(row.get("source_output_dir", "")).strip()
            for row in rows
            if str(row.get("source_output_dir", "")).strip()
        }
        if len(source_output_dirs) != 1:
            result["status"] = "FAIL"
            result["errors"].append(
                f"Inconsistent source_output_dir values in {csv_path.name}: {sorted(source_output_dirs)}"
            )

    if has_source_cache:
        source_cache_dirs = {
            str(row.get("source_cache_dir", "")).strip()
            for row in rows
            if str(row.get("source_cache_dir", "")).strip()
        }
        for cache_dir in source_cache_dirs:
            tail = Path(cache_dir).name.lower()
            if tail and tail != split:
                result["status"] = "FAIL"
                result["errors"].append(
                    f"source_cache_dir tail mismatch in {csv_path.name}: {cache_dir}"
                )

    if has_source_ann:
        source_ann_dirs = {
            str(row.get("source_annotations_dir", "")).strip()
            for row in rows
            if str(row.get("source_annotations_dir", "")).strip()
        }
        for ann_dir in source_ann_dirs:
            tail = Path(ann_dir).name.lower()
            if tail and tail != split:
                result["status"] = "FAIL"
                result["errors"].append(
                    f"source_annotations_dir tail mismatch in {csv_path.name}: {ann_dir}"
                )

    for path_str in sorted(
        source_output_dirs.union(source_cache_dirs, source_ann_dirs)
    ):
        marker = _detect_suspicious_path(Path(path_str))
        if marker is None:
            continue
        msg = f"Suspicious provenance path ({marker}) in {csv_path.name}: {path_str}"
        if allow_historical_inputs:
            result["warnings"].append(msg)
        else:
            result["status"] = "FAIL"
            result["errors"].append(msg)

    return result, rows, image_ids, bbox_ids, next(iter(source_output_dirs), None)


def _validate_split_audit(
    split: str,
    stage3c_dir: Path,
    expected_rows: int,
) -> dict:
    result = {
        "split": split,
        "audit_path": str(stage3c_dir / f"governance_target_audit_{split}.json"),
        "status": "PASS",
        "errors": [],
    }
    audit_path = stage3c_dir / f"governance_target_audit_{split}.json"
    if not audit_path.exists():
        result["status"] = "FAIL"
        result["errors"].append(f"Missing Stage3C audit JSON: {audit_path}")
        return result

    try:
        audit = _read_json(audit_path)
    except Exception as exc:
        result["status"] = "FAIL"
        result["errors"].append(f"Invalid JSON in {audit_path}: {exc}")
        return result

    if str(audit.get("stage")) != "3C":
        result["status"] = "FAIL"
        result["errors"].append(
            f"Invalid stage in {audit_path.name}: {audit.get('stage')}"
        )

    if str(audit.get("split")) != split:
        result["status"] = "FAIL"
        result["errors"].append(
            f"Split mismatch in {audit_path.name}: audit={audit.get('split')} expected={split}"
        )

    agg_total_raw = (audit.get("aggregation") or {}).get("total_boxes")
    if agg_total_raw is not None:
        try:
            agg_total = int(agg_total_raw)
        except (TypeError, ValueError):
            result["status"] = "FAIL"
            result["errors"].append(
                f"Invalid aggregation.total_boxes in {audit_path.name}: {agg_total_raw}"
            )
        else:
            if agg_total != expected_rows:
                result["status"] = "FAIL"
                result["errors"].append(
                    f"Row-count mismatch for {split}: csv={expected_rows} audit={agg_total}"
                )

    provenance = audit.get("provenance")
    if not isinstance(provenance, dict):
        result["status"] = "FAIL"
        result["errors"].append(f"Missing provenance block in {audit_path.name}")
    else:
        missing_keys = [
            key
            for key in _REQUIRED_PROVENANCE_KEYS
            if str(provenance.get(key, "")).strip() == ""
        ]
        if missing_keys:
            result["status"] = "FAIL"
            result["errors"].append(
                f"Missing provenance keys in {audit_path.name}: {', '.join(missing_keys)}"
            )

    return result


def _validate_cross_split_isolation(
    split_images: Dict[str, Set[str]],
    split_boxes: Dict[str, Set[str]],
    split_output_root: Dict[str, str],
) -> List[str]:
    errors: List[str] = []
    split_names = sorted(split_images.keys())
    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            overlap_images = split_images[left] & split_images[right]
            if overlap_images:
                errors.append(
                    f"image_id overlap between {left} and {right} (n={len(overlap_images)})"
                )
            overlap_boxes = split_boxes[left] & split_boxes[right]
            if overlap_boxes:
                errors.append(
                    f"bbox_id overlap between {left} and {right} (n={len(overlap_boxes)})"
                )

    roots = set(split_output_root.values())
    if len(roots) > 1:
        errors.append(
            "Cross-root source_output_dir detected across splits: "
            + str(split_output_root)
        )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="B9 focused local verification (lightweight dry-run)."
    )
    parser.add_argument(
        "--stage3c_dir",
        default="outputs/stage3c_audit",
        help="Directory containing governance_features_{split}.csv and governance_target_audit_{split}.json",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["train", "valid", "test"],
        help="Splits to validate.",
    )
    parser.add_argument(
        "--output_json",
        default="outputs/stage3_light_verify/b9_focused_verification.json",
        help="Path to write verification report JSON.",
    )
    parser.add_argument(
        "--allow_historical_inputs",
        action="store_true",
        help="Allow suspicious historical/stale paths (default: fail-fast).",
    )
    parser.add_argument(
        "--fail_on_warnings",
        action="store_true",
        help="Treat warnings as failures.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    stage3c_dir = Path(args.stage3c_dir)
    output_json = Path(args.output_json)

    report = {
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "stage": "B9_focused_verification",
        "mode": "dry-run-lightweight",
        "repo_root": str(repo_root),
        "stage3c_dir": str(stage3c_dir),
        "allow_historical_inputs": bool(args.allow_historical_inputs),
        "static_compile": [],
        "split_checks": [],
        "audit_checks": [],
        "cross_split": {"status": "PASS", "errors": []},
        "warnings": [],
        "errors": [],
        "overall_status": "PASS",
    }

    print("[B9] Focused local verification (dry-run, no training/eval compute)")
    print(f"[B9] repo_root={repo_root}")

    compile_results, compile_errors, compile_warnings = _compile_and_guard_validate(
        repo_root
    )
    report["static_compile"] = compile_results
    report["errors"].extend(compile_errors)
    report["warnings"].extend(compile_warnings)

    marker = _detect_suspicious_path(stage3c_dir)
    if marker is not None:
        msg = f"stage3c_dir looks suspicious ({marker}): {stage3c_dir}"
        if args.allow_historical_inputs:
            report["warnings"].append(msg)
        else:
            report["errors"].append(msg)

    split_images: Dict[str, Set[str]] = {}
    split_boxes: Dict[str, Set[str]] = {}
    split_output_root: Dict[str, str] = {}

    for split in args.splits:
        split_norm = split.strip().lower()
        if split_norm not in {"train", "valid", "test"}:
            report["errors"].append(f"Unsupported split name: {split}")
            continue

        csv_path = stage3c_dir / f"governance_features_{split_norm}.csv"
        split_result, rows, image_ids, bbox_ids, source_root = _validate_split_csv(
            split=split_norm,
            csv_path=csv_path,
            allow_historical_inputs=args.allow_historical_inputs,
        )
        report["split_checks"].append(split_result)
        report["warnings"].extend(split_result.get("warnings", []))
        report["errors"].extend(split_result.get("errors", []))

        if rows:
            split_images[split_norm] = image_ids
            split_boxes[split_norm] = bbox_ids
            if source_root:
                split_output_root[split_norm] = source_root

            audit_result = _validate_split_audit(
                split=split_norm,
                stage3c_dir=stage3c_dir,
                expected_rows=len(rows),
            )
            report["audit_checks"].append(audit_result)
            report["errors"].extend(audit_result.get("errors", []))

    if len(split_images) >= 2:
        cross_errors = _validate_cross_split_isolation(
            split_images=split_images,
            split_boxes=split_boxes,
            split_output_root=split_output_root,
        )
        if cross_errors:
            report["cross_split"] = {"status": "FAIL", "errors": cross_errors}
            report["errors"].extend(cross_errors)

    if report["errors"]:
        report["overall_status"] = "FAIL"
    elif args.fail_on_warnings and report["warnings"]:
        report["overall_status"] = "FAIL"

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[B9] static checks: {len(report['static_compile'])} scripts")
    print(f"[B9] split checks : {len(report['split_checks'])} splits")
    print(f"[B9] warnings     : {len(report['warnings'])}")
    print(f"[B9] errors       : {len(report['errors'])}")
    print(f"[B9] report       : {output_json}")
    print(f"[B9] OVERALL      : {report['overall_status']}")

    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
