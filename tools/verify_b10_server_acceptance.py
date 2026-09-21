#!/usr/bin/env python
"""B10 server acceptance gate verifier (read-only).

This script validates whether a completed Point123 server run is acceptable for
CAS-quality claim based on explicit PASS/FAIL criteria.

It does not retrain or modify artifacts. It only reads outputs and writes a
compact acceptance report under run_root/summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _as_float(value, default: Optional[float] = None) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _check_file(path: Path, label: str, errors: List[str], checks: List[dict]) -> bool:
    ok = path.exists()
    checks.append(
        {
            "check": f"file_exists::{label}",
            "status": _status(ok),
            "path": str(path),
        }
    )
    if not ok:
        errors.append(f"Missing required file: {path}")
    return ok


def _validate_stage3b_split(
    split: str,
    summary_path: Path,
    expected_provider: str,
    require_strict_anatomy: bool,
    min_pss_implemented_coverage: float,
    allow_path_warnings: bool,
    errors: List[str],
    checks: List[dict],
) -> None:
    if not summary_path.exists():
        return

    payload = _read_json(summary_path)
    gate_ok = str(payload.get("gate")) == "PASS"
    checks.append(
        {
            "check": f"stage3b_gate::{split}",
            "status": _status(gate_ok),
            "path": str(summary_path),
            "gate": payload.get("gate"),
        }
    )
    if not gate_ok:
        errors.append(f"[Stage3B/{split}] gate is not PASS")

    stage_ok = str(payload.get("stage")) == "3B"
    split_ok = str(payload.get("split")) == split
    checks.append(
        {
            "check": f"stage3b_stage_split::{split}",
            "status": _status(stage_ok and split_ok),
            "stage": payload.get("stage"),
            "split": payload.get("split"),
        }
    )
    if not stage_ok:
        errors.append(f"[Stage3B/{split}] invalid stage value: {payload.get('stage')}")
    if not split_ok:
        errors.append(
            f"[Stage3B/{split}] split mismatch: payload={payload.get('split')}"
        )

    pss_cov = _as_float(payload.get("pss_implemented_coverage"), default=None)
    pss_ok = pss_cov is not None and pss_cov >= min_pss_implemented_coverage
    checks.append(
        {
            "check": f"stage3b_pss_coverage::{split}",
            "status": _status(pss_ok),
            "value": pss_cov,
            "min_required": min_pss_implemented_coverage,
        }
    )
    if not pss_ok:
        errors.append(
            f"[Stage3B/{split}] pss_implemented_coverage={pss_cov} "
            f"< min_required={min_pss_implemented_coverage}"
        )

    mode = str(payload.get("anatomy_quality_mode", ""))
    provider = str(payload.get("anatomy_provider_mode", ""))
    strict_gate = bool((payload.get("pilar_gates") or {}).get("strict_anatomy_gate"))

    if require_strict_anatomy:
        anatomy_ok = mode == "strict" and provider == expected_provider and strict_gate
        checks.append(
            {
                "check": f"stage3b_strict_anatomy::{split}",
                "status": _status(anatomy_ok),
                "mode": mode,
                "provider": provider,
                "strict_anatomy_gate": strict_gate,
            }
        )
        if not anatomy_ok:
            errors.append(
                f"[Stage3B/{split}] strict anatomy requirement failed "
                f"(mode={mode}, provider={provider}, strict_gate={strict_gate})"
            )

    path_warnings = payload.get("path_warnings") or []
    path_ok = allow_path_warnings or (len(path_warnings) == 0)
    checks.append(
        {
            "check": f"stage3b_path_warnings::{split}",
            "status": _status(path_ok),
            "count": len(path_warnings),
        }
    )
    if not path_ok:
        errors.append(f"[Stage3B/{split}] path_warnings present ({len(path_warnings)})")

    prov = payload.get("provenance")
    prov_ok = isinstance(prov, dict)
    checks.append(
        {
            "check": f"stage3b_provenance::{split}",
            "status": _status(prov_ok),
        }
    )
    if not prov_ok:
        errors.append(f"[Stage3B/{split}] missing provenance block")


def _validate_stage3c_split(
    split: str,
    audit_path: Path,
    expected_provider: str,
    require_strict_anatomy: bool,
    allow_path_warnings: bool,
    errors: List[str],
    checks: List[dict],
) -> None:
    if not audit_path.exists():
        return

    payload = _read_json(audit_path)
    gate_ok = str(payload.get("gate")) == "PASS"
    checks.append(
        {
            "check": f"stage3c_gate::{split}",
            "status": _status(gate_ok),
            "path": str(audit_path),
            "gate": payload.get("gate"),
        }
    )
    if not gate_ok:
        errors.append(f"[Stage3C/{split}] gate is not PASS")

    stage_ok = str(payload.get("stage")) == "3C"
    split_ok = str(payload.get("split")) == split
    checks.append(
        {
            "check": f"stage3c_stage_split::{split}",
            "status": _status(stage_ok and split_ok),
            "stage": payload.get("stage"),
            "split": payload.get("split"),
        }
    )
    if not stage_ok:
        errors.append(f"[Stage3C/{split}] invalid stage value: {payload.get('stage')}")
    if not split_ok:
        errors.append(
            f"[Stage3C/{split}] split mismatch: payload={payload.get('split')}"
        )

    anatomy = payload.get("anatomy") or {}
    if require_strict_anatomy:
        mode = str(anatomy.get("quality_mode", ""))
        provider = str(anatomy.get("provider_mode", ""))
        strict_gate = bool(anatomy.get("strict_anatomy_gate", False))
        anatomy_ok = mode == "strict" and provider == expected_provider and strict_gate
        checks.append(
            {
                "check": f"stage3c_strict_anatomy::{split}",
                "status": _status(anatomy_ok),
                "mode": mode,
                "provider": provider,
                "strict_anatomy_gate": strict_gate,
            }
        )
        if not anatomy_ok:
            errors.append(
                f"[Stage3C/{split}] strict anatomy requirement failed "
                f"(mode={mode}, provider={provider}, strict_gate={strict_gate})"
            )

    path_warnings = payload.get("path_warnings") or []
    path_ok = allow_path_warnings or (len(path_warnings) == 0)
    checks.append(
        {
            "check": f"stage3c_path_warnings::{split}",
            "status": _status(path_ok),
            "count": len(path_warnings),
        }
    )
    if not path_ok:
        errors.append(f"[Stage3C/{split}] path_warnings present ({len(path_warnings)})")

    prov = payload.get("provenance")
    prov_ok = isinstance(prov, dict)
    checks.append(
        {
            "check": f"stage3c_provenance::{split}",
            "status": _status(prov_ok),
        }
    )
    if not prov_ok:
        errors.append(f"[Stage3C/{split}] missing provenance block")


def _validate_stage3d_outputs(
    run_root: Path,
    expected_stage3c_dir: Path,
    allow_path_warnings: bool,
    errors: List[str],
    checks: List[dict],
) -> None:
    new_dir = run_root / "stage3d_ab" / "new"

    baseline_csv = new_dir / "baseline_results.csv"
    if baseline_csv.exists():
        rows = _read_csv(baseline_csv)
        has_b4_test = any(
            str(r.get("baseline")) == "B4" and str(r.get("split")) == "test"
            for r in rows
        )
        checks.append(
            {
                "check": "stage3d_b4_test_present",
                "status": _status(has_b4_test),
                "path": str(baseline_csv),
                "n_rows": len(rows),
            }
        )
        if not has_b4_test:
            errors.append("[Stage3D/new] baseline_results.csv missing B4 test rows")

    prov_path = new_dir / "stage3d_run_provenance.json"
    if prov_path.exists():
        prov = _read_json(prov_path)
        split_iso_ok = str(prov.get("split_isolation")) == "PASS"
        checks.append(
            {
                "check": "stage3d_split_isolation",
                "status": _status(split_iso_ok),
                "value": prov.get("split_isolation"),
            }
        )
        if not split_iso_ok:
            errors.append("[Stage3D/new] split_isolation is not PASS")

        path_warnings = prov.get("path_warnings") or []
        path_ok = allow_path_warnings or (len(path_warnings) == 0)
        checks.append(
            {
                "check": "stage3d_path_warnings",
                "status": _status(path_ok),
                "count": len(path_warnings),
            }
        )
        if not path_ok:
            errors.append(f"[Stage3D/new] path_warnings present ({len(path_warnings)})")

        expected = str(expected_stage3c_dir.resolve())
        actual = str(prov.get("stage3c_dir", ""))
        src_ok = actual == expected
        checks.append(
            {
                "check": "stage3d_source_stage3c_dir",
                "status": _status(src_ok),
                "expected": expected,
                "actual": actual,
            }
        )
        if not src_ok:
            errors.append(
                f"[Stage3D/new] stage3c_dir mismatch: expected={expected} actual={actual}"
            )


def _validate_stage3d4_outputs(
    run_root: Path,
    expected_stage3c_dir: Path,
    allow_path_warnings: bool,
    errors: List[str],
    checks: List[dict],
) -> None:
    new_dir = run_root / "stage3d4_ab" / "new"

    res_csv = new_dir / "tier_routing_results.csv"
    if res_csv.exists():
        rows = _read_csv(res_csv)
        header = list(rows[0].keys()) if rows else []
        required_cols = ["compare_baseline_source", "source_stage3c_dir"]
        has_cols = all(col in header for col in required_cols)
        checks.append(
            {
                "check": "stage3d4_results_schema",
                "status": _status(bool(rows) and has_cols),
                "path": str(res_csv),
                "n_rows": len(rows),
            }
        )
        if not rows:
            errors.append("[Stage3D.4/new] tier_routing_results.csv is empty")
        if not has_cols:
            errors.append(
                "[Stage3D.4/new] tier_routing_results.csv missing provenance columns"
            )

        expected = str(expected_stage3c_dir.resolve())
        source_values = {
            str(r.get("source_stage3c_dir", "")).strip()
            for r in rows
            if str(r.get("source_stage3c_dir", "")).strip()
        }
        source_ok = source_values == {expected}
        checks.append(
            {
                "check": "stage3d4_source_stage3c_dir",
                "status": _status(source_ok),
                "expected": expected,
                "actual_values": sorted(source_values),
            }
        )
        if not source_ok:
            errors.append(
                "[Stage3D.4/new] source_stage3c_dir values mismatch expected run root"
            )

    prov_path = new_dir / "stage3d4_run_provenance.json"
    if prov_path.exists():
        prov = _read_json(prov_path)
        split_iso_ok = str(prov.get("split_isolation")) == "PASS"
        checks.append(
            {
                "check": "stage3d4_split_isolation",
                "status": _status(split_iso_ok),
                "value": prov.get("split_isolation"),
            }
        )
        if not split_iso_ok:
            errors.append("[Stage3D.4/new] split_isolation is not PASS")

        path_warnings = prov.get("path_warnings") or []
        path_ok = allow_path_warnings or (len(path_warnings) == 0)
        checks.append(
            {
                "check": "stage3d4_path_warnings",
                "status": _status(path_ok),
                "count": len(path_warnings),
            }
        )
        if not path_ok:
            errors.append(
                f"[Stage3D.4/new] path_warnings present ({len(path_warnings)})"
            )


def _write_md(report: dict, md_path: Path) -> None:
    lines: List[str] = []
    lines.append("# B10 Server Acceptance Report")
    lines.append("")
    lines.append(f"- generated_at_utc: {report.get('generated_at_utc')}")
    lines.append(f"- run_root: {report.get('run_root')}")
    lines.append(f"- dataset: {report.get('dataset')}")
    lines.append(f"- overall_status: **{report.get('overall_status')}**")
    lines.append("")

    checks = report.get("checks", [])
    lines.append("## Check Summary")
    lines.append("")
    lines.append(f"- total_checks: {len(checks)}")
    lines.append(
        f"- pass_checks: {sum(1 for c in checks if c.get('status') == 'PASS')}"
    )
    lines.append(
        f"- fail_checks: {sum(1 for c in checks if c.get('status') == 'FAIL')}"
    )
    lines.append("")

    errors = report.get("errors", [])
    lines.append("## Blocking Errors")
    lines.append("")
    if errors:
        for err in errors:
            lines.append(f"- {err}")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Non-blocking Warnings")
    lines.append("")
    warnings = report.get("warnings", [])
    if warnings:
        for w in warnings:
            lines.append(f"- {w}")
    else:
        lines.append("- None")
    lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify B10 server execution acceptance gate."
    )
    parser.add_argument("--run_root", required=True, help="Run root directory")
    parser.add_argument("--dataset", default="oiaddr", help="Dataset key")
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["train", "valid", "test"],
        help="Splits to validate",
    )
    parser.add_argument(
        "--expected_anatomy_provider",
        default="external_pretrained",
        help="Expected provider for strict-quality claim",
    )
    parser.add_argument(
        "--require_strict_anatomy",
        action="store_true",
        help="Require strict external anatomy gating",
    )
    parser.add_argument(
        "--min_pss_implemented_coverage",
        type=float,
        default=0.50,
        help="Minimum accepted PSS implemented coverage per split",
    )
    parser.add_argument(
        "--allow_path_warnings",
        action="store_true",
        help="Allow path warnings without failing the gate",
    )
    parser.add_argument(
        "--report_json",
        default=None,
        help="Optional JSON report path. Default: <run_root>/summary/b10_server_acceptance_report.json",
    )
    parser.add_argument(
        "--report_md",
        default=None,
        help="Optional Markdown report path. Default: <run_root>/summary/b10_server_acceptance_report.md",
    )
    args = parser.parse_args()

    run_root = Path(args.run_root)
    stage3c_dir = run_root / "stage3c" / args.dataset
    summary_dir = run_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    report_json = (
        Path(args.report_json)
        if args.report_json
        else summary_dir / "b10_server_acceptance_report.json"
    )
    report_md = (
        Path(args.report_md)
        if args.report_md
        else summary_dir / "b10_server_acceptance_report.md"
    )

    errors: List[str] = []
    warnings: List[str] = []
    checks: List[dict] = []

    if not run_root.exists():
        errors.append(f"run_root not found: {run_root}")

    required_common = [
        run_root / "summary" / "point123_ab_compare_summary.json",
        run_root / "summary" / "point123_ab_compare_summary.md",
        run_root / "stage3d_ab" / "old" / "baseline_results.csv",
        run_root / "stage3d_ab" / "new" / "baseline_results.csv",
        run_root / "stage3d_ab" / "new" / "stage3d_run_provenance.json",
        run_root / "stage3d_ab" / "new" / "lgg_test_metrics.json",
        run_root / "stage3d4_ab" / "old" / "tier_routing_results.csv",
        run_root / "stage3d4_ab" / "new" / "tier_routing_results.csv",
        run_root / "stage3d4_ab" / "new" / "tier_routing_decision.md",
        run_root / "stage3d4_ab" / "new" / "stage3d4_run_provenance.json",
    ]
    for path in required_common:
        _check_file(path, path.name, errors, checks)

    for split in args.splits:
        split = split.strip().lower()
        if split not in {"train", "valid", "test"}:
            errors.append(f"Unsupported split name: {split}")
            continue

        s3b_json = (
            run_root / "stage3b" / args.dataset / split / f"evidence_audit_{split}.json"
        )
        s3c_split_json = (
            run_root
            / "stage3c"
            / args.dataset
            / split
            / f"governance_target_audit_{split}.json"
        )
        s3c_flat_csv = stage3c_dir / f"governance_features_{split}.csv"
        s3c_flat_json = stage3c_dir / f"governance_target_audit_{split}.json"

        _check_file(s3b_json, f"stage3b_{split}", errors, checks)
        _check_file(s3c_split_json, f"stage3c_split_audit_{split}", errors, checks)
        _check_file(s3c_flat_csv, f"stage3c_flat_csv_{split}", errors, checks)
        _check_file(s3c_flat_json, f"stage3c_flat_audit_{split}", errors, checks)

        _validate_stage3b_split(
            split=split,
            summary_path=s3b_json,
            expected_provider=args.expected_anatomy_provider,
            require_strict_anatomy=bool(args.require_strict_anatomy),
            min_pss_implemented_coverage=float(args.min_pss_implemented_coverage),
            allow_path_warnings=bool(args.allow_path_warnings),
            errors=errors,
            checks=checks,
        )
        _validate_stage3c_split(
            split=split,
            audit_path=s3c_flat_json,
            expected_provider=args.expected_anatomy_provider,
            require_strict_anatomy=bool(args.require_strict_anatomy),
            allow_path_warnings=bool(args.allow_path_warnings),
            errors=errors,
            checks=checks,
        )

    _validate_stage3d_outputs(
        run_root=run_root,
        expected_stage3c_dir=stage3c_dir,
        allow_path_warnings=bool(args.allow_path_warnings),
        errors=errors,
        checks=checks,
    )
    _validate_stage3d4_outputs(
        run_root=run_root,
        expected_stage3c_dir=stage3c_dir,
        allow_path_warnings=bool(args.allow_path_warnings),
        errors=errors,
        checks=checks,
    )

    overall_status = "PASS" if not errors else "FAIL"

    report = {
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "stage": "B10_server_acceptance",
        "run_root": str(run_root),
        "dataset": args.dataset,
        "expected_anatomy_provider": args.expected_anatomy_provider,
        "require_strict_anatomy": bool(args.require_strict_anatomy),
        "min_pss_implemented_coverage": float(args.min_pss_implemented_coverage),
        "allow_path_warnings": bool(args.allow_path_warnings),
        "checks": checks,
        "warnings": warnings,
        "errors": errors,
        "overall_status": overall_status,
    }

    report_json.parent.mkdir(parents=True, exist_ok=True)
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    report_md.parent.mkdir(parents=True, exist_ok=True)
    _write_md(report, report_md)

    print("[B10] Server acceptance verification")
    print(f"[B10] run_root   : {run_root}")
    print(f"[B10] checks     : {len(checks)}")
    print(f"[B10] errors     : {len(errors)}")
    print(f"[B10] report json: {report_json}")
    print(f"[B10] report md  : {report_md}")
    print(f"[B10] OVERALL    : {overall_status}")

    return 0 if overall_status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
