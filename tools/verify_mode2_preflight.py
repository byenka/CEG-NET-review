#!/usr/bin/env python
"""verify_mode2_preflight.py — Mode 2 preflight verifier (B1).

Read-only predicate gate that MUST pass before any Mode 2 training run.
Encodes the canonical checks from
``start29/Hasil analisa/Mode2_implementation_checklist Mei 31_claude.md`` §B1.

Mode 2 may only be started against a Mode 1 LOCKED baseline:
``outputs/point123_2026-05-30_point123_external_pss_stage3d_final_clean``.
Inputs from ``outputs/artefak/`` are refused unless ``--allow_historical_inputs``
is set (which is intended for OLD-baseline compare only, not active Mode 2 input).

Exit code 0 = GO, 1 = NO-GO.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_here = Path(__file__).resolve().parent
_root = _here.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


# Canonical Stage3C CSV header columns (Mode 1 LOCKED schema).
REQUIRED_STAGE3C_COLS: List[str] = [
    "bbox_id",
    "image_id",
    "split",
    "run_repo_root",
    "source_cache_dir",
    "source_annotations_dir",
    "source_kg_dir",
    "source_output_dir",
    "anatomy_provider_mode",
    "anatomy_quality_mode",
    "anatomy_checkpoint",
    "pss_source_json",
    "compare_baseline_source",
    "anatomy_provider_name",
    "anatomy_provider_method",
    "anatomy_fallback_to_approximate",
    "gt_class",
    "gt_supervised",
    "majority_class",
    "ev_sas",
    "ev_sdca",
    "ev_aps",
    "ev_pss",
    "ev_pss_status",
    "radial_zone",
]


# --- helpers -----------------------------------------------------------------


def _norm(path: str) -> str:
    return str(Path(path).resolve()).replace("\\", "/").lower()


def _read_header(path: Path) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return next(csv.reader(f))


def _col_std(path: Path, col: str) -> Optional[float]:
    vals: List[float] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            v = (row.get(col) or "").strip()
            if not v or v.lower() in {"nan", "none"}:
                continue
            try:
                vals.append(float(v))
            except ValueError:
                continue
    return statistics.pstdev(vals) if len(vals) >= 2 else None


def _has_value_substring(path: Path, col: str, needle: str) -> bool:
    needle_l = needle.lower()
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if needle_l in (row.get(col) or "").lower():
                return True
    return False


def _any_supervised(path: Path) -> bool:
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = (row.get("gt_supervised") or "").strip()
            if v in {"1", "true", "True"}:
                return True
    return False


def _any_real_pss_source(path: Path) -> bool:
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = (row.get("pss_source_json") or "").strip()
            if v and v.lower() != "legacy_unknown":
                return True
    return False


# --- check primitives --------------------------------------------------------


Check = Dict[str, object]


def _pass(name: str, message: str = "") -> Check:
    return {"name": name, "status": "PASS", "message": message}


def _fail(name: str, message: str) -> Check:
    return {"name": name, "status": "FAIL", "message": message}


def check_mode1_lock_file(path: Path) -> Check:
    return _pass("mode1_lock_file_exists", str(path)) if path.exists() else _fail(
        "mode1_lock_file_exists", f"missing: {path}"
    )


def check_b10_report(run_root: Path) -> List[Check]:
    report = run_root / "summary" / "b10_server_acceptance_report.json"
    if not report.exists():
        return [
            _fail("b10_report_exists", f"missing: {report}"),
            _fail("b10_overall_pass", "report missing"),
            _fail("b10_errors_empty", "report missing"),
            _fail("b10_all_checks_pass", "report missing"),
        ]
    try:
        data = json.loads(report.read_text(encoding="utf-8"))
    except Exception as exc:
        return [
            _fail("b10_report_exists", f"unreadable: {exc}"),
            _fail("b10_overall_pass", "unreadable"),
            _fail("b10_errors_empty", "unreadable"),
            _fail("b10_all_checks_pass", "unreadable"),
        ]
    overall = data.get("overall_status")
    errors = data.get("errors") or []
    checks = data.get("checks") or []
    non_pass = [
        c for c in checks if isinstance(c, dict) and c.get("status") != "PASS"
    ]
    return [
        _pass("b10_report_exists", str(report)),
        _pass("b10_overall_pass", "PASS")
        if overall == "PASS"
        else _fail("b10_overall_pass", f"overall_status={overall!r}"),
        _pass("b10_errors_empty", f"errors=[] (len={len(errors)})")
        if len(errors) == 0
        else _fail("b10_errors_empty", f"len(errors)={len(errors)}: {errors[:3]}"),
        _pass("b10_all_checks_pass", f"checks={len(checks)} all PASS")
        if not non_pass and len(checks) > 0
        else _fail(
            "b10_all_checks_pass",
            f"non-PASS={len(non_pass)} of {len(checks)}; first={non_pass[:1]}",
        ),
    ]


def check_stage3c_layout(stage3c_dir: Path, allow_historical: bool) -> List[Check]:
    results: List[Check] = []
    norm = _norm(str(stage3c_dir))
    if "/outputs/artefak/" in norm and not allow_historical:
        results.append(
            _fail(
                "stage3c_not_artefak",
                f"stage3c_dir under outputs/artefak/: {stage3c_dir} "
                "(use --allow_historical_inputs only for historical baseline compare)",
            )
        )
    else:
        results.append(_pass("stage3c_not_artefak", str(stage3c_dir)))
    sample: Optional[Path] = None
    for split in ("train", "valid", "test"):
        p = stage3c_dir / f"governance_features_{split}.csv"
        if p.exists():
            results.append(_pass(f"stage3c_csv_{split}_exists", str(p)))
            sample = sample or p
        else:
            results.append(_fail(f"stage3c_csv_{split}_exists", f"missing: {p}"))
    if sample is None:
        results.append(_fail("stage3c_schema_columns", "no CSV available"))
        return results
    try:
        header = _read_header(sample)
    except Exception as exc:
        results.append(_fail("stage3c_schema_columns", f"header read error: {exc}"))
        return results
    missing = [c for c in REQUIRED_STAGE3C_COLS if c not in header]
    if missing:
        results.append(
            _fail(
                "stage3c_schema_columns",
                f"missing canonical cols ({len(missing)}): "
                f"{missing[:6]}{' ...' if len(missing) > 6 else ''}",
            )
        )
    else:
        results.append(
            _pass(
                "stage3c_schema_columns",
                f"{len(REQUIRED_STAGE3C_COLS)} canonical cols present in {sample.name}",
            )
        )
    return results


def check_stage3c_data(stage3c_dir: Path) -> List[Check]:
    results: List[Check] = []
    train_csv = stage3c_dir / "governance_features_train.csv"
    if not train_csv.exists():
        for n in (
            "stage3c_gt_supervised_any",
            "stage3c_ev_aps_nonconstant",
            "stage3c_ev_pss_nonconstant",
            "stage3c_anatomy_external_pretrained",
            "stage3c_anatomy_strict",
            "stage3c_pss_source_json",
        ):
            results.append(_fail(n, "train CSV missing"))
        return results
    any_sup = False
    for split in ("train", "valid", "test"):
        p = stage3c_dir / f"governance_features_{split}.csv"
        if p.exists() and _any_supervised(p):
            any_sup = True
            break
    results.append(
        _pass("stage3c_gt_supervised_any", "at least one gt_supervised=1 found")
        if any_sup
        else _fail("stage3c_gt_supervised_any", "no gt_supervised=1 row found")
    )
    aps_std = _col_std(train_csv, "ev_aps")
    results.append(
        _pass("stage3c_ev_aps_nonconstant", f"std={aps_std:.6f}")
        if aps_std is not None and aps_std > 1e-9
        else _fail(
            "stage3c_ev_aps_nonconstant",
            f"std={aps_std} (APS still constant — anatomy is approximate fallback?)",
        )
    )
    pss_std = _col_std(train_csv, "ev_pss")
    results.append(
        _pass("stage3c_ev_pss_nonconstant", f"std={pss_std:.6f}")
        if pss_std is not None and pss_std > 1e-9
        else _fail(
            "stage3c_ev_pss_nonconstant",
            f"std={pss_std} (PSS still placeholder 0.5?)",
        )
    )
    has_external = _has_value_substring(
        train_csv, "anatomy_provider_mode", "external_pretrained"
    ) or _has_value_substring(
        train_csv, "anatomy_provider_name", "external_pretrained"
    )
    results.append(
        _pass("stage3c_anatomy_external_pretrained", "external_pretrained found")
        if has_external
        else _fail(
            "stage3c_anatomy_external_pretrained",
            "no rows match external_pretrained in anatomy_provider_mode/name",
        )
    )
    has_strict = _has_value_substring(train_csv, "anatomy_quality_mode", "strict")
    results.append(
        _pass("stage3c_anatomy_strict", "anatomy_quality_mode contains 'strict'")
        if has_strict
        else _fail(
            "stage3c_anatomy_strict",
            "no rows with anatomy_quality_mode containing 'strict'",
        )
    )
    results.append(
        _pass("stage3c_pss_source_json", "at least one real pss_source_json value")
        if _any_real_pss_source(train_csv)
        else _fail(
            "stage3c_pss_source_json",
            "all rows have pss_source_json empty or 'legacy_unknown'",
        )
    )
    return results


def check_lgg_checkpoint(pkl_path: Path) -> List[Check]:
    results: List[Check] = []
    if not pkl_path.exists():
        results.append(_fail("lgg_checkpoint_exists", f"missing: {pkl_path}"))
        results.append(_fail("lgg_b3_torch_key", "checkpoint missing"))
        results.append(_fail("lgg_b3_state_dict_nonempty", "checkpoint missing"))
        return results
    results.append(_pass("lgg_checkpoint_exists", str(pkl_path)))
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
    except Exception as exc:
        results.append(_fail("lgg_b3_torch_key", f"unreadable: {exc}"))
        results.append(_fail("lgg_b3_state_dict_nonempty", "unreadable"))
        return results
    if "B3_LGG_TORCH" not in data:
        results.append(
            _fail(
                "lgg_b3_torch_key",
                f"B3_LGG_TORCH missing; have keys={list(data.keys())[:6]}",
            )
        )
        results.append(_fail("lgg_b3_state_dict_nonempty", "B3_LGG_TORCH missing"))
        return results
    results.append(_pass("lgg_b3_torch_key", "B3_LGG_TORCH present"))
    obj = data.get("B3_LGG_TORCH") or {}
    sd = obj.get("state_dict") if isinstance(obj, dict) else None
    if not isinstance(sd, dict) or not sd:
        results.append(
            _fail(
                "lgg_b3_state_dict_nonempty",
                f"state_dict missing/empty (type={type(sd).__name__})",
            )
        )
    else:
        results.append(_pass("lgg_b3_state_dict_nonempty", f"{len(sd)} params"))
    return results


def check_baseline(baseline_path: Path, expected: float, tol: float) -> List[Check]:
    results: List[Check] = []
    if not baseline_path.exists():
        results.append(_fail("baseline_audit_exists", f"missing: {baseline_path}"))
        results.append(_fail("baseline_mAP_50_matches", "baseline audit missing"))
        return results
    results.append(_pass("baseline_audit_exists", str(baseline_path)))
    try:
        data = json.loads(baseline_path.read_text(encoding="utf-8"))
    except Exception as exc:
        results.append(_fail("baseline_mAP_50_matches", f"unreadable: {exc}"))
        return results
    m = (data.get("map_metrics") or {}).get("mAP_50")
    if m is None:
        results.append(_fail("baseline_mAP_50_matches", "map_metrics.mAP_50 missing"))
        return results
    if abs(float(m) - expected) <= tol:
        results.append(
            _pass(
                "baseline_mAP_50_matches",
                f"mAP_50={float(m):.6f} (expected={expected:.6f}, tol={tol})",
            )
        )
    else:
        results.append(
            _fail(
                "baseline_mAP_50_matches",
                f"mAP_50={float(m):.6f} expected={expected:.6f} tol={tol}",
            )
        )
    return results


def check_gt_class_leakage(stage3c_dir: Path) -> Check:
    try:
        from tools.verify_gt_class_leakage_fix import verify_split  # type: ignore
    except Exception as exc:
        return _fail(
            "gt_class_leakage_verifier",
            f"cannot import verify_gt_class_leakage_fix: {exc}",
        )
    total_fabricated = 0
    splits_checked: List[str] = []
    has_supervised_col = True
    for split in ("train", "valid", "test"):
        p = stage3c_dir / f"governance_features_{split}.csv"
        if not p.exists():
            continue
        try:
            r = verify_split(p)
        except Exception as exc:
            return _fail(
                "gt_class_leakage_verifier",
                f"verify_split({split}) error: {exc}",
            )
        splits_checked.append(split)
        total_fabricated += int(r.get("fabricated_labels") or 0)
        if not r.get("has_gt_supervised_col"):
            has_supervised_col = False
    if not splits_checked:
        return _fail("gt_class_leakage_verifier", "no Stage3C CSV to check")
    if total_fabricated == 0 and has_supervised_col:
        return _pass(
            "gt_class_leakage_verifier",
            f"fabricated=0 across {splits_checked}, gt_supervised_col=True",
        )
    return _fail(
        "gt_class_leakage_verifier",
        f"fabricated={total_fabricated} has_gt_supervised_col={has_supervised_col}",
    )


# --- main --------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mode 2 preflight verifier (B1) — read-only.",
    )
    parser.add_argument("--mode1_lock_file", required=True)
    parser.add_argument("--mode1_run_root", required=True)
    parser.add_argument("--stage3c_dir", required=True)
    parser.add_argument("--lgg_checkpoint", required=True)
    parser.add_argument("--baseline_audit", required=True)
    parser.add_argument("--expected_baseline_map50", type=float, default=0.241129)
    parser.add_argument("--baseline_map50_tol", type=float, default=1e-6)
    parser.add_argument("--report_json", default=None)
    parser.add_argument("--report_md", default=None)
    parser.add_argument(
        "--allow_historical_inputs",
        action="store_true",
        help="Permit outputs/artefak/ Stage3C input. For OLD baseline compare only.",
    )
    args = parser.parse_args()

    results: List[Check] = []
    results.append(check_mode1_lock_file(Path(args.mode1_lock_file)))
    results.extend(check_b10_report(Path(args.mode1_run_root)))
    results.extend(
        check_stage3c_layout(
            Path(args.stage3c_dir), allow_historical=args.allow_historical_inputs
        )
    )
    results.extend(check_stage3c_data(Path(args.stage3c_dir)))
    results.extend(check_lgg_checkpoint(Path(args.lgg_checkpoint)))
    results.extend(
        check_baseline(
            Path(args.baseline_audit),
            args.expected_baseline_map50,
            args.baseline_map50_tol,
        )
    )
    results.append(check_gt_class_leakage(Path(args.stage3c_dir)))

    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    overall = "PASS" if n_fail == 0 else "FAIL"

    print("[Mode2Preflight] === Mode 2 Preflight (B1) ===")
    print(f"[Mode2Preflight] mode1_lock_file = {args.mode1_lock_file}")
    print(f"[Mode2Preflight] mode1_run_root  = {args.mode1_run_root}")
    print(f"[Mode2Preflight] stage3c_dir     = {args.stage3c_dir}")
    print(f"[Mode2Preflight] lgg_checkpoint  = {args.lgg_checkpoint}")
    print(
        f"[Mode2Preflight] baseline_audit  = {args.baseline_audit}  "
        f"(expected mAP_50={args.expected_baseline_map50})"
    )
    for r in results:
        print(f"  [{r['status']}] {r['name']}: {r['message']}")
    print(
        f"[Mode2Preflight] Checks: {len(results)}  PASS: {n_pass}  FAIL: {n_fail}"
    )
    print(f"[Mode2Preflight] OVERALL: {overall}")

    if args.report_json:
        out = Path(args.report_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "tool": "verify_mode2_preflight",
                    "mode1_lock_file": args.mode1_lock_file,
                    "mode1_run_root": args.mode1_run_root,
                    "stage3c_dir": args.stage3c_dir,
                    "lgg_checkpoint": args.lgg_checkpoint,
                    "baseline_audit": args.baseline_audit,
                    "expected_baseline_map50": args.expected_baseline_map50,
                    "baseline_map50_tol": args.baseline_map50_tol,
                    "checks": results,
                    "n_pass": n_pass,
                    "n_fail": n_fail,
                    "overall_status": overall,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[Mode2Preflight] JSON report: {out}")

    if args.report_md:
        out = Path(args.report_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Mode 2 Preflight Report (B1)",
            "",
            f"- **OVERALL: {overall}**",
            f"- Checks: {len(results)} (PASS={n_pass}, FAIL={n_fail})",
            f"- mode1_lock_file: `{args.mode1_lock_file}`",
            f"- mode1_run_root: `{args.mode1_run_root}`",
            f"- stage3c_dir: `{args.stage3c_dir}`",
            f"- lgg_checkpoint: `{args.lgg_checkpoint}`",
            f"- baseline_audit: `{args.baseline_audit}` (expected mAP_50={args.expected_baseline_map50})",
            "",
            "| # | Status | Check | Message |",
            "|---|---|---|---|",
        ]
        for i, r in enumerate(results, 1):
            msg = str(r["message"]).replace("|", "/").replace("\n", " ")
            lines.append(f"| {i} | {r['status']} | `{r['name']}` | {msg} |")
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[Mode2Preflight] MD report: {out}")

    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
