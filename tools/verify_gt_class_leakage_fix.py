#!/usr/bin/env python


from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

# Categories that correspond to a valid IoU>=0.5 GT match (supervised gt_class).
_SUPERVISED_CATEGORIES = {"TP_EXACT", "CLINICALLY_CLOSE_MISMATCH", "UNSAFE_MISMATCH"}
# Categories that must NEVER carry a supervised gt_class (no valid GT match).
_UNRESOLVED_CATEGORIES = {"LOCALIZATION_WEAK", "BACKGROUND_FP", "LOW_CONF_CANDIDATE"}
_LESION_CLASSES = {"MA", "HE", "Hard_EX", "Soft_EX"}


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _is_class(value: str) -> bool:
    return (value or "").strip() in _LESION_CLASSES


def verify_split(path: Path) -> Dict:
    rows = _read_rows(path)
    n = len(rows)
    has_col = bool(rows) and "gt_class" in rows[0]
    has_supervised_col = bool(rows) and "gt_supervised" in rows[0]

    fabricated = 0  # unresolved-category rows that still carry a gt_class label
    fabricated_by_cat: Dict[str, int] = {}
    supervised_rows = 0
    supervised_gt_eq_majority = 0
    empty_gt = 0
    cat_counts: Dict[str, int] = {}

    for r in rows:
        cat = (r.get("category") or "").strip()
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        gt = (r.get("gt_class") or "").strip()
        maj = (r.get("majority_class") or "").strip()
        if not _is_class(gt):
            empty_gt += 1
        if cat in _UNRESOLVED_CATEGORIES and _is_class(gt):
            fabricated += 1
            fabricated_by_cat[cat] = fabricated_by_cat.get(cat, 0) + 1
        if cat in _SUPERVISED_CATEGORIES:
            supervised_rows += 1
            if _is_class(gt) and gt == maj:
                supervised_gt_eq_majority += 1

    # The parser keeps only non-R4 rows with a valid gt_class. After the fix that
    # is exactly the supervised (IoU>=0.5) set.
    kept_supervised = sum(
        1
        for r in rows
        if _is_class((r.get("gt_class") or "").strip())
        and (r.get("tier_target") or "").strip() != "R4"
    )

    # Overall gt==majority across ALL rows (the old leakage headline metric).
    overall_eq = sum(
        1
        for r in rows
        if _is_class((r.get("gt_class") or "").strip())
        and (r.get("gt_class") or "").strip() == (r.get("majority_class") or "").strip()
    )

    return {
        "path": str(path),
        "n_rows": n,
        "has_gt_class_col": has_col,
        "has_gt_supervised_col": has_supervised_col,
        "empty_gt_rate": (empty_gt / n) if n else 0.0,
        "overall_gt_eq_majority_rate": (overall_eq / n) if n else 0.0,
        "fabricated_labels": fabricated,
        "fabricated_by_category": fabricated_by_cat,
        "supervised_rows": supervised_rows,
        "supervised_gt_eq_majority_rate": (
            supervised_gt_eq_majority / supervised_rows if supervised_rows else 0.0
        ),
        "kept_supervised_rows_for_stage5": kept_supervised,
        "category_counts": cat_counts,
        # PASS iff no unresolved-category box carries a fabricated label AND the
        # new gt_supervised column exists (i.e. CSV was produced by fixed Stage 3C).
        "pass": fabricated == 0 and has_supervised_col,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify gt_class leakage fix.")
    parser.add_argument(
        "--stage3c_dir",
        default="outputs/stage3c_audit",
        help="Directory containing governance_features_{split}.csv",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["train", "valid", "test"],
        help="Splits to check (missing files are skipped).",
    )
    args = parser.parse_args()

    base = Path(args.stage3c_dir)
    results = []
    for split in args.splits:
        p = base / f"governance_features_{split}.csv"
        if not p.exists():
            print(f"[verify] SKIP (missing): {p}")
            continue
        results.append(verify_split(p))

    if not results:
        print("[verify] ERROR: no governance_features_*.csv found.")
        return 2

    all_pass = True
    for r in results:
        verdict = "PASS" if r["pass"] else "FAIL"
        if not r["pass"]:
            all_pass = False
        print("")
        print(f"[verify] {Path(r['path']).name}: {verdict}")
        print(f"           rows={r['n_rows']}  gt_supervised_col={r['has_gt_supervised_col']}")
        print(
            f"           empty_gt_rate={r['empty_gt_rate']:.3f}  "
            f"overall_gt==majority={r['overall_gt_eq_majority_rate']:.3f}"
        )
        print(
            f"           fabricated_labels(non-matched cats with gt_class)="
            f"{r['fabricated_labels']}  {r['fabricated_by_category']}"
        )
        print(
            f"           supervised_rows(IoU>=0.5)={r['supervised_rows']}  "
            f"supervised gt==majority={r['supervised_gt_eq_majority_rate']:.3f} "
            f"(legit-high for TP_EXACT)"
        )
        print(
            f"           kept_supervised_rows_for_stage5={r['kept_supervised_rows_for_stage5']}"
        )
        print(f"           category_counts={r['category_counts']}")

    print("")
    if all_pass:
        print("[verify] OVERALL: PASS - no fabricated gt_class in non-matched categories.")
        print("[verify] Stage 5 proxy reclass_acc is now measured on real GT only.")
        return 0
    print("[verify] OVERALL: FAIL - leakage still present (regenerate Stage 3C with the fix).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
