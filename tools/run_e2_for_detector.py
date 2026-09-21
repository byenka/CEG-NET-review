#!/usr/bin/env python
"""run_e2_for_detector.py — E2 orchestrator for ONE comparison detector.

Chains the detector-agnostic governance pipeline by REUSING the existing,
proven tools verbatim (no reimplementation, no Mode-1 modification):

  1. export    : tools/export_compare_predictions.py  (checkpoint -> DetectorBox cache)
  2. stage3c   : tools/audit_stage3c_governance_targets.py  (cache -> governance_features
                 _{valid,test}.csv)  [run once per split, separate clean dirs]
  3. benchmark : tools/eval_fp_reduction_benchmark.py  (CSV -> fp_benchmark.json with
                 KG-gate vs M0 FP-suppression)

Then run tools/aggregate_e2_governance.py across detectors for the verdict.

NO-OVERWRITE: every stage refuses to clobber prior results —
  * export uses its own guard (export_summary.json sentinel),
  * stage3c uses --require_clean_output_dir (separate dir per split),
  * benchmark output is guarded here before the call.
All under a NEW namespace (outputs/detcmp/<tag>_*), separate from Mode 1.

Use --dry_run to print the exact commands WITHOUT executing (recommended first).

Anatomy/KG/PSS: defaults are the LIGHT 'approximate' provider (no ONNX needed),
which is internally consistent across detectors. To compare against the Mode-1
headline number, pass the SAME settings Mode 1 used:
  --anatomy_provider external_pretrained --anatomy_checkpoint <onnx> \
  --anatomy_quality_mode strict --pss_dir <tta pss dir>

Usage:
  python tools/run_e2_for_detector.py --tag retinanet \
    --checkpoint outputs/detcmp/retinanet/best.pt \
    --out_root outputs/detcmp --kg_dir outputs/stage3_audit \
    --data_root data/oiaddr --use_class_probs --dry_run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
_root = _here.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from tools._safe_output import guard_file  # noqa: E402

PY = sys.executable or "python"


def _run(cmd, dry: bool, label: str) -> None:
    print(f"\n[e2] === {label} ===")
    print("[e2] $ " + " ".join(str(c) for c in cmd))
    if dry:
        return
    res = subprocess.run([str(c) for c in cmd])
    if res.returncode != 0:
        print(f"[e2] STAGE FAILED ({label}) rc={res.returncode}; aborting.")
        raise SystemExit(res.returncode)


def main() -> int:
    ap = argparse.ArgumentParser(description="E2 orchestrator for one detector.")
    ap.add_argument("--tag", required=True, help="short detector tag (dir prefix).")
    ap.add_argument("--checkpoint", required=True, help="best.pt from train_detector_compare.py")
    ap.add_argument("--out_root", default="outputs/detcmp",
                    help="namespace root for E2 artifacts (separate from Mode 1).")
    ap.add_argument("--data_root", default="data/oiaddr")
    ap.add_argument("--kg_dir", default="outputs/stage3_audit",
                    help="Stage 3A KG artifacts dir (same KG as Mode 1).")
    ap.add_argument("--splits", nargs="+", default=["valid", "test"],
                    choices=["valid", "test"])
    ap.add_argument("--score_thresh", type=float, default=0.05)
    ap.add_argument("--detections_per_img", type=int, default=300)
    ap.add_argument("--use_class_probs", action="store_true")
    # anatomy / pss pass-through (default light)
    ap.add_argument("--anatomy_provider", default="approximate",
                    choices=["approximate", "radial_zone", "external_pretrained"])
    ap.add_argument("--anatomy_quality_mode", default="relaxed",
                    choices=["relaxed", "strict"])
    ap.add_argument("--anatomy_checkpoint", default=None)
    ap.add_argument("--anatomy_device", default="cpu")
    ap.add_argument("--pss_dir", default=None)
    # benchmark
    ap.add_argument("--tp_ret_target", type=float, default=0.80)
    ap.add_argument("--positive", choices=["tp_exact", "near_lesion"], default="tp_exact")
    ap.add_argument("--require_cuda", action="store_true")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow replacing prior E2 artifacts for this --tag.")
    ap.add_argument("--dry_run", action="store_true",
                    help="print commands without executing.")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    cache_dir = out_root / f"{args.tag}_cache"
    s3c_root = out_root / f"{args.tag}_stage3c"
    bench_json = out_root / f"{args.tag}_fp_benchmark.json"

    # Guard the final benchmark output up-front (fail fast before heavy compute).
    if not args.dry_run:
        guard_file(bench_json, overwrite=args.overwrite, tag="[e2]")

    # ---- 1. export ----
    export_cmd = [PY, "tools/export_compare_predictions.py",
                  "--checkpoint", args.checkpoint,
                  "--splits", *args.splits,
                  "--score_thresh", args.score_thresh,
                  "--detections_per_img", args.detections_per_img,
                  "--data_root", args.data_root,
                  "--output_dir", cache_dir]
    if args.use_class_probs:
        export_cmd.append("--use_class_probs")
    if args.require_cuda:
        export_cmd.append("--require_cuda")
    if args.overwrite:
        export_cmd.append("--overwrite")
    _run(export_cmd, args.dry_run, "1/3 export predictions")

    # ---- 2. stage3c per split (separate clean dirs => no-overwrite) ----
    for split in args.splits:
        ann_dir = Path(args.data_root) / "xml" / split
        s3c_out = s3c_root / split
        s3c_cmd = [PY, "tools/audit_stage3c_governance_targets.py",
                   "--annotations_dir", ann_dir,
                   "--cache_dir", cache_dir / split,
                   "--kg_dir", args.kg_dir,
                   "--split", split,
                   "--output_dir", s3c_out,
                   "--anatomy_provider", args.anatomy_provider,
                   "--anatomy_quality_mode", args.anatomy_quality_mode,
                   "--anatomy_device", args.anatomy_device]
        if args.anatomy_checkpoint:
            s3c_cmd += ["--anatomy_checkpoint", args.anatomy_checkpoint]
        if args.pss_dir:
            s3c_cmd += ["--pss_dir", args.pss_dir]
        if not args.overwrite:
            s3c_cmd.append("--require_clean_output_dir")
        _run(s3c_cmd, args.dry_run, f"2/3 stage3c governance_features ({split})")

    # ---- 3. benchmark ----
    valid_csv = s3c_root / "valid" / "governance_features_valid.csv"
    test_csv = s3c_root / "test" / "governance_features_test.csv"
    bench_cmd = [PY, "tools/eval_fp_reduction_benchmark.py",
                 "--valid_csv", valid_csv, "--test_csv", test_csv,
                 "--tp_ret_target", args.tp_ret_target,
                 "--positive", args.positive,
                 "--output_json", bench_json]
    _run(bench_cmd, args.dry_run, "3/3 FP-reduction benchmark")

    print(f"\n[e2] detector '{args.tag}' pipeline {'PLANNED (dry-run)' if args.dry_run else 'DONE'}.")
    print(f"[e2]   cache:     {cache_dir}")
    print(f"[e2]   stage3c:   {s3c_root}/{{valid,test}}/governance_features_*.csv")
    print(f"[e2]   benchmark: {bench_json}")
    print(f"[e2] After >=2 detectors: python tools/aggregate_e2_governance.py "
          f"--glob '{out_root}/*_fp_benchmark.json'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
