#!/usr/bin/env python
"""export_yolo_predictions.py — E2 (Point 4): export an ULTRALYTICS detector
(YOLOv8 / YOLO11 / RT-DETR) to the SAME DetectorBox JSON cache the governance
pipeline consumes — so a modern 2026 detector can flow through the identical
Stage3C builder + FP-reduction benchmark as the torchvision detectors.

Why a separate exporter (not tools/export_compare_predictions.py):
  * export_compare_predictions.py does torch.load(ckpt)["model_state_dict"] +
    cfg["model_type"] in the torchvision factory SUPPORTED set, then
    load_state_dict on a torchvision module. An ultralytics best.pt has a totally
    different structure (an ultralytics model object), so it CANNOT pass through
    that path. This adapter bridges the gap.

Format fidelity: we reuse the EXACT DetectorBox formatter the Mode-1 / E2
exporters use (``_predictions_to_detector_boxes``) and the SAME zone assignment
(``estimate_landmarks_approx``), and we iterate ``DRDetectionDataset`` so the
image set, order and stems are IDENTICAL to every other detector. Downstream
Stage3C treats the YOLO cache exactly like any other detector.

Class probabilities (honesty contract):
  Ultralytics post-NMS exposes only the top-1 class + confidence per box (the
  full per-class vector is not retained after NMS). We therefore write
  ``class_probs=None`` -> the formatter falls back to the documented synthetic
  score-residual distribution and stamps ``probs_source="synthetic_score_residual"``
  per box — EXACTLY like the existing fcos / ssd300 rows. The real-class-prob
  (SDCA) HEADLINE stays on the Mode-1 Faster R-CNN; this YOLO row demonstrates
  DETECTOR-AGNOSTIC FP-suppression of the same governance gate, with SDCA from
  the same honest proxy already used for fcos/ssd. (A future real-prob YOLO path
  would need pre-NMS class-logit extraction — out of scope here, documented.)

Operating point parity: --score_thresh 0.05 and --detections_per_img 300 mirror
the torchvision export so the candidate pool per image is comparable. NMS IoU is
the ultralytics default (0.7), recorded in export_summary.json.

Does NOT touch Mode 1. NEVER overwrites a previous cache (guarded).

Usage (server, after the YOLO best.pt exists):
  python tools/export_yolo_predictions.py \
    --checkpoint runs/detect/outputs/detcmp_yolo/yolo11s/yolo11s/weights/best.pt \
    --splits valid test \
    --score_thresh 0.05 --detections_per_img 300 --imgsz 800 \
    --data_root data/oiaddr \
    --output_dir outputs/0506/yolo_e2/yolo11s_cache --device 0
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

_here = Path(__file__).resolve().parent
_root = _here.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch

from src.data.coco_adapter import DRDetectionDataset
from src.data.zoning import estimate_landmarks_approx
# Reuse the EXACT DetectorBox formatter the Mode-1 / E2 exporters use.
from tools.export_detector_predictions import _predictions_to_detector_boxes
from tools._safe_output import guard_output_dir

# YOLO data.yaml class order (tools/convert_voc_to_yolo.py): 0=MA 1=HE 2=Hard_EX 3=Soft_EX.
# DetectorBox LABEL_MAP is 1-indexed (1=MA ... 4=Soft_EX) -> label = yolo_cls + 1.
_YOLO_TO_DETECTORBOX_LABEL = lambda c: int(c) + 1  # noqa: E731


def _load_yolo(checkpoint: str):
    """Load an ultralytics model (YOLO or RT-DETR). Clear error if missing."""
    try:
        from ultralytics import YOLO, RTDETR
    except Exception as exc:  # pragma: no cover - depends on server env
        print("[yolo-export] ERROR: ultralytics not installed. "
              "Run: pip install ultralytics")
        print(f"[yolo-export] import error: {exc}")
        raise SystemExit(2)
    name = Path(checkpoint).name.lower()
    if "rtdetr" in name or "rt-detr" in name:
        return RTDETR(checkpoint), "rtdetr"
    return YOLO(checkpoint), "yolo"


@torch.no_grad()
def export_split(model, family, split, output_dir, data_root, device,
                 score_thresh, detections_per_img, imgsz, nms_iou,
                 max_images: Optional[int]) -> dict:
    split_dir = Path(output_dir) / split
    split_dir.mkdir(parents=True, exist_ok=True)
    ds = DRDetectionDataset(split=split, data_root=data_root, max_images=max_images)

    total, n_synth = 0, 0
    manifest = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*estimate_landmarks_approx.*")
        for idx in range(len(ds)):
            s = ds._samples[idx]
            stem, img_path = s["stem"], s["img_path"]
            # Predict on the ORIGINAL image file; ultralytics returns xyxy in the
            # original (un-letterboxed) pixel space, matching torchvision coords.
            res = model.predict(source=img_path, imgsz=imgsz, conf=score_thresh,
                                iou=nms_iou, max_det=detections_per_img,
                                device=device, verbose=False)[0]
            H, W = int(res.orig_shape[0]), int(res.orig_shape[1])
            lm = estimate_landmarks_approx(image_w=W, image_h=H)

            b = res.boxes
            if b is None or b.shape[0] == 0:
                boxes = torch.zeros(0, 4)
                labels = torch.zeros(0, dtype=torch.long)
                scores = torch.zeros(0)
            else:
                boxes = b.xyxy.detach().cpu().float()
                scores = b.conf.detach().cpu().float()
                cls = b.cls.detach().cpu()
                labels = torch.tensor(
                    [_YOLO_TO_DETECTORBOX_LABEL(c) for c in cls.tolist()],
                    dtype=torch.long)

            # class_probs=None -> synthetic score-residual (honest, == fcos/ssd path)
            det_boxes = _predictions_to_detector_boxes(
                stem, boxes, labels, scores, lm, class_probs=None)
            (split_dir / f"{stem}.json").write_text(
                json.dumps(det_boxes, separators=(",", ":")), encoding="utf-8")
            total += len(det_boxes)
            n_synth += len(det_boxes)
            manifest.append({"stem": stem, "n_pred": len(det_boxes),
                             "image_w": W, "image_h": H})
            if (idx + 1) % 50 == 0 or (idx + 1) == len(ds):
                print(f"  [{idx+1}/{len(ds)}] last={stem} n_pred={len(det_boxes)}")

    mpath = Path(output_dir) / f"{split}_manifest.csv"
    with open(mpath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "n_pred", "image_w", "image_h"])
        w.writeheader(); w.writerows(manifest)
    avg = total / max(len(ds), 1)
    print(f"[yolo-export] {split}: {len(ds)} imgs, {total} preds "
          f"({avg:.1f} avg/img)  probs_source=synthetic (100%) -> {split_dir}")
    return {"split": split, "n_images": len(ds), "total_preds": total,
            "n_real_prob_boxes": 0, "n_synth_prob_boxes": n_synth,
            "real_prob_coverage": 0.0, "output_dir": str(split_dir),
            "manifest_csv": str(mpath)}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="E2 (Point 4): export ultralytics detector -> DetectorBox cache.")
    ap.add_argument("--checkpoint", required=True,
                    help="ultralytics best.pt (YOLOv8/YOLO11/RT-DETR trained on OIA-DDR).")
    ap.add_argument("--splits", nargs="+", default=["valid", "test"],
                    choices=["train", "valid", "test"])
    ap.add_argument("--score_thresh", type=float, default=0.05,
                    help="conf threshold (mirror torchvision export 0.05).")
    ap.add_argument("--detections_per_img", type=int, default=300,
                    help="max_det (mirror torchvision export 300).")
    ap.add_argument("--imgsz", type=int, default=800,
                    help="inference resolution; 800 matches Mode-1 r5b min_size.")
    ap.add_argument("--nms_iou", type=float, default=0.7,
                    help="ultralytics NMS IoU (default 0.7); recorded for the record.")
    ap.add_argument("--data_root", default="data/oiaddr")
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--device", default="0", help="ultralytics device (e.g. 0 or cpu).")
    ap.add_argument("--max_images", type=int, default=None)
    ap.add_argument("--use_class_probs", action="store_true",
                    help="accepted for CLI parity with export_compare_predictions; "
                         "ultralytics retains no post-NMS class vector, so probs are "
                         "synthetic regardless (documented per box).")
    ap.add_argument("--require_cuda", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.require_cuda and not torch.cuda.is_available():
        print("[yolo-export] ERROR: --require_cuda set but CUDA unavailable.")
        return 2
    out = guard_output_dir(args.output_dir, overwrite=args.overwrite,
                           sentinels=("export_summary.json",), tag="[yolo-export]")
    model, family = _load_yolo(args.checkpoint)
    print(f"[yolo-export] {family} loaded from {args.checkpoint}  "
          f"imgsz={args.imgsz} conf={args.score_thresh} max_det={args.detections_per_img}")
    if args.use_class_probs:
        print("[yolo-export] note: --use_class_probs given, but ultralytics keeps no "
              "post-NMS per-class vector -> probs_source=synthetic_score_residual "
              "(same honest proxy as fcos/ssd). Real-prob YOLO = future work.")

    stats = [export_split(model, family, sp, out, args.data_root, args.device,
                          args.score_thresh, args.detections_per_img, args.imgsz,
                          args.nms_iou, args.max_images) for sp in args.splits]

    summary = {"timestamp": datetime.now().isoformat(timespec="seconds"),
               "tool": "export_yolo_predictions", "checkpoint": args.checkpoint,
               "family": family, "model_type": Path(args.checkpoint).parent.name,
               "score_thresh": args.score_thresh,
               "detections_per_img": args.detections_per_img,
               "imgsz": args.imgsz, "nms_iou": args.nms_iou,
               "probs_source": "synthetic_score_residual (post-NMS class vector "
                               "unavailable in ultralytics)",
               "splits": stats}
    (out / "export_summary.json").write_text(json.dumps(summary, indent=2),
                                             encoding="utf-8")
    print(f"[yolo-export] DONE -> {out/'export_summary.json'}")
    print(f"[yolo-export] next: tools/audit_stage3c_governance_targets.py "
          f"--cache_dir {out}/<split>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
