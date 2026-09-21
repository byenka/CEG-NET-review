#!/usr/bin/env python
"""export_compare_predictions.py — E2 step 1: export a COMPARISON detector's
predictions to the DetectorBox JSON cache that the governance pipeline
(Stage3C builder + benchmark) consumes — for the detector-agnostic test.

Why a NEW exporter (not the Mode-1 tools/export_detector_predictions.py):
  * the Mode-1 exporter only knows fasterrcnn/retinanet and is part of the
    LOCKED Mode-1 bridge — we must NOT modify it.
  * comparison checkpoints come from tools/train_detector_compare.py and may be
    fcos / ssd300 / *_v2; this exporter rebuilds ANY of the 6 stock detectors via
    src/models/detector_compare_factory.py using the checkpoint's saved config.

Format fidelity: the DetectorBox records are produced by the SAME helper the
Mode-1 exporter uses (``_predictions_to_detector_boxes`` imported from
tools/export_detector_predictions.py) — identical JSON schema, identical zone
assignment. So the downstream Stage3C builder treats every detector uniformly.

Real class probabilities (richer SDCA / uncertainty signal):
  fasterrcnn / fasterrcnn_v2 -> predict_with_class_probs (GeneralizedRCNN path)
  retinanet / retinanet_v2   -> predict_retinanet_with_class_probs
  fcos / ssd300              -> no real-prob fn -> synthetic (score-residual)
Each box records its ``probs_source`` so the comparison is honest per detector.

Does NOT touch Mode 1. NEVER overwrites a previous cache (guarded).

Usage:
  python tools/export_compare_predictions.py \
    --checkpoint outputs/detcmp/retinanet/best.pt \
    --splits valid test --use_class_probs \
    --score_thresh 0.05 --detections_per_img 300 \
    --output_dir outputs/detcmp/retinanet_cache
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
from src.models.detector_compare_factory import build_detector, SUPPORTED
# Reuse the EXACT DetectorBox formatter the Mode-1 exporter uses (read-only import).
from tools.export_detector_predictions import _predictions_to_detector_boxes
from tools._safe_output import guard_output_dir

# model_type -> which real-class-prob function applies (None => synthetic fallback)
_REALPROB_FRCNN = {"fasterrcnn", "fasterrcnn_v2"}
_REALPROB_RETINA = {"retinanet", "retinanet_v2"}


def _load_compare_model(checkpoint: str, device: torch.device,
                        score_thresh: float, detections_per_img: int):
    """Rebuild a comparison detector from a train_detector_compare.py checkpoint."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    mt = cfg.get("model_type")
    if mt not in SUPPORTED:
        raise ValueError(f"checkpoint model_type='{mt}' not in {SUPPORTED}. "
                         f"Is this a tools/train_detector_compare.py checkpoint?")
    num_classes = cfg.get("num_classes", 5)
    anchor_sizes = cfg.get("anchor_sizes")
    anchor_ratios = cfg.get("anchor_ratios")
    # "model_default" sentinel -> let the factory pick the family default
    if anchor_sizes == "model_default":
        anchor_sizes = None
    if anchor_ratios == "model_default":
        anchor_ratios = None
    model = build_detector(mt, num_classes=num_classes, score_thresh=score_thresh,
                           detections_per_img=detections_per_img,
                           anchor_sizes=anchor_sizes, anchor_ratios=anchor_ratios,
                           pretrained_backbone=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[e2-export] {mt} loaded  num_classes={num_classes}  "
          f"score_thresh={score_thresh}  det/img={detections_per_img}  "
          f"epoch={ckpt.get('epoch','?')}  val_mAP50={ckpt.get('val_mAP_50','?')}")
    return model.to(device).eval(), mt


def _realprob_fn(model_type: str, use_class_probs: bool):
    """Return the real-class-prob inference fn for this detector, or None."""
    if not use_class_probs:
        return None
    try:
        from src.governance.multiclass_inference import (
            predict_with_class_probs, predict_retinanet_with_class_probs,
        )
    except Exception as exc:  # pragma: no cover
        print(f"[e2-export] WARN: multiclass_inference unavailable ({exc}); "
              f"using synthetic probs.")
        return None
    if model_type in _REALPROB_FRCNN:
        return predict_with_class_probs
    if model_type in _REALPROB_RETINA:
        return predict_retinanet_with_class_probs
    print(f"[e2-export] note: model_type={model_type} has no real-prob path; "
          f"synthetic score-residual probs will be used.")
    return None


@torch.no_grad()
def export_split(model, model_type, split, output_dir, device, use_class_probs,
                 data_root, max_images: Optional[int]) -> dict:
    split_dir = Path(output_dir) / split
    split_dir.mkdir(parents=True, exist_ok=True)
    ds = DRDetectionDataset(split=split, data_root=data_root, max_images=max_images)
    realfn = _realprob_fn(model_type, use_class_probs)

    n_real, n_synth, total = 0, 0, 0
    manifest = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*estimate_landmarks_approx.*")
        for idx in range(len(ds)):
            image, _ = ds[idx]
            stem = ds._samples[idx]["stem"]
            _, H, W = image.shape
            lm = estimate_landmarks_approx(image_w=W, image_h=H)
            class_probs = None
            if realfn is not None:
                try:
                    out = realfn(model, [image], device=device)[0]
                    boxes, labels = out["boxes"], out["labels"]
                    scores, class_probs = out["scores"], out["class_probs"]
                except Exception as exc:
                    if idx == 0:
                        print(f"[e2-export] WARN: real-prob path failed "
                              f"({type(exc).__name__}: {exc}); falling back to "
                              f"standard forward + synthetic probs.")
                    o = model([image.to(device)])[0]
                    boxes = o.get("boxes", torch.zeros(0, 4)).cpu()
                    labels = o.get("labels", torch.zeros(0, dtype=torch.long)).cpu()
                    scores = o.get("scores", torch.zeros(0)).cpu()
                    class_probs = None
            else:
                o = model([image.to(device)])[0]
                boxes = o.get("boxes", torch.zeros(0, 4)).cpu()
                labels = o.get("labels", torch.zeros(0, dtype=torch.long)).cpu()
                scores = o.get("scores", torch.zeros(0)).cpu()

            det_boxes = _predictions_to_detector_boxes(
                stem, boxes, labels, scores, lm, class_probs=class_probs)
            (split_dir / f"{stem}.json").write_text(
                json.dumps(det_boxes, separators=(",", ":")), encoding="utf-8")
            total += len(det_boxes)
            for b in det_boxes:
                if b.get("probs_source") == "real_roi_softmax":
                    n_real += 1
                else:
                    n_synth += 1
            manifest.append({"stem": stem, "n_pred": len(det_boxes),
                             "image_w": W, "image_h": H})
            if (idx + 1) % 50 == 0 or (idx + 1) == len(ds):
                print(f"  [{idx+1}/{len(ds)}] last={stem} n_pred={len(det_boxes)}")

    mpath = Path(output_dir) / f"{split}_manifest.csv"
    with open(mpath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "n_pred", "image_w", "image_h"])
        w.writeheader(); w.writerows(manifest)
    cov = n_real / max(total, 1)
    print(f"[e2-export] {split}: {len(ds)} imgs, {total} preds  "
          f"real_prob={n_real} synth={n_synth} (real_cov={100*cov:.1f}%) -> {split_dir}")
    return {"split": split, "n_images": len(ds), "total_preds": total,
            "n_real_prob_boxes": n_real, "n_synth_prob_boxes": n_synth,
            "real_prob_coverage": round(cov, 4), "output_dir": str(split_dir),
            "manifest_csv": str(mpath)}


def main() -> int:
    ap = argparse.ArgumentParser(description="E2: export comparison detector -> DetectorBox cache.")
    ap.add_argument("--checkpoint", required=True,
                    help="best.pt from tools/train_detector_compare.py")
    ap.add_argument("--splits", nargs="+", default=["valid", "test"],
                    choices=["train", "valid", "test"])
    ap.add_argument("--score_thresh", type=float, default=0.05)
    ap.add_argument("--detections_per_img", type=int, default=300)
    ap.add_argument("--use_class_probs", action="store_true",
                    help="use real class-prob path where available (frcnn/retina); "
                         "synthetic fallback for fcos/ssd or on failure.")
    ap.add_argument("--data_root", default="data/oiaddr")
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--max_images", type=int, default=None)
    ap.add_argument("--require_cuda", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.require_cuda and not torch.cuda.is_available():
        print("[e2-export] ERROR: --require_cuda set but CUDA unavailable.")
        return 2
    out = guard_output_dir(args.output_dir, overwrite=args.overwrite,
                           sentinels=("export_summary.json",), tag="[e2-export]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[e2-export] device={device}  checkpoint={args.checkpoint}")

    model, mt = _load_compare_model(args.checkpoint, device,
                                    args.score_thresh, args.detections_per_img)
    stats = [export_split(model, mt, sp, out, device, args.use_class_probs,
                          args.data_root, args.max_images) for sp in args.splits]

    summary = {"timestamp": datetime.now().isoformat(timespec="seconds"),
               "tool": "export_compare_predictions", "checkpoint": args.checkpoint,
               "model_type": mt, "score_thresh": args.score_thresh,
               "detections_per_img": args.detections_per_img,
               "use_class_probs": args.use_class_probs, "splits": stats}
    (out / "export_summary.json").write_text(json.dumps(summary, indent=2),
                                             encoding="utf-8")
    print(f"[e2-export] DONE -> {out/'export_summary.json'}")
    print(f"[e2-export] next: build governance_features via "
          f"tools/audit_stage3c_governance_targets.py --cache_dir {out}/<split>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
