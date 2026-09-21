#!/usr/bin/env python
"""
export_detector_predictions.py
Phase 19N — Stage 2 Bridge: detector checkpoint → DetectorBox JSON Cache

Converts detector predictions into the DetectorBox JSON format
expected by detector_backend.py's cached_predictions mode.

This is the handoff from Stage 1 (frozen detector) to Stage 2 (governance
pipeline). Once this cache exists, the governance pipeline can run without
re-running the detector.

── DetectorBox representation for a single 4-class detector ─────────────────

    bbox_id:          "<stem>_b<NNNN>"  (zero-padded 4 digits)
    head_predictions: []  (empty — detector is single-head, not multi-head)
    ensemble_probs:   predicted class gets `score`, other 3 get (1-score)/3
    assigned_zone:    ETDRS zone via approximate landmark estimate
    detector_raw:     non-contract metadata (label, score, bbox) for debugging
    r2a_raw:          legacy compatibility alias of detector_raw

SAS note (evidence.py compute_sas):
    head_predictions=[] triggers the max_ensemble fallback automatically.
    SAS = max(ensemble_probs) = detection confidence score.
    This is more informative than SAS=1.0 (which a synthetic single head
    would always produce by trivially agreeing with itself).
    Callers see used_fallback=True — this is expected and documented.

Zone note:
    AnatomyProvider is not yet implemented (Stage 2 scope).
    estimate_landmarks_approx() is used as placeholder:
      fovea = image centre, disc = fovea + 3.5 DD rightward (right-eye assumed).
    WARNING: eye laterality is ignored. Replace with real AnatomyProvider before
    governance ablations that depend on zone-specific APS signals.

Output layout:
    <output_dir>/
        <split>/
            <image_stem>.json   — list of DetectorBox dicts (one per prediction)
        <split>_manifest.csv    — per-image prediction count summary
    export_summary.json         — run metadata and aggregate stats

Usage:
    python tools/export_detector_predictions.py \\
    --checkpoint outputs/detector_stage1/stage1c_fasterrcnn_r5b_ext25/fasterrcnn_best.pt \
    --model_type fasterrcnn \\
      --splits valid test \\
      --score_thresh 0.05 \\
      --detections_per_img 300 \\
    --output_dir outputs/detector_cache/exported

    # Then in governance pipeline:
    from src.detector_backend import get_detector_outputs
    preds = get_detector_outputs(
        image_ids, mode="cached_predictions",
        cache_dir="outputs/detector_cache/exported/test",
    )
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if int(os.environ.get("OMP_NUM_THREADS", "0") or "0") <= 0:
    os.environ["OMP_NUM_THREADS"] = "1"
if "MKL_NUM_THREADS" not in os.environ:
    os.environ["MKL_NUM_THREADS"] = "1"

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, "..")
if _root not in sys.path:
    sys.path.insert(0, _root)

import torch
from torch.utils.data import DataLoader

try:
    from src.data.coco_adapter import (
        DRDetectionDataset,
        LABEL_NAMES,
        NUM_CLASSES,
        collate_fn,
    )
    from src.data.zoning import (
        AnatomicalLandmarks,
        assign_point_to_zone,
        estimate_landmarks_approx,
    )
except ModuleNotFoundError:
    from lgovnet.src.data.coco_adapter import (
        DRDetectionDataset,
        LABEL_NAMES,
        NUM_CLASSES,
        collate_fn,
    )
    from lgovnet.src.data.zoning import (
        AnatomicalLandmarks,
        assign_point_to_zone,
        estimate_landmarks_approx,
    )

_DATA_ROOT = os.path.join(_root, "data")

# 1-indexed torchvision label → canonical class name
LABEL_MAP: Dict[int, str] = {i + 1: n for i, n in enumerate(LABEL_NAMES)}
ALL_CLASSES: Tuple[str, ...] = tuple(LABEL_NAMES)  # ("MA", "HE", "Hard_EX", "Soft_EX")


# ──────────────────────────────────────────────────────────────────────────────
# Model loader (mirrors audit_detector_stage1a.py — kept in sync)
# ──────────────────────────────────────────────────────────────────────────────


def _load_fasterrcnn_model(
    ckpt: dict,
    device: torch.device,
    score_thresh: float = 0.05,
    detections_per_img: int = 300,
) -> torch.nn.Module:
    try:
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.rpn import AnchorGenerator

        try:
            from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

            _weights_v2 = True
        except ImportError:
            _weights_v2 = False
    except ImportError as exc:
        raise ImportError(f"torchvision is required: {exc}") from exc

    cfg = ckpt.get("config", {})
    num_classes = cfg.get("num_classes", NUM_CLASSES + 1)

    anchor_sizes = cfg.get("anchor_sizes", ((8,), (16,), (32,), (64,), (128,)))
    anchor_ratios = cfg.get("anchor_ratios", ((0.5, 1.0, 2.0),) * 5)
    anchor_gen = AnchorGenerator(
        sizes=tuple(tuple(s) for s in anchor_sizes),
        aspect_ratios=tuple(tuple(r) for r in anchor_ratios),
    )

    if _weights_v2:
        model = fasterrcnn_resnet50_fpn(weights=None, rpn_anchor_generator=anchor_gen)
    else:
        model = fasterrcnn_resnet50_fpn(
            pretrained=False, rpn_anchor_generator=anchor_gen
        )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(ckpt["model_state_dict"])

    model.roi_heads.score_thresh = score_thresh
    model.roi_heads.detections_per_img = detections_per_img

    print(
        f"[Export] FasterRCNN loaded: num_classes={num_classes}"
        f"  score_thresh={score_thresh}"
        f"  detections_per_img={detections_per_img}"
    )
    return model.to(device).eval()


def _load_retinanet_model(
    ckpt: dict,
    device: torch.device,
    score_thresh: float = 0.05,
    detections_per_img: int = 300,
) -> torch.nn.Module:
    try:
        from torchvision.models.detection import retinanet_resnet50_fpn
        from torchvision.models.detection.retinanet import RetinaNetClassificationHead

        try:
            from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights

            _weights_v2 = True
        except ImportError:
            _weights_v2 = False
    except ImportError as exc:
        raise ImportError(f"torchvision is required: {exc}") from exc

    cfg = ckpt.get("config", {})
    num_classes = cfg.get("num_classes", NUM_CLASSES + 1)

    if _weights_v2:
        model = retinanet_resnet50_fpn(weights=None)
    else:
        model = retinanet_resnet50_fpn(pretrained=False)

    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=num_classes,
    )
    model.load_state_dict(ckpt["model_state_dict"])

    model.score_thresh = score_thresh
    model.detections_per_img = detections_per_img

    print(
        f"[Export] RetinaNet loaded: num_classes={num_classes}"
        f"  score_thresh={score_thresh}"
        f"  detections_per_img={detections_per_img}"
    )
    return model.to(device).eval()


# ──────────────────────────────────────────────────────────────────────────────
# DetectorBox conversion
# ──────────────────────────────────────────────────────────────────────────────


def _make_ensemble_probs(label_name: str, score: float) -> Dict[str, float]:
    """Construct ensemble_probs for a single-head detector prediction.

    Predicted class gets `score`. Other 3 classes share (1-score) equally.
    Sum is always exactly 1.0 (floating-point exact within ~1e-16).

    LEGACY synthetic fallback. With `--use_class_probs` (Pilar 1, gap G1) the
    REAL ROI-head softmax is written to `detector_class_probs` and
    `ensemble_probs` is populated from it instead.
    """
    rest = (1.0 - score) / 3.0
    return {cls: (score if cls == label_name else rest) for cls in ALL_CLASSES}


def _make_ensemble_probs_from_real(class_probs_row: List[float]) -> Dict[str, float]:
    """Build ensemble_probs from the REAL multinomial vector returned by
    predict_with_class_probs. Column k → lesion class ALL_CLASSES[k].
    Renormalised to sum=1.0 to absorb float drift.
    """
    s = float(sum(class_probs_row))
    if s <= 0.0:
        return {cls: 0.25 for cls in ALL_CLASSES}
    return {cls: float(class_probs_row[k]) / s for k, cls in enumerate(ALL_CLASSES)}


def _assign_zone(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    landmarks: AnatomicalLandmarks,
) -> str:
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return assign_point_to_zone(cx, cy, landmarks)


def _predictions_to_detector_boxes(
    stem: str,
    boxes: torch.Tensor,  # [N, 4] float32 pixel coords
    labels: torch.Tensor,  # [N] int64 1-indexed
    scores: torch.Tensor,  # [N] float32
    landmarks: AnatomicalLandmarks,
    class_probs: Optional[torch.Tensor] = None,  # [N, num_lesion_classes]
) -> List[dict]:
    """Convert detector output to DetectorBox JSON list.

    When `class_probs` is provided (Pilar 1, `--use_class_probs`), the REAL
    multinomial distribution from the ROI head is written to
    `detector_class_probs`, and `ensemble_probs` is populated from it.

    `head_predictions` is kept EMPTY in both branches. Emitting a single-head
    entry makes compute_sas() compute head-agreement = 1/1 = 1.0 for every
    box (SAS becomes a dead constant — observed in logserver7). With an empty
    list, compute_sas() falls back to max(ensemble_probs), which on the REAL
    probabilities is a genuine, varying detector-confidence signal.
    """
    result = []
    has_real = class_probs is not None and class_probs.numel() > 0

    for i in range(boxes.shape[0]):
        x1, y1, x2, y2 = [float(v) for v in boxes[i]]
        label_id = int(labels[i])
        score = float(scores[i])

        label_name = LABEL_MAP.get(label_id)
        if label_name is None:
            continue  # skip background or unknown label

        zone = _assign_zone(x1, y1, x2, y2, landmarks)
        raw_meta = {
            "label": label_id,
            "label_name": label_name,
            "score": round(score, 6),
            "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
        }

        if has_real:
            row = class_probs[i].tolist()
            ensemble_probs = _make_ensemble_probs_from_real(row)
            box_record = {
                "bbox_id": f"{stem}_b{i:04d}",
                "head_predictions": [],  # empty → SAS = max(real ensemble_probs)
                "ensemble_probs": ensemble_probs,
                "detector_class_probs": dict(ensemble_probs),
                "probs_source": "real_roi_softmax",
                "assigned_zone": zone,
                "detector_raw": raw_meta,
                "r2a_raw": raw_meta,
            }
        else:
            ensemble_probs = _make_ensemble_probs(label_name, score)
            box_record = {
                "bbox_id": f"{stem}_b{i:04d}",
                "head_predictions": [],  # SAS uses max_ensemble fallback
                "ensemble_probs": ensemble_probs,
                "probs_source": "synthetic_score_residual",
                "assigned_zone": zone,
                "detector_raw": raw_meta,
                "r2a_raw": raw_meta,  # legacy alias
            }
        result.append(box_record)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Per-split export
# ──────────────────────────────────────────────────────────────────────────────


def export_split(
    model: torch.nn.Module,
    split: str,
    output_dir: str,
    device: torch.device,
    max_images: Optional[int] = None,
    data_root: Optional[str] = None,
    use_class_probs: bool = False,
    model_type: str = "fasterrcnn",
    tta_hflip: bool = False,
) -> dict:
    """Run model inference on `split` and write per-image DetectorBox JSON files.

    When `use_class_probs=True` (Pilar 1, gap G1), inference uses
    `predict_with_class_probs` which retains the full multinomial probability
    vector per kept box. This activates real SDCA / e_pred signal downstream.

    When `tta_hflip=True`, the class-probability path uses horizontal-flip TTA
    and inverse-flips boxes back to original coordinates.

    Returns aggregate stats dict for summary reporting.
    """
    split_dir = Path(output_dir) / split
    split_dir.mkdir(parents=True, exist_ok=True)

    multiclass_fn = None
    if use_class_probs:
        try:
            from lgovnet.src.governance.multiclass_inference import (
                predict_with_class_probs,
                predict_with_class_probs_tta_hflip,
                predict_retinanet_with_class_probs,
            )
        except ImportError:
            from src.governance.multiclass_inference import (
                predict_with_class_probs,
                predict_with_class_probs_tta_hflip,
                predict_retinanet_with_class_probs,
            )
        if tta_hflip:
            multiclass_fn = predict_with_class_probs_tta_hflip
        else:
            multiclass_fn = (
                predict_with_class_probs
                if model_type == "fasterrcnn"
                else predict_retinanet_with_class_probs
            )
        print(
            f"[Export] Pilar 1 ACTIVE — real class probabilities "
            f"(model_type={model_type}; field `detector_class_probs`)"
        )
        if tta_hflip:
            print(
                "[Export] TTA ACTIVE — horizontal-flip view with inverse-flipped "
                "boxes in original image frame."
            )

    ds = DRDetectionDataset(
        split=split, data_root=data_root or _DATA_ROOT, max_images=max_images
    )
    n_images = len(ds)

    manifest_rows = []
    total_preds = 0
    total_real_prob_boxes = 0
    total_synth_prob_boxes = 0

    print(f"\n[Export] ── {split} split: {n_images} images ──────────────────")

    # Suppress repeated estimate_landmarks_approx warnings
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*estimate_landmarks_approx.*",
            category=UserWarning,
        )
        print(
            "[Export] Zone assignment: estimate_landmarks_approx() "
            "(fovea=centre, disc=centre+3.5DD right). "
            "Eye laterality ignored — placeholder until AnatomyProvider."
        )

        with torch.no_grad():
            for idx in range(n_images):
                image, _ = ds[idx]
                stem = ds._samples[idx]["stem"]
                _, H, W = image.shape

                # Approximate landmarks from image dimensions
                lm = estimate_landmarks_approx(image_w=W, image_h=H)

                if multiclass_fn is not None:
                    out = multiclass_fn(model, [image], device=device)[0]
                    boxes = out["boxes"]
                    labels = out["labels"]
                    scores = out["scores"]
                    class_probs = out["class_probs"]
                else:
                    output = model([image.to(device)])[0]
                    boxes = output.get("boxes", torch.zeros(0, 4)).cpu()
                    labels = output.get(
                        "labels", torch.zeros(0, dtype=torch.long)
                    ).cpu()
                    scores = output.get("scores", torch.zeros(0)).cpu()
                    class_probs = None

                det_boxes = _predictions_to_detector_boxes(
                    stem, boxes, labels, scores, lm, class_probs=class_probs
                )

                json_path = split_dir / f"{stem}.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(det_boxes, f, separators=(",", ":"))

                n_pred = len(det_boxes)
                total_preds += n_pred

                # Pilar 1 activation diagnostics:
                # count how many exported boxes carry real detector multinomial probs
                # versus synthetic score-residual fallback.
                for b in det_boxes:
                    src = b.get("probs_source", "synthetic_score_residual")
                    if src == "real_roi_softmax":
                        total_real_prob_boxes += 1
                    else:
                        total_synth_prob_boxes += 1

                manifest_rows.append(
                    {"stem": stem, "n_pred": n_pred, "image_w": W, "image_h": H}
                )

                if (idx + 1) % 50 == 0 or (idx + 1) == n_images:
                    print(f"  [{idx + 1}/{n_images}] last={stem}  n_pred={n_pred}")

    # Write manifest CSV
    manifest_path = Path(output_dir) / f"{split}_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["stem", "n_pred", "image_w", "image_h"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    avg_pred = total_preds / max(n_images, 1)
    real_cov = total_real_prob_boxes / max(total_preds, 1)
    print(
        f"[Export] {split}: {n_images} images, {total_preds} total preds"
        f"  ({avg_pred:.1f} avg/img)"
        f"  → {split_dir}/"
    )
    print(
        f"[Export] {split}: prob_source real={total_real_prob_boxes} "
        f"synthetic={total_synth_prob_boxes} "
        f"(real_coverage={100.0 * real_cov:.1f}%)"
    )

    if use_class_probs and total_real_prob_boxes == 0:
        print(
            "[Export] WARN: --use_class_probs is set but 0 boxes were exported with "
            "probs_source='real_roi_softmax'. Check model_type/checkpoint compatibility."
        )

    return {
        "split": split,
        "n_images": n_images,
        "total_preds": total_preds,
        "avg_preds_per_image": round(avg_pred, 1),
        "n_real_prob_boxes": total_real_prob_boxes,
        "n_synth_prob_boxes": total_synth_prob_boxes,
        "real_prob_coverage": round(real_cov, 4),
        "output_dir": str(split_dir),
        "manifest_csv": str(manifest_path),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Export detector predictions to DetectorBox JSON cache."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to detector checkpoint.",
    )
    parser.add_argument(
        "--model_type",
        default="fasterrcnn",
        choices=["fasterrcnn", "retinanet"],
        help="Detector family used by the checkpoint (default: fasterrcnn).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["valid", "test"],
        choices=["train", "valid", "test"],
        help="Which data splits to export (default: valid test).",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.05,
        help="Minimum detection score (default: 0.05).",
    )
    parser.add_argument(
        "--detections_per_img",
        type=int,
        default=300,
        help="Max detections per image (default: 300 for governance candidate budget).",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/detector_cache/exported",
        help="Root cache directory. Per-split subdirs are created automatically.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Cap on images per split (default: None = all images).",
    )
    parser.add_argument(
        "--data_root",
        default=_DATA_ROOT,
        help="Dataset root containing images/ and xml/ subdirs (default: data/).",
    )
    parser.add_argument(
        "--use_class_probs",
        action="store_true",
        help="(Pilar 1, gap G1) Use predict_with_class_probs to export the REAL "
        "multinomial probability vector per kept box. Adds a "
        "`detector_class_probs` field and populates `ensemble_probs` from "
        "real probs instead of the synthetic (1-score)/(C-1) fallback. "
        "Required for real SDCA / e_pred signal. Faster R-CNN only; "
        "RetinaNet falls back to synthetic.",
    )
    parser.add_argument(
        "--tta_hflip",
        action="store_true",
        help=(
            "Enable horizontal-flip TTA predictor for the real class-probability "
            "path. Boxes are inverse-flipped back to original coordinates. "
            "Requires --use_class_probs and model_type=fasterrcnn."
        ),
    )
    parser.add_argument(
        "--strict_real_probs",
        action="store_true",
        help=(
            "Fail the export if real class-probability coverage is below "
            "--min_real_prob_coverage on any requested split. "
            "For Faster R-CNN this requires --use_class_probs."
        ),
    )
    parser.add_argument(
        "--min_real_prob_coverage",
        type=float,
        default=0.80,
        help=(
            "Minimum required real-probability coverage in strict mode "
            "(default: 0.80)."
        ),
    )
    args = parser.parse_args()

    if args.tta_hflip:
        if args.model_type != "fasterrcnn":
            raise ValueError(
                "--tta_hflip currently supports only model_type=fasterrcnn"
            )
        if not args.use_class_probs:
            raise ValueError("--tta_hflip requires --use_class_probs")

    if args.strict_real_probs:
        if args.model_type != "fasterrcnn":
            raise ValueError(
                "--strict_real_probs currently supports only model_type=fasterrcnn"
            )
        if not args.use_class_probs:
            raise ValueError("--strict_real_probs requires --use_class_probs")
        if not (0.0 <= float(args.min_real_prob_coverage) <= 1.0):
            raise ValueError("--min_real_prob_coverage must be in [0,1]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Export] === Detector → DetectorBox Cache Export ===")
    print(
        f"[Export] device={device}  model_type={args.model_type}  splits={args.splits}"
    )
    print(
        f"[Export] score_thresh={args.score_thresh}  detections_per_img={args.detections_per_img}"
    )
    print(f"[Export] checkpoint={args.checkpoint}")
    print(f"[Export] output_dir={args.output_dir}")
    print(
        f"[Export] use_class_probs={args.use_class_probs}  tta_hflip={args.tta_hflip}"
    )

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    print(
        f"[Export] Checkpoint epoch={ckpt.get('epoch', '?')}  val_loss={ckpt.get('val_loss', '?')}"
    )

    if args.model_type == "retinanet":
        model = _load_retinanet_model(
            ckpt,
            device,
            score_thresh=args.score_thresh,
            detections_per_img=args.detections_per_img,
        )
    else:
        model = _load_fasterrcnn_model(
            ckpt,
            device,
            score_thresh=args.score_thresh,
            detections_per_img=args.detections_per_img,
        )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    all_stats = []
    for split in args.splits:
        stats = export_split(
            model=model,
            split=split,
            output_dir=args.output_dir,
            device=device,
            max_images=args.max_images,
            data_root=args.data_root,
            use_class_probs=args.use_class_probs,
            model_type=args.model_type,
            tta_hflip=args.tta_hflip,
        )
        all_stats.append(stats)

    strict_fail_summaries = []
    if args.strict_real_probs:
        for s in all_stats:
            cov = float(s.get("real_prob_coverage", 0.0))
            if cov < float(args.min_real_prob_coverage):
                strict_fail_summaries.append((s["split"], cov))

    # Write summary JSON
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": args.checkpoint,
        "model_type": args.model_type,
        "checkpoint_epoch": ckpt.get("epoch", "?"),
        "score_thresh": args.score_thresh,
        "detections_per_img": args.detections_per_img,
        "use_class_probs": args.use_class_probs,
        "tta_hflip": args.tta_hflip,
        "zone_assignment": "estimate_landmarks_approx (placeholder — eye laterality ignored)",
        "head_predictions_mode": "empty list → compute_sas() max_ensemble fallback",
        "splits": all_stats,
    }
    summary_path = Path(args.output_dir) / "export_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Export] ══════════════════ DONE ══════════════════")
    for s in all_stats:
        print(
            f"  {s['split']:8s}  {s['n_images']:4d} images  "
            f"{s['total_preds']:7d} preds  {s['avg_preds_per_image']:6.1f}/img  "
            f"real_prob={s.get('n_real_prob_boxes', 0)} "
            f"({100.0 * float(s.get('real_prob_coverage', 0.0)):.1f}%)"
        )
    print(f"  Summary: {summary_path}")
    print(
        "\n[Export] Next step: set detector_mode=cached_predictions, "
        f"cache_dir=<output_dir>/<split> in experiment config."
    )

    if strict_fail_summaries:
        print("\n[Export] STRICT REAL-PROB GATE: FAIL")
        for split, cov in strict_fail_summaries:
            print(
                f"  split={split}: real_coverage={100.0 * cov:.1f}% "
                f"(< {100.0 * float(args.min_real_prob_coverage):.1f}%)"
            )
        print(
            "[Export] Action: confirm checkpoint/model_type compatibility and "
            "re-run with --use_class_probs."
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
