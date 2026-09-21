#!/usr/bin/env python
"""train_yolo_compare.py — OPTIONAL modern comparators (YOLOv8 / RT-DETR) for the
model-level comparison (E1), via ultralytics. Writes a ``compare_results.json``
in the SAME schema as tools/train_detector_compare.py so the aggregator
(tools/aggregate_detector_compare.py) can include these rows in Table 10.

Honesty contract: YOLOv8/RT-DETR are stock ultralytics architectures — no custom
module, no detector novelty claimed. They run on the SAME OIA-DDR split (converted
by tools/convert_voc_to_yolo.py, identical samples/classes). Fairness caveat: the
YOLO training recipe (augmentation, input size) differs from the torchvision
detectors, so YOLO rows are reported as "modern reference" with that note, not as
a perfectly matched controlled cell.

Does NOT touch Mode 1. NEVER overwrites a previous result dir (guarded).

Setup (server):
    pip install ultralytics
    python tools/convert_voc_to_yolo.py --data_root data/oiaddr --out data/oiaddr_yolo --link

Usage:
    python tools/train_yolo_compare.py --model yolov8s.pt \
        --data data/oiaddr_yolo/data.yaml \
        --output_dir outputs/detcmp_yolo/yolov8s \
        --epochs 100 --imgsz 1280 --batch 8 --device 0 --eval_test
    python tools/train_yolo_compare.py --model rtdetr-l.pt ... (RT-DETR)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
_root = _here.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from tools._safe_output import guard_output_dir  # noqa: E402

CLASS_NAMES = {0: "MA", 1: "HE", 2: "Hard_EX", 3: "Soft_EX"}


def _load_model(model_name: str):
    """Return an ultralytics model instance (YOLO or RTDETR). Clear error if the
    package is missing."""
    try:
        from ultralytics import YOLO, RTDETR
    except Exception as exc:  # pragma: no cover - depends on server env
        print("[yolo] ERROR: ultralytics not installed. Run: pip install ultralytics")
        print(f"[yolo] import error: {exc}")
        raise SystemExit(2)
    if "rtdetr" in model_name.lower():
        return RTDETR(model_name), "rtdetr"
    return YOLO(model_name), "yolo"


def _extract_map50(metrics) -> dict:
    """Pull mAP@0.5 + per-class AP50 from an ultralytics metrics object,
    defensively (attribute names vary slightly across versions)."""
    out = {"mAP_50": None, "per_class_AP50": {}}
    box = getattr(metrics, "box", None)
    if box is None:
        return out
    out["mAP_50"] = float(getattr(box, "map50", float("nan")))
    # per-class AP50: box.ap50 indexed by box.ap_class_index
    ap50 = getattr(box, "ap50", None)
    idx = getattr(box, "ap_class_index", None)
    try:
        if ap50 is not None and idx is not None:
            for cls_i, ap in zip(list(idx), list(ap50)):
                name = CLASS_NAMES.get(int(cls_i), str(int(cls_i)))
                out["per_class_AP50"][name] = float(ap)
    except Exception:
        pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Train a stock YOLO/RT-DETR comparator (E1).")
    ap.add_argument("--model", required=True,
                    help="ultralytics weights, e.g. yolov8s.pt / yolov8m.pt / rtdetr-l.pt")
    ap.add_argument("--data", required=True, help="path to data.yaml from converter")
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=100,
                    help="MAX epoch budget (a CAP, not a fixed count). With "
                         "--patience early-stopping the run stops when val mAP "
                         "plateaus, so this is NOT '4x more training' than the "
                         "torchvision detectors — it converges and stops. YOLO's "
                         "heavy built-in augmentation (mosaic/mixup) needs a "
                         "larger cap than the lightly-augmented torchvision E1 "
                         "models (which peak ~epoch 18-25). Fairness = each "
                         "detector reported at its OWN best-val checkpoint.")
    ap.add_argument("--patience", type=int, default=30,
                    help="early-stopping patience (epochs w/o val improvement); "
                         "makes the epoch budget adaptive/fair across recipes. "
                         "0 disables (train the full --epochs cap).")
    ap.add_argument("--imgsz", type=int, default=800,
                    help="input resolution (px, longest-side letterbox). "
                         "DEFAULT 800 = cocok dgn Mode 1 r5b (min_size=800, "
                         "torchvision default, terverifikasi dari checkpoint). "
                         "JANGAN naikkan hanya untuk YOLO tanpa menaikkan "
                         "--min_size di torchvision juga — unfair utk lesi kecil.")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--eval_test", action="store_true",
                    help="also evaluate the best weights on the test split.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = guard_output_dir(args.output_dir, overwrite=args.overwrite, tag="[yolo]")
    model, family = _load_model(args.model)
    run_name = Path(args.model).stem
    # patience=0 -> disable early stop (use full cap); else stop on val plateau.
    patience = args.patience if args.patience > 0 else args.epochs
    print(f"[yolo] model={args.model} ({family})  data={args.data}  out={out}")
    print(f"[yolo] epoch_cap={args.epochs}  patience={patience}  "
          f"(early-stop -> adaptive budget; best.pt selected on val)")

    # ultralytics writes runs under project/name; keep it inside our output_dir.
    train_metrics = model.train(
        data=str(args.data), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, device=args.device, patience=patience,
        project=str(out), name=run_name, exist_ok=args.overwrite,
        verbose=True,
    )

    config = {
        "model_type": run_name,
        "family": family,
        "model_variant": "stock_comparison",
        "architecture_label": f"{run_name} (stock ultralytics, comparison)",
        "imgsz": args.imgsz, "epoch_cap": args.epochs, "patience": patience,
        "batch": args.batch,
        "no_custom_module": True,
        "note": "stock ultralytics detector; recipe differs from torchvision E1. "
                "Epoch is a CAP with early-stopping (patience) -> adaptive budget; "
                "best.pt selected on val. Modern reference row, not a perfectly "
                "matched controlled cell.",
    }

    # valid mAP from the training run's best epoch
    valid_eval = _extract_map50(train_metrics)
    results = {"model_type": run_name, "config": config,
               "best_val_mAP_50": valid_eval.get("mAP_50"),
               "valid": valid_eval}

    if args.eval_test:
        test_metrics = model.val(data=str(args.data), split="test",
                                 imgsz=args.imgsz, device=args.device,
                                 project=str(out), name=f"{run_name}_test",
                                 exist_ok=args.overwrite)
        results["test"] = _extract_map50(test_metrics)

    (out / "compare_results.json").write_text(json.dumps(results, indent=2),
                                              encoding="utf-8")
    print(f"\n[yolo] === {run_name} DONE ===")
    print(f"[yolo] valid mAP_50 = {results['best_val_mAP_50']}")
    if args.eval_test:
        print(f"[yolo] test  mAP_50 = {results['test'].get('mAP_50')}")
    print(f"[yolo] wrote {out/'compare_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
