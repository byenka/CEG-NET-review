#!/usr/bin/env python
"""train_detector_compare.py — train a STOCK comparison detector on OIA-DDR and
report mAP@0.5 (model-level comparison E1). Self-contained: own training loop +
own mAP eval. Does NOT touch Mode 1 (no Mode-1 checkpoint/script/audit tool used).

All comparison detectors are stock torchvision architectures (see
src/models/detector_compare_factory.py). We claim NO detector novelty; this
positions our (also stock) detector and supports the detector-agnostic
governance test (E2).

Outputs (under --output_dir):
  best.pt                 — best-mAP checkpoint (compatible config dict)
  compare_results.json    — per-class AP50 + mAP_50 on valid (and test if given)
  train_log.csv           — per-epoch train_loss + val_mAP_50

Usage:
  python tools/train_detector_compare.py \
    --model_type fcos --data_root data/oiaddr \
    --output_dir outputs/detcmp/m3_fcos_r50fpn \
    --epochs 25 --batch_size 2 --lr 1e-4 --require_cuda
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

_here = Path(__file__).resolve().parent
_root = _here.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch
from torch.utils.data import DataLoader

from src.data.coco_adapter import DRDetectionDataset, collate_fn  # type: ignore
from src.models.detector_compare_factory import (
    build_detector, detector_config, SUPPORTED,
    DEFAULT_FRCNN_ANCHOR_SIZES, DEFAULT_FRCNN_ANCHOR_RATIOS,
)
from tools._safe_output import guard_output_dir

CLASS_IDS = (1, 2, 3, 4)
CLASS_NAMES = {1: "MA", 2: "HE", 3: "Hard_EX", 4: "Soft_EX"}


# ---------- mAP@0.5 (manual, per-class) ----------

def _iou(a, b):
    import numpy as np
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=float)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def _ap_for_class(preds: List[Dict], targets: List[Dict], cls: int) -> float:
    import numpy as np
    scores, tps, n_gt = [], [], 0
    for p, t in zip(preds, targets):
        gmask = t["labels"] == cls
        gb = t["boxes"][gmask]
        n_gt += int(gb.shape[0])
        dmask = p["labels"] == cls
        db = p["boxes"][dmask]; ds = p["scores"][dmask]
        if db.shape[0] == 0:
            continue
        order = np.argsort(-ds)
        db, ds = db[order], ds[order]
        used = np.zeros(gb.shape[0], dtype=bool)
        iou = _iou(db, gb) if gb.shape[0] else np.zeros((db.shape[0], 0))
        for i in range(db.shape[0]):
            scores.append(float(ds[i]))
            hit = 0
            if iou.shape[1] > 0:
                j = int(np.argmax(iou[i]))
                if iou[i, j] >= 0.5 and not used[j]:
                    used[j] = True; hit = 1
            tps.append(hit)
    if n_gt == 0:
        return float("nan")
    if not scores:
        return 0.0
    order = np.argsort(-np.array(scores))
    tp = np.array(tps)[order]
    fp = 1 - tp
    tp_c = np.cumsum(tp); fp_c = np.cumsum(fp)
    recall = tp_c / n_gt
    precision = tp_c / np.maximum(tp_c + fp_c, 1)
    # 101-point interpolation
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        p = precision[recall >= r]
        ap += (p.max() if p.size else 0.0) / 101
    return float(ap)


@torch.no_grad()
def evaluate_map50(model, loader, device) -> Dict:
    import numpy as np
    model.eval()
    preds, targets = [], []
    for images, tgts in loader:
        images = [im.to(device) for im in images]
        out = model(images)
        for o in out:
            preds.append({"boxes": o["boxes"].cpu().numpy(),
                          "labels": o["labels"].cpu().numpy(),
                          "scores": o["scores"].cpu().numpy()})
        for t in tgts:
            targets.append({"boxes": t["boxes"].numpy() if hasattr(t["boxes"], "numpy") else np.asarray(t["boxes"]),
                            "labels": t["labels"].numpy() if hasattr(t["labels"], "numpy") else np.asarray(t["labels"])})
    aps = {CLASS_NAMES[c]: _ap_for_class(preds, targets, c) for c in CLASS_IDS}
    valid_aps = [v for v in aps.values() if v == v]  # drop nan
    mean = float(sum(valid_aps) / len(valid_aps)) if valid_aps else 0.0
    return {"mAP_50": mean, "per_class_AP50": aps}


def main() -> int:
    ap = argparse.ArgumentParser(description="Train a stock comparison detector (E1).")
    ap.add_argument("--model_type", required=True, choices=list(SUPPORTED))
    ap.add_argument("--data_root", default="data/oiaddr")
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--score_thresh", type=float, default=0.02)
    ap.add_argument("--detections_per_img", type=int, default=200)
    ap.add_argument("--min_size", type=int, default=800,
                    help="shorter-side resize (px). DEFAULT 800 = Mode 1 r5b "
                         "operating resolution (torchvision default, tidak di-override "
                         "di skrip asli kita — terverifikasi dari checkpoint). "
                         "Samakan ini ke YOLO --imgsz. SSD300 fixed 300 (diabaikan).")
    ap.add_argument("--max_size", type=int, default=1333,
                    help="longer-side cap (px). DEFAULT 1333 (torchvision/r5b).")
    ap.add_argument("--freeze_backbone_epochs", type=int, default=5)
    ap.add_argument("--eval_test", action="store_true",
                    help="also evaluate the best checkpoint on the test split.")
    ap.add_argument("--require_cuda", action="store_true")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow replacing a previous run at --output_dir "
                         "(default: refuse if best.pt/compare_results.json exist).")
    args = ap.parse_args()

    if args.require_cuda and not torch.cuda.is_available():
        print("[compare] ERROR: --require_cuda set but CUDA unavailable.")
        return 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Never clobber a previous detector run unless explicitly told to.
    guard_output_dir(args.output_dir, overwrite=args.overwrite, tag="[compare]")
    print(f"[compare] model_type={args.model_type}  device={device}")

    train_ds = DRDetectionDataset(split="train", data_root=str(args.data_root))
    valid_ds = DRDetectionDataset(split="valid", data_root=str(args.data_root))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"))
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"))

    model = build_detector(args.model_type, num_classes=5,
                           score_thresh=args.score_thresh,
                           detections_per_img=args.detections_per_img,
                           min_size=args.min_size, max_size=args.max_size,
                           pretrained_backbone=True).to(device)
    if args.model_type != "ssd300":
        act = getattr(model.transform, "min_size", "?")
        print(f"[compare] resolution: min_size={act} max_size={args.max_size}px "
              f"(Mode 1 r5b = 800/1333)")

    def set_backbone_frozen(frozen: bool):
        if hasattr(model, "backbone"):
            for p in model.backbone.parameters():
                p.requires_grad = not frozen

    set_backbone_frozen(args.freeze_backbone_epochs > 0)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-4)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    cfg = detector_config(args.model_type, 5, args.score_thresh,
                          args.detections_per_img,
                          DEFAULT_FRCNN_ANCHOR_SIZES, DEFAULT_FRCNN_ANCHOR_RATIOS,
                          min_size=args.min_size, max_size=args.max_size)

    best_map, best_state = -1.0, None
    log_rows = []
    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_backbone_epochs + 1 and args.freeze_backbone_epochs > 0:
            set_backbone_frozen(False)
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                    lr=args.lr * 0.1, weight_decay=1e-4)
            print(f"[compare] epoch {epoch}: backbone unfrozen, lr={args.lr*0.1:.2e}")
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for images, tgts in train_loader:
            images = [im.to(device) for im in images]
            tgts = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()}
                    for t in tgts]
            opt.zero_grad()
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    ld = model(images, tgts); loss = sum(ld.values())
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                ld = model(images, tgts); loss = sum(ld.values())
                loss.backward(); opt.step()
            tot += float(loss.item()); nb += 1
        ev = evaluate_map50(model, valid_loader, device)
        dt = time.time() - t0
        print(f"[compare] epoch {epoch:>2}/{args.epochs}  train_loss={tot/max(nb,1):.4f}  "
              f"val_mAP50={ev['mAP_50']:.4f}  {dt:.0f}s")
        log_rows.append({"epoch": epoch, "train_loss": tot/max(nb,1),
                         "val_mAP_50": ev["mAP_50"]})
        if ev["mAP_50"] > best_map:
            best_map = ev["mAP_50"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"model_state_dict": best_state, "config": cfg,
                        "epoch": epoch, "val_mAP_50": ev["mAP_50"],
                        "per_class_AP50": ev["per_class_AP50"]},
                       args.output_dir / "best.pt")

    # final results (reload best, eval valid for per-class, optionally test)
    results = {"model_type": args.model_type, "config": cfg,
               "best_val_mAP_50": best_map}
    if best_state is not None:
        model.load_state_dict(best_state)
    val_eval = evaluate_map50(model, valid_loader, device)
    results["valid"] = val_eval
    if args.eval_test:
        test_ds = DRDetectionDataset(split="test", data_root=str(args.data_root))
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=collate_fn, num_workers=args.num_workers,
                                 pin_memory=(device.type == "cuda"))
        results["test"] = evaluate_map50(model, test_loader, device)

    (args.output_dir / "compare_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    with open(args.output_dir / "train_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_mAP_50"])
        w.writeheader(); w.writerows(log_rows)

    print(f"\n[compare] === {args.model_type} DONE ===")
    print(f"[compare] best val mAP_50 = {best_map:.4f}")
    print(f"[compare] per-class AP50 (valid): "
          + "  ".join(f"{k}={v:.3f}" for k, v in val_eval['per_class_AP50'].items()))
    if args.eval_test:
        print(f"[compare] test mAP_50 = {results['test']['mAP_50']:.4f}")
    print(f"[compare] wrote {args.output_dir/'compare_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
