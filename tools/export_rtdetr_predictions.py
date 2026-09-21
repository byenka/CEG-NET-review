#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""export_rtdetr_predictions.py — RT-DETR (HuggingFace) -> DetectorBox cache.

Menjembatani checkpoint hasil ``agustus/scripts/finetune_rtdetr_oiaddr.py`` ke
cache DetectorBox yang sama dengan yang dipakai Mode-1 / E2, sehingga RT-DETR
bisa masuk panel detector-agnostic (Tabel 11) lewat jalur yang sudah ada:

    export_rtdetr_predictions.py  ->  audit_stage3c_governance_targets.py
                                  ->  eval_fp_reduction_benchmark.py

KEBIJAKAN SIGMOID vs SOFTMAX (terkunci, DECISION_sigmoid_vs_softmax_2026-07-29.md,
sama dengan ``extract_evidence_outputs()`` di skrip fine-tune):

  * skor deteksi  s_i = logits.sigmoid().max(-1)   <- NATIVE, dipakai box/threshold/M0.
                                                      JANGAN di-softmax; itu merusak
                                                      kalibrasi deteksi dan baseline M0.
  * distribusi    p_i = softmax(4 logit kelas)     <- simpleks Delta^4 untuk SAS, SDCA,
                                                      PSS, dan JSD multi-sumber. JSD
                                                      hanya sahih pada distribusi yang
                                                      berjumlah 1, jadi sigmoid tidak bisa.

Karena RT-DETR punya logit kelas per-query yang NYATA, cache ditandai
``probs_source="real_roi_softmax"`` (bukan proxy sintetis seperti YOLO/FCOS).
Inilah yang membuat RT-DETR berguna sebagai titik uji fidelitas evidence.

RT-DETR bersifat end-to-end tanpa NMS. Setiap query menghasilkan paling banyak satu
deteksi (kelas = argmax atas 4 logit), sehingga satu kotak memetakan tepat ke satu
distribusi kelas. Itu sesuai kontrak DetectorBox.

Contoh:
    python tools/export_rtdetr_predictions.py \\
      --checkpoint outputs/rtdetr_oiaddr_r18/best.pt \\
      --splits valid test --data_root data/oiaddr \\
      --score_thresh 0.05 --detections_per_img 300 \\
      --output_dir outputs/0506/rtdetr_e2/rtdetr_r18_cache --device cuda
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from PIL import Image

from src.data.coco_adapter import DRDetectionDataset
from src.data.zoning import estimate_landmarks_approx
# Formatter DetectorBox yang PERSIS dipakai exporter Mode-1 / E2.
from tools.export_detector_predictions import _predictions_to_detector_boxes
from tools._safe_output import guard_output_dir

# Urutan kanonik kelas; DetectorBox LABEL_MAP 1-indexed (1=MA .. 4=Soft_EX),
# sedangkan head RT-DETR 0-indexed -> label = query_class + 1.
CANON = ["MA", "HE", "Hard_EX", "Soft_EX"]


def _load_rtdetr(checkpoint: str, base_model: Optional[str], device: str):
    """Muat checkpoint fine-tune -> (model eval, processor, meta provenance)."""
    try:
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
    except ImportError as exc:
        print("[rtdetr-export] ERROR: transformers tidak terpasang. "
              'Jalankan: pip install "transformers>=4.45" timm')
        raise SystemExit(2) from exc

    ckpt_path = Path(checkpoint)
    if not ckpt_path.is_file():
        print(f"[rtdetr-export] ERROR: checkpoint tidak ada: {ckpt_path}")
        raise SystemExit(2)

    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    saved_args = ck.get("args", {}) or {}
    base = base_model or saved_args.get("model")
    if not base:
        print("[rtdetr-export] ERROR: --base_model tidak diberikan dan checkpoint "
              "tidak menyimpan args['model'].")
        raise SystemExit(2)

    id2label_ck = ck.get("id2label") or {}
    order = [id2label_ck[k] for k in sorted(id2label_ck, key=int)] if id2label_ck else []
    if order and order != CANON:
        print(f"[rtdetr-export] PERINGATAN: urutan kelas checkpoint {order} "
              f"BERBEDA dari kanonik {CANON}. Pemetaan label akan salah. Berhenti.")
        raise SystemExit(3)

    id2label = {i: c for i, c in enumerate(CANON)}
    model = AutoModelForObjectDetection.from_pretrained(
        base, num_labels=len(CANON), id2label=id2label,
        label2id={c: i for i, c in enumerate(CANON)},
        ignore_mismatched_sizes=True,
    )
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    if missing or unexpected:
        print(f"[rtdetr-export] state_dict: {len(missing)} missing, "
              f"{len(unexpected)} unexpected (0/0 yang diharapkan)")
        if missing:
            print(f"[rtdetr-export]   contoh missing: {list(missing)[:3]}")
    model.to(device).eval()

    proc = AutoImageProcessor.from_pretrained(base)
    meta = {
        "checkpoint": str(ckpt_path),
        "base_model": str(base),
        "checkpoint_epoch": ck.get("epoch"),
        "checkpoint_val_map50": ck.get("map50"),
        "train_img_size": saved_args.get("img_size"),
    }
    print(f"[rtdetr-export] model dimuat dari {base}")
    print(f"[rtdetr-export] checkpoint epoch={meta['checkpoint_epoch']} "
          f"val_mAP50={meta['checkpoint_val_map50']}")
    return model, proc, meta


def _queries_to_detections(logits: torch.Tensor, pred_boxes: torch.Tensor,
                           width: int, height: int, score_thresh: float,
                           max_det: int):
    """Satu query -> paling banyak satu deteksi. Tanpa NMS (RT-DETR end-to-end).

    logits      [Q, 4] logit kelas mentah
    pred_boxes  [Q, 4] cxcywh ternormalisasi [0,1]
    return      (boxes_xyxy [N,4] px, labels [N] 1-indexed, scores [N], probs [N,4])
    """
    # Kebijakan terkunci: skor = sigmoid NATIVE, distribusi = softmax Delta^4.
    sig = logits.sigmoid()
    scores, cls0 = sig.max(dim=-1)
    probs = torch.softmax(logits, dim=-1)

    keep = scores >= score_thresh
    scores, cls0, probs = scores[keep], cls0[keep], probs[keep]
    boxes = pred_boxes[keep]

    if scores.numel() > max_det:
        top = torch.topk(scores, max_det).indices
        scores, cls0, probs, boxes = scores[top], cls0[top], probs[top], boxes[top]

    # cxcywh ternormalisasi -> xyxy absolut. Processor melakukan resize persegi
    # tanpa padding (do_pad=false), jadi koordinat ternormalisasi pada frame hasil
    # resize identik dengan pada citra asli; kalikan saja dengan (W, H).
    cx, cy, w, h = boxes.unbind(-1)
    xyxy = torch.stack([(cx - w / 2) * width, (cy - h / 2) * height,
                        (cx + w / 2) * width, (cy + h / 2) * height], dim=-1)
    xyxy[:, 0::2] = xyxy[:, 0::2].clamp(0, width)
    xyxy[:, 1::2] = xyxy[:, 1::2].clamp(0, height)

    # buang kotak berluas nol setelah clamp (kontrak DetectorBox: x1<x2, y1<y2)
    valid = (xyxy[:, 2] > xyxy[:, 0]) & (xyxy[:, 3] > xyxy[:, 1])
    return xyxy[valid].cpu().float(), (cls0[valid] + 1).cpu().long(), \
        scores[valid].cpu().float(), probs[valid].cpu().float()


@torch.no_grad()
def export_split(model, proc, split, output_dir, data_root, device,
                 score_thresh, detections_per_img, imgsz,
                 max_images: Optional[int]) -> dict:
    split_dir = Path(output_dir) / split
    split_dir.mkdir(parents=True, exist_ok=True)
    ds = DRDetectionDataset(split=split, data_root=data_root, max_images=max_images)

    total, manifest = 0, []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*estimate_landmarks_approx.*")
        for idx in range(len(ds)):
            s = ds._samples[idx]
            stem, img_path = s["stem"], s["img_path"]
            img = Image.open(img_path).convert("RGB")
            W, H = img.width, img.height
            lm = estimate_landmarks_approx(image_w=W, image_h=H)

            enc = proc(images=[img], return_tensors="pt")
            pv = enc["pixel_values"].to(device)
            pm = enc.get("pixel_mask")
            out = model(pixel_values=pv,
                        pixel_mask=pm.to(device) if pm is not None else None)

            boxes, labels, scores, probs = _queries_to_detections(
                out.logits[0], out.pred_boxes[0], W, H,
                score_thresh, detections_per_img)

            # class_probs terisi -> formatter menandai probs_source=real_roi_softmax
            det_boxes = _predictions_to_detector_boxes(
                stem, boxes, labels, scores, lm, class_probs=probs)
            (split_dir / f"{stem}.json").write_text(
                json.dumps(det_boxes, separators=(",", ":")), encoding="utf-8")

            total += len(det_boxes)
            manifest.append({"stem": stem, "n_pred": len(det_boxes),
                             "image_w": W, "image_h": H})
            if (idx + 1) % 50 == 0 or (idx + 1) == len(ds):
                print(f"  [{idx+1}/{len(ds)}] last={stem} n_pred={len(det_boxes)}",
                      flush=True)

    mpath = Path(output_dir) / f"{split}_manifest.csv"
    with open(mpath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "n_pred", "image_w", "image_h"])
        w.writeheader()
        w.writerows(manifest)

    avg = total / max(len(ds), 1)
    print(f"[rtdetr-export] {split}: {len(ds)} gambar, {total} deteksi "
          f"({avg:.1f} rata-rata/gambar)  probs_source=real_roi_softmax (100%) "
          f"-> {split_dir}")
    return {"split": split, "n_images": len(ds), "total_preds": total,
            "n_real_prob_boxes": total, "n_synth_prob_boxes": 0,
            "real_prob_coverage": 1.0, "output_dir": str(split_dir),
            "manifest_csv": str(mpath)}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Export RT-DETR (HuggingFace) ke cache DetectorBox untuk panel E2.")
    ap.add_argument("--checkpoint", required=True,
                    help="best.pt dari finetune_rtdetr_oiaddr.py")
    ap.add_argument("--base_model", default=None,
                    help="id/direktori model dasar. Default: dibaca dari checkpoint "
                         "args['model']. Untuk mode offline arahkan ke folder lokal.")
    ap.add_argument("--splits", nargs="+", default=["valid", "test"],
                    choices=["train", "valid", "test"])
    ap.add_argument("--data_root", default="data/oiaddr")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--score_thresh", type=float, default=0.05)
    ap.add_argument("--detections_per_img", type=int, default=300)
    ap.add_argument("--imgsz", type=int, default=None,
                    help="Sisi persegi processor. Default: img_size saat training.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_images", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[rtdetr-export] CUDA tidak tersedia -> pakai CPU (lambat).")
        device = "cpu"

    out = guard_output_dir(Path(args.output_dir), overwrite=args.overwrite,
                           sentinels=("export_summary.json",), tag="[rtdetr-export]")

    model, proc, meta = _load_rtdetr(args.checkpoint, args.base_model, device)

    imgsz = args.imgsz or meta.get("train_img_size") or 640
    try:
        proc.size = {"height": int(imgsz), "width": int(imgsz)}
    except Exception:
        print(f"[rtdetr-export] PERINGATAN: gagal menyetel processor.size={imgsz}")
    print(f"[rtdetr-export] device={device} imgsz={imgsz} "
          f"score_thresh={args.score_thresh} cap={args.detections_per_img}")
    print("[rtdetr-export] kebijakan: skor=sigmoid native, class_probs=softmax "
          "atas 4 logit kelas (Delta^4) -> real_roi_softmax")

    splits_meta = []
    for sp in args.splits:
        print(f"\n[rtdetr-export] ── split {sp} ─────────────────────────")
        splits_meta.append(export_split(
            model, proc, sp, out, args.data_root, device,
            args.score_thresh, args.detections_per_img, imgsz, args.max_images))

    summary = {
        "tool": "export_rtdetr_predictions",
        "family": "rtdetr",
        "model_type": "rtdetr",
        "checkpoint": meta["checkpoint"],
        "base_model": meta["base_model"],
        "checkpoint_epoch": meta["checkpoint_epoch"],
        "checkpoint_val_map50": meta["checkpoint_val_map50"],
        "data_root": args.data_root,
        "imgsz": imgsz,
        "score_thresh": args.score_thresh,
        "detections_per_img": args.detections_per_img,
        "nms": "none (RT-DETR end-to-end)",
        "probs_source": "real_roi_softmax (softmax over 4 per-query class logits; "
                        "detection score kept as native sigmoid max)",
        "policy_ref": "agustus/DECISION_sigmoid_vs_softmax_2026-07-29.md",
        "zone_assignment": "estimate_landmarks_approx (placeholder, sama dengan exporter lain)",
        "splits": splits_meta,
    }
    (out / "export_summary.json").write_text(json.dumps(summary, indent=2),
                                             encoding="utf-8")
    print(f"\n[rtdetr-export] SELESAI -> {out / 'export_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
