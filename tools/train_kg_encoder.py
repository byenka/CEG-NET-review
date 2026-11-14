#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, "..")
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import torch

from lgovnet.src.kg.builder import build_graph_from_artifacts
from lgovnet.src.kg.encoder import (
    build_encoder_inputs,
    build_gat_encoder,
    compute_compat,
    compute_sdm,
    normalize_sdca,
)
from lgovnet.src.governance.gate import LGG
from lgovnet.src.losses.gate_loss import GateLoss
from lgovnet.src.losses.sdgl import SDGLLoss
from lgovnet.src.train_step import run_lgovnet_step
from lgovnet.src.detector_backend import load_cached_predictions
from lgovnet.src.data.xml_parser import parse_xml_file


def _raw_to_kg():

    from lgovnet.src.data.ontology import map_raw_to_kg

    return map_raw_to_kg


def ensure_head_predictions(det_all: dict) -> dict:

    n_box = n_fill = n_skip = 0
    for boxes in det_all.values():
        for b in boxes:
            n_box += 1
            if b.get("head_predictions"):
                continue
            raw = b.get("detector_raw") or b.get("r2a_raw") or {}
            bb = raw.get("bbox")
            if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                n_skip += 1
                continue
            probs = dict(b.get("detector_class_probs") or b.get("ensemble_probs") or {})
            head = raw.get("label_name") or (max(probs, key=probs.get) if probs else "")
            b["head_predictions"] = [{
                "head": head,
                "bbox": [float(v) for v in bb],
                "class_probs": probs,
                "score": float(raw.get("score", 0.0)),
            }]
            n_fill += 1
    print(f"[kgtrain] head_predictions disintesis dari detector_raw untuk {n_fill}/{n_box} kotak"
          + (f", {n_skip} kotak tanpa bbox dilewati" if n_skip else ""), flush=True)
    if n_fill == 0 and n_box:
        print("[kgtrain] PERINGATAN: tidak ada yang disintesis; periksa isi cache.", flush=True)
    return det_all


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_gt_boxes(xml_dir: Path, image_ids, raw_map) -> dict:

    out, n_box, n_skip = {}, 0, 0
    for iid in image_ids:
        parsed = parse_xml_file(xml_dir / f"{iid}.xml")
        boxes = []
        if parsed:
            for obj in parsed.get("objects", []):
                cls = raw_map(str(obj.get("label_raw", "")))
                if cls is None:
                    n_skip += 1
                    continue
                boxes.append({"bbox": list(obj["bbox"]), "class_label": cls})
        out[iid] = boxes
        n_box += len(boxes)
    print(f"[kgtrain] GT: {n_box} kotak pada {len(out)} citra, {n_skip} label tak terpetakan",
          flush=True)
    return out


def export_artifacts(encoder, kg_artifacts, out_dir: Path, kg_dir: Path, device) -> dict:

    encoder.eval()
    with torch.no_grad():
        enc = build_encoder_inputs(kg_artifacts, device=device)
        emb = encoder(enc["x"], enc["edge_index"], enc["edge_type_ids"], enc["edge_weights"])
        sdm_raw = compute_sdm(emb, enc["node_names"])
        sdm_norm = normalize_sdca(sdm_raw, method="log_scaled")
        compat = compute_compat(emb, encoder.W_compat, enc["node_names"],
                                compat_prior=encoder.compat_prior)

    # PEMISAH KUNCI WAJIB "__". tools/audit_stage3c_governance_targets.py memuat artefak ini
    # dengan `tuple(k.split("__", 1)) for k in data if "__" in k`; memakai pemisah lain membuat
    # Stage 3C memuat NOL pasangan, lalu setiap SDCA jatuh ke fallback 0.0 tanpa satu pun error.
    # Kegagalan itu senyap dan hanya terlihat dari baris "Loaded: sdm_norm (0 pairs)".
    sdm_out = {f"{a}__{b}": {"raw": float(v["raw"]), "norm": float(v["norm"]),
                             "source": "gat_encoder_trained"}
               for (a, b), v in sdm_norm.items()}
    compat_out = {f"{a}__{b}": float(v) for (a, b), v in compat.items()}

    (out_dir / "sdm_norm_encoder_trained.json").write_text(
        json.dumps(sdm_out, indent=2), encoding="utf-8")
    # Nama yang DICARI Stage 3C. Isinya identik dengan berkas di atas; field "source"
    # menyatakan asalnya supaya tidak tertukar dengan prior klinis.
    (out_dir / "sdm_norm_clinical_prior.json").write_text(
        json.dumps(sdm_out, indent=2), encoding="utf-8")
    (out_dir / "compat_scores.json").write_text(
        json.dumps(compat_out, indent=2), encoding="utf-8")
    for name in ("nodes.json", "edges.json"):
        src = kg_dir / name
        if src.is_file():
            shutil.copy2(src, out_dir / name)
    torch.save(encoder.state_dict(), out_dir / "kg_encoder_trained.pt")
    encoder.train()
    return {"sdm_norm": sdm_out, "compat": compat_out}


def main() -> int:
    p = argparse.ArgumentParser(description="Latih encoder GAT KG dan ekspor SDM/Compat.")
    p.add_argument("--kg_dir", default="outputs/stage3_audit",
                   help="Direktori artefak Stage 3A (nodes.json, edges.json).")
    p.add_argument("--cache_dir", required=True,
                   help="Cache DetectorBox split latih, mis. outputs/detector_cache/primary/oiaddr/train")
    p.add_argument("--annotations_dir", required=True,
                   help="Direktori XML Pascal VOC untuk split yang sama.")
    p.add_argument("--output_dir", required=True,
                   help="Direktori keluaran BARU. Menolak menimpa kecuali --force.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_images", type=int, default=4, help="Citra per langkah optimisasi.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lambda_kg", type=float, default=0.1,
                   help="Bobot L_KG. BESAR = SDM ditarik kembali ke prior ICDR. Sweep nilai ini.")
    p.add_argument("--lambda_gate", type=float, default=1.0)
    p.add_argument("--eta", type=float, default=1.0, help="Skala tau_SDM untuk target lunak R2.")
    p.add_argument("--train_gate", action="store_true", default=True,
                   help="Ikut melatih LGG (bawaan). Gate acak menyuntik derau ke gradien encoder.")
    p.add_argument("--freeze_gate", dest="train_gate", action="store_false")
    p.add_argument("--max_images", type=int, default=0, help="0 = semua.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true", help="Izinkan menulis ke direktori tidak kosong.")
    args = p.parse_args()

    kg_dir, out_dir = Path(args.kg_dir), Path(args.output_dir)
    cache_dir, xml_dir = Path(args.cache_dir), Path(args.annotations_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise SystemExit(f"[kgtrain] BERHENTI: {out_dir} tidak kosong. "
                         "Pakai direktori lain, atau --force bila memang disengaja.")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    if str(device) != args.device:
        print(f"[kgtrain] CUDA tidak tersedia; memakai {device}", flush=True)
    set_seed(args.seed)

    # ── daftar citra: irisan antara cache detektor dan anotasi ────────────────
    cached = sorted(q.stem for q in cache_dir.glob("*.json"))
    have_xml = {q.stem for q in xml_dir.glob("*.xml")}
    image_ids = [i for i in cached if i in have_xml]
    if not image_ids:
        raise SystemExit(f"[kgtrain] BERHENTI: tidak ada irisan antara {cache_dir} dan {xml_dir}")
    if args.max_images:
        image_ids = image_ids[:args.max_images]
    print(f"[kgtrain] {len(image_ids)} citra dipakai "
          f"(cache {len(cached)}, xml {len(have_xml)})", flush=True)

    kg_artifacts = build_graph_from_artifacts(kg_dir / "nodes.json", kg_dir / "edges.json")
    raw_map = _raw_to_kg()
    gt_boxes_all = load_gt_boxes(xml_dir, image_ids, raw_map)
    det_all = ensure_head_predictions(load_cached_predictions(image_ids, str(cache_dir)))

    encoder = build_gat_encoder().to(device)
    gate = LGG().to(device)
    sdgl_fn = SDGLLoss(lambda_kg=args.lambda_kg, eta=args.eta)
    gate_loss_fn = GateLoss()

    params = list(encoder.parameters()) + (list(gate.parameters()) if args.train_gate else [])
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    cfg = {"lambda_gate": args.lambda_gate, "lambda_kg": args.lambda_kg}

    t0, step, history = time.perf_counter(), 0, []
    for epoch in range(args.epochs):
        encoder.train()
        gate.train()
        order = list(image_ids)
        random.shuffle(order)
        # tier_stats WAJIB dicatat. Bila tidak ada kotak yang mendarat di R1/R2/R3, L_SDGL
        # bernilai nol dan satu-satunya gaya yang menggerakkan encoder adalah L_KG, yang
        # menarik embedding ke prior ICDR. Training seperti itu tidak bisa menaikkan porsi KG,
        # dan kegagalannya struktural, bukan empiris. Angka ini yang memberitahu kita.
        sums, nb = {"total": 0.0, "sdgl": 0.0, "gate": 0.0, "kg": 0.0}, 0
        tiers = {"num_r1": 0, "num_r2": 0, "num_r3": 0, "num_r4": 0}
        for s in range(0, len(order), args.batch_images):
            batch = order[s:s + args.batch_images]
            det = {i: det_all.get(i, []) for i in batch}
            if not any(det.values()):
                continue
            out = run_lgovnet_step(
                batch, det, {i: gt_boxes_all.get(i, []) for i in batch},
                kg_artifacts, encoder, gate, sdgl_fn, gate_loss_fn, step=step, cfg=cfg,
            )
            loss = out["loss_total"]
            if not torch.is_tensor(loss) or not loss.requires_grad:
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            step += 1
            nb += 1
            for k, key in (("total", "loss_total"), ("sdgl", "loss_sdgl"),
                           ("gate", "loss_gate"), ("kg", "loss_kg")):
                v = out.get(key)
                sums[k] += float(v.detach()) if torch.is_tensor(v) else float(v or 0.0)
            ts = out.get("tier_stats") or {}
            for tk in tiers:
                tiers[tk] += int(ts.get(tk, 0) or 0)
        n_sup = tiers["num_r1"] + tiers["num_r2"] + tiers["num_r3"]
        rec = {"epoch": epoch, "n_batches": nb, "step": step, **tiers,
               "n_supervised": n_sup,
               **{k: (v / nb if nb else None) for k, v in sums.items()}}
        history.append(rec)
        print(f"[kgtrain] epoch {epoch:3d} batches={nb:4d} "
              f"total={rec['total']:.4f} sdgl={rec['sdgl']:.4f} "
              f"gate={rec['gate']:.4f} kg={rec['kg']:.4f} | "
              f"R1={tiers['num_r1']} R2={tiers['num_r2']} R3={tiers['num_r3']} "
              f"R4={tiers['num_r4']} disupervisi={n_sup} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
        if epoch == 0 and n_sup == 0:
            print("[kgtrain] PERINGATAN: nol kotak di R1/R2/R3, jadi L_SDGL = 0. "
                  "Encoder hanya digerakkan L_KG (tarikan ke prior ICDR) dan L_gate. "
                  "Porsi KG tidak akan naik; hentikan dan periksa sumber tier dulu.", flush=True)

    art = export_artifacts(encoder, kg_artifacts, out_dir, kg_dir, device)
    (out_dir / "train_log.json").write_text(json.dumps(
        {"args": vars(args), "n_images": len(image_ids), "history": history,
         "sdm_norm_trained": art["sdm_norm"]}, indent=2), encoding="utf-8")

    print(f"[kgtrain] SDM terlatih (norm): "
          f"{ {k: round(v['norm'], 4) for k, v in art['sdm_norm'].items()} }", flush=True)
    print(f"[kgtrain] selesai {time.perf_counter() - t0:.0f}s -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
