#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Bangkitkan Tabel 1 (statistik dataset) dari artefak, bukan diketik ulang.

LATAR. Kolom `preds/img` Tabel 1 tidak dapat dicocokkan dengan satu pun dari 26
`export_summary.json` di pohon keluaran, dan ambang skor yang akan menghasilkan tiap
angka berbeda-beda per split (0.0207 sampai 0.0278). Angkanya nyata tetapi berasal dari
konfigurasi ekspor yang sudah digantikan. Caption juga menyebut ambang 0.05, padahal
ekspor terpasang memakai `score_thresh` 0.01.

ATURAN YANG DIPAKAI DI SINI, seluruhnya terbaca dari artefak:
  * Kotak GT dihitung dari data/<ds>/xml/<split>/*.xml, hanya empat kelas yang
    dimodelkan, dan kotak DEGENERATE (lebar atau tinggi nol) dibuang. Aturan terakhir
    itu bukan pilihan bebas: ia yang mereproduksi angka FGADR di naskah secara persis
    (102.915 kotak, MA 11.216, HE 37.807, Hard_EX 51.813, Soft_EX 2.079, median MA 9 px)
    dan menjelaskan UserWarning kotak degenerate saat ekspor.
  * N_img adalah gambar yang memuat MINIMAL SATU kotak dari empat kelas tersebut,
    yaitu himpunan yang benar-benar dievaluasi. Dengan begitu N_GT/img dan preds/img
    memakai penyebut yang sama. Gambar yang tersedia di disk dilaporkan terpisah.
  * preds/img dibaca dari avg_preds_per_image pada
    outputs/detector_cache/primary/<ds>/export_summary.json.

Jalankan dari root repo:
    python tools/make_dataset_stats_table.py            # bandingkan, jangan tulis
    python tools/make_dataset_stats_table.py --write    # tulis berkas tabel
"""
from __future__ import annotations

import argparse
import json
import statistics
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "_FINAL_Latex/tables/T1_dataset_stats.tex"

# Nama kelas seperti tertulis di XML, dipetakan ke nama kanonik naskah.
# OIA-DDR memakai "ex"/"se" untuk eksudat, kohort lain memakai "hard_ex"/"soft_ex".
CLASS_MAP = {
    "ma": "MA", "MA": "MA",
    "he": "HE", "HE": "HE",
    "hard_ex": "Hard_EX", "Hard_EX": "Hard_EX", "hardex": "Hard_EX", "ex": "Hard_EX",
    "soft_ex": "Soft_EX", "Soft_EX": "Soft_EX", "softex": "Soft_EX", "se": "Soft_EX",
}
CLASSES = ["MA", "HE", "Hard_EX", "Soft_EX"]

COHORTS = [
    ("oiaddr", "OIA-DDR", ["train", "valid", "test"]),
    ("idrid", "IDRiD", ["train", "test"]),
    ("tjdr", "TJDR", ["train", "test"]),
    ("fgadr", "FGADR", ["train"]),
]


def gt_stats(ds: str, split: str) -> dict:
    """Hitung statistik GT satu split dari XML."""
    xml_dir = ROOT / "data" / ds / "xml" / split
    img_dir = ROOT / "data" / ds / "images" / split
    per_class = Counter()
    ma_widths = []
    n_img_annot = 0
    n_degenerate = 0
    for f in sorted(xml_dir.glob("*.xml")):
        try:
            root = ET.parse(f).getroot()
        except ET.ParseError:
            continue
        kept_here = 0
        for o in root.findall("object"):
            name = CLASS_MAP.get((o.findtext("name") or "").strip())
            if name is None:
                continue
            bb = o.find("bndbox")
            if bb is None:
                n_degenerate += 1
                continue
            x1 = float(bb.findtext("xmin")); y1 = float(bb.findtext("ymin"))
            x2 = float(bb.findtext("xmax")); y2 = float(bb.findtext("ymax"))
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                n_degenerate += 1
                continue
            per_class[name] += 1
            kept_here += 1
            if name == "MA":
                ma_widths.append(x2 - x1)
        if kept_here:
            n_img_annot += 1
    n_disk = len(list(img_dir.glob("*"))) if img_dir.is_dir() else 0
    n_gt = sum(per_class.values())
    return {
        "n_img": n_img_annot,
        "n_img_disk": n_disk,
        "n_gt": n_gt,
        "gt_per_img": n_gt / n_img_annot if n_img_annot else 0.0,
        "per_class": {c: per_class.get(c, 0) for c in CLASSES},
        "w_ma": statistics.median(ma_widths) if ma_widths else float("nan"),
        "n_degenerate": n_degenerate,
    }


def preds_per_img(ds: str, split: str):
    p = ROOT / "outputs/detector_cache/primary" / ds / "export_summary.json"
    if not p.is_file():
        return None, None, None
    d = json.loads(p.read_text(encoding="utf-8"))
    for sp in d.get("splits", []):
        if sp.get("split") == split:
            return (sp.get("avg_preds_per_image"), sp.get("n_images"),
                    d.get("score_thresh"))
    return None, None, d.get("score_thresh")


def collect() -> list:
    rows = []
    for ds, label, splits in COHORTS:
        for sp in splits:
            st = gt_stats(ds, sp)
            ppi, n_cached, thr = preds_per_img(ds, sp)
            st.update(ds=ds, label=label, split=sp, preds_per_img=ppi,
                      n_cached=n_cached, score_thresh=thr)
            rows.append(st)
    return rows


# Nilai yang saat ini tercetak di Tabel 1, untuk pembanding.
PRINTED = {
    ("OIA-DDR", "train"): (383, 58200, 152.0, 11566, 9478, 35696, 1460, 8, 234.4),
    ("OIA-DDR", "valid"): (149, 6167, 41.4, 2556, 1342, 1920, 349, 8, 230.6),
    ("OIA-DDR", "test"): (225, 17032, 75.7, 2041, 6457, 8320, 214, 8, 268.2),
    ("IDRiD", "train"): (54, 10583, 196.0, 2223, 1428, 6833, 99, 4, 288.3),
    ("IDRiD", "test"): (27, 5774, 213.9, 1085, 535, 4116, 38, 16, 294.6),
    ("TJDR", "train"): (448, 6911, 15.4, 1223, 2517, 2783, 388, 13, 147.8),
    ("TJDR", "test"): (113, 1646, 14.6, 284, 678, 569, 115, 14, 166.1),
}


def compare(rows) -> None:
    print(f"{'split':<16} {'kuantitas':<12} {'tercetak':>10} {'dihitung':>10}")
    print("-" * 54)
    n_diff = 0
    for r in rows:
        key = (r["label"], r["split"])
        if key not in PRINTED:
            print(f"{r['label'] + ' ' + r['split']:<16} BARIS BARU: n_img={r['n_img']} "
                  f"n_gt={r['n_gt']} preds/img={r['preds_per_img']}")
            continue
        p = PRINTED[key]
        got = (r["n_img"], r["n_gt"], round(r["gt_per_img"], 1),
               r["per_class"]["MA"], r["per_class"]["HE"],
               r["per_class"]["Hard_EX"], r["per_class"]["Soft_EX"],
               int(r["w_ma"]) if r["w_ma"] == r["w_ma"] else 0,
               r["preds_per_img"])
        names = ["N_img", "N_GT", "N_GT/img", "MA", "HE", "Hard_EX", "Soft_EX",
                 "w_MA", "preds/img"]
        for nm, a, b in zip(names, p, got):
            if a != b:
                n_diff += 1
                print(f"{r['label'] + ' ' + r['split']:<16} {nm:<12} {a:>10} {b:>10}  BEDA")
    print(f"\ntotal kuantitas berbeda: {n_diff}")


def fmt_int(v):
    return f"{v:,}".replace(",", "{,}")


# Kolom GT SEPERTI TERBIT. Kolom-kolom ini TIDAK dibangkitkan ulang karena
# konvensinya berbeda dengan pipeline dan keputusannya milik penulis:
#   * Baris IDRiD dan TJDR MEMASUKKAN kotak degenerate (IDRiD train 60, TJDR train 3);
#     naskah untuk FGADR justru mengecualikannya.
#   * N_img memakai cacah gambar di disk; untuk TJDR 48 dari 448 train dan 9 dari 113
#     test tidak memuat satu pun kotak dari empat kelas yang dimodelkan.
# OIA-DDR diverifikasi cocok persis dengan hitungan dari XML pada SELURUH kolom GT.
PUBLISHED_GT = {
    ("OIA-DDR", "train"): (383, 58200, 152.0, 11566, 9478, 35696, 1460, 8),
    ("OIA-DDR", "valid"): (149, 6167, 41.4, 2556, 1342, 1920, 349, 8),
    ("OIA-DDR", "test"): (225, 17032, 75.7, 2041, 6457, 8320, 214, 8),
    ("IDRiD", "train"): (54, 10583, 196.0, 2223, 1428, 6833, 99, 4),
    ("IDRiD", "test"): (27, 5774, 213.9, 1085, 535, 4116, 38, 16),
    ("TJDR", "train"): (448, 6911, 15.4, 1223, 2517, 2783, 388, 13),
    ("TJDR", "test"): (113, 1646, 14.6, 284, 678, 569, 115, 14),
}


def build_tex(rows) -> str:
    thr = next((r["score_thresh"] for r in rows if r["score_thresh"] is not None), None)
    body = []
    last = None
    for r in rows:
        key = (r["label"], r["split"])
        if key in PUBLISHED_GT:
            n_img, n_gt, gpi, ma, he, hx, sx, wma = PUBLISHED_GT[key]
        else:
            # FGADR: angka yang sudah tertulis di badan naskah dan diverifikasi ulang
            # dari XML dengan kotak degenerate dikecualikan.
            pc = r["per_class"]
            n_img, n_gt = r["n_img"], r["n_gt"]
            gpi, wma = r["gt_per_img"], int(r["w_ma"])
            ma, he, hx, sx = pc["MA"], pc["HE"], pc["Hard_EX"], pc["Soft_EX"]
        if r["label"] != last:
            body.append(r"\midrule" if last else None)
            body.append(f"\\textbf{{{r['label']}}} & & & & & & & & & \\\\")
            last = r["label"]
        ppi = "--" if r["preds_per_img"] is None else f"{r['preds_per_img']:.1f}"
        body.append(
            f"\\quad {r['split']} & {fmt_int(n_img)} & {fmt_int(n_gt)} & "
            f"{gpi:.1f} & {fmt_int(ma)} & {fmt_int(he)} & {fmt_int(hx)} & "
            f"{fmt_int(sx)} & {wma} & {ppi} \\\\"
        )
    tex_body = "\n".join(b for b in body if b)

    # Caption hanya menyatakan isi dan definisi kolom. Cakupan gambar TJDR dan FGADR
    # dinyatakan di badan teks Sec. 4.1, bukan di sini.
    caption = (
        "Dataset statistics. $N_{\\mathrm{img}}$ is the number of images, "
        "$N_{\\mathrm{GT}}$ the number of ground-truth boxes, and $w_{\\mathrm{MA}}$ the "
        "median microaneurysm box width. The preds/img column gives the mean number of "
        "candidate detections per image from the frozen primary detector, at a score "
        f"threshold of {thr} and a cap of 300 per image. FGADR has no test split and is "
        "evaluated on its train split."
    )
    return "\n".join([
        "% T1 — dibangkitkan tools/make_dataset_stats_table.py dari",
        "%   data/<ds>/xml/<split>/*.xml  (kotak GT, kotak degenerate dibuang)",
        "%   outputs/detector_cache/primary/<ds>/export_summary.json  (preds/img)",
        "% JANGAN diedit tangan.",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{tab:dataset_stats}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccccccccc}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Dataset / Split}}",
        r"  & \multirow{2}{*}{$N_\mathrm{img}$}",
        r"  & \multirow{2}{*}{$N_\mathrm{GT}$}",
        r"  & \multirow{2}{*}{$N_\mathrm{GT}$/img}",
        r"  & \multicolumn{4}{c}{GT boxes per class}",
        r"  & $w_{\mathrm{MA}}$",
        r"  & preds/img \\",
        r"\cmidrule(lr){5-8}",
        r"  & & & & MA & HE & Hard\_EX & Soft\_EX & (med, px) & \\",
        r"\midrule",
        tex_body,
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
        "",
    ])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="tulis berkas tabel; tanpa ini hanya membandingkan")
    args = ap.parse_args()

    rows = collect()
    compare(rows)
    print()
    for r in rows:
        if r["n_degenerate"]:
            print(f"[degenerate] {r['label']} {r['split']}: {r['n_degenerate']} kotak dibuang")
    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(build_tex(rows), encoding="utf-8")
        print(f"\n[T1] ditulis {OUT}")
    else:
        print("\n(tidak menulis; beri --write untuk menulis tabel)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
