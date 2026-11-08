#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Bangkitkan Tabel 1 (statistik dataset) dari artefak, bukan diketik ulang.

KOLOM DAN SUMBERNYA:
  * N_source : cacah berkas anotasi XML yang didistribusikan penyedia,
               data/<ds>/xml/<split>/*.xml.
  * N_eval   : cacah gambar pada cache deteksi primer per-gambar, dan
  * preds/img: rata-rata deteksi per gambar pada cache yang sama. Cache dibaca dari
               results/detector_outputs/primary_<ds>.zip (repositori kode; di workspace
               lewat _UPLOAD_GITHUB/). Zip TJDR dan FGADR memuat gambar bebas-lesi
               (9 dari 113 TJDR test, 105 dari 1.842 FGADR), sehingga preds/img TJDR test
               205,1 dan FGADR 292,6, bukan 215,4 dan 294,1 dari export_summary.json lama
               yang hanya memuat gambar berlesi.
  * Kolom GT : seperti terbit (PUBLISHED_GT). Konvensinya berbeda antar kohort dan
               keputusannya milik penulis: baris IDRiD dan TJDR MEMASUKKAN kotak
               degenerate (IDRiD train 60, TJDR train 3), FGADR mengecualikannya. Bila
               XML tersedia, mode bandingkan menghitung ulang kolom GT dari XML dan
               melaporkan setiap selisih. N_GT/img dihitung atas N_source.

Komentar kepala tabel dipertahankan seperti terbit; sumber preds/img yang sebenarnya
adalah cache per-gambar di atas, bukan export_summary.json yang disebut komentar itu.

Jalankan dari root repo:
    python tools/make_dataset_stats_table.py            # bandingkan, jangan tulis
    python tools/make_dataset_stats_table.py --write    # tulis berkas tabel
"""
from __future__ import annotations

import argparse
import json
import statistics
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "_FINAL_Latex/tables/T1_dataset_stats.tex"
# Repositori kode lebih dulu, lalu salinannya di dalam workspace riset.
ZIP_DIRS = (ROOT / "results/detector_outputs", ROOT / "_UPLOAD_GITHUB/results/detector_outputs")

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

# Kolom GT SEPERTI TERBIT: (N_source, N_GT, N_GT/img, MA, HE, Hard_EX, Soft_EX, w_MA).
# OIA-DDR dan FGADR cocok persis dengan hitungan dari XML. w_MA TJDR test = 13, median
# 13,0 dari XML; nilai 14 pada versi lama generator ini adalah salah ketik yang sudah
# dikoreksi di tabel terbit.
PUBLISHED_GT = {
    ("OIA-DDR", "train"): (383, 58200, 152.0, 11566, 9478, 35696, 1460, 8),
    ("OIA-DDR", "valid"): (149, 6167, 41.4, 2556, 1342, 1920, 349, 8),
    ("OIA-DDR", "test"): (225, 17032, 75.7, 2041, 6457, 8320, 214, 8),
    ("IDRiD", "train"): (54, 10583, 196.0, 2223, 1428, 6833, 99, 4),
    ("IDRiD", "test"): (27, 5774, 213.9, 1085, 535, 4116, 38, 16),
    ("TJDR", "train"): (448, 6911, 15.4, 1223, 2517, 2783, 388, 13),
    ("TJDR", "test"): (113, 1646, 14.6, 284, 678, 569, 115, 13),
    ("FGADR", "train"): (1842, 102915, 55.9, 11216, 37807, 51813, 2079, 9),
}


def gt_stats(ds: str, split: str) -> dict:
    """Hitung statistik GT satu split dari XML (kotak degenerate dibuang)."""
    xml_dir = ROOT / "data" / ds / "xml" / split
    per_class = Counter()
    ma_widths = []
    n_xml = 0
    n_degenerate = 0
    for f in sorted(xml_dir.glob("*.xml")):
        try:
            root = ET.parse(f).getroot()
        except ET.ParseError:
            continue
        n_xml += 1
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
            if name == "MA":
                ma_widths.append(x2 - x1)
    n_gt = sum(per_class.values())
    return {
        "n_source": n_xml,
        "n_gt": n_gt,
        "gt_per_img": n_gt / n_xml if n_xml else 0.0,
        "per_class": {c: per_class.get(c, 0) for c in CLASSES},
        "w_ma": statistics.median(ma_widths) if ma_widths else float("nan"),
        "n_degenerate": n_degenerate,
    }


def cache_counts(ds: str, split: str):
    """(N_eval, preds/img) dari cache primer per-gambar; (None, None) bila zip tidak ada."""
    for d in ZIP_DIRS:
        z = d / f"primary_{ds}.zip"
        if not z.is_file():
            continue
        n = []
        with zipfile.ZipFile(z) as f:
            for name in f.namelist():
                parts = name.split("/")
                if len(parts) == 3 and parts[1] == split and name.endswith(".json"):
                    n.append(len(json.loads(f.read(name))))
        return len(n), (sum(n) / len(n) if n else None)
    return None, None


def collect() -> list:
    rows = []
    for ds, label, splits in COHORTS:
        for sp in splits:
            st = gt_stats(ds, sp)
            n_eval, ppi = cache_counts(ds, sp)
            st.update(ds=ds, label=label, split=sp, n_eval=n_eval, preds_per_img=ppi)
            rows.append(st)
    return rows


def compare(rows) -> None:
    """Hitungan XML lawan kolom GT terbit; dilewati bila XML tidak tersedia."""
    names = ["N_source", "N_GT", "N_GT/img", "MA", "HE", "Hard_EX", "Soft_EX", "w_MA"]
    print(f"{'split':<16} {'kuantitas':<12} {'terbit':>10} {'dari XML':>10}")
    print("-" * 54)
    for r in rows:
        key = (r["label"], r["split"])
        if not r["n_source"]:
            print(f"{key[0] + ' ' + key[1]:<16} XML tidak tersedia, kolom GT memakai nilai terbit")
            continue
        got = (r["n_source"], r["n_gt"], round(r["gt_per_img"], 1),
               r["per_class"]["MA"], r["per_class"]["HE"],
               r["per_class"]["Hard_EX"], r["per_class"]["Soft_EX"],
               int(r["w_ma"]) if r["w_ma"] == r["w_ma"] else 0)
        for nm, a, b in zip(names, PUBLISHED_GT[key], got):
            if a != b:
                print(f"{key[0] + ' ' + key[1]:<16} {nm:<12} {a:>10} {b:>10}  BEDA")
        print(f"{key[0] + ' ' + key[1]:<16} N_eval={r['n_eval']} preds/img={r['preds_per_img']}")


def fmt_int(v):
    return f"{v:,}".replace(",", "{,}")


def build_tex(rows) -> str:
    body = []
    last = None
    for r in rows:
        n_src, n_gt, gpi, ma, he, hx, sx, wma = PUBLISHED_GT[(r["label"], r["split"])]
        if r["n_eval"] is None:
            raise SystemExit(f"[T1] cache primer {r['ds']} tidak ditemukan di {ZIP_DIRS}")
        if r["label"] != last:
            body.append(r"\midrule" if last else None)
            body.append(f"\\textbf{{{r['label']}}} & & & & & & & & & & \\\\")
            last = r["label"]
        body.append(
            f"\\quad {r['split']} & {fmt_int(n_src)} & {fmt_int(r['n_eval'])} & {fmt_int(n_gt)} & "
            f"{gpi:.1f} & {fmt_int(ma)} & {fmt_int(he)} & {fmt_int(hx)} & "
            f"{fmt_int(sx)} & {wma} & {r['preds_per_img']:.1f} \\\\"
        )
    tex_body = "\n".join(b for b in body if b)

    caption = (
        "Dataset statistics. $N_{\\mathrm{source}}$ is the number of annotated images the\n"
        "provider distributes and $N_{\\mathrm{eval}}$ the number entering the cached evaluation "
        "set. The two differ only on the TJDR train split, which no reported result\n"
        "uses. $N_{\\mathrm{GT}}$ counts ground-truth boxes and $w_{\\mathrm{MA}}$ is the\n"
        "median microaneurysm box width. $N_{\\mathrm{GT}}$/img is taken over\n"
        "$N_{\\mathrm{source}}$ and preds/img over $N_{\\mathrm{eval}}$, preds being the\n"
        "candidate detections of the frozen primary detector. FGADR has no test split and is\n"
        "evaluated on its train split."
    )
    return "\n".join([
        "%   data/<ds>/xml/<split>/*.xml  (ground-truth boxes; degenerate boxes dropped)",
        "%   outputs/detector_cache/primary/<ds>/export_summary.json  (preds/img)",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{tab:dataset_stats}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccccccc}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Dataset / Split}}",
        r"  & \multirow{2}{*}{$N_\mathrm{source}$}",
        r"  & \multirow{2}{*}{$N_\mathrm{eval}$}",
        r"  & \multirow{2}{*}{$N_\mathrm{GT}$}",
        r"  & \multirow{2}{*}{$N_\mathrm{GT}$/img}",
        r"  & \multicolumn{4}{c}{GT boxes per class}",
        r"  & $w_{\mathrm{MA}}$",
        r"  & preds/img \\",
        r"\cmidrule(lr){6-9}",
        r"  & & & & & MA & HE & Hard\_EX & Soft\_EX & (med, px) & \\",
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
