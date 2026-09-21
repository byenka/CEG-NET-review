#!/usr/bin/env python
"""sanitize_xml_paths.py — buat salinan bersih anotasi XML tanpa elemen <path>.

Elemen <path> pada anotasi IDRiD dan TJDR memuat direktori pengguna lokal mesin penulis.
Elemen itu tidak dipakai pipeline mana pun, yang membaca <filename>, sehingga dapat dibuang
tanpa mengubah perilaku.

Berkas asli TIDAK DISENTUH. Salinan ditulis ke data/<ds>/xml_clean/<split>/ dengan struktur
split yang sama.

Penghapusan dilakukan di tingkat bita, bukan lewat parser XML, agar indentasi, urutan elemen,
dan akhiran baris tetap identik dengan aslinya. Hanya baris <path>...</path> yang hilang.

Setiap berkas diverifikasi sesudah ditulis: dapat diurai, tidak lagi memuat direktori pengguna,
dan seluruh objek (kelas dan empat koordinat) serta ukuran citra identik dengan aslinya.

Jalankan dari root repo:
  python tools/sanitize_xml_paths.py            # tulis salinan bersih
  python tools/sanitize_xml_paths.py --check    # hanya laporkan, tanpa menulis
"""
from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COHORTS = ["idrid", "tjdr"]
REPORT = ROOT / "outputs/0506/xml_provenance/sanitize_report.json"

# baris utuh <path>...</path> beserta indentasi dan akhiran barisnya
PATH_LINE = re.compile(rb"[ \t]*<path>.*?</path>[ \t]*\r?\n", re.S)
USERDIR = re.compile(rb"[A-Za-z]:[/\\]+Users[/\\]+", re.I)


def signature(data: bytes):
    """Kelas, koordinat, dan ukuran citra; dipakai untuk membandingkan sebelum dan sesudah."""
    root = ET.fromstring(data)
    size = root.find("size")
    dims = None
    if size is not None:
        dims = (size.findtext("width"), size.findtext("height"), size.findtext("depth"))
    objs = []
    for o in root.findall("object"):
        bb = o.find("bndbox")
        objs.append((
            (o.findtext("name") or "").strip(),
            bb.findtext("xmin"), bb.findtext("ymin"),
            bb.findtext("xmax"), bb.findtext("ymax"),
        ) if bb is not None else ((o.findtext("name") or "").strip(),))
    return dims, objs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="laporkan saja, jangan menulis")
    args = ap.parse_args()

    report = {"tool": "tools/sanitize_xml_paths.py", "write": not args.check, "cohorts": {}}
    grand = {"files": 0, "stripped": 0, "clean_already": 0, "mismatch": 0, "leak_after": 0}

    for coh in COHORTS:
        src_base = ROOT / f"data/{coh}/xml"
        dst_base = ROOT / f"data/{coh}/xml_clean"
        if not src_base.is_dir():
            report["cohorts"][coh] = {"error": "direktori xml tidak ada"}
            continue

        n = stripped = already = mismatch = leak_after = 0
        for src in sorted(src_base.rglob("*.xml")):
            raw = src.read_bytes()
            n += 1
            out, k = PATH_LINE.subn(b"", raw)
            if k:
                stripped += 1
            else:
                already += 1

            # verifikasi isi tidak berubah selain elemen path
            try:
                if signature(raw) != signature(out):
                    mismatch += 1
                    continue
            except ET.ParseError:
                mismatch += 1
                continue
            if USERDIR.search(out):
                leak_after += 1

            if not args.check:
                rel = src.relative_to(src_base)
                dst = dst_base / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(out)

        report["cohorts"][coh] = {
            "files": n, "path_element_stripped": stripped, "already_clean": already,
            "content_mismatch": mismatch, "user_dir_remaining": leak_after,
            "destination": f"data/{coh}/xml_clean",
        }
        for key, val in (("files", n), ("stripped", stripped), ("clean_already", already),
                         ("mismatch", mismatch), ("leak_after", leak_after)):
            grand[key] += val
        print(f"{coh:6s} berkas={n:5d} <path> dihapus={stripped:5d} "
              f"sudah bersih={already:4d} isi berubah={mismatch} sisa direktori pengguna={leak_after}")

    report["total"] = grand
    report["passed"] = grand["mismatch"] == 0 and grand["leak_after"] == 0
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nTOTAL {grand['files']} berkas, {grand['stripped']} dibersihkan, "
          f"{grand['mismatch']} isi berubah, {grand['leak_after']} masih bocor")
    print("VONIS:", "LOLOS" if report["passed"] else "GAGAL")
    print(f"ditulis {REPORT}")


if __name__ == "__main__":
    main()
