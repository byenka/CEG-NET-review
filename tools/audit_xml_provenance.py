#!/usr/bin/env python
"""audit_xml_provenance.py — audit provenans anotasi XML untuk keempat kohort.

Memeriksa apa yang dapat diverifikasi dari berkas itu sendiri, tanpa mengakses penyedia
dataset: jumlah berkas dan objek, kosakata kelas, validitas koordinat terhadap ukuran citra
yang dideklarasikan, kotak berukuran nol, duplikat persis, varian skema XML, kebocoran path
absolut, tumpang tindih citra antar-split, dan rekonsiliasi dengan laporan konversi bila ada.

Skrip ini TIDAK mengubah apa pun. Ia hanya membaca.

Keluaran: outputs/0506/xml_provenance/audit.json
Jalankan dari root repo:
  python tools/audit_xml_provenance.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/0506/xml_provenance/audit.json"

COHORTS = ["oiaddr", "idrid", "tjdr", "fgadr"]
USERPATH = re.compile(r"[A-Za-z]:[/\\]+Users[/\\]+[^/\\<]+", re.I)
FGADR_REPORT = ROOT / "data/fgadr/fgadr_xml_conversion_report.csv"


def audit_file(p: Path):
    """Kembalikan (objek, masalah) untuk satu berkas XML."""
    root = ET.parse(p).getroot()
    size = root.find("size")
    W = int(size.find("width").text) if size is not None else None
    H = int(size.find("height").text) if size is not None else None

    path_el = root.find("path")
    rec = {
        "folder": (root.findtext("folder") or "").strip(),
        "has_path": path_el is not None,
        "has_segmented": root.find("segmented") is not None,
        "path_text": (path_el.text or "").strip() if path_el is not None else "",
        "width": W, "height": H,
    }

    objs, issues = [], Counter()
    seen = set()
    for o in root.findall("object"):
        name = (o.findtext("name") or "").strip()
        bb = o.find("bndbox")
        if bb is None:
            issues["objek_tanpa_bndbox"] += 1
            continue
        try:
            x1 = float(bb.findtext("xmin")); y1 = float(bb.findtext("ymin"))
            x2 = float(bb.findtext("xmax")); y2 = float(bb.findtext("ymax"))
        except (TypeError, ValueError):
            issues["koordinat_tak_terbaca"] += 1
            continue
        objs.append(name)
        if x2 <= x1 or y2 <= y1:
            issues["kotak_degenerat"] += 1
        if x1 < 0 or y1 < 0:
            issues["koordinat_negatif"] += 1
        if W and H and (x2 > W or y2 > H):
            issues["keluar_batas"] += 1
        key = (name, x1, y1, x2, y2)
        if key in seen:
            issues["duplikat_persis"] += 1
        seen.add(key)

    rec["objects"] = objs
    rec["distinct"] = len(seen)
    rec["issues"] = dict(issues)
    return rec


def main() -> None:
    report = {"tool": "tools/audit_xml_provenance.py", "cohorts": {}}

    for coh in COHORTS:
        base = ROOT / f"data/{coh}/xml"
        if not base.is_dir():
            report["cohorts"][coh] = {"error": "direktori xml tidak ada"}
            continue

        splits = sorted(d.name for d in base.iterdir() if d.is_dir())
        per_split, classes, issues = {}, Counter(), Counter()
        folders, schema = Counter(), Counter()
        leak_files = 0
        ids_by_split = {}
        nobj_total = 0

        ndist_total = 0
        for sp in splits:
            files = sorted((base / sp).glob("*.xml"))
            ids, nobj, ndist = set(), 0, 0
            for f in files:
                rec = audit_file(f)
                ids.add(f.stem)
                nobj += len(rec["objects"])
                ndist += rec["distinct"]
                classes.update(rec["objects"])
                for k, v in rec["issues"].items():
                    issues[k] += v
                folders[rec["folder"]] += 1
                schema["with_path" if rec["has_path"] else "no_path"] += 1
                schema["with_segmented" if rec["has_segmented"] else "no_segmented"] += 1
                if rec["has_path"] and USERPATH.search(rec["path_text"]):
                    leak_files += 1
            per_split[sp] = {
                "files": len(files), "objects": nobj, "distinct_boxes": ndist,
                "duplicate_ratio": round(nobj / ndist, 3) if ndist else None,
            }
            ids_by_split[sp] = ids
            nobj_total += nobj
            ndist_total += ndist

        overlap = {}
        sl = list(ids_by_split)
        for i in range(len(sl)):
            for j in range(i + 1, len(sl)):
                inter = ids_by_split[sl[i]] & ids_by_split[sl[j]]
                if inter:
                    overlap[f"{sl[i]}|{sl[j]}"] = len(inter)

        mask_dir = ROOT / f"data/{coh}/images_mask"
        report["cohorts"][coh] = {
            "splits": per_split,
            "files_total": sum(v["files"] for v in per_split.values()),
            "objects_total": nobj_total,
            "distinct_boxes_total": ndist_total,
            "duplicate_objects": nobj_total - ndist_total,
            "classes": dict(classes.most_common()),
            "coordinate_issues": dict(issues),
            "schema_variant": dict(schema),
            "folder_tags": dict(folders.most_common(5)),
            "absolute_path_leak_files": leak_files,
            "split_id_overlap": overlap,
            "source_masks_present": mask_dir.is_dir(),
            "source_mask_files": (sum(1 for _ in mask_dir.rglob("*")
                                      if _.is_file()) if mask_dir.is_dir() else 0),
        }
        print(f"{coh:8s} berkas={report['cohorts'][coh]['files_total']:5d} "
              f"objek={nobj_total:7d} kelas={len(classes)} "
              f"masalah_koordinat={sum(issues.values())} bocor={leak_files}")

    # rekonsiliasi FGADR terhadap laporan konversi
    if FGADR_REPORT.exists():
        rows = list(csv.DictReader(FGADR_REPORT.open(newline="", encoding="utf-8")))
        kept = sum(int(r["num_kept"]) for r in rows if r["num_kept"])
        comp = sum(int(r["num_components"]) for r in rows if r["num_components"])
        resized = sum(1 for r in rows if r["mask_width"] and r["image_width"]
                      and (r["mask_width"] != r["image_width"]
                           or r["mask_height"] != r["image_height"]))
        report["fgadr_conversion_report"] = {
            "file": "data/fgadr/fgadr_xml_conversion_report.csv",
            "sha256": hashlib.sha256(FGADR_REPORT.read_bytes()).hexdigest(),
            "rows": len(rows),
            "images": len({r["image_file"] for r in rows}),
            "status_counts": dict(Counter(r["status"] for r in rows)),
            "components_found": comp,
            "components_kept": kept,
            "components_dropped": comp - kept,
            "mask_size_differs_from_image": resized,
            "xml_objects_in_repo": report["cohorts"]["fgadr"]["objects_total"],
            "reconciles": kept == report["cohorts"]["fgadr"]["objects_total"],
        }
        r = report["fgadr_conversion_report"]
        print(f"\nFGADR rekonsiliasi: komponen disimpan {r['components_kept']} "
              f"vs objek XML {r['xml_objects_in_repo']} -> "
              f"{'COCOK' if r['reconciles'] else 'TIDAK COCOK'}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nditulis {OUT}")


if __name__ == "__main__":
    main()
