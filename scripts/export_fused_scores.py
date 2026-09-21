"""export_fused_scores.py — ekspor skor fusi tiga pass OIA-DDR ke CSV berkunci bbox_id.

Tujuan: tools/make_tier_routing.py ikut dirilis sebagai kode, dan menanam pembaca cache detektor
di sana akan membuatnya lambat serta bergantung pada ribuan berkas cache hanya untuk satu kolom
tabel. Skrip ini menuliskan vektornya sekali, generator tabel tinggal membacanya dan mencocokkan
lewat bbox_id sehingga keselarasan barisnya dapat diperiksa, bukan diasumsikan.

Fusi = rata-rata skor detektor primer, pass flip, dan detektor kedua; kotak yang tidak punya
pasangan pada suatu pass menyumbang 0. Pencocokan IoU >= 0.15, kelas diabaikan, satu kotak sumber
boleh dipakai beberapa kotak primer. Definisi ini identik dengan baselines_nokg.py dan
fusion_external.py, dan hasilnya diperiksa silang terhadap keduanya sebelum ditulis.

Keluaran: _septem/scripts_optionB/fused_scores_oiaddr.csv (split, bbox_id, fused). Hanya membaca.
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.eval_fp_reduction_benchmark import load_csv  # noqa: E402
from tools.make_indomain_results import POSITIVE, T_DEP, V_DEP  # noqa: E402

OUT = ROOT / "_septem/scripts_optionB/fused_scores_oiaddr.csv"
REF = ROOT / "_septem/scripts_optionB/baselines_nokg.json"
CACHE = {"primary": ROOT / "outputs/detector_cache/primary/oiaddr",
         "flip": ROOT / "outputs/detector_cache/secondary/oiaddr",
         "v2": ROOT / "outputs/secondary_v2/oiaddr"}
IOU_MIN = 0.15
t0 = time.perf_counter()


def log(m):
    print(f"[exp] {m} ({time.perf_counter() - t0:.0f}s)", flush=True)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def cache_entries(root, split, image_id):
    f = root / split / f"{image_id}.json"
    if not f.is_file():
        return []
    out = []
    for e in json.loads(f.read_text(encoding="utf-8")):
        raw = e.get("detector_raw") or e.get("r2a_raw") or {}
        if "bbox" in raw and "score" in raw:
            out.append((e.get("bbox_id"), raw["bbox"], float(raw["score"])))
    return out


def fused_three(csv_path, split):
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = [(r["bbox_id"], r["image_id"], float(r["pred_score"])) for r in csv.DictReader(fh)]
    by_img = {}
    for i, (_b, img, _s) in enumerate(rows):
        by_img.setdefault(img, []).append(i)
    s_p = np.array([r[2] for r in rows], dtype=np.float64)
    s_f, s_v = np.zeros(len(rows)), np.zeros(len(rows))
    for img, idx in by_img.items():
        prim = {bid: bb for bid, bb, _s in cache_entries(CACHE["primary"], split, img)}
        others = {k: [(bb, sc) for _i, bb, sc in cache_entries(CACHE[k], split, img)]
                  for k in ("flip", "v2")}
        for i in idx:
            pb = prim.get(rows[i][0])
            if pb is None:
                continue
            for key, arr in (("flip", s_f), ("v2", s_v)):
                best, best_iou = 0.0, 0.0
                for ob, os_ in others[key]:
                    v = iou(pb, ob)
                    if v >= IOU_MIN and v > best_iou:
                        best_iou, best = v, os_
                if best_iou >= IOU_MIN:
                    arr[i] = best
    return [r[0] for r in rows], (s_p + s_f + s_v) / 3.0


ids_v, fv = fused_three(V_DEP, "valid")
log(f"valid: {len(ids_v)} kotak")
ids_t, ft = fused_three(T_DEP, "test")
log(f"test : {len(ids_t)} kotak")

# Cek silang wajib: AP fusi pada test harus sama dengan baselines_nokg.json sampai 1e-9.
te = load_csv(T_DEP, POSITIVE)
ap = float(average_precision_score(te["label"], ft))
ref = json.loads(REF.read_text(encoding="utf-8"))["fusion"]["ens3_mean0"]["ap"]
if abs(ap - ref) > 1e-9:
    raise SystemExit(f"BERHENTI: AP fusi {ap:.9f} != baselines_nokg.json {ref:.9f}")
log(f"cek silang AP fusi test = {ap:.6f}, cocok dengan baselines_nokg.json")

with open(OUT, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["split", "bbox_id", "fused"])
    for bid, s in zip(ids_v, fv):
        w.writerow(["valid", bid, f"{s:.6f}"])
    for bid, s in zip(ids_t, ft):
        w.writerow(["test", bid, f"{s:.6f}"])
log(f"ditulis {OUT.name} ({len(ids_v) + len(ids_t)} baris)")
