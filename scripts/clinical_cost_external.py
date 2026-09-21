"""clinical_cost_external.py — apakah kebutaan biaya klinis fusi bertahan di luar OIA-DDR?

Latar. clinical_cost_fusion.py menunjukkan pada OIA-DDR bahwa rata-rata skor tiga pass
mengungguli setiap varian gate pada AUPRC, tetapi memisahkan konfusi grade-shifting dari
within-grade hanya pada tingkat kebetulan (AUROC 0.483, CI memuat 0.5), sementara SDCA memisahkan
0.994. Klaim itu kini bertumpu pada SATU dataset. Reviewer akan menyebutnya kebetulan.

Skrip ini mengulang uji yang sama pada tiga kohort eksternal dengan protokol zero-shot naskah.

Protokol, identik dengan Tabel 13:
  * kelas positif  = UNSAFE_MISMATCH (grade-shifting, seharusnya disupres)
  * kelas negatif  = CLINICALLY_CLOSE_MISMATCH (within-grade, seharusnya dipertahankan)
  * AUROC dengan UNSAFE sebagai positif; 0.5 = tanpa informasi ke arah mana pun
  * CI: bootstrap image-clustered 2.000 resampel, seed 0

Arah tiap sinyal berbeda dan harus dibaca hati-hati:
  * SDCA          tinggi = berbahaya   -> informatif mendekati 1
  * skor detektor tinggi = pertahankan -> sadar klinis berarti mendekati 0
  * skor fusi     sama dengan di atas
Karena itu yang dilaporkan juga |AUROC - 0.5|, yaitu banyaknya informasi biaya klinis yang dibawa
sinyal itu, bebas arah.

DAYA STATISTIK BERBEDA JAUH ANTAR KOHORT, dan ini wajib ikut dilaporkan:
  FGADR 360 positif (lebih kuat dari OIA-DDR), TJDR 45, IDRiD hanya 25.
  Satu kohort kecil bukan bukti; yang bermakna adalah POLA di ketiganya ditambah daya FGADR.

Batas keras cache, bukan pilihan desain: secondary_v2 hanya ada untuk idrid dan tjdr, sehingga
FGADR hanya bisa fusi DUA pass (primer + flip). Jumlah pass dicatat per kohort.

Efisiensi: skor fusi hanya dihitung untuk citra yang memuat sekurangnya satu baris terpilih,
bukan seluruh split. FGADR 510.825 baris tetapi hanya 1.941 yang dipakai.

Keluaran: _septem/scripts_optionB/clinical_cost_external.json. Hanya membaca.
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.make_indomain_results import N_BOOT  # noqa: E402

csv.field_size_limit(10 ** 9)

REF = ROOT / "outputs/0506/evsds_pss/domain_shift_5seed.json"
OUT = ROOT / "_septem/scripts_optionB/clinical_cost_external.json"
CACHE = {"primary": ROOT / "outputs/detector_cache/primary",
         "flip": ROOT / "outputs/detector_cache/secondary",
         "v2": ROOT / "outputs/secondary_v2"}
KEY = {"OIA-DDR": "oiaddr", "IDRiD": "idrid", "TJDR": "tjdr", "FGADR": "fgadr"}
IOU_MIN = 0.15
UNSAFE, CLOSE = "UNSAFE_MISMATCH", "CLINICALLY_CLOSE_MISMATCH"
t0 = time.perf_counter()


def log(m):
    print(f"[cce] {m} ({time.perf_counter() - t0:.0f}s)", flush=True)


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


def select_rows(csv_path):
    """Baris dari dua kategori itu saja. Mengembalikan daftar dict ringkas."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            cat = r.get("category", "")
            if cat not in (UNSAFE, CLOSE):
                continue
            rows.append({"bbox_id": r["bbox_id"], "image_id": r["image_id"],
                         "score": float(r["pred_score"]), "sdca": float(r["ev_sdca"]),
                         "unsafe": 1 if cat == UNSAFE else 0})
    return rows


def fused_for_rows(rows, key, split, use_v2):
    """Skor fusi hanya untuk citra yang memuat baris terpilih."""
    by_img = {}
    for i, r in enumerate(rows):
        by_img.setdefault(r["image_id"], []).append(i)
    fused = np.zeros(len(rows), dtype=np.float64)
    srcs = ["flip"] + (["v2"] if use_v2 else [])
    npass = 1 + len(srcs)
    n_noprim = 0
    for img, idx in by_img.items():
        prim = {bid: bb for bid, bb, _s in cache_entries(CACHE["primary"] / key, split, img)}
        if not prim:
            n_noprim += 1
        others = {k: [(bb, sc) for _i, bb, sc in cache_entries(CACHE[k] / key, split, img)]
                  for k in srcs}
        for i in idx:
            total = rows[i]["score"]
            pb = prim.get(rows[i]["bbox_id"])
            if pb is not None:
                for k in srcs:
                    best, best_iou = 0.0, 0.0
                    for ob, os_ in others[k]:
                        v = iou(pb, ob)
                        if v >= IOU_MIN and v > best_iou:
                            best_iou, best = v, os_
                    if best_iou >= IOU_MIN:
                        total += best
            fused[i] = total / npass
    return fused, npass, n_noprim


def boot_auroc(score, y, img, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(img)
    groups = [np.where(img == im)[0] for im in uniq]
    n = len(groups)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[k] for k in rng.integers(0, n, size=n)])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        vals.append(roc_auc_score(yy, score[idx]))
    if not vals:
        return [None, None], 0
    lo, hi = (float(x) for x in np.percentile(vals, [2.5, 97.5]))
    return [lo, hi], len(vals)


ref = json.loads(REF.read_text(encoding="utf-8"))
res = {"note": "AUROC dengan UNSAFE_MISMATCH sebagai kelas positif; 0.5 = tanpa informasi",
       "n_bootstrap": N_BOOT, "cohorts": []}

for c in ref["cohorts"]:
    name = c["name"]
    key = KEY[name]
    csv_path = ROOT / c["csv"]
    split = csv_path.parent.name
    use_v2 = (CACHE["v2"] / key / split).is_dir()

    rows = select_rows(csv_path)
    if not rows:
        log(f"{name}: tidak ada baris dua kategori itu; dilewati")
        continue
    y = np.array([r["unsafe"] for r in rows], dtype=int)
    img = np.array([r["image_id"] for r in rows])
    n_u, n_c = int(y.sum()), int((1 - y).sum())
    log(f"{name}: UNSAFE={n_u} CLOSE={n_c} pada {len(set(img))} citra, split={split}, "
        f"v2={'ada' if use_v2 else 'TIDAK ADA'}")
    if n_u == 0 or n_c == 0:
        log(f"{name}: satu kelas kosong; AUROC tidak terdefinisi, dilewati")
        continue

    fused, npass, n_noprim = fused_for_rows(rows, key, split, use_v2)
    sig = {
        "detector_score": np.array([r["score"] for r in rows], dtype=np.float64),
        f"fused_{npass}_pass": fused,
        "kg_sdca": np.array([r["sdca"] for r in rows], dtype=np.float64),
    }
    rec = {"name": name, "split": split, "n_unsafe": n_u, "n_close": n_c,
           "n_images": int(len(set(img))), "n_pass_fusion": npass,
           "images_without_primary_cache": n_noprim, "signals": {}}
    for sname, s in sig.items():
        a = float(roc_auc_score(y, s))
        ci, nv = boot_auroc(s, y, img)
        rec["signals"][sname] = {
            "auroc": a, "info_abs": abs(a - 0.5), "ci95": ci, "n_boot_valid": nv,
            "ci_contains_half": (ci[0] is not None and ci[0] < 0.5 < ci[1]),
            "mean_unsafe": float(s[y == 1].mean()), "mean_close": float(s[y == 0].mean()),
        }
        lo = f"{ci[0]:.3f}" if ci[0] is not None else "NA"
        hi = f"{ci[1]:.3f}" if ci[1] is not None else "NA"
        log(f"{name} {sname:18s} AUROC {a:.4f} |info| {abs(a - 0.5):.4f} CI [{lo}, {hi}]"
            + ("  (memuat 0.5)" if rec["signals"][sname]["ci_contains_half"] else ""))
    res["cohorts"].append(rec)

OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
log(f"selesai -> {OUT}")
