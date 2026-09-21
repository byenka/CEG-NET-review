"""clinical_cost_fusion.py — apakah fusi skor membeli dimensi biaya klinis, atau hanya peringkat?

Pertanyaan yang menentukan. baselines_nokg.py dan fusion_external.py menunjukkan rata-rata skor
tiga pass mengungguli setiap varian gate pada AUPRC di keempat kohort. Kalau fusi juga membawa
sinyal biaya klinis, maka kontribusi naskah ini habis. Kalau fusi buta terhadap biaya klinis
sama seperti skor detektor mentah (T13: AUROC 0.479, kebetulan), maka yang dibeli knowledge graph
adalah dimensi yang TIDAK dimiliki metode berbasis skor mana pun, berapa pun AUPRC-nya.

Protokol mengikuti Tabel 13 persis: pada OIA-DDR test, pisahkan UNSAFE_MISMATCH (grade-shifting,
seharusnya DISUPRES) dari CLINICALLY_CLOSE_MISMATCH (within-grade, seharusnya DIPERTAHANKAN),
lalu ukur AUROC tiap sinyal dengan UNSAFE sebagai kelas positif.

Arah yang benar berbeda per sinyal, dan ini harus dibaca hati-hati:
  * SDCA: tinggi = berbahaya, jadi sinyal yang informatif memberi AUROC MENDEKATI 1.
  * skor detektor, skor fusi, skor gate: tinggi = pertahankan, jadi sinyal yang sadar klinis
    memberi UNSAFE nilai RENDAH, yaitu AUROC MENDEKATI 0.
  * AUROC 0.5 berarti tidak ada informasi sama sekali, ke arah mana pun.
Karena itu yang dilaporkan juga |AUROC - 0.5|, yaitu banyaknya informasi biaya klinis yang dibawa
sinyal itu, bebas arah.

CI: bootstrap berpasangan image-clustered 2.000 resampel (seed 0), sama dengan tabel lain.
Keluaran: _septem/scripts_optionB/clinical_cost_fusion.json. Hanya membaca.
"""
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.eval_fp_reduction_benchmark import apply_mlp, fit_mlp, load_csv  # noqa: E402
from tools.make_indomain_results import (  # noqa: E402
    N_BOOT, POSITIVE, SEEDS, T_DEP, V_DEP, gate_matrix,
)

OUT = ROOT / "_septem/scripts_optionB/clinical_cost_fusion.json"
CACHE = {"primary": ROOT / "outputs/detector_cache/primary/oiaddr",
         "flip": ROOT / "outputs/detector_cache/secondary/oiaddr",
         "v2": ROOT / "outputs/secondary_v2/oiaddr"}
IOU_MIN = 0.15
EV = ("ev_sas", "ev_sdca", "ev_aps", "ev_pss", "ev_sds", "ev_sds_nsrc")
UNSAFE, CLOSE = "UNSAFE_MISMATCH", "CLINICALLY_CLOSE_MISMATCH"
t0 = time.perf_counter()


def log(m):
    print(f"[cc] {m} ({time.perf_counter() - t0:.0f}s)", flush=True)


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
    return (s_p + s_f + s_v) / 3.0


def categories(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as fh:
        return np.array([r.get("category", "") for r in csv.DictReader(fh)])


def boot_auroc(score, y, img, n_boot=N_BOOT, seed=0):
    """CI image-clustered untuk AUROC; kelompok yang jadi satu kelas saja dilewati."""
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
        return {"ci95": [None, None], "n_valid": 0}
    lo, hi = (float(x) for x in np.percentile(vals, [2.5, 97.5]))
    return {"ci95": [lo, hi], "n_valid": len(vals), "mean": float(np.mean(vals))}


va, te = load_csv(V_DEP, POSITIVE), load_csv(T_DEP, POSITIVE)
yv = va["label"].astype(np.float64)
cat = categories(T_DEP)
if len(cat) != len(te["label"]):
    raise SystemExit("BERHENTI: panjang kolom category tidak sama dengan CSV")

sel = np.where((cat == UNSAFE) | (cat == CLOSE))[0]
y = (cat[sel] == UNSAFE).astype(int)          # 1 = berbahaya (grade-shifting)
img = te["image_id"][sel]
log(f"UNSAFE n={int(y.sum())}, CLINICALLY-CLOSE n={int((1 - y).sum())}, total {len(y)}")

fv, ft = fused_three(V_DEP, "valid"), fused_three(T_DEP, "test")
log("skor fusi tiga pass dihitung untuk valid dan test")

signals = {
    "detector_score": te["pred_score"][sel],
    "kg_sdca": te["ev_sdca"][sel],
    "fused_three_pass": ft[sel],
}

# Titik per-deteksi diekspor supaya figur ROC tidak perlu menghitung ulang skor fusi, yang
# berarti membaca ribuan berkas cache hanya untuk menggambar satu kurva. Tanpa kurva fusi,
# Gambar 10 memperlihatkan dua sinyal sementara Tabel 13 di bawahnya memuat tiga.
PTS = ROOT / "_septem/scripts_optionB/clinical_cost_points.csv"
with open(PTS, "w", newline="", encoding="utf-8") as fh:
    _w = csv.writer(fh)
    _w.writerow(["unsafe", "detector_score", "fused_three_pass", "kg_sdca"])
    for _i in range(len(y)):
        _w.writerow([int(y[_i]), f"{signals['detector_score'][_i]:.6f}",
                     f"{signals['fused_three_pass'][_i]:.6f}", f"{signals['kg_sdca'][_i]:.6f}"])
log(f"titik per-deteksi -> {PTS.name} ({len(y)} baris)")

# gate: dipasang pada valid dengan protokol naskah, lalu skornya diuji pada dua kategori itu
gate_specs = {"full_gate": None, "gate_fused": "fused"}
res = {"n_unsafe": int(y.sum()), "n_close": int((1 - y).sum()),
       "note": "AUROC dengan UNSAFE sebagai kelas positif; 0.5 = tanpa informasi",
       "signals": {}, "gates": {}}

for name, s in signals.items():
    a = float(roc_auc_score(y, s))
    b = boot_auroc(np.asarray(s, dtype=np.float64), y, img)
    arr = np.asarray(s, dtype=np.float64)
    # Tabel 13 menulis mean$\pm$sd tetapi tidak mencatat konvensi sd-nya. Kedua konvensi
    # disimpan supaya kolom baru memakai konvensi yang SAMA dengan dua kolom lama; yang
    # benar adalah yang mereproduksi 0.309 dan 0.274 pada baris skor detektor.
    res["signals"][name] = {
        "auroc": a, "info_abs": abs(a - 0.5), "ci95": b["ci95"],
        "mean_unsafe": float(arr[y == 1].mean()),
        "mean_close": float(arr[y == 0].mean()),
        "sd_unsafe_ddof0": float(arr[y == 1].std(ddof=0)),
        "sd_unsafe_ddof1": float(arr[y == 1].std(ddof=1)),
        "sd_close_ddof0": float(arr[y == 0].std(ddof=0)),
        "sd_close_ddof1": float(arr[y == 0].std(ddof=1)),
    }
    log(f"{name:18s} AUROC {a:.4f} |info| {abs(a - 0.5):.4f} "
        f"CI [{b['ci95'][0]:.4f}, {b['ci95'][1]:.4f}] | "
        f"unsafe {arr[y == 1].mean():.3f}+-{arr[y == 1].std(ddof=0):.3f}/"
        f"{arr[y == 1].std(ddof=1):.3f} close {arr[y == 0].mean():.3f}+-"
        f"{arr[y == 0].std(ddof=0):.3f}/{arr[y == 0].std(ddof=1):.3f}")

if "--signals-only" in sys.argv:
    # Menulis ulang hanya bagian sinyal; blok gate yang sudah ada dipertahankan apa adanya
    # supaya lari cepat ini tidak menghapus hasil 10 pemasangan MLP yang mahal.
    if OUT.is_file():
        prev = json.loads(OUT.read_text(encoding="utf-8"))
        res["gates"] = prev.get("gates", {})
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    log(f"signals-only selesai -> {OUT}")
    raise SystemExit(0)

for gname, mode in gate_specs.items():
    Xv = (np.stack([va[c] for c in EV] + [fv], 1) if mode == "fused"
          else gate_matrix(va, "full_gate", None))
    Xt = (np.stack([te[c] for c in EV] + [ft], 1) if mode == "fused"
          else gate_matrix(te, "full_gate", None))
    aur = []
    for seed in SEEDS:
        m = fit_mlp(Xv, yv, seed=seed)
        st = apply_mlp(Xt, m)[sel]
        aur.append(float(roc_auc_score(y, st)))
        log(f"{gname} seed {seed}: AUROC {aur[-1]:.4f}")
    res["gates"][gname] = {"auroc_per_seed": aur, "auroc_mean": statistics.mean(aur),
                           "auroc_sd": statistics.stdev(aur),
                           "info_abs_mean": statistics.mean(abs(a - 0.5) for a in aur)}

OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
log(f"selesai -> {OUT}")
