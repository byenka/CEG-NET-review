"""fusion_external.py — apakah gating masih menambah nilai DI ATAS fusi skor, di luar OIA-DDR?

Lanjutan dari baselines_nokg.py dan gate_on_fusion.py, yang menjawab pertanyaan itu hanya pada
OIA-DDR. Reviewer akan menyebut hasil satu dataset sebagai kebetulan, jadi perbandingan yang sama
diulang pada tiga kohort eksternal dengan protokol zero-shot naskah.

Protokol (sama persis dengan T12 dan naskah):
  * gate dipasang SEKALI pada OIA-DDR valid (fit_mlp, 16 tanh, 1500 iterasi, lr 0.1), 5 seed,
    lalu diterapkan tanpa perubahan ke setiap kohort. Tidak ada refit per kohort.
  * AUPRC = average precision (sklearn), bebas urutan skor kembar.
  * bootstrap berpasangan image-clustered, 2.000 resampel, bootstrap_seed 0, per seed gate.
  * tidak ada titik kerja (tau*) untuk kohort eksternal: naskah melaporkan kohort eksternal
    sebagai hasil PERINGKAT saja, karena tau* dari domain sumber tidak transfer.

Batas keras yang ditemukan pada cache, BUKAN pilihan desain:
  * secondary_v2 (detektor kedua) hanya ada untuk oiaddr, idrid dan tjdr. FGADR tidak punya.
    Karena itu FGADR hanya bisa fusi DUA pass (primer + flip), dan skor fusi di sana bukan
    besaran yang sama dengan yang dipakai memasang gate_fused pada OIA-DDR valid (tiga pass).
    gate_fused karena itu TIDAK dilaporkan untuk FGADR; yang dilaporkan hanya baseline fusi
    dua pass dan gate penuh, sebagai deskriptif.
  * split cache mengikuti CSV resmi tiap kohort: FGADR memakai split train (seluruh kohort
    dipakai sebagai evaluasi zero-shot), tiga kohort lain memakai test.

Keluaran: _septem/scripts_optionB/fusion_external.json. Hanya membaca.
"""
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.eval_fp_reduction_benchmark import apply_mlp, fit_mlp, load_csv  # noqa: E402
from tools.make_indomain_results import (  # noqa: E402
    N_BOOT, POSITIVE, SEEDS, V_DEP, gate_matrix, paired_bootstrap_ap,
)

REF = ROOT / "outputs/0506/evsds_pss/domain_shift_5seed.json"
OUT = ROOT / "_septem/scripts_optionB/fusion_external.json"
IOU_MIN = 0.15
EV = ("ev_sas", "ev_sdca", "ev_aps", "ev_pss", "ev_sds", "ev_sds_nsrc")
KEY = {"OIA-DDR": "oiaddr", "IDRiD": "idrid", "TJDR": "tjdr", "FGADR": "fgadr"}
CACHE = {"primary": ROOT / "outputs/detector_cache/primary",
         "flip": ROOT / "outputs/detector_cache/secondary",
         "v2": ROOT / "outputs/secondary_v2"}
# FGADR jauh lebih besar (510k baris); bootstrap di sana dibatasi ke satu pembanding.
t0 = time.perf_counter()


def log(msg):
    print(f"[ext] {msg} ({time.perf_counter() - t0:.0f}s)", flush=True)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def cache_entries(root: Path, split: str, image_id: str):
    f = root / split / f"{image_id}.json"
    if not f.is_file():
        return []
    out = []
    for e in json.loads(f.read_text(encoding="utf-8")):
        raw = e.get("detector_raw") or e.get("r2a_raw") or {}
        if "bbox" in raw and "score" in raw:
            out.append((e.get("bbox_id"), raw["bbox"], float(raw["score"])))
    return out


def fused_scores(csv_path: Path, key: str, split: str, use_v2: bool):
    """Skor fusi per baris CSV. Kotak primer dicocokkan lewat bbox_id, sumber lain lewat IoU."""
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = [(r["bbox_id"], r["image_id"], float(r["pred_score"])) for r in csv.DictReader(fh)]
    by_img = {}
    for i, (_b, img, _s) in enumerate(rows):
        by_img.setdefault(img, []).append(i)
    n = len(rows)
    s_p = np.array([r[2] for r in rows], dtype=np.float64)
    s_f, s_v = np.zeros(n), np.zeros(n)
    m_f, m_v = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    no_prim, no_box = 0, 0
    srcs = [("flip", s_f, m_f)] + ([("v2", s_v, m_v)] if use_v2 else [])
    for img, idx in by_img.items():
        prim = {bid: bb for bid, bb, _s in cache_entries(CACHE["primary"] / key, split, img)}
        if not prim:
            no_prim += 1
            continue
        others = {k: [(bb, sc) for _i, bb, sc in cache_entries(CACHE[k] / key, split, img)]
                  for k, _a, _m in srcs}
        for i in idx:
            pb = prim.get(rows[i][0])
            if pb is None:
                no_box += 1
                continue
            for k, arr, mask in srcs:
                best, best_iou = 0.0, 0.0
                for ob, os_ in others[k]:
                    v = iou(pb, ob)
                    if v >= IOU_MIN and v > best_iou:
                        best_iou, best = v, os_
                if best_iou >= IOU_MIN:
                    arr[i], mask[i] = best, True
    npass = 3 if use_v2 else 2
    fused = (s_p + s_f + s_v) / npass if use_v2 else (s_p + s_f) / 2.0
    cov = {"n_pass": npass, "flip_matched": float(m_f.mean()),
           "v2_matched": float(m_v.mean()) if use_v2 else None,
           "images_without_primary_cache": no_prim, "rows_without_primary_box": no_box}
    return fused, cov


ref = json.loads(REF.read_text(encoding="utf-8"))
cohorts = [{"name": c["name"], "csv": ROOT / c["csv"]} for c in ref["cohorts"]]
for c in cohorts:
    c["key"] = KEY[c["name"]]
    c["split"] = c["csv"].parent.name
    c["use_v2"] = (CACHE["v2"] / c["key"] / c["split"]).is_dir()
    log(f"{c['name']}: split={c['split']} v2={'ada' if c['use_v2'] else 'TIDAK ADA'}")

# ── gate dipasang sekali pada OIA-DDR valid ──────────────────────────────────
va = load_csv(V_DEP, POSITIVE)
yv = va["label"].astype(np.float64)
fv, cov_v = fused_scores(V_DEP, "oiaddr", "valid", True)
log(f"valid fusi tiga pass: flip {cov_v['flip_matched']:.3f} v2 {cov_v['v2_matched']:.3f}")
Xv_fused = np.stack([va[c] for c in EV] + [fv], 1)
Xv_full = gate_matrix(va, "full_gate", None)
models = {"gate_fused": [fit_mlp(Xv_fused, yv, seed=s) for s in SEEDS],
          "full_gate": [fit_mlp(Xv_full, yv, seed=s) for s in SEEDS]}
log("gate dipasang pada OIA-DDR valid (5 seed, dua varian)")

res = {"protocol": {"fit_split": "OIA-DDR valid", "seeds": list(SEEDS), "n_boot": N_BOOT,
                    "estimator": "average_precision", "iou_min": IOU_MIN,
                    "note": "tidak ada tau*; kohort eksternal dilaporkan sebagai peringkat"},
       "cohorts": []}

for c in cohorts:
    te = load_csv(c["csv"], POSITIVE)
    y, img = te["label"], te["image_id"]
    m0 = float(average_precision_score(y, te["pred_score"]))
    fused, cov = fused_scores(c["csv"], c["key"], c["split"], c["use_v2"])
    ap_fused = float(average_precision_score(y, fused))
    log(f"{c['name']}: n={len(y)} M0 {m0:.5f} fusi{cov['n_pass']} {ap_fused:.5f} "
        f"(flip {cov['flip_matched']:.3f})")

    rec = {"name": c["name"], "key": c["key"], "split": c["split"], "n_rows": int(len(y)),
           "n_pass_fusion": cov["n_pass"], "coverage": cov, "m0_ap": m0, "fused_ap": ap_fused,
           "fused_minus_m0": ap_fused - m0, "gates": {}, "bootstrap": {}}

    # gate_fused hanya sah bila skor fusi memakai jumlah pass yang sama dengan saat fitting
    variants = ["full_gate"] + (["gate_fused"] if c["use_v2"] else [])
    scores = {}
    for name in variants:
        X = (np.stack([te[k] for k in EV] + [fused], 1) if name == "gate_fused"
             else gate_matrix(te, "full_gate", None))
        aps = []
        scores[name] = {}
        for i, s in enumerate(SEEDS):
            st = apply_mlp(X, models[name][i])
            scores[name][s] = st
            aps.append(float(average_precision_score(y, st)))
        rec["gates"][name] = {
            "ap_per_seed": aps, "ap_mean": statistics.mean(aps), "ap_sd": statistics.stdev(aps),
            "gain_vs_fused_mean": statistics.mean(a - ap_fused for a in aps),
            "n_gain_vs_fused_pos": sum(1 for a in aps if a > ap_fused),
            "gain_vs_m0_mean": statistics.mean(a - m0 for a in aps)}
        log(f"{c['name']} {name}: AP {statistics.mean(aps):.5f} +- {statistics.stdev(aps):.5f} "
            f"| vs fusi {statistics.mean(a - ap_fused for a in aps):+.5f} "
            f"({sum(1 for a in aps if a > ap_fused)}/5 seed)")

    # bootstrap: prioritas gate_fused vs fusi; full_gate vs fusi hanya bila kohort tidak besar
    todo = [v for v in ("gate_fused", "full_gate") if v in scores]
    if rec["n_rows"] > 100000:
        todo = [v for v in todo if v == ("gate_fused" if c["use_v2"] else "full_gate")]
        log(f"{c['name']}: {rec['n_rows']} baris, bootstrap dibatasi ke {todo}")
    for name in todo:
        rows = []
        for i, s in enumerate(SEEDS):
            b = paired_bootstrap_ap(scores[name][s], fused, y, img, N_BOOT, 0)
            b["seed"] = s
            b["gain_point"] = float(rec["gates"][name]["ap_per_seed"][i] - ap_fused)
            rows.append(b)
            log(f"{c['name']} boot {name} seed {s} - fusi: {b['gain_point']:+.4f} "
                f"[{b['gain_ci95'][0]:+.4f}, {b['gain_ci95'][1]:+.4f}]")
        rec["bootstrap"][f"{name}_minus_fused"] = rows
        rec["gates"][name]["n_ci_above_0"] = sum(1 for b in rows if b["gain_ci95"][0] > 1e-6)
        rec["gates"][name]["n_ci_below_0"] = sum(1 for b in rows if b["gain_ci95"][1] < -1e-6)
    res["cohorts"].append(rec)
    del te

OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
log(f"selesai -> {OUT}")
