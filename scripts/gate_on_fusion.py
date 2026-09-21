"""gate_on_fusion.py — apakah evidence masih menambah nilai DI ATAS ensembling skor?

Latar: baselines_nokg.py menunjukkan fusi skor tiga pass (primer + flip + detektor kedua, kotak
tanpa pasangan dianggap 0) mencapai AUPRC lebih tinggi daripada gate penuh. Pertanyaan lanjutan
yang akan diajukan reviewer: apakah gate hanya menduplikasi ensembling, atau tetap menambah nilai
bila skor fusi itu sendiri yang menjadi masukannya?

Varian yang diuji (protokol sama dengan naskah: fit pada OIA-DDR valid, 5 seed, AP, bootstrap
image-clustered 2.000 seed 0; tau* dari valid pada retensi TP 0.80):
  fused              skor fusi tiga pass saja (pembanding)
  gate_fused         MLP pada [SAS, SDCA, APS, PSS, ev_sds, ev_sds_nsrc, skor fusi]
  gate_fused_raw     idem, ditambah skor detektor mentah
Pembanding bootstrap: setiap varian gate vs skor fusi, per seed.

Fungsi pencocokan kotak disalin dari baselines_nokg.py supaya skrip ini berdiri sendiri
(mengimpor modul itu akan menjalankan ulang seluruh perhitungannya).
Keluaran: _septem/scripts_optionB/gate_on_fusion.json. Hanya membaca.
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
from tools.eval_fp_reduction_benchmark import (  # noqa: E402
    apply_mlp, eval_at_threshold, fit_mlp, load_csv, threshold_for_tp_ret,
)
from tools.make_indomain_results import (  # noqa: E402
    N_BOOT, POSITIVE, SEEDS, TP_RET, T_DEP, V_DEP, paired_bootstrap_ap,
)

OUT = ROOT / "_septem/scripts_optionB/gate_on_fusion.json"
CACHE = {"primary": ROOT / "outputs/detector_cache/primary/oiaddr",
         "flip": ROOT / "outputs/detector_cache/secondary/oiaddr",
         "v2": ROOT / "outputs/secondary_v2/oiaddr"}
IOU_MIN = 0.15
EV = ("ev_sas", "ev_sdca", "ev_aps", "ev_pss", "ev_sds", "ev_sds_nsrc")
t0 = time.perf_counter()


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
    """Rata-rata skor primer, flip, dan detektor kedua; kotak tanpa pasangan dianggap 0."""
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


va, te = load_csv(V_DEP, POSITIVE), load_csv(T_DEP, POSITIVE)
yv, yt, img = va["label"], te["label"], te["image_id"]
fv, ft = fused_three(V_DEP, "valid"), fused_three(T_DEP, "test")
ap_f = float(average_precision_score(yt, ft))
tau_f = threshold_for_tp_ret(fv, yv, TP_RET)
ev_f = eval_at_threshold(ft, yt, tau_f)
print(f"[fuse] skor fusi: AP {ap_f:.5f} TP {ev_f['tp_ret']:.4f} FP {ev_f['fp_supp']:.4f} "
      f"({time.perf_counter() - t0:.0f}s)", flush=True)

res = {"fused": {"ap": ap_f, "tp_ret": ev_f["tp_ret"], "fp_supp": ev_f["fp_supp"],
                 "precision": ev_f["precision_after"], "tau": tau_f},
       "m0_ap": float(average_precision_score(yt, te["pred_score"])), "gates": {}, "bootstrap": {}}


def matrix(d, fused, with_raw):
    cols = [d[c] for c in EV] + [fused]
    if with_raw:
        cols.append(d["pred_score"])
    return np.stack(cols, 1)


for name, with_raw in (("gate_fused", False), ("gate_fused_raw", True)):
    aps, ops, scores = [], [], {}
    Xv, Xt = matrix(va, fv, with_raw), matrix(te, ft, with_raw)
    for seed in SEEDS:
        m = fit_mlp(Xv, yv.astype(np.float64), seed=seed)
        sv, st = apply_mlp(Xv, m), apply_mlp(Xt, m)
        scores[seed] = st
        aps.append(float(average_precision_score(yt, st)))
        e = eval_at_threshold(st, yt, threshold_for_tp_ret(sv, yv, TP_RET))
        ops.append((e["tp_ret"], e["fp_supp"], e["precision_after"]))
        print(f"[fuse] {name} seed {seed}: AP {aps[-1]:.5f} ({time.perf_counter() - t0:.0f}s)", flush=True)
    res["gates"][name] = {"ap_per_seed": aps, "ap_mean": statistics.mean(aps),
                          "ap_sd": statistics.stdev(aps),
                          "gain_vs_fused_mean": statistics.mean(a - ap_f for a in aps),
                          "n_gain_vs_fused_pos": sum(1 for a in aps if a > ap_f),
                          "tp_ret_mean": statistics.mean(o[0] for o in ops),
                          "fp_supp_mean": statistics.mean(o[1] for o in ops),
                          "precision_mean": statistics.mean(o[2] for o in ops)}
    rows = []
    for seed in SEEDS:
        b = paired_bootstrap_ap(scores[seed], ft, yt, img, N_BOOT, 0)
        b["seed"] = seed
        b["gain_point"] = float(aps[seed] - ap_f) if isinstance(seed, int) and seed < len(aps) else None
        rows.append(b)
        print(f"[fuse] boot {name} seed {seed} - fused: {b['gain_point']:+.4f} "
              f"[{b['gain_ci95'][0]:+.4f}, {b['gain_ci95'][1]:+.4f}] ({time.perf_counter() - t0:.0f}s)",
              flush=True)
    res["bootstrap"][f"{name}_minus_fused"] = rows
    res["gates"][name]["n_ci_above_0"] = sum(1 for b in rows if b["gain_ci95"][0] > 1e-6)

OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
print(f"[fuse] selesai {time.perf_counter() - t0:.0f}s -> {OUT}")
