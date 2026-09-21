"""baselines_nokg.py — pembanding tanpa knowledge graph pada kotak kandidat yang sama.

Pertanyaan reviewer yang ingin dijawab: apakah gain gate hanya berasal dari dua pass tambahan
(flip test-time augmentation dan detektor kedua), bukan dari evidence knowledge graph?

Semua skor dihitung pada boks kandidat detektor primer yang SAMA (cache beku), split test OIA-DDR,
label governance_retain, AUPRC = average precision, bootstrap image-clustered 2.000 resampel
(seed 0) seperti Tabel 2 dan 8.

Pembanding:
  m0              skor detektor primer
  tta_mean0       rata-rata skor primer dan pass flip; kotak tanpa pasangan dianggap 0
  tta_meanavail   rata-rata atas pass yang punya kotak cocok
  ens2_mean0      rata-rata skor primer dan detektor kedua; tanpa pasangan dianggap 0
  ens2_meanavail  rata-rata atas sumber yang tersedia
  ens3_mean0      rata-rata primer + flip + detektor kedua; tanpa pasangan dianggap 0
  nokg_gate       MLP protokol naskah pada [SAS, PSS, ev_sds, ev_sds_nsrc, skor] (tanpa SDCA dan APS)
Pembanding internal: kg_gate dan full_gate dipasang ulang dengan protokol yang sama (dicek terhadap
outputs/0506/indomain_5seed/indomain_results.json).

Pencocokan kotak: IoU >= 0.15, kelas diabaikan, satu kotak sumber boleh dipakai beberapa kotak primer
(sama dengan resep SAS/PSS pipeline).
Keluaran: _septem/scripts_optionB/baselines_nokg.json dan tabel ringkas ke stdout. Hanya membaca.
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
    N_BOOT, POSITIVE, SEEDS, TP_RET, T_DEP, V_DEP, gate_matrix, paired_bootstrap_ap,
)

OUT = ROOT / "_septem/scripts_optionB/baselines_nokg.json"
CACHE = {
    "primary": ROOT / "outputs/detector_cache/primary/oiaddr",
    "flip": ROOT / "outputs/detector_cache/secondary/oiaddr",
    "v2": ROOT / "outputs/secondary_v2/oiaddr",
}
IOU_MIN = 0.15
NOKG = ("ev_sas", "ev_pss", "ev_sds", "ev_sds_nsrc")
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


def cache_entries(root: Path, split: str, image_id: str):
    """[(bbox_id, bbox, score)] dari satu berkas cache."""
    f = root / split / f"{image_id}.json"
    if not f.is_file():
        return []
    out = []
    for e in json.loads(f.read_text(encoding="utf-8")):
        raw = e.get("detector_raw") or e.get("r2a_raw") or {}
        if "bbox" in raw and "score" in raw:
            out.append((e.get("bbox_id"), raw["bbox"], float(raw["score"])))
    return out


def cache_boxes(root: Path, split: str, image_id: str):
    return [(b, s) for _i, b, s in cache_entries(root, split, image_id)]


def fused_scores(csv_path: Path, split: str):
    """Skor fusi per baris CSV (urutan baris dipertahankan)."""
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = [(r["bbox_id"], r["image_id"], float(r["pred_score"])) for r in csv.DictReader(fh)]
    by_img = {}
    for i, (_bid, img, _s) in enumerate(rows):
        by_img.setdefault(img, []).append(i)
    n = len(rows)
    s_p = np.array([r[2] for r in rows], dtype=np.float64)
    s_flip, s_v2 = np.zeros(n), np.zeros(n)
    m_flip, m_v2 = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    missing_primary, missing_box = 0, 0
    for img, idx in by_img.items():
        # kotak primer dicari lewat bbox_id, bukan urutan baris
        prim = {bid: bb for bid, bb, _s in cache_entries(CACHE["primary"], split, img)}
        if not prim:
            missing_primary += 1
            continue
        others = {k: cache_boxes(CACHE[k], split, img) for k in ("flip", "v2")}
        for i in idx:
            pb = prim.get(rows[i][0])
            if pb is None:
                missing_box += 1
                continue
            for key, sc_arr, mask in (("flip", s_flip, m_flip), ("v2", s_v2, m_v2)):
                best, best_iou = 0.0, 0.0
                for ob, os_ in others[key]:
                    v = iou(pb, ob)
                    if v >= IOU_MIN and v > best_iou:
                        best_iou, best = v, os_
                if best_iou >= IOU_MIN:
                    sc_arr[i], mask[i] = best, True
    out = {
        "m0": s_p,
        "tta_mean0": (s_p + s_flip) / 2.0,
        "tta_meanavail": np.where(m_flip, (s_p + s_flip) / 2.0, s_p),
        "ens2_mean0": (s_p + s_v2) / 2.0,
        "ens2_meanavail": np.where(m_v2, (s_p + s_v2) / 2.0, s_p),
        "ens3_mean0": (s_p + s_flip + s_v2) / 3.0,
    }
    cov = {"flip_matched": float(m_flip.mean()), "v2_matched": float(m_v2.mean()),
           "images_without_primary_cache": missing_primary, "rows_without_primary_box": missing_box}
    return out, cov


def nokg_matrix(d):
    return np.stack([d[c] for c in NOKG] + [d["pred_score"]], 1)


va, te = load_csv(V_DEP, POSITIVE), load_csv(T_DEP, POSITIVE)
yv, yt, img = va["label"], te["label"], te["image_id"]
m0 = float(average_precision_score(yt, te["pred_score"]))
print(f"[base] test {len(yt)} kotak, M0 AP {m0:.5f} ({time.perf_counter() - t0:.0f}s)", flush=True)

test_fused, cov_test = fused_scores(T_DEP, "test")
print(f"[base] cakupan test: flip {cov_test['flip_matched']:.3f}, v2 {cov_test['v2_matched']:.3f} "
      f"({time.perf_counter() - t0:.0f}s)", flush=True)
valid_fused, cov_valid = (None, None)
if (CACHE["flip"] / "valid").is_dir() and (CACHE["v2"] / "valid").is_dir():
    valid_fused, cov_valid = fused_scores(V_DEP, "valid")
    print(f"[base] cakupan valid: flip {cov_valid['flip_matched']:.3f}, v2 {cov_valid['v2_matched']:.3f}",
          flush=True)
else:
    print("[base] cache valid tidak lengkap: titik kerja untuk baseline fusi dilewati", flush=True)

res = {"m0_ap": m0, "coverage_test": cov_test, "coverage_valid": cov_valid, "fusion": {}, "gates": {}}
for name, s in test_fused.items():
    if name == "m0":
        continue
    rec = {"ap": float(average_precision_score(yt, s))}
    if valid_fused is not None:
        tau = threshold_for_tp_ret(valid_fused[name], yv, TP_RET)
        ev = eval_at_threshold(s, yt, tau)
        rec.update({"tau": tau, "tp_ret": ev["tp_ret"], "fp_supp": ev["fp_supp"],
                    "precision": ev["precision_after"]})
    res["fusion"][name] = rec
    print(f"[base] {name:15s} AP {rec['ap']:.5f} ({time.perf_counter() - t0:.0f}s)", flush=True)

gate_scores = {}
for name in ("nokg_gate", "kg_gate", "full_gate"):
    aps, ops = [], []
    gate_scores[name] = {}
    for seed in SEEDS:
        if name == "nokg_gate":
            Xv, Xt = nokg_matrix(va), nokg_matrix(te)
        else:
            Xv, Xt = gate_matrix(va, name, None), gate_matrix(te, name, None)
        m = fit_mlp(Xv, yv.astype(np.float64), seed=seed)
        sv, st = apply_mlp(Xv, m), apply_mlp(Xt, m)
        gate_scores[name][seed] = st
        aps.append(float(average_precision_score(yt, st)))
        ev = eval_at_threshold(st, yt, threshold_for_tp_ret(sv, yv, TP_RET))
        ops.append((ev["tp_ret"], ev["fp_supp"], ev["precision_after"]))
        print(f"[base] {name} seed {seed}: AP {aps[-1]:.5f} ({time.perf_counter() - t0:.0f}s)", flush=True)
    res["gates"][name] = {
        "ap_per_seed": aps, "ap_mean": statistics.mean(aps), "ap_sd": statistics.stdev(aps),
        "gain_vs_m0_mean": statistics.mean(a - m0 for a in aps),
        "tp_ret_mean": statistics.mean(o[0] for o in ops),
        "fp_supp_mean": statistics.mean(o[1] for o in ops),
        "precision_mean": statistics.mean(o[2] for o in ops)}

# bootstrap: setiap pembanding fusi vs M0, dan gate penuh vs pembanding fusi terbaik (per seed)
boot = {}
for name, s in test_fused.items():
    if name == "m0":
        continue
    b = paired_bootstrap_ap(s, test_fused["m0"], yt, img, N_BOOT, 0)
    b["gain_point"] = float(average_precision_score(yt, s) - m0)
    boot[f"{name}_minus_m0"] = b
    print(f"[base] boot {name}-m0: {b['gain_point']:+.4f} [{b['gain_ci95'][0]:+.4f}, {b['gain_ci95'][1]:+.4f}] "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)
best_fusion = max((k for k in test_fused if k != "m0"), key=lambda k: res["fusion"][k]["ap"])
res["best_fusion"] = best_fusion
for gname in ("full_gate", "nokg_gate"):
    rows = []
    for seed in SEEDS:
        b = paired_bootstrap_ap(gate_scores[gname][seed], test_fused[best_fusion], yt, img, N_BOOT, 0)
        b["seed"] = seed
        b["gain_point"] = float(average_precision_score(yt, gate_scores[gname][seed])
                                - res["fusion"][best_fusion]["ap"])
        rows.append(b)
        print(f"[base] boot {gname} seed {seed} - {best_fusion}: {b['gain_point']:+.4f} "
              f"[{b['gain_ci95'][0]:+.4f}, {b['gain_ci95'][1]:+.4f}] ({time.perf_counter() - t0:.0f}s)",
              flush=True)
    boot[f"{gname}_minus_{best_fusion}"] = rows
res["bootstrap"] = boot
OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
print(f"[base] selesai {time.perf_counter() - t0:.0f}s -> {OUT}")
