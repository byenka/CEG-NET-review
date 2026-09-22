#!/usr/bin/env python
"""eval_multidetector_ensemble.py — baseline ensemble lima detektor dari cache.

Tabel 8 membandingkan gate terhadap ensemble tiga pass, yaitu pass primer, pass flip, dan satu
detektor sekunder. Reviewer yang menanyakan deep ensemble berhak menanyakan pembanding ensemble
yang lebih kuat. Lima detektor lain sudah terlatih dan cache deteksinya lengkap untuk OIA-DDR
valid dan test, sehingga pembanding itu dapat dihitung tanpa menjalankan ulang model apa pun.

Ini BUKAN deep ensemble dalam arti Lakshminarayanan (arsitektur sama, inisialisasi berbeda);
kelimanya arsitektur berbeda. Deep ensemble sejati menuntut pelatihan ulang beberapa kali dan
tidak tersedia dari arsip. Yang ini adalah ensemble heterogen, dan pada tugas peringkat ia
pembanding yang lebih kuat daripada rata-rata tiga pass.

Skor ensemble didefinisikan pada kotak kandidat detektor PRIMER, mengikuti konvensi naskah:
tiap anggota dicocokkan ke kotak primer pada IoU minimal 0.15, dan anggota yang tidak punya
kotak yang cocok menyumbang skor nol.

Varian yang dilaporkan:
  E5      rata-rata lima detektor detcmp
  E6      rata-rata primer ditambah lima detektor detcmp
  E6+bukti gate penuh dengan skor E6 menggantikan masukan skor mentah

Protokol identik tools/eval_fp_reduction_benchmark.py: target governance_retain, gate dipasang
pada OIA-DDR validasi, lima benih, bootstrap berklaster-citra 2.000 resample benih 0.

Keluaran: outputs/0506/selective_conformal/multidetector_ensemble.json
Jalankan dari root repo:
  python tools/eval_multidetector_ensemble.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.eval_fp_reduction_benchmark import (  # noqa: E402
    EVIDENCE_FEATURES, apply_mlp, auprc, fit_mlp, load_csv,
)
from tools.eval_selective_conformal_baselines import (  # noqa: E402
    ALPHA, BOOT_SEED, N_BOOT, aurc, build_buckets, conformal_threshold,
    coverage_at, e_aurc,
)

BASE = ROOT / "outputs/0506"
VALID_CSV = BASE / "evsds_v2sec/oiaddr/valid/governance_features_valid.csv"
TEST_CSV = BASE / "evsds_v2sec/oiaddr/test/governance_features_test.csv"
PRIMARY = ROOT / "outputs/detector_cache/primary/oiaddr"
MEMBERS = ["fasterrcnn", "fasterrcnn_v2", "fcos", "retinanet", "retinanet_v2"]
DETCMP = ROOT / "outputs/detcmp"
OUT = BASE / "selective_conformal/multidetector_ensemble.json"

IOU_MATCH = 0.15
SEEDS = (0, 1, 2, 3, 4)
TASK = "governance_retain"


def load_boxes(path: Path):
    """{image_id: (bbox_ids, Nx4 array, scores)} dari satu direktori cache split."""
    out = {}
    for f in sorted(path.glob("*.json")):
        rows = json.loads(f.read_text(encoding="utf-8"))
        ids, bb, sc = [], [], []
        for r in rows:
            raw = r.get("detector_raw") or {}
            box = raw.get("bbox")
            if not box:
                continue
            ids.append(r["bbox_id"])
            bb.append(box)
            sc.append(float(raw.get("score", 0.0)))
        out[f.stem] = (ids, np.asarray(bb, dtype=np.float64).reshape(-1, 4),
                       np.asarray(sc, dtype=np.float64))
    return out


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    ar_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = ar_a[:, None] + ar_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def member_scores(split: str, csv_ids: np.ndarray) -> dict:
    """Skor tiap anggota pada kotak primer, selaras urutan csv_ids."""
    prim = load_boxes(PRIMARY / split)
    # peta bbox_id primer -> (image, indeks baris)
    loc = {}
    for img, (ids, _, _) in prim.items():
        for j, bid in enumerate(ids):
            loc[bid] = (img, j)
    missing = [b for b in csv_ids if b not in loc]
    if missing:
        raise SystemExit(f"[ens] {len(missing)} bbox_id CSV tidak ada di cache primer {split}")

    out = {}
    prim_sc = np.array([prim[loc[b][0]][2][loc[b][1]] for b in csv_ids])
    out["primary"] = prim_sc

    for m in MEMBERS:
        cache = DETCMP / f"{m}_cache" / split
        if not cache.is_dir():
            raise SystemExit(f"[ens] cache tidak ada: {cache}")
        other = load_boxes(cache)
        per_img = {}
        n_img_missing = 0
        for img, (_, pb, _) in prim.items():
            if img not in other:
                n_img_missing += 1
                per_img[img] = np.zeros(pb.shape[0])
                continue
            _, ob, osc = other[img]
            M = iou_matrix(pb, ob)
            best = M.argmax(1) if ob.shape[0] else np.zeros(pb.shape[0], dtype=int)
            bi = M.max(1) if ob.shape[0] else np.zeros(pb.shape[0])
            s = np.where(bi >= IOU_MATCH, osc[best] if ob.shape[0] else 0.0, 0.0)
            per_img[img] = s
        out[m] = np.array([per_img[loc[b][0]][loc[b][1]] for b in csv_ids])
        cov = float((out[m] > 0).mean())
        print(f"    {split:5s} {m:16s} citra tanpa cache={n_img_missing:3d} "
              f"kotak tercocokkan={cov:.3f}")
    return out


def bbox_ids_of(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as fh:
        return np.array([r["bbox_id"] for r in csv.DictReader(fh)])


def paired_delta(a, b, label, buckets, fn, n_boot=N_BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    n_img = len(buckets)
    d = []
    for _ in range(n_boot):
        pick = rng.integers(0, n_img, n_img)
        i = np.concatenate([buckets[p] for p in pick])
        y = label[i]
        s = int(y.sum())
        if s == 0 or s == y.size:
            continue
        d.append(fn(a[i], y) - fn(b[i], y))
    if not d:
        return None
    return {"delta": float(np.mean(d)),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def main() -> None:
    valid = load_csv(VALID_CSV, positive=TASK)
    test = load_csv(TEST_CSV, positive=TASK)
    print("membangun skor anggota:")
    sv = member_scores("valid", bbox_ids_of(VALID_CSV))
    st = member_scores("test", bbox_ids_of(TEST_CSV))

    lab_i = test["label"]
    lab_f = lab_i.astype(np.float64)
    buckets = build_buckets(test["image_id"])
    print(f"\nOIA-DDR test n={lab_i.size} positif={int(lab_i.sum())}")

    e5_v = np.mean([sv[m] for m in MEMBERS], axis=0)
    e5_t = np.mean([st[m] for m in MEMBERS], axis=0)
    e6_v = np.mean([sv["primary"]] + [sv[m] for m in MEMBERS], axis=0)
    e6_t = np.mean([st["primary"]] + [st[m] for m in MEMBERS], axis=0)

    ev = list(EVIDENCE_FEATURES)
    def stack(d, score, extra):
        return np.stack([d[f] for f in ev] + [score, d["ev_sds"], d["ev_sds_nsrc"]], 1)

    methods = {
        "Detector score (S)": {"v": [valid["pred_score"]], "t": [test["pred_score"]]},
        "Five-detector ensemble (E5)": {"v": [e5_v], "t": [e5_t]},
        "Six-detector ensemble (E6)": {"v": [e6_v], "t": [e6_t]},
    }
    for tag, s_v, s_t in (("Full gate (S+E)", valid["pred_score"], test["pred_score"]),
                          ("E6-input gate (E6+E)", e6_v, e6_t)):
        Xv, Xt = stack(valid, s_v, None), stack(test, s_t, None)
        a, b = [], []
        for seed in SEEDS:
            m = fit_mlp(Xv, valid["label"].astype(np.float64), seed=seed)
            a.append(apply_mlp(Xv, m)); b.append(apply_mlp(Xt, m))
        methods[tag] = {"v": a, "t": b}

    rep = {"task": TASK, "alpha": ALPHA, "iou_match": IOU_MATCH, "members": MEMBERS,
           "seeds": list(SEEDS), "n_boot": N_BOOT, "cohort": "OIA-DDR test",
           "note": "ensemble heterogen, bukan deep ensemble arsitektur-sama",
           "methods": {}}
    print()
    for name, d in methods.items():
        a_, ea_, ap_, cv_ = [], [], [], []
        for r in range(len(d["t"])):
            a_.append(aurc(d["t"][r], lab_f)); ea_.append(e_aurc(d["t"][r], lab_f))
            ap_.append(auprc(d["t"][r], lab_i))
            thr = conformal_threshold(d["v"][r], valid["label"], ALPHA)
            cv_.append(coverage_at(d["t"][r], lab_i, thr)["empirical_coverage"])
        rec = {"auprc": float(np.mean(ap_)), "aurc": float(np.mean(a_)),
               "e_aurc": float(np.mean(ea_)),
               "conformal_empirical_coverage": float(np.mean(cv_)),
               "conformal_coverage_gap": float(np.mean(cv_)) - (1 - ALPHA),
               "n_seeds": len(d["t"])}
        rep["methods"][name] = rec
        print(f"  {name:30s} AUPRC={rec['auprc']:.4f} AURC={rec['aurc']:.4f} "
              f"cakupan={rec['conformal_empirical_coverage']:.3f} "
              f"(selisih {rec['conformal_coverage_gap']:+.3f})")

    print()
    rep["paired"] = {}
    for a_name, b_name in (("Full gate (S+E)", "Six-detector ensemble (E6)"),
                           ("E6-input gate (E6+E)", "Six-detector ensemble (E6)")):
        A, B = methods[a_name]["t"][0], methods[b_name]["t"][0]
        for mname, fn in (("auprc", lambda s, y: auprc(s, y.astype(np.int64))),
                          ("aurc", lambda s, y: aurc(s, y.astype(np.float64)))):
            dd = paired_delta(A, B, lab_i, buckets, fn)
            rep["paired"][f"{a_name} vs {b_name} [{mname}]"] = dd
            excl = dd["ci95"][1] < 0 or dd["ci95"][0] > 0
            print(f"  {mname.upper():5s} {a_name} - {b_name} = {dd['delta']:+.4f} "
                  f"CI[{dd['ci95'][0]:+.4f},{dd['ci95'][1]:+.4f}] "
                  f"{'signifikan' if excl else 'CI memuat nol'}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nditulis {OUT}")


if __name__ == "__main__":
    main()
