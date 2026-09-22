#!/usr/bin/env python
"""eval_ensemble_external.py — ensemble multi-detektor lawan gate, keempat kohort.

Menutup keberatan yang belum terjawab: ensemble enam detektor mengalahkan gate di OIA-DDR;
apakah itu bertahan di bawah pergeseran domain?

Protokol nol-tembak, identik naskah: gate dipasang pada OIA-DDR validasi, lima benih,
diterapkan tanpa perubahan. Skor ensemble didefinisikan pada kotak kandidat detektor PRIMER;
tiap anggota dicocokkan pada IoU minimal 0,15, anggota tanpa kotak cocok menyumbang nol.

Keluaran: _LGOVNET_NEW_SEP/results/ensemble_external.json
Jalankan dari root repo:
  python _LGOVNET_NEW_SEP/scripts/eval_ensemble_external.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.eval_fp_reduction_benchmark import (  # noqa: E402
    EVIDENCE_FEATURES, apply_mlp, auprc, fit_mlp, load_csv,
)
from tools.eval_selective_conformal_baselines import (  # noqa: E402
    ALPHA, BOOT_SEED, N_BOOT, aurc, build_buckets, conformal_threshold, coverage_at, e_aurc,
)
from tools.eval_multidetector_ensemble import (  # noqa: E402
    IOU_MATCH, MEMBERS, iou_matrix, load_boxes,
)

BASE = ROOT / "outputs/0506"
PRIM = ROOT / "outputs/detector_cache/primary"
DETCMP = ROOT / "outputs/detcmp"
OUT = ROOT / "_LGOVNET_NEW_SEP/results/ensemble_external.json"

SEEDS = (0, 1, 2, 3, 4)
TASK = "governance_retain"
BOOT_MAX_N = 120_000

# (nama, csv bukti, direktori cache primer, sufiks cache detcmp, split)
COHORTS = [
    ("OIA-DDR test", BASE / "evsds_v2sec/oiaddr/test/governance_features_test.csv",
     PRIM / "oiaddr/test", "", "test"),
    ("IDRiD", BASE / "evsds_pss/idrid/test/governance_features_test.csv",
     PRIM / "idrid/test", "_idrid", "test"),
    ("TJDR", BASE / "evsds_pss/tjdr/test/governance_features_test.csv",
     PRIM / "tjdr/test", "_tjdr", "test"),
    ("FGADR", BASE / "evsds_pss/fgadr/train/governance_features_train.csv",
     PRIM / "fgadr/train", "_fgadr", "train"),
]
VALID_CSV = BASE / "evsds_v2sec/oiaddr/valid/governance_features_valid.csv"
VALID_PRIM = PRIM / "oiaddr/valid"


def bbox_ids_of(path: Path) -> list:
    with path.open(newline="", encoding="utf-8") as fh:
        return [r["bbox_id"] for r in csv.DictReader(fh)]


def member_scores(prim_dir: Path, suffix: str, split: str, csv_ids: list) -> dict:
    """Skor tiap anggota pada kotak primer, selaras urutan csv_ids."""
    prim = load_boxes(prim_dir)
    loc = {}
    for img, (ids, _, _) in prim.items():
        for j, bid in enumerate(ids):
            loc[bid] = (img, j)
    miss = [b for b in csv_ids if b not in loc]
    if miss:
        raise SystemExit(f"[ens] {len(miss)} bbox_id CSV tak ada di cache primer {prim_dir}")

    out = {"primary": np.array([prim[loc[b][0]][2][loc[b][1]] for b in csv_ids])}
    for m in MEMBERS:
        cache = DETCMP / f"{m}_cache{suffix}" / split
        if not cache.is_dir():
            raise SystemExit(f"[ens] cache tidak ada: {cache}")
        other = load_boxes(cache)
        per_img, n_missing = {}, 0
        for img, (_, pb, _) in prim.items():
            if img not in other:
                n_missing += 1
                per_img[img] = np.zeros(pb.shape[0])
                continue
            _, ob, osc = other[img]
            M = iou_matrix(pb, ob)
            if ob.shape[0]:
                best, bi = M.argmax(1), M.max(1)
                per_img[img] = np.where(bi >= IOU_MATCH, osc[best], 0.0)
            else:
                per_img[img] = np.zeros(pb.shape[0])
        out[m] = np.array([per_img[loc[b][0]][loc[b][1]] for b in csv_ids])
        print(f"    {m:16s} citra tanpa cache={n_missing:4d} "
              f"kotak tercocokkan={float((out[m] > 0).mean()):.3f}")
    return out


def paired_delta(a, b, label, buckets, fn, n_boot=N_BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    n_img, d = len(buckets), []
    for _ in range(n_boot):
        i = np.concatenate([buckets[p] for p in rng.integers(0, n_img, n_img)])
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
    print("membangun skor anggota pada OIA-DDR validasi:")
    sv = member_scores(VALID_PRIM, "", "valid", bbox_ids_of(VALID_CSV))
    e5_v = np.mean([sv[m] for m in MEMBERS], axis=0)
    e6_v = np.mean([sv["primary"]] + [sv[m] for m in MEMBERS], axis=0)

    ev = list(EVIDENCE_FEATURES)
    def stack(d, score):
        return np.stack([d[f] for f in ev] + [score, d["ev_sds"], d["ev_sds_nsrc"]], 1)

    Xv_raw, Xv_e6 = stack(valid, valid["pred_score"]), stack(valid, e6_v)
    gate_raw = [fit_mlp(Xv_raw, valid["label"].astype(np.float64), seed=s) for s in SEEDS]
    gate_e6 = [fit_mlp(Xv_e6, valid["label"].astype(np.float64), seed=s) for s in SEEDS]

    rep = {"task": TASK, "alpha": ALPHA, "iou_match": IOU_MATCH, "members": MEMBERS,
           "seeds": list(SEEDS), "protocol": "nol-tembak, gate dipasang pada OIA-DDR validasi",
           "cohorts": {}}

    for name, csv_p, prim_dir, suffix, split in COHORTS:
        if not csv_p.exists():
            rep["cohorts"][name] = {"error": f"csv tidak ada: {csv_p}"}
            continue
        print(f"\n=== {name} ===")
        test = load_csv(csv_p, positive=TASK)
        st = member_scores(prim_dir, suffix, split, bbox_ids_of(csv_p))
        lab_i = test["label"]
        lab_f = lab_i.astype(np.float64)
        e5_t = np.mean([st[m] for m in MEMBERS], axis=0)
        e6_t = np.mean([st["primary"]] + [st[m] for m in MEMBERS], axis=0)
        Xt_raw, Xt_e6 = stack(test, test["pred_score"]), stack(test, e6_t)

        methods = {
            "S (detector score)": ([valid["pred_score"]], [test["pred_score"]]),
            "E5 (five detectors)": ([e5_v], [e5_t]),
            "E6 (six detectors)": ([e6_v], [e6_t]),
            "S+E (full gate)": ([apply_mlp(Xv_raw, m) for m in gate_raw],
                                [apply_mlp(Xt_raw, m) for m in gate_raw]),
            "E6+E (E6-input gate)": ([apply_mlp(Xv_e6, m) for m in gate_e6],
                                     [apply_mlp(Xt_e6, m) for m in gate_e6]),
        }

        entry = {"n": int(lab_i.size), "n_positive": int(lab_i.sum()),
                 "prevalence": float(lab_f.mean()), "methods": {}}
        for mname, (v_list, t_list) in methods.items():
            ap, ar, cv = [], [], []
            for r in range(len(t_list)):
                ap.append(auprc(t_list[r], lab_i))
                ar.append(aurc(t_list[r], lab_f))
                thr = conformal_threshold(v_list[r], valid["label"], ALPHA)
                cv.append(coverage_at(t_list[r], lab_i, thr)["empirical_coverage"])
            rec = {"auprc": float(np.mean(ap)), "aurc": float(np.mean(ar)),
                   "conformal_coverage": float(np.mean(cv)),
                   "conformal_gap": float(np.mean(cv)) - (1 - ALPHA),
                   "n_seeds": len(t_list)}
            entry["methods"][mname] = rec
            print(f"  {mname:24s} AUPRC={rec['auprc']:.4f} AURC={rec['aurc']:.4f} "
                  f"cakupan={rec['conformal_coverage']:.3f} ({rec['conformal_gap']:+.3f})")

        if lab_i.size <= BOOT_MAX_N:
            buckets = build_buckets(test["image_id"])
            entry["paired_vs_E6"] = {}
            for a_name in ("S+E (full gate)", "E6+E (E6-input gate)"):
                A = methods[a_name][1][0]
                dd = paired_delta(A, e6_t, lab_i, buckets,
                                  lambda s, y: auprc(s, y.astype(np.int64)))
                entry["paired_vs_E6"][a_name] = dd
                excl = dd["ci95"][1] < 0 or dd["ci95"][0] > 0
                print(f"  AUPRC {a_name} - E6 = {dd['delta']:+.4f} "
                      f"CI[{dd['ci95'][0]:+.4f},{dd['ci95'][1]:+.4f}] "
                      f"{'signifikan' if excl else 'CI memuat nol'}")
        else:
            entry["paired_vs_E6"] = "dilewati, n melebihi batas bootstrap"
            print(f"  (bootstrap dilewati, n={lab_i.size} > {BOOT_MAX_N})")

        rep["cohorts"][name] = entry

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nditulis {OUT}")


if __name__ == "__main__":
    main()
