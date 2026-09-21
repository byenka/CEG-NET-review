#!/usr/bin/env python
"""eval_selective_fusion_headtohead.py — gate lawan rata-rata tiga pass pada sumbu baru.

Tabel 8 menunjukkan rata-rata tiga pass (AUPRC 0.722) mengungguli gate penuh (0.699) pada
OIA-DDR. Pertanyaan yang belum dijawab: apakah urutan itu bertahan pada metrik selective
prediction dan pada akurasi titik operasi conformal, dua sumbu yang berbeda dari AUPRC.

Skor fusi hanya tersedia untuk OIA-DDR (valid dan test), sehingga perbandingan ini in-domain.
Protokol identik dengan tools/eval_fp_reduction_benchmark.py: gate dipasang pada validasi,
lima benih, bootstrap berklaster-citra 2.000 resample benih 0.

Keluaran: outputs/0506/selective_conformal/fusion_headtohead.json
Jalankan dari root repo:
  python tools/eval_selective_fusion_headtohead.py
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
VALID = BASE / "evsds_v2sec/oiaddr/valid/governance_features_valid.csv"
TEST = BASE / "evsds_v2sec/oiaddr/test/governance_features_test.csv"
FUSED = ROOT / "_septem/scripts_optionB/fused_scores_oiaddr.csv"
OUT = BASE / "selective_conformal/fusion_headtohead.json"

TASK = "governance_retain"
SEEDS = (0, 1, 2, 3, 4)


def bbox_ids(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as fh:
        return np.array([r["bbox_id"] for r in csv.DictReader(fh)])


def fused_for(split: str, ids: np.ndarray) -> np.ndarray:
    """Skor fusi selaras urutan baris CSV bukti. Gagal keras bila ada yang tak cocok."""
    table = {}
    with FUSED.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["split"] == split:
                table[r["bbox_id"]] = float(r["fused"])
    missing = [i for i in ids if i not in table]
    if missing:
        raise SystemExit(f"[fusi] {len(missing)} bbox_id tanpa skor fusi pada split {split}")
    return np.array([table[i] for i in ids], dtype=np.float64)


def paired_aurc_delta(a: np.ndarray, b: np.ndarray, label: np.ndarray, buckets,
                      n_boot: int = N_BOOT, seed: int = BOOT_SEED):
    """CI selisih AURC(a) - AURC(b) pada resample berpasangan yang sama."""
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
        d.append(aurc(a[i], y) - aurc(b[i], y))
    if not d:
        return None
    return {"delta": float(np.mean(d)),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def main() -> None:
    valid = load_csv(VALID, positive=TASK)
    test = load_csv(TEST, positive=TASK)
    fv = fused_for("valid", bbox_ids(VALID))
    ft = fused_for("test", bbox_ids(TEST))
    assert fv.size == valid["label"].size and ft.size == test["label"].size

    lab_i = test["label"]
    lab_f = lab_i.astype(np.float64)
    buckets = build_buckets(test["image_id"])
    print(f"OIA-DDR test n={lab_i.size} positif={int(lab_i.sum())}")

    ev = list(EVIDENCE_FEATURES)
    Xv_raw = np.stack([valid[f] for f in ev] + [valid["pred_score"], valid["ev_sds"],
                                                valid["ev_sds_nsrc"]], 1)
    Xt_raw = np.stack([test[f] for f in ev] + [test["pred_score"], test["ev_sds"],
                                               test["ev_sds_nsrc"]], 1)
    Xv_fus = np.stack([valid[f] for f in ev] + [fv, valid["ev_sds"], valid["ev_sds_nsrc"]], 1)
    Xt_fus = np.stack([test[f] for f in ev] + [ft, test["ev_sds"], test["ev_sds_nsrc"]], 1)

    methods = {"Three-pass mean (F)": {"valid": [fv], "test": [ft], "learned": False}}
    for tag, Xv, Xt in (("Full gate (S+E)", Xv_raw, Xt_raw),
                        ("Fusion-input gate (F+E)", Xv_fus, Xt_fus)):
        sv, st = [], []
        for seed in SEEDS:
            m = fit_mlp(Xv, valid["label"].astype(np.float64), seed=seed)
            sv.append(apply_mlp(Xv, m))
            st.append(apply_mlp(Xt, m))
        methods[tag] = {"valid": sv, "test": st, "learned": True}

    report = {"task": TASK, "cohort": "OIA-DDR test", "alpha": ALPHA,
              "seeds": list(SEEDS), "n_boot": N_BOOT, "methods": {}}
    for name, d in methods.items():
        a, ea, ap, cov, ret = [], [], [], [], []
        for r in range(len(d["test"])):
            st, sv = d["test"][r], d["valid"][r]
            a.append(aurc(st, lab_f))
            ea.append(e_aurc(st, lab_f))
            ap.append(auprc(st, lab_i))
            thr = conformal_threshold(sv, valid["label"], ALPHA)
            c = coverage_at(st, lab_i, thr)
            cov.append(c["empirical_coverage"])
            ret.append(c["retained_fraction"])
        rec = {"aurc": float(np.mean(a)), "aurc_sd": float(np.std(a)),
               "e_aurc": float(np.mean(ea)), "auprc": float(np.mean(ap)),
               "conformal_empirical_coverage": float(np.mean(cov)),
               "conformal_coverage_gap": float(np.mean(cov)) - (1.0 - ALPHA),
               "conformal_retained_fraction": float(np.mean(ret)),
               "n_seeds": len(d["test"])}
        report["methods"][name] = rec
        print(f"  {name:26s} AURC={rec['aurc']:.4f} E-AURC={rec['e_aurc']:.4f} "
              f"AUPRC={rec['auprc']:.4f} cakupan={rec['conformal_empirical_coverage']:.3f}")

    # selisih berpasangan terhadap fusi, benih 0 untuk gate
    report["paired_vs_fusion"] = {}
    for name in ("Full gate (S+E)", "Fusion-input gate (F+E)"):
        dd = paired_aurc_delta(methods[name]["test"][0], ft, lab_i, buckets)
        report["paired_vs_fusion"][name] = dd
        sign = "lebih baik" if dd["delta"] < 0 else "lebih buruk"
        excl = (dd["ci95"][1] < 0) or (dd["ci95"][0] > 0)
        print(f"  AURC({name}) - AURC(fusi) = {dd['delta']:+.4f} "
              f"CI[{dd['ci95'][0]:+.4f},{dd['ci95'][1]:+.4f}] -> {sign}, "
              f"{'signifikan' if excl else 'CI memuat nol'}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nditulis {OUT}")


if __name__ == "__main__":
    main()
