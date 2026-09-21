#!/usr/bin/env python
"""eval_selective_conformal_baselines.py — baseline keluarga yang belum diukur.

Naskah menyitir conformal prediction dan selective prediction dengan reject option
sebagai keluarga pesaing, tetapi Tabel 8 tidak pernah mengukurnya. Skrip ini menutup
celah itu memakai protokol yang sama persis dengan tools/eval_fp_reduction_benchmark.py:
target governance_retain, gate dipasang pada OIA-DDR validasi, lima benih, bootstrap
berklaster-citra.

Tiga sumbu yang ditambahkan:

  A. Selective prediction (reject option). Kurva risiko-cakupan dan AURC. Pada cakupan c
     kita menahan c fraksi teratas menurut skor dan mengukur laju kesalahan di antara yang
     ditahan. AURC adalah rata-rata risiko atas seluruh cakupan; makin kecil makin baik.
     E-AURC mengurangkan AURC pemeringkat sempurna, sehingga membandingkan metode pada
     prevalensi yang berbeda menjadi sah.

  B. Split conformal untuk titik operasi. Naskah memilih tau* lewat kuantil validasi agar
     retensi TP mencapai 0,80. Split conformal menjawab pertanyaan yang sama dengan jaminan
     cakupan sampel-hingga: pada miscoverage alpha, ambangnya adalah kuantil
     ceil((n+1)(1-alpha))/n dari nonkonformitas positif kalibrasi. Kita laporkan cakupan
     nominal lawan empiris, in-domain dan di bawah pergeseran.

  C. Kalibrasi isotonik di samping temperature scaling. Keduanya monoton sehingga TIDAK
     mengubah peringkat; nilainya ada pada ECE dan Brier. Invariansi AUPRC diperiksa
     sebagai kontrol, bukan diasumsikan.

MC dropout dan deep ensemble berbasis bobot TIDAK dapat dijalankan dari artefak beku: keduanya
menuntut eksekusi ulang detektor, yang dikecualikan protokol detektor beku. Keluarga ensemble
diwakili oleh flip TTA, ensemble dua detektor, dan rata-rata tiga pass yang sudah ada di Tabel 8.

Keluaran: outputs/0506/selective_conformal/baselines.json
Jalankan dari root repo:
  python tools/eval_selective_conformal_baselines.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.eval_fp_reduction_benchmark import (  # noqa: E402
    apply_temperature, auprc, build_methods, fit_temperature, load_csv,
)

BASE = ROOT / "outputs/0506"
VALID = BASE / "evsds_v2sec/oiaddr/valid/governance_features_valid.csv"
COHORTS = {
    "OIA-DDR test": BASE / "evsds_v2sec/oiaddr/test/governance_features_test.csv",
    "IDRiD": BASE / "evsds_pss/idrid/test/governance_features_test.csv",
    "TJDR": BASE / "evsds_pss/tjdr/test/governance_features_test.csv",
    "FGADR": BASE / "evsds_pss/fgadr/train/governance_features_train.csv",
}
OUT = BASE / "selective_conformal/baselines.json"

TASK = "governance_retain"
SEEDS = (0, 1, 2, 3, 4)
ALPHA = 0.20           # miscoverage; 1-alpha = 0.80 = titik operasi retensi TP naskah
N_BOOT = 2000
BOOT_SEED = 0
BOOT_MAX_N = 120_000   # di atas ini bootstrap dilewati; dilaporkan apa adanya


# ---------------------------------------------------------------- selective prediction
def risk_coverage(score: np.ndarray, label: np.ndarray):
    """Risiko pada tiap cakupan. Risiko = fraksi negatif di antara yang ditahan."""
    order = np.argsort(-score, kind="mergesort")
    y = label[order]
    k = np.arange(1, y.size + 1, dtype=np.float64)
    risk = np.cumsum(1 - y) / k
    return k / y.size, risk


def aurc(score: np.ndarray, label: np.ndarray) -> float:
    _, risk = risk_coverage(score, label)
    return float(risk.mean())


def aurc_optimal(label: np.ndarray) -> float:
    """AURC pemeringkat sempurna: seluruh positif lebih dulu."""
    n = label.size
    n_pos = int(label.sum())
    k = np.arange(1, n + 1, dtype=np.float64)
    errors = np.maximum(k - n_pos, 0.0)
    return float((errors / k).mean())


def e_aurc(score: np.ndarray, label: np.ndarray) -> float:
    return aurc(score, label) - aurc_optimal(label)


# ---------------------------------------------------------------- split conformal
def conformal_threshold(cal_score: np.ndarray, cal_label: np.ndarray, alpha: float) -> float:
    """Ambang tahan-atau-tolak dengan jaminan cakupan positif >= 1-alpha.

    Nonkonformitas positif kalibrasi adalah -skor, sehingga bebas skala. Ambangnya
    kuantil ceil((n+1)(1-alpha))/n, koreksi sampel-hingga yang baku pada split conformal.
    """
    s = cal_score[cal_label == 1]
    if s.size == 0:
        return float(np.min(cal_score))
    nc = -s
    n = nc.size
    level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    tau = float(np.quantile(nc, level, method="higher"))
    return -tau


def coverage_at(score: np.ndarray, label: np.ndarray, thr: float) -> dict:
    keep = score >= thr
    n_pos = max(int(label.sum()), 1)
    return {
        "empirical_coverage": float(((label == 1) & keep).sum() / n_pos),
        "retained_fraction": float(keep.mean()),
    }


# ---------------------------------------------------------------- kalibrasi
def ece(prob: np.ndarray, label: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1], right=False), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        total += m.mean() * abs(prob[m].mean() - label[m].mean())
    return float(total)


def brier(prob: np.ndarray, label: np.ndarray) -> float:
    return float(np.mean((prob - label) ** 2))


def to_unit(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    return (x - lo) / (hi - lo) if hi > lo else np.full_like(x, 0.5)


# ---------------------------------------------------------------- bootstrap
def build_buckets(image_id: np.ndarray):
    """Indeks baris per citra, dihitung SEKALI. Membangunnya di dalam loop resample
    membuat biaya bootstrap naik sebesar jumlah citra."""
    _, inv = np.unique(image_id, return_inverse=True)
    order = np.argsort(inv, kind="mergesort")
    sorted_inv = inv[order]
    n_img = int(inv.max()) + 1
    starts = np.searchsorted(sorted_inv, np.arange(n_img), side="left")
    ends = np.searchsorted(sorted_inv, np.arange(n_img), side="right")
    return [order[s:e] for s, e in zip(starts, ends)]


def aurc_ci(score, label, buckets, n_boot=N_BOOT, seed=BOOT_SEED):
    if label.size > BOOT_MAX_N:
        return None
    rng = np.random.default_rng(seed)
    n_img = len(buckets)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, n_img, n_img)
        i = np.concatenate([buckets[p] for p in pick])
        y = label[i]
        s = int(y.sum())
        if s == 0 or s == y.size:
            continue
        vals.append(aurc(score[i], y))
    if not vals:
        return None
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


# ---------------------------------------------------------------- main
def main() -> None:
    valid = load_csv(VALID, positive=TASK)
    report = {
        "task": TASK, "alpha": ALPHA, "seeds": list(SEEDS),
        "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
        "protocol": "identik dengan tools/eval_fp_reduction_benchmark.py",
        "not_feasible": {
            "mc_dropout": "menuntut eksekusi ulang detektor; dikecualikan protokol detektor beku",
            "weight_space_deep_ensemble": "menuntut beberapa detektor terlatih; "
                                          "keluarga ensemble diwakili flip TTA, ensemble dua "
                                          "detektor, dan rata-rata tiga pass di Tabel 8",
        },
        "cohorts": {},
    }

    for coh, path in COHORTS.items():
        if not path.exists():
            report["cohorts"][coh] = {"error": f"tidak ada: {path}"}
            continue
        test = load_csv(path, positive=TASK)
        lab = test["label"].astype(np.float64)
        n_pos = int(lab.sum())
        entry = {"n": int(lab.size), "n_positive": n_pos,
                 "prevalence": float(lab.mean()), "methods": {}}
        print(f"\n=== {coh}  n={lab.size}  positif={n_pos} ({lab.mean():.3f}) ===", flush=True)
        buckets = build_buckets(test["image_id"]) if lab.size <= BOOT_MAX_N else None

        # kumpulkan skor per metode atas lima benih
        per_method = {}
        # Metode non-learned identik lintas benih; salinannya diabaikan saat merata-rata.
        for seed in SEEDS:
            for name, family, sv, st in build_methods(valid, test, mlp_seed=seed):
                d = per_method.setdefault(name, {"family": family, "valid": [], "test": []})
                d["valid"].append(sv)
                d["test"].append(st)
        for name, d in per_method.items():
            learned = d["family"] in ("governance", "learned-linear", "learned-score")
            n_rep = len(d["test"]) if learned else 1
            a_list, ea_list, ap_list, cov_list, ret_list = [], [], [], [], []
            for r in range(n_rep):
                st, sv = d["test"][r], d["valid"][r]
                a_list.append(aurc(st, lab))
                ea_list.append(e_aurc(st, lab))
                ap_list.append(auprc(st, test["label"]))
                thr = conformal_threshold(sv, valid["label"], ALPHA)
                c = coverage_at(st, test["label"], thr)
                cov_list.append(c["empirical_coverage"])
                ret_list.append(c["retained_fraction"])
            rec = {
                "family": d["family"], "n_seeds": n_rep,
                "aurc": float(np.mean(a_list)), "aurc_sd": float(np.std(a_list)),
                "e_aurc": float(np.mean(ea_list)),
                "auprc": float(np.mean(ap_list)),
                "conformal_empirical_coverage": float(np.mean(cov_list)),
                "conformal_retained_fraction": float(np.mean(ret_list)),
                "conformal_nominal_coverage": 1.0 - ALPHA,
            }
            rec["conformal_coverage_gap"] = rec["conformal_empirical_coverage"] - (1.0 - ALPHA)
            if buckets is not None:
                rec["aurc_ci95"] = aurc_ci(d["test"][0], test["label"], buckets)
            entry["methods"][name] = rec
            print(f"  {name:26s} AURC={rec['aurc']:.4f} E-AURC={rec['e_aurc']:.4f} "
                  f"AUPRC={rec['auprc']:.4f} cakupan={rec['conformal_empirical_coverage']:.3f} "
                  f"(nominal {1-ALPHA:.2f})", flush=True)

        # ---- kalibrasi: mentah lawan temperature lawan isotonik
        T = fit_temperature(valid["pred_score"], valid["label"])
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(valid["pred_score"], valid["label"].astype(np.float64))
        variants = {
            "raw score": to_unit(test["pred_score"]),
            f"temperature (T={T:.2f})": to_unit(apply_temperature(test["pred_score"], T)),
            "isotonic": np.clip(iso.predict(test["pred_score"]), 0.0, 1.0),
        }
        entry["calibration"] = {}
        for vname, p in variants.items():
            entry["calibration"][vname] = {
                "ece": ece(p, lab), "brier": brier(p, lab),
                "auprc": auprc(p, test["label"]),
            }
            print(f"  [kalibrasi] {vname:22s} ECE={entry['calibration'][vname]['ece']:.4f} "
                  f"Brier={entry['calibration'][vname]['brier']:.4f} "
                  f"AUPRC={entry['calibration'][vname]['auprc']:.4f}")
        report["cohorts"][coh] = entry

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nditulis {OUT}")


if __name__ == "__main__":
    main()
