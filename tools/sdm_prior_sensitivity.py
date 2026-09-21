#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Uji sensitivitas gate terhadap prior klinis SDM.

MOTIVASI. Enam angka pada `outputs/stage3_audit/sdm_norm_clinical_prior.json`
ditetapkan dari aturan ICDR, bukan dipelajari dari data. Pertanyaan pertama yang akan
diajukan reviewer: seberapa besar perolehan gate bergantung pada keenam angka itu?
Tanpa jawaban terukur, Kontribusi 1 berdiri di atas tetapan yang tak teruji.

Satu angka khususnya tidak dapat diturunkan dari aturan yang tertulis di naskah.
Naskah menyatakan biaya "0 bila satu kelompok keparahan, dan bertambah menurut jumlah
batas grade yang dilewati" — itu menjelaskan raw 0, 1.0, dan 2.0, tetapi TIDAK
menjelaskan Hard_EX-Soft_EX = 1.5. Padahal pasangan itulah yang paling sering muncul
setelah pasangan satu-grade: 18.772 dari 64.214 deteksi (29,2%) memakai nilai 0.75.
Jadi angka yang tak terdokumentasi itu justru menanggung beban.

CARA KERJA. SDCA seluruhnya ditentukan oleh pasangan (kelas prediksi, runner-up):
    SDCA_i = SDM_norm(c_i, r_i).
Kolom prob_ma/prob_he/prob_hardex/prob_softex pada CSV governance menyimpan distribusi
kelasnya, sehingga pasangan itu dapat dibangun ulang dan SDCA dihitung ulang di bawah
prior alternatif TANPA menjalankan ulang detektor. Verifikasi: rekonstruksi ini cocok
persis dengan kolom ev_sdca yang tersimpan pada 64.203 dari 64.214 baris (99,98%);
11 baris sisanya adalah deteksi yang pred_class-nya bukan argmax distribusi tersimpan.
Supaya seluruh varian dibandingkan pada pijakan yang identik, SEMUA varian (termasuk
baseline) memakai SDCA hasil rekonstruksi, dan selisih terhadap kolom terpasang
dilaporkan terpisah sebagai kontrol.

Gate, pembagian data, dan metrik mengikuti tools/eval_fp_reduction_benchmark.py apa
adanya: AUPRC pada tugas governance_retain, gate MLP dipasang pada split valid dan
dievaluasi pada split test.

Jalankan dari root repo:
    python tools/sdm_prior_sensitivity.py
    python tools/sdm_prior_sensitivity.py --seeds 5 --out outputs/0506/sdm_sensitivity.json
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.eval_fp_reduction_benchmark import (  # noqa: E402
    auprc, build_methods, load_csv,
)

POS = "governance_retain"
BASE = ROOT / "outputs/0506/evsds_v2sec/oiaddr"
VALID_CSV = BASE / "valid/governance_features_valid.csv"
TEST_CSV = BASE / "test/governance_features_test.csv"
SDM_JSON = ROOT / "outputs/stage3_audit/sdm_norm_clinical_prior.json"

CLASSES = ["MA", "HE", "Hard_EX", "Soft_EX"]
PROB_COL = {"MA": "prob_ma", "HE": "prob_he",
            "Hard_EX": "prob_hardex", "Soft_EX": "prob_softex"}
PAIRS = [tuple(sorted(p)) for p in itertools.combinations(CLASSES, 2)]

M0, GATE = "M0_score", "KG-gate-MLP (ours)"


# ---------- prior klinis ----------

def load_locked_prior() -> dict:
    raw = json.loads(SDM_JSON.read_text(encoding="utf-8"))
    out = {}
    for key, v in raw.items():
        a, b = key.split("__")
        out[tuple(sorted((a, b)))] = float(v["norm"])
    return out


def _norm(raw: dict) -> dict:
    """Rescale by max, persis seperti Eq. 1 naskah."""
    mx = max(raw.values())
    return {k: (v / mx if mx > 0 else 0.0) for k, v in raw.items()}


def prior_variants(locked: dict, n_shuffle: int = 5) -> list:
    """Daftar (nama, keterangan, prior) yang diuji."""
    same_group = {("HE", "MA")}   # satu-satunya pasangan satu-grade (kunci terurut)
    variants = [
        ("locked", "prior ICDR terpasang {0, 0.50, 0.75, 1.00}", dict(locked)),
    ]

    # (1) Hanya langkah bulat: menghapus satu-satunya nilai yang tidak diturunkan
    #     naskah (Hard_EX-Soft_EX raw 1.5 -> 1.0).
    raw_int = {("HE", "MA"): 0.0, ("Hard_EX", "MA"): 1.0, ("MA", "Soft_EX"): 2.0,
               ("HE", "Hard_EX"): 1.0, ("HE", "Soft_EX"): 2.0,
               ("Hard_EX", "Soft_EX"): 1.0}
    variants.append(("integer_steps",
                     "Hard_EX-Soft_EX dijadikan satu langkah penuh (0.75 -> 0.50)",
                     _norm(raw_int)))

    # (2) Biner: 0 bila satu kelompok keparahan, 1 bila tidak. Membuang seluruh
    #     penjenjangan dan menyisakan dikotomi dalam-grade / lintas-grade.
    variants.append(("binary",
                     "0 bila satu kelompok keparahan, 1 bila lintas-grade",
                     {p: (0.0 if p in same_group else 1.0) for p in PAIRS}))

    # (3) Seragam: SDCA menjadi konstanta, jadi pilar itu tidak membawa informasi.
    #     Ini batas bawah yang benar untuk membaca kontribusi priornya.
    variants.append(("uniform",
                     "seluruh pasangan bernilai sama (SDCA tanpa informasi)",
                     {p: 0.5 for p in PAIRS}))

    # (4) Acak: nilai yang sama, pasangan yang diacak. Menguji apakah yang penting
    #     URUTAN klinisnya atau sekadar sebaran nilainya.
    vals = [locked[p] for p in PAIRS]
    for s in range(n_shuffle):
        rng = np.random.default_rng(1000 + s)
        perm = list(rng.permutation(len(PAIRS)))
        variants.append((f"shuffled_s{s}",
                         "nilai yang sama, pasangan diacak",
                         {PAIRS[i]: vals[perm[i]] for i in range(len(PAIRS))}))
    return variants


# ---------- rekonstruksi pasangan (prediksi, runner-up) ----------

def read_pairs(path: Path) -> tuple:
    """Kembalikan (daftar pasangan terurut per baris, ev_sdca tersimpan)."""
    pairs, stored = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pc = r.get("pred_class")
            try:
                probs = {c: float(r[PROB_COL[c]]) for c in CLASSES}
            except (TypeError, ValueError, KeyError):
                probs = None
            if pc in CLASSES and probs is not None:
                runner = max((c for c in CLASSES if c != pc), key=lambda c: probs[c])
                pairs.append(tuple(sorted((pc, runner))))
            else:
                pairs.append(None)
            try:
                stored.append(float(r.get("ev_sdca", "nan")))
            except (TypeError, ValueError):
                stored.append(np.nan)
    return pairs, np.asarray(stored, dtype=np.float64)


def sdca_from_prior(pairs: list, prior: dict, fallback: np.ndarray) -> np.ndarray:
    """Petakan tiap baris ke nilai priornya; baris tanpa pasangan memakai nilai asli."""
    out = np.empty(len(pairs), dtype=np.float64)
    for i, p in enumerate(pairs):
        out[i] = prior[p] if p is not None and p in prior else fallback[i]
    return out


# ---------- pelaksana ----------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5,
                    help="jumlah seed inisialisasi MLP per varian")
    ap.add_argument("--shuffles", type=int, default=5,
                    help="jumlah prior teracak")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "outputs/0506/sdm_prior_sensitivity.json")
    args = ap.parse_args()

    for p in (VALID_CSV, TEST_CSV, SDM_JSON):
        if not p.is_file():
            print(f"[sdm-sens] berkas tidak ada: {p}")
            return 2

    print("[sdm-sens] memuat CSV governance ...", flush=True)
    valid = load_csv(VALID_CSV, POS)
    test = load_csv(TEST_CSV, POS)
    vp, v_stored = read_pairs(VALID_CSV)
    tp, t_stored = read_pairs(TEST_CSV)
    assert len(vp) == len(valid["label"]), "urutan baris valid tidak sejajar"
    assert len(tp) == len(test["label"]), "urutan baris test tidak sejajar"

    locked = load_locked_prior()

    # Kontrol: seberapa jauh rekonstruksi menyimpang dari kolom terpasang.
    rec_t = sdca_from_prior(tp, locked, t_stored)
    n_diff = int(np.sum(np.abs(rec_t - t_stored) > 1e-9))
    print(f"[sdm-sens] rekonstruksi vs kolom terpasang (test): "
          f"{len(tp) - n_diff}/{len(tp)} cocok persis, {n_diff} berbeda")

    variants = prior_variants(locked, n_shuffle=args.shuffles)
    results = []
    for name, desc, prior in variants:
        valid["ev_sdca"] = sdca_from_prior(vp, prior, v_stored)
        test["ev_sdca"] = sdca_from_prior(tp, prior, t_stored)
        gates, m0s = [], []
        for seed in range(args.seeds):
            got = {n: st for n, _f, _sv, st in build_methods(valid, test, mlp_seed=seed)
                   if n in (M0, GATE)}
            gates.append(auprc(got[GATE], test["label"]))
            m0s.append(auprc(got[M0], test["label"]))
        g = np.asarray(gates); m = np.asarray(m0s)
        row = {
            "variant": name,
            "description": desc,
            "prior": {f"{a}__{b}": prior[(a, b)] for a, b in PAIRS},
            "gate_auprc_mean": float(g.mean()),
            "gate_auprc_sd": float(g.std(ddof=1)) if len(g) > 1 else 0.0,
            "gate_auprc_per_seed": [float(x) for x in g],
            "m0_auprc": float(m.mean()),
            "gain_mean": float((g - m).mean()),
            "gain_sd": float((g - m).std(ddof=1)) if len(g) > 1 else 0.0,
        }
        results.append(row)
        print(f"[sdm-sens] {name:<16} gate={row['gate_auprc_mean']:.4f} "
              f"(sd {row['gate_auprc_sd']:.4f})  gain vs M0={row['gain_mean']:+.4f}",
              flush=True)

    base = next(r for r in results if r["variant"] == "locked")
    for r in results:
        r["delta_vs_locked"] = r["gate_auprc_mean"] - base["gate_auprc_mean"]

    payload = {
        "task": POS,
        "cohort": "OIA-DDR test (in-domain)",
        "valid_csv": str(VALID_CSV.relative_to(ROOT)).replace("\\", "/"),
        "test_csv": str(TEST_CSV.relative_to(ROOT)).replace("\\", "/"),
        "n_seeds": args.seeds,
        "n_test_rows": int(len(tp)),
        "reconstruction_mismatch_rows": n_diff,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[sdm-sens] ditulis -> {args.out}")

    print("\nringkasan (AUPRC gate, selisih terhadap prior terpasang):")
    for r in sorted(results, key=lambda x: -x["gate_auprc_mean"]):
        print(f"  {r['variant']:<16} {r['gate_auprc_mean']:.4f}  "
              f"{r['delta_vs_locked']:+.4f}   {r['description']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
