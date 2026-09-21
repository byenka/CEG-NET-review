#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Hitung sekali angka domain-shift 5 seed, simpan sebagai artefak JSON.

Memisahkan perhitungan dari penggambaran: script figur membaca JSON ini, sehingga
figur dapat direproduksi tanpa memuat ulang CSV 334 MB dan setiap angka di figur
bisa ditelusuri ke satu berkas.

Keluaran: outputs/0506/evsds_pss/domain_shift_5seed.json

Jalankan dari root repo:
    python journal/figures/scripts/_figdata_domain_shift.py
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.eval_fp_reduction_benchmark import load_csv, build_methods, auprc  # noqa: E402

BASE = ROOT / "outputs/0506"
POS = "governance_retain"
SEEDS = [0, 1, 2, 3, 4]
OUT = BASE / "evsds_pss/domain_shift_5seed.json"

# PSS nyata di semua kohort. In-domain memakai evsds_v2sec karena PSS di sana
# memang sudah nyata (59% 'implemented'); eksternal memakai evsds_pss.
COHORTS = [
    ("OIA-DDR", "in-domain", BASE / "evsds_v2sec/oiaddr/test/governance_features_test.csv", 0.58),
    ("IDRiD", "zero-shot", BASE / "evsds_pss/idrid/test/governance_features_test.csv", 0.64),
    ("TJDR", "zero-shot", BASE / "evsds_pss/tjdr/test/governance_features_test.csv", 0.51),
    ("FGADR", "zero-shot", BASE / "evsds_pss/fgadr/train/governance_features_train.csv", 0.50),
]

M0, B4, B6 = "M0_score", "KG-gate-MLP (ours)", "KG-gate+SDS+corrob (B6)"


def main() -> int:
    valid = load_csv(BASE / "evsds_v2sec/oiaddr/valid/governance_features_valid.csv", POS)
    out = {
        "tool": "_figdata_domain_shift",
        "positive_label": POS,
        "seeds": SEEDS,
        "valid_csv": "outputs/0506/evsds_v2sec/oiaddr/valid/governance_features_valid.csv",
        "note": ("PSS nyata di semua kohort. Eksternal dari evsds_pss (PSS dibangkitkan "
                 "dengan resep run terkunci); in-domain dari evsds_v2sec yang PSS-nya "
                 "memang sudah nyata."),
        "cohorts": [],
    }

    for name, role, path, pss_cov in COHORTS:
        if not path.is_file():
            print(f"[figdata] LEWAT {name}: {path} tidak ada")
            continue
        print(f"[figdata] memuat {name} ...", flush=True)
        test = load_csv(path, POS)
        m0 = None
        b4s, b6s = [], []
        for seed in SEEDS:
            r = {n: auprc(stt, test["label"])
                 for n, _f, _sv, stt in build_methods(valid, test, mlp_seed=seed)
                 if n in (M0, B4, B6)}
            m0 = r[M0]
            b4s.append(r[B4])
            b6s.append(r[B6])
        g4 = [x - m0 for x in b4s]
        g6 = [x - m0 for x in b6s]
        g64 = [b - a for a, b in zip(b4s, b6s)]
        rec = {
            "name": name, "role": role,
            "csv": str(path.relative_to(ROOT)).replace("\\", "/"),
            "n_boxes": int(test["label"].size),
            "pss_coverage": pss_cov,
            "m0_auprc": m0,
            "b4_auprc_per_seed": b4s, "b6_auprc_per_seed": b6s,
            "b4_auprc_mean": st.mean(b4s), "b6_auprc_mean": st.mean(b6s),
            "gain_b4_m0": {"per_seed": g4, "mean": st.mean(g4), "sd": st.pstdev(g4),
                           "n_pos": sum(1 for x in g4 if x > 0)},
            "gain_b6_m0": {"per_seed": g6, "mean": st.mean(g6), "sd": st.pstdev(g6),
                           "n_pos": sum(1 for x in g6 if x > 0)},
            "gain_b6_b4": {"per_seed": g64, "mean": st.mean(g64), "sd": st.pstdev(g64),
                           "n_pos": sum(1 for x in g64 if x > 0)},
        }
        out["cohorts"].append(rec)
        print(f"  {name}: M0={m0:.4f}  B6-M0={rec['gain_b6_m0']['mean']:+.4f}"
              f" ({rec['gain_b6_m0']['n_pos']}/5)  B6-B4={rec['gain_b6_b4']['mean']:+.4f}"
              f" ({rec['gain_b6_b4']['n_pos']}/5)", flush=True)
        del test

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[figdata] ditulis -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
