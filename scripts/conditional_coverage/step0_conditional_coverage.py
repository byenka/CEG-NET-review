#!/usr/bin/env python
"""step0_conditional_coverage.py — Langkah 0 dari rancangan ulang CEG.

Menguji apakah ambang conformal yang dikalibrasi MARGINAL pada OIA-DDR validasi memberi
cakupan mendekati nominal DI DALAM tiap stratum korroborasi pada OIA-DDR test.

Protokol dikunci sebelum eksekusi:
  _LGOVNET_NEW_SEP/PROTOCOL_LOCK_step0_conditional_coverage.md

Kriteria (dikunci):
  MATI   : setiap metode |selisih| <= 0,05 di setiap stratum
  LANJUT : sekurangnya satu metode |selisih| > 0,15 pada stratum dengan >= 200 positif
  ABU-ABU: selisih maksimum di antara keduanya

Keluaran: _LGOVNET_NEW_SEP/results/step0_conditional_coverage.json
Jalankan dari root repo:
  python _LGOVNET_NEW_SEP/scripts/step0_conditional_coverage.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.eval_fp_reduction_benchmark import (  # noqa: E402
    EVIDENCE_FEATURES, apply_mlp, fit_mlp, load_csv,
)
from tools.eval_selective_conformal_baselines import conformal_threshold  # noqa: E402
from tools.eval_multidetector_ensemble import (  # noqa: E402
    MEMBERS, TEST_CSV, VALID_CSV, bbox_ids_of, member_scores,
)

OUT = ROOT / "_LGOVNET_NEW_SEP/results/step0_conditional_coverage.json"
ALPHA = 0.20
NOMINAL = 1.0 - ALPHA
SEEDS = (0, 1, 2, 3, 4)
TASK = "governance_retain"
MIN_POS = 200          # stratum di bawah ini tidak memicu kriteria
KILL = 0.05
PROCEED = 0.15


def strata(scores: dict) -> np.ndarray:
    """k = jumlah anggota detcmp yang mencocokkan kotak (skor > 0)."""
    return np.sum([(scores[m] > 0).astype(int) for m in MEMBERS], axis=0)


def coverage_in(mask, score, label, thr):
    pos = mask & (label == 1)
    n = int(pos.sum())
    if n == 0:
        return None, 0
    return float((score[pos] >= thr).mean()), n


def main() -> None:
    valid = load_csv(VALID_CSV, positive=TASK)
    test = load_csv(TEST_CSV, positive=TASK)
    print("membangun skor anggota:")
    sv = member_scores("valid", bbox_ids_of(VALID_CSV))
    st = member_scores("test", bbox_ids_of(TEST_CSV))

    kv, kt = strata(sv), strata(st)
    lab_v, lab_t = valid["label"], test["label"]

    e5_v = np.mean([sv[m] for m in MEMBERS], axis=0)
    e5_t = np.mean([st[m] for m in MEMBERS], axis=0)
    e6_v = np.mean([sv["primary"]] + [sv[m] for m in MEMBERS], axis=0)
    e6_t = np.mean([st["primary"]] + [st[m] for m in MEMBERS], axis=0)

    ev = list(EVIDENCE_FEATURES)
    Xv = np.stack([valid[f] for f in ev] + [valid["pred_score"], valid["ev_sds"],
                                            valid["ev_sds_nsrc"]], 1)
    Xt = np.stack([test[f] for f in ev] + [test["pred_score"], test["ev_sds"],
                                           test["ev_sds_nsrc"]], 1)
    gv, gt = [], []
    for s in SEEDS:
        m = fit_mlp(Xv, lab_v.astype(np.float64), seed=s)
        gv.append(apply_mlp(Xv, m)); gt.append(apply_mlp(Xt, m))

    methods = {
        "S (detector score)": ([valid["pred_score"]], [test["pred_score"]]),
        "S+E (full gate)": (gv, gt),
        "E5 (five detectors)": ([e5_v], [e5_t]),
        "E6 (six detectors)": ([e6_v], [e6_t]),
    }

    rep = {"alpha": ALPHA, "nominal": NOMINAL, "task": TASK, "min_positive": MIN_POS,
           "kill_threshold": KILL, "proceed_threshold": PROCEED,
           "protocol": "_LGOVNET_NEW_SEP/PROTOCOL_LOCK_step0_conditional_coverage.md",
           "strata_definition": "k = jumlah dari 5 detektor detcmp yang cocok pada IoU>=0.15",
           "strata_test": {}, "methods": {}, "mondrian_preview": {}}

    print(f"\n=== stratum pada OIA-DDR test (n={lab_t.size}) ===")
    for k in range(6):
        m = kt == k
        npos = int((m & (lab_t == 1)).sum())
        rep["strata_test"][str(k)] = {
            "n": int(m.sum()), "n_positive": npos,
            "positive_rate": float(lab_t[m].mean()) if m.sum() else None}
        print(f"  k={k}  n={int(m.sum()):6d}  positif={npos:6d}  "
              f"laju={lab_t[m].mean() if m.sum() else float('nan'):.3f}")

    max_gap_eligible = 0.0
    max_gap_any = 0.0
    print(f"\n=== selisih cakupan per stratum (nominal {NOMINAL:.2f}) ===")
    for name, (v_list, t_list) in methods.items():
        per_seed_gaps = {str(k): [] for k in range(6)}
        marg = []
        for r in range(len(t_list)):
            thr = conformal_threshold(v_list[r], lab_v, ALPHA)
            cov_all, _ = coverage_in(np.ones_like(lab_t, dtype=bool), t_list[r], lab_t, thr)
            marg.append(cov_all)
            for k in range(6):
                c, n = coverage_in(kt == k, t_list[r], lab_t, thr)
                if c is not None:
                    per_seed_gaps[str(k)].append(c - NOMINAL)
        rec = {"marginal_coverage": float(np.mean(marg)),
               "marginal_gap": float(np.mean(marg)) - NOMINAL, "per_stratum": {}}
        line = []
        for k in range(6):
            g = per_seed_gaps[str(k)]
            if not g:
                rec["per_stratum"][str(k)] = None; line.append("   -  "); continue
            gm = float(np.mean(g))
            npos = rep["strata_test"][str(k)]["n_positive"]
            rec["per_stratum"][str(k)] = {"gap": gm, "n_positive": npos,
                                          "eligible": npos >= MIN_POS}
            max_gap_any = max(max_gap_any, abs(gm))
            if npos >= MIN_POS:
                max_gap_eligible = max(max_gap_eligible, abs(gm))
            line.append(f"{gm:+.3f}")
        rep["methods"][name] = rec
        print(f"  {name:22s} marginal={rec['marginal_gap']:+.3f} | "
              f"k=0..5: {' '.join(f'{x:>7s}' if isinstance(x, str) else x for x in line)}")

    # ---- pratinjau Mondrian, BUKAN bagian kriteria
    print("\n=== pratinjau Mondrian (kalibrasi per stratum), bukan kriteria ===")
    for name, (v_list, t_list) in methods.items():
        gaps = []
        for r in range(len(t_list)):
            per_k = []
            for k in range(6):
                mv, mt = kv == k, kt == k
                if (mv & (lab_v == 1)).sum() < 20 or (mt & (lab_t == 1)).sum() == 0:
                    continue
                thr = conformal_threshold(v_list[r][mv], lab_v[mv], ALPHA)
                c, _ = coverage_in(mt, t_list[r], lab_t, thr)
                if c is not None:
                    per_k.append(abs(c - NOMINAL))
            if per_k:
                gaps.append(max(per_k))
        if gaps:
            rep["mondrian_preview"][name] = {"max_abs_gap": float(np.mean(gaps))}
            print(f"  {name:22s} selisih mutlak terburuk antar-stratum = {np.mean(gaps):.3f}")

    # ---- vonis
    if max_gap_eligible > PROCEED:
        verdict = "LANJUT"
    elif max_gap_any <= KILL:
        verdict = "MATI"
    else:
        verdict = "ABU-ABU"
    rep["max_abs_gap_eligible_strata"] = max_gap_eligible
    rep["max_abs_gap_any_stratum"] = max_gap_any
    rep["verdict"] = verdict

    print(f"\nselisih mutlak terbesar (stratum >= {MIN_POS} positif) = {max_gap_eligible:.3f}")
    print(f"selisih mutlak terbesar (stratum mana pun)            = {max_gap_any:.3f}")
    print(f"VONIS: {verdict}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"ditulis {OUT}")


if __name__ == "__main__":
    main()
