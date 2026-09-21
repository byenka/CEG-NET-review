"""net_benefit_isotonic_check.py — uji KEADILAN untuk klaim manfaat bersih v2.

KEBERATAN YANG DIUJI. Pada net_benefit_v2, gate mengalahkan fusi skor pada
manfaat bersih, padahal fusi menang pada AUPRC. Sebabnya kalibrasi: gate
dilatih dengan NLL sehingga keluarannya sudah probabilitas (T=1.00, ECE 0.0225),
sedangkan fusi skor tidak pernah dilatih menjadi probabilitas (ECE 0.1864).
Mengadu keduanya pada ambang probabilitas tetap berarti mengadu pengklasifikasi
terlatih melawan skor mentah. Penskalaan suhu hanya satu parameter dan menyisakan
ECE fusi 0.1114.

UJI YANG BENAR: beri tiap pembanding kalibrator TERKUAT yang wajar, yaitu
regresi isotonik yang dipasang pada OIA-DDR VALIDASI, lalu ulangi DCA.
  Bila fusi terkalibrasi isotonik MENYUSUL gate, klaim v2 adalah artefak
  kalibrasi lemah dan TIDAK BOLEH masuk naskah.
  Bila fusi tetap tertinggal, klaim bertahan dalam bentuk konservatif.

CATATAN KEJUJURAN: isotonik untuk GATE dipasang pada skor validasi yang bersifat
in-sample (gate memang dipasang pada validasi), sehingga menguntungkan gate.
Karena itu perbandingan yang dilaporkan sebagai kesimpulan adalah yang
KONSERVATIF: fusi dengan kalibrasi terbaiknya melawan gate TANPA isotonik.

Hanya membaca cache skor. Keluaran: net_benefit_isotonic_check.json.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "_septem/scripts_optionB"
z = np.load(D / "net_benefit_v2_scores.npz")
yv, yt = z["y_valid"], z["y_test"]
PT = [round(x, 3) for x in np.arange(0.05, 0.80, 0.05)]
SYS = ["detector_score", "score_fusion", "full_gate", "gate_on_fusion"]
prev = float(yt.mean())


def nb(p, y, pt):
    keep = p >= pt
    n = len(y)
    return float((keep & (y == 1)).sum()) / n - float((keep & (y == 0)).sum()) / n * (pt / (1 - pt))


def ece(p, y, bins=10):
    e = np.linspace(0, 1, bins + 1)
    t = 0.0
    for i in range(bins):
        m = (p >= e[i]) & (p <= e[i + 1] if i == bins - 1 else p < e[i + 1])
        if m.any():
            t += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(t)


out = {"prevalence": prev, "pt_grid": PT, "curves": {}, "diag": {},
       "treat_all": [prev - (1 - prev) * (p / (1 - p)) for p in PT]}
iso_t = {}
for s in SYS:
    sv, st = z[f"valid__{s}"], z[f"test__{s}"]
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir.fit(sv, yv)
    pt_iso = np.clip(ir.predict(st), 0.0, 1.0)
    iso_t[s] = pt_iso
    out["curves"][f"{s}__iso"] = [nb(pt_iso, yt, p) for p in PT]
    out["diag"][s] = {"ece_iso": ece(pt_iso, yt),
                      "brier_iso": float(np.mean((pt_iso - yt) ** 2)),
                      "retained_frac_iso": [float((pt_iso >= p).mean()) for p in PT]}
    print(f"{s:<16s} ECE_iso={out['diag'][s]['ece_iso']:.4f} "
          f"Brier_iso={out['diag'][s]['brier_iso']:.4f}")

v2 = json.loads((D / "net_benefit_v2.json").read_text(encoding="utf-8"))
print("\n=== UJI KONSERVATIF: fusi ISOTONIK lawan gate MENTAH ===")
print(f"{'pt':>5s}{'fusi_iso':>11s}{'gateFusi_mth':>14s}{'gatePenuh_mth':>15s}{'selisih':>10s}{'menang':>12s}")
n_gate_win = 0
for i, p in enumerate(PT):
    fi = out["curves"]["score_fusion__iso"][i]
    gf = v2["curves"]["gate_on_fusion__raw"][i]
    fg = v2["curves"]["full_gate__raw"][i]
    dd = gf - fi
    w = "gate" if dd > 0 else "fusi"
    if dd > 0:
        n_gate_win += 1
    print(f"{p:5.2f}{fi:+11.4f}{gf:+14.4f}{fg:+15.4f}{dd:+10.4f}{w:>12s}")
out["conservative"] = {"gate_on_fusion_raw_beats_fusion_iso_at": n_gate_win, "of": len(PT)}
print(f"\ngate-atas-fusi (mentah) mengungguli fusi ISOTONIK pada {n_gate_win}/{len(PT)} titik p_t")
best_iso = {}
for i, p in enumerate(PT):
    c = {"treat_all": out["treat_all"][i], "treat_none": 0.0}
    for s in SYS:
        c[s] = out["curves"][f"{s}__iso"][i]
    best_iso[str(p)] = max(c, key=c.get)
out["best_system_per_pt_iso"] = best_iso
print("\n=== SEMUA ISOTONIK, siapa terbaik per p_t ===")
for p in PT:
    print(f"  pt={p:.2f} -> {best_iso[str(p)]}")
(D / "net_benefit_isotonic_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("\nselesai -> net_benefit_isotonic_check.json")
