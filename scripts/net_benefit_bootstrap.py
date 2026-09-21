"""net_benefit_bootstrap.py — apakah margin manfaat bersih bertahan terhadap ketidakpastian?

MENGAPA SKRIP INI WAJIB. Setelah fusi skor diberi kalibrasi isotonik yang adil,
keunggulan gate menyusut ke desimal KETIGA (+0.0006 sampai +0.0078). Seluruh
naskah menopang klaimnya dengan bootstrap berklaster-citra 2.000 resample.
Margin sekecil itu TIDAK BOLEH masuk naskah tanpa selang kepercayaan.

Protokol disamakan dengan naskah: resample CITRA dengan pengembalian (bukan
kotak, karena kotak dalam satu citra berkorelasi), 2.000 resample, benih 0,
selang persentil 95%.

Tiga kontras, masing-masing pada seluruh grid p_t:
  A KONSERVATIF  gate_on_fusion MENTAH  -  score_fusion ISOTONIK
                 (isotonik gate bersifat in-sample sehingga menguntungkan gate;
                  kontras ini menolak keuntungan itu dan karenanya dijadikan
                  kesimpulan yang dilaporkan)
  B SIMETRIS     gate_on_fusion ISOTONIK - score_fusion ISOTONIK
  C GATE TERPASANG full_gate MENTAH      - score_fusion ISOTONIK

Hanya membaca cache. Keluaran: net_benefit_bootstrap.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.eval_fp_reduction_benchmark import load_csv  # noqa: E402
from tools.make_indomain_results import POSITIVE, T_DEP  # noqa: E402

D = ROOT / "_septem/scripts_optionB"
N_BOOT, BSEED = 2000, 0
PT = [round(x, 3) for x in np.arange(0.05, 0.80, 0.05)]

z = np.load(D / "net_benefit_v2_scores.npz")
yv, yt = z["y_valid"], z["y_test"]
img = load_csv(T_DEP, POSITIVE)["image_id"]
assert len(img) == len(yt), "image_id dan label tidak sepanjang yang sama"


def iso(name):
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir.fit(z[f"valid__{name}"], yv)
    return np.clip(ir.predict(z[f"test__{name}"]), 0.0, 1.0)


P = {"fusion_iso": iso("score_fusion"),
     "gatefus_iso": iso("gate_on_fusion"),
     "gatefus_raw": z["test__gate_on_fusion"],
     "fullgate_raw": z["test__full_gate"]}

uniq, inv = np.unique(img, return_inverse=True)
groups = [np.flatnonzero(inv == k) for k in range(len(uniq))]
print(f"[bs] test n={len(yt)} citra={len(uniq)} resample={N_BOOT} benih={BSEED}", flush=True)

pos = {k: (v[:, None] >= np.array(PT)[None, :]) for k, v in P.items()}
ypos = (yt == 1)
w = np.array([p / (1 - p) for p in PT])


def nb_rows(idx):
    n = len(idx)
    yb = ypos[idx]
    out = {}
    for k, keepm in pos.items():
        km = keepm[idx]
        tp = (km & yb[:, None]).sum(0)
        fp = (km & ~yb[:, None]).sum(0)
        out[k] = tp / n - (fp / n) * w
    return out


base = nb_rows(np.arange(len(yt)))
CON = {"A_konservatif": ("gatefus_raw", "fusion_iso"),
       "B_simetris": ("gatefus_iso", "fusion_iso"),
       "C_gate_terpasang": ("fullgate_raw", "fusion_iso")}
acc = {c: np.empty((N_BOOT, len(PT))) for c in CON}
rng = np.random.default_rng(BSEED)
for b in range(N_BOOT):
    pick = rng.integers(0, len(uniq), len(uniq))
    idx = np.concatenate([groups[i] for i in pick])
    r = nb_rows(idx)
    for c, (a, bb) in CON.items():
        acc[c][b] = r[a] - r[bb]
    if (b + 1) % 500 == 0:
        print(f"[bs] {b + 1}/{N_BOOT}", flush=True)

res = {"n": int(len(yt)), "n_images": int(len(uniq)), "n_boot": N_BOOT,
       "bootstrap_seed": BSEED, "pt_grid": PT, "contrasts": {}}
for c, (a, bb) in CON.items():
    obs = base[a] - base[bb]
    lo = np.percentile(acc[c], 2.5, axis=0)
    hi = np.percentile(acc[c], 97.5, axis=0)
    excl = (lo > 0) | (hi < 0)
    res["contrasts"][c] = {
        "numerator": a, "denominator": bb,
        "observed": obs.tolist(), "ci_lo": lo.tolist(), "ci_hi": hi.tolist(),
        "excludes_zero": excl.tolist(),
        "n_pt_excluding_zero": int(excl.sum()),
        "n_pt_favoring_numerator": int((obs > 0).sum()),
    }
    print(f"\n=== {c}: {a} - {bb} ===")
    print(f"{'pt':>5s}{'selisih':>11s}{'CI95 bawah':>13s}{'CI95 atas':>12s}{'kecualikan 0':>14s}")
    for i, p in enumerate(PT):
        print(f"{p:5.2f}{obs[i]:+11.4f}{lo[i]:+13.4f}{hi[i]:+12.4f}"
              f"{('YA' if excl[i] else 'tidak'):>14s}")
    print(f"  -> CI mengecualikan nol pada {int(excl.sum())}/{len(PT)} titik p_t")

(D / "net_benefit_bootstrap.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
print("\nselesai -> net_benefit_bootstrap.json")
