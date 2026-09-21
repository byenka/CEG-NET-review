"""target_recalibration.py — apakah sedikit sampel berlabel dari domain target memulihkan titik kerja?

Naskah MENGAKUI keterbatasan ini: tau* yang dipasang pada domain sumber tidak transfer, dan
rekalibrasi domain target disebut sebagai pekerjaan masa depan. Skrip ini mengukurnya.

Rancangan (sengaja TIDAK optimistis):
  Memilih tau pada data yang sama lalu mengujinya di situ akan melebih-lebihkan. Karena itu tiap
  kohort dibelah PER CITRA menjadi dua bagian. Kalibrasi pada bagian A, evaluasi pada bagian B,
  lalu ditukar, lalu dirata-ratakan. Ini meniru penempatan nyata: sedikit sampel berlabel dari
  lokasi target, bukan seluruh kohort.

Dibandingkan tiga hal pada bagian evaluasi yang sama:
  tau_sumber  : dipilih pada OIA-DDR valid (yang dipakai naskah sekarang)
  tau_target  : dipilih pada separuh kohort target yang TIDAK dievaluasi
  tau_oracle  : dipilih pada bagian evaluasi itu sendiri (batas atas, tidak dapat dicapai
                di penempatan nyata; hanya untuk menunjukkan seberapa jauh tau_target dari ideal)

KRITERIA DIPATOK DI MUKA, sebelum melihat hasil:
  BERHASIL bila tau_target mencapai retensi TP dalam +-0.03 dari sasaran 0.80 pada bagian
  evaluasi, di KETIGA kohort eksternal, sementara tau_sumber tidak.
  Hasil dilaporkan apa pun keluarannya.

Gate: protokol naskah (fit_mlp pada OIA-DDR valid, konfigurasi deployed, 5 seed).
Keluaran: _septem/scripts_optionB/target_recalibration.json. Hanya membaca.
"""
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.eval_fp_reduction_benchmark import (  # noqa: E402
    apply_mlp, eval_at_threshold, fit_mlp, load_csv, threshold_for_tp_ret,
)
from tools.make_indomain_results import (  # noqa: E402
    POSITIVE, SEEDS, TP_RET, V_DEP, gate_matrix,
)

REF = ROOT / "outputs/0506/evsds_pss/domain_shift_5seed.json"
OUT = ROOT / "_septem/scripts_optionB/target_recalibration.json"
SPLIT_SEED = 0
TOL = 0.03
t0 = time.perf_counter()


def log(m):
    print(f"[recal] {m} ({time.perf_counter() - t0:.0f}s)", flush=True)


def halves(image_id):
    """Belah per CITRA, bukan per kotak: kotak dari satu citra tidak boleh terpisah."""
    uniq = np.unique(image_id)
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(uniq))
    a = set(uniq[perm[: len(uniq) // 2]].tolist())
    mask_a = np.array([i in a for i in image_id])
    return mask_a, ~mask_a


va = load_csv(V_DEP, POSITIVE)
yv = va["label"]
Xv = gate_matrix(va, "full_gate", None)
models = [fit_mlp(Xv, yv.astype(np.float64), seed=s) for s in SEEDS]
sv = [apply_mlp(Xv, m) for m in models]
tau_src = [threshold_for_tp_ret(s, yv, TP_RET) for s in sv]
log(f"gate dipasang pada OIA-DDR valid, 5 seed; tau_sumber = "
    f"{[round(t, 4) for t in tau_src]}")

ref = json.loads(REF.read_text(encoding="utf-8"))
res = {"tp_ret_target": TP_RET, "tolerance": TOL, "split_seed": SPLIT_SEED,
       "design": "belah per citra, kalibrasi di satu bagian, evaluasi di bagian lain, lalu tukar",
       "cohorts": []}

for c in ref["cohorts"]:
    name = c["name"]
    d = load_csv(ROOT / c["csv"], POSITIVE)
    y, img = d["label"], d["image_id"]
    X = gate_matrix(d, "full_gate", None)
    mA, mB = halves(img)
    rec = {"name": name, "n": int(len(y)), "n_images": int(len(np.unique(img))),
           "n_A": int(mA.sum()), "n_B": int(mB.sum()), "per_seed": []}

    for i, m in enumerate(models):
        st = apply_mlp(X, m)
        row = {"seed": SEEDS[i]}
        for lab, (cal, ev) in (("A_cal_B_eval", (mA, mB)), ("B_cal_A_eval", (mB, mA))):
            tau_tgt = threshold_for_tp_ret(st[cal], y[cal], TP_RET)
            tau_orc = threshold_for_tp_ret(st[ev], y[ev], TP_RET)
            out = {}
            for tname, tau in (("source", tau_src[i]), ("target", tau_tgt), ("oracle", tau_orc)):
                e = eval_at_threshold(st[ev], y[ev], tau)
                out[tname] = {"tau": float(tau), "tp_ret": e["tp_ret"],
                              "fp_supp": e["fp_supp"], "precision": e["precision_after"],
                              "abs_err_tp": abs(e["tp_ret"] - TP_RET)}
            row[lab] = out
        rec["per_seed"].append(row)

    for tname in ("source", "target", "oracle"):
        errs = [r[lab][tname]["abs_err_tp"] for r in rec["per_seed"]
                for lab in ("A_cal_B_eval", "B_cal_A_eval")]
        tps = [r[lab][tname]["tp_ret"] for r in rec["per_seed"]
               for lab in ("A_cal_B_eval", "B_cal_A_eval")]
        fps = [r[lab][tname]["fp_supp"] for r in rec["per_seed"]
               for lab in ("A_cal_B_eval", "B_cal_A_eval")]
        rec[tname] = {"tp_ret_mean": statistics.mean(tps), "fp_supp_mean": statistics.mean(fps),
                      "abs_err_tp_mean": statistics.mean(errs),
                      "within_tol_frac": sum(1 for e in errs if e <= TOL) / len(errs)}
    log(f"{name}: sumber TP {rec['source']['tp_ret_mean']:.3f} (err {rec['source']['abs_err_tp_mean']:.3f}) | "
        f"target TP {rec['target']['tp_ret_mean']:.3f} (err {rec['target']['abs_err_tp_mean']:.3f}, "
        f"dalam toleransi {rec['target']['within_tol_frac']:.0%}) | "
        f"oracle TP {rec['oracle']['tp_ret_mean']:.3f}")
    res["cohorts"].append(rec)
    del d, X

ext = [c for c in res["cohorts"] if c["name"] != "OIA-DDR"]
res["verdict"] = {
    "target_within_tol_all_external": all(c["target"]["abs_err_tp_mean"] <= TOL for c in ext),
    "source_within_tol_all_external": all(c["source"]["abs_err_tp_mean"] <= TOL for c in ext),
}
res["verdict"]["criterion_met"] = (res["verdict"]["target_within_tol_all_external"]
                                   and not res["verdict"]["source_within_tol_all_external"])
OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
log(f"kriteria terpenuhi: {res['verdict']['criterion_met']} -> {OUT.name}")
