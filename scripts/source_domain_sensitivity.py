"""source_domain_sensitivity.py — apakah hasil gate artefak karena selalu dipasang di OIA-DDR?

Keberatan reviewer yang dijawab: seluruh hasil gating memasang gate pada OIA-DDR validasi. Bila
pemetaan bukti-ke-keandalan itu khas OIA-DDR, hasilnya tidak berarti banyak. Skrip ini memasang
gate pada kohort LAIN lalu mengujinya pada OIA-DDR test, memakai CSV bukti yang sudah ada.
Tidak ada Stage 3C baru, tidak ada server.

KRITERIA DIPATOK DI MUKA, sebelum melihat hasil:
  BERHASIL bila gate yang dipasang di FGADR mengungguli skor detektor pada OIA-DDR test di
  sekurangnya 4 dari 5 seed, dengan rerata delta AUPRC di atas nol.
  HASIL DILAPORKAN APA PUN KELUARANNYA. Hasil negatif TIDAK membantah klaim naskah mana pun:
  naskah mengklaim lapisan post-hoc yang dipasang pada validasi domain penempatan, bukan
  pemetaan universal. Negatif berarti keterbatasan yang presisi, yaitu gate berpindah sebagai
  RESEP dan bukan sebagai objek terlatih.

Sumber yang dipakai dan alasannya:
  FGADR train  510.825 kotak, 1.737 citra  -> sumber utama, data terbanyak
  TJDR  test    22.397 kotak,   104 citra  -> sumber kedua
  IDRiD TIDAK dipakai sebagai sumber: hanya 27 citra, sehingga bootstrap image-clustered terlalu
  lemah untuk mendukung klaim apa pun. Ia tetap sah sebagai target.

Confound yang WAJIB diungkap bersama hasilnya:
  prevalensi positif berbeda jauh antar sumber (FGADR 0.175, TJDR 0.143, OIA-DDR valid 0.356),
  sehingga eksperimen ini menguji pergeseran domain DAN pergeseran prior kelas sekaligus.
  Prevalensi UJI konstan karena semua dievaluasi pada OIA-DDR test yang sama.

Keluaran: _septem/scripts_optionB/source_domain_sensitivity.json. Hanya membaca.
"""
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.eval_fp_reduction_benchmark import apply_mlp, fit_mlp, load_csv  # noqa: E402
from tools.make_indomain_results import (  # noqa: E402
    N_BOOT, POSITIVE, SEEDS, T_DEP, V_DEP, gate_matrix, paired_bootstrap_ap,
)

OUT = ROOT / "_septem/scripts_optionB/source_domain_sensitivity.json"
SOURCES = [
    ("OIA-DDR valid", "outputs/0506/evsds_v2sec/oiaddr/valid/governance_features_valid.csv"),
    ("FGADR train", "outputs/0506/evsds_pss/fgadr/train/governance_features_train.csv"),
    ("TJDR test", "outputs/0506/evsds_pss/tjdr/test/governance_features_test.csv"),
]
t0 = time.perf_counter()


def log(m):
    print(f"[src] {m} ({time.perf_counter() - t0:.0f}s)", flush=True)


te = load_csv(T_DEP, POSITIVE)
yt, img = te["label"], te["image_id"]
m0 = float(average_precision_score(yt, te["pred_score"]))
Xt = {g: gate_matrix(te, g, None) for g in ("kg_gate", "full_gate")}
log(f"target tetap: OIA-DDR test n={len(yt)} prevalensi={yt.mean():.4f} M0 AP={m0:.5f}")

res = {"target": "OIA-DDR test", "m0_ap": m0, "n_target": int(len(yt)),
       "prevalence_target": float(yt.mean()), "sources": []}

for sname, spath in SOURCES:
    d = load_csv(ROOT / spath, POSITIVE)
    ys = d["label"]
    rec = {"name": sname, "n": int(len(ys)), "prevalence": float(ys.mean()),
           "n_images": int(len(np.unique(d["image_id"]))), "gates": {}}
    log(f"sumber {sname}: n={rec['n']} prevalensi={rec['prevalence']:.4f} citra={rec['n_images']}")

    for g in ("kg_gate", "full_gate"):
        Xs = gate_matrix(d, g, None)
        aps, scores = [], {}
        for s in SEEDS:
            m = fit_mlp(Xs, ys.astype(np.float64), seed=s)
            st = apply_mlp(Xt[g], m)
            scores[s] = st
            aps.append(float(average_precision_score(yt, st)))
        gain = [a - m0 for a in aps]
        rec["gates"][g] = {
            "ap_per_seed": aps, "ap_mean": statistics.mean(aps), "ap_sd": statistics.stdev(aps),
            "gain_vs_m0_mean": statistics.mean(gain),
            "n_gain_pos": sum(1 for x in gain if x > 0),
        }
        log(f"  {sname} -> {g}: AP {statistics.mean(aps):.5f} +- {statistics.stdev(aps):.5f} "
            f"| vs M0 {statistics.mean(gain):+.5f} ({sum(1 for x in gain if x > 0)}/5 seed)")
        if g == "full_gate":
            rows = []
            for s in SEEDS:
                b = paired_bootstrap_ap(scores[s], te["pred_score"], yt, img, N_BOOT, 0)
                b["seed"] = s
                rows.append(b)
            rec["gates"][g]["bootstrap_vs_m0"] = rows
            rec["gates"][g]["n_ci_above_0"] = sum(1 for b in rows if b["gain_ci95"][0] > 1e-6)
            log(f"  {sname} -> {g}: CI di atas nol pada "
                f"{rec['gates'][g]['n_ci_above_0']}/5 seed")
    res["sources"].append(rec)
    del d

fg = next((s for s in res["sources"] if s["name"].startswith("FGADR")), None)
res["verdict"] = {
    "criterion": "gate FGADR unggul atas M0 pada >=4 dari 5 seed dengan rerata gain > 0",
    "criterion_met": bool(fg and fg["gates"]["full_gate"]["n_gain_pos"] >= 4
                          and fg["gates"]["full_gate"]["gain_vs_m0_mean"] > 0),
}
OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
log(f"kriteria terpenuhi: {res['verdict']['criterion_met']} -> {OUT.name}")
