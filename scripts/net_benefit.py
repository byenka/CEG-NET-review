"""net_benefit.py — CACAT RANCANGAN, JANGAN DIPAKAI. Gunakan net_benefit_v2.py.

==========================================================================
PERINGATAN (ditambahkan 2026-09-17, setelah skrip ini dijalankan)
==========================================================================
Skrip ini memilih ambang tiap sistem sebagai KUANTIL skor yang menyamai p_t
(lihat net_benefit() di bawah). Akibatnya fraksi deteksi yang dipertahankan
terpaku tepat pada p_t, sehingga TP/n <= p_t dan manfaat bersih sebuah sistem
TERKUNCI di bawah p_t berapa pun bagusnya sistem itu.

Karena manfaat bersih retain-semua melebihi p_t pada p_t <= 0.20, kemenangan
retain-semua di daerah itu DIPAKSA oleh aritmetika dan BUKAN temuan klinis.
Terverifikasi: pada p_t=0.05 batas sistem 0.0500 lawan retain-semua 0.3495.

Decision curve analysis baku (Vickers dan Elkin) memakai aturan
    pertahankan bila p_hat >= p_t
dengan p_hat PROBABILITAS TERKALIBRASI, justru supaya fraksi retensi bebas
bergerak dan tidak terikat pada preferensi biaya.

ANGKA DARI net_benefit.json TIDAK SAH DIKUTIP. Khususnya kesimpulan
"skor fusi mengungguli gate penuh pada 15/15 titik p_t" dan "gate-atas-fusi
terbaik hanya pada p_t 0.30-0.55" adalah artefak rancangan ini.

Pengganti yang sah:
    net_benefit_v2.py             DCA baku + kalibrasi penskalaan suhu
    net_benefit_isotonic_check.py uji keadilan, beri baseline kalibrator terbaik
    net_benefit_bootstrap.py      CI bootstrap berklaster-citra atas selisih NB

Berkas ini SENGAJA TIDAK DIHAPUS supaya koreksinya terlihat, bukan tersembunyi.
==========================================================================

Deskripsi asli (dipertahankan apa adanya sebagai catatan sejarah):
kurva manfaat bersih, nilai klinis tanpa pembaca manusia.

Keberatan yang dijawab: "nilai klinisnya belum ditunjukkan". Selama ini jawabannya dilempar ke
studi pembaca, yang menuntut dokter mata dan tidak dapat dikerjakan di sini. Analisis manfaat
bersih adalah cara baku menunjukkan kegunaan lintas rasio biaya, seluruhnya dari skor dan label
yang sudah ada.

Definisi (Vickers dan Elkin, decision curve analysis), diterapkan pada tugas triase
retain-versus-suppress:
    NB(p_t) = TP/n  -  (FP/n) * p_t/(1 - p_t)
dengan ambang pada tiap sistem dipilih sebagai kuantil skor yang menyamai p_t, dan p_t adalah
probabilitas ambang: rasio p_t/(1-p_t) menyatakan berapa banyak retain keliru yang bersedia
ditukar dengan satu retain benar yang terlewat.

Pembanding: retain-semua (batas atas naif), retain-tidak-ada (nol), skor detektor, skor fusi
bila tersedia, gate penuh, dan gate di atas skor fusi. Skor fusi dibaca dari
fused_scores_oiaddr.csv bila ada.

Ini DESKRIPTIF: tidak ada gagal atau berhasil. Yang dilaporkan adalah rentang p_t tempat tiap
sistem memberi manfaat bersih tertinggi.

Keluaran: _septem/scripts_optionB/net_benefit.json. Hanya membaca.
"""
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.eval_fp_reduction_benchmark import apply_mlp, fit_mlp, load_csv  # noqa: E402
from tools.make_indomain_results import (  # noqa: E402
    POSITIVE, SEEDS, T_DEP, V_DEP, gate_matrix,
)

FUSED = ROOT / "_septem/scripts_optionB/fused_scores_oiaddr.csv"
OUT = ROOT / "_septem/scripts_optionB/net_benefit.json"
EV = ("ev_sas", "ev_sdca", "ev_aps", "ev_pss", "ev_sds", "ev_sds_nsrc")
PT = [round(x, 3) for x in np.arange(0.05, 0.80, 0.05)]
t0 = time.perf_counter()


def log(m):
    print(f"[nb] {m} ({time.perf_counter() - t0:.0f}s)", flush=True)


def load_fused(split, bbox_ids):
    if not FUSED.is_file():
        return None
    m = {}
    with open(FUSED, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["split"] == split:
                m[r["bbox_id"]] = float(r["fused"])
    if any(b not in m for b in bbox_ids):
        return None
    return np.array([m[b] for b in bbox_ids], dtype=np.float64)


def bbox_ids(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as fh:
        return [r["bbox_id"] for r in csv.DictReader(fh)]


def net_benefit(score, y, pt):
    """Ambang = kuantil skor yang menyamai p_t, sehingga semua sistem dibandingkan adil."""
    n = len(y)
    tau = float(np.quantile(score, 1.0 - pt))
    keep = score >= tau
    tp = float((keep & (y == 1)).sum())
    fp = float((keep & (y == 0)).sum())
    return tp / n - (fp / n) * (pt / (1.0 - pt))


va, te = load_csv(V_DEP, POSITIVE), load_csv(T_DEP, POSITIVE)
yv, yt = va["label"], te["label"]
fv = load_fused("valid", bbox_ids(V_DEP))
ft = load_fused("test", bbox_ids(T_DEP))
log(f"test n={len(yt)} positif={int(yt.sum())} ({yt.mean():.3f}); "
    f"skor fusi {'ada' if ft is not None else 'tidak ada'}")

systems = {"detector_score": te["pred_score"]}
if ft is not None:
    systems["score_fusion"] = ft

Xv, Xt = gate_matrix(va, "full_gate", None), gate_matrix(te, "full_gate", None)
gate_scores = []
for s in SEEDS:
    m = fit_mlp(Xv, yv.astype(np.float64), seed=s)
    gate_scores.append(apply_mlp(Xt, m))
systems["full_gate"] = np.mean(gate_scores, axis=0)
log("gate penuh dipasang, 5 seed (skor dirata-ratakan antar seed untuk kurva)")

if ft is not None:
    Xvf = np.stack([va[c] for c in EV] + [fv], 1)
    Xtf = np.stack([te[c] for c in EV] + [ft], 1)
    gf = [apply_mlp(Xtf, fit_mlp(Xvf, yv.astype(np.float64), seed=s)) for s in SEEDS]
    systems["gate_on_fusion"] = np.mean(gf, axis=0)
    log("gate di atas skor fusi dipasang, 5 seed")

prev = float(yt.mean())
curves = {"treat_all": [prev - (1 - prev) * (p / (1 - p)) for p in PT],
          "treat_none": [0.0 for _ in PT]}
for name, s in systems.items():
    curves[name] = [net_benefit(np.asarray(s, dtype=np.float64), yt, p) for p in PT]

best = {}
for i, p in enumerate(PT):
    vals = {k: v[i] for k, v in curves.items()}
    best[str(p)] = max(vals, key=vals.get)

res = {"pt_grid": PT, "prevalence": prev, "n": int(len(yt)),
       "curves": curves, "best_system_per_pt": best,
       "note": "NB = TP/n - (FP/n) * pt/(1-pt); ambang = kuantil skor yang menyamai pt"}
OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")

log("manfaat bersih pada beberapa p_t:")
for i, p in enumerate(PT):
    if i % 3:
        continue
    row = "  ".join(f"{k}={curves[k][i]:+.4f}" for k in curves if k != "treat_none")
    log(f"  pt={p:.2f}  {row}  -> terbaik: {best[str(p)]}")
log(f"selesai -> {OUT.name}")
