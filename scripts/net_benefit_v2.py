"""net_benefit_v2.py — kurva manfaat bersih BAKU, memperbaiki cacat rancangan v1.

MENGAPA VERSI INI ADA. net_benefit.py (v1) memilih ambang tiap sistem sebagai
KUANTIL skor yang menyamai p_t, sehingga fraksi deteksi yang dipertahankan
dipatok tepat p_t. Akibatnya TP/n <= p_t, sehingga manfaat bersih sebuah sistem
TERKUNCI di bawah p_t berapa pun bagusnya sistem itu. Karena manfaat bersih
retain-semua melebihi p_t pada p_t <= 0.20, kemenangan retain-semua di daerah
itu DIPAKSA oleh aritmetika dan bukan temuan klinis. Terverifikasi: pada
p_t=0.05 batas sistem 0.0500 sedangkan retain-semua 0.3495.

Decision curve analysis baku (Vickers dan Elkin) memakai aturan
    pertahankan bila p_hat >= p_t
dengan p_hat PROBABILITAS TERKALIBRASI, justru supaya fraksi retensi bebas
bergerak dan tidak terikat pada preferensi biaya. Itu yang dikerjakan di sini.

Kalibrasi: penskalaan suhu (fit_temperature), sudah menjadi konvensi repo dan
sudah menjadi baseline bernama di Bagian 4.5. Suhu dipasang pada OIA-DDR
VALIDASI, tidak pernah pada test.

KEBERATAN YANG DIUNGKAP, BUKAN DISEMBUNYIKAN: skor gate pada validasi bersifat
in-sample, karena gate memang dipasang pada validasi (protokol naskah). Suhu
untuk gate karena itu dipasang pada skor in-sample, sedangkan suhu untuk skor
detektor tidak. Supaya kalibrasi tidak diam-diam mengerjakan pekerjaan berat,
kurva dilaporkan DUA KALI: mentah dan terkalibrasi. Bila keduanya sepakat
secara kualitatif, keberatan itu tidak menggigit.

DESKRIPTIF: tidak ada gagal atau berhasil. Yang dilaporkan adalah rentang p_t
tempat tiap sistem memberi manfaat bersih tertinggi, PLUS fraksi retensi pada
tiap p_t, yang pada v1 secara patologis terpaku pada p_t.

Keluaran: _septem/scripts_optionB/net_benefit_v2.json (dan cache skor .npz).
Hanya membaca data.
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.eval_fp_reduction_benchmark import (  # noqa: E402
    apply_mlp, apply_temperature, fit_mlp, fit_temperature, load_csv,
)
from tools.make_indomain_results import (  # noqa: E402
    POSITIVE, SEEDS, T_DEP, V_DEP, gate_matrix,
)

FUSED = ROOT / "_septem/scripts_optionB/fused_scores_oiaddr.csv"
OUT = ROOT / "_septem/scripts_optionB/net_benefit_v2.json"
CACHE = ROOT / "_septem/scripts_optionB/net_benefit_v2_scores.npz"
EV6 = ("ev_sas", "ev_sdca", "ev_aps", "ev_pss", "ev_sds", "ev_sds_nsrc")
PT = [round(x, 3) for x in np.arange(0.05, 0.80, 0.05)]
t0 = time.perf_counter()


def log(m):
    print(f"[nb2] {m} ({time.perf_counter() - t0:.0f}s)", flush=True)


def load_fused(split, ids):
    m = {}
    with open(FUSED, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["split"] == split:
                m[r["bbox_id"]] = float(r["fused"])
    if any(b not in m for b in ids):
        return None
    return np.array([m[b] for b in ids], dtype=np.float64)


def bbox_ids(p):
    with open(p, newline="", encoding="utf-8") as fh:
        return [r["bbox_id"] for r in csv.DictReader(fh)]


def net_benefit(p_hat, y, pt):
    """DCA BAKU: ambang adalah p_t itu sendiri pada probabilitas, bukan kuantil."""
    n = len(y)
    keep = p_hat >= pt
    tp = float((keep & (y == 1)).sum())
    fp = float((keep & (y == 0)).sum())
    return {
        "nb": tp / n - (fp / n) * (pt / (1.0 - pt)),
        "retained_frac": float(keep.mean()),
        "tp_frac": tp / n,
        "fp_frac": fp / n,
    }


def ece(p_hat, y, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    tot = 0.0
    for i in range(bins):
        m = (p_hat >= edges[i]) & (p_hat < edges[i + 1] if i < bins - 1
                                   else p_hat <= edges[i + 1])
        if not m.any():
            continue
        tot += m.mean() * abs(y[m].mean() - p_hat[m].mean())
    return float(tot)


va, te = load_csv(V_DEP, POSITIVE), load_csv(T_DEP, POSITIVE)
yv, yt = va["label"].astype(np.float64), te["label"].astype(np.float64)
fv, ft = load_fused("valid", bbox_ids(V_DEP)), load_fused("test", bbox_ids(T_DEP))
prev = float(yt.mean())
log(f"valid n={len(yv)} test n={len(yt)} prevalensi_test={prev:.4f}; "
    f"skor fusi {'ada' if ft is not None else 'TIDAK ADA'}")

pairs = {"detector_score": (va["pred_score"].astype(np.float64),
                            te["pred_score"].astype(np.float64))}
if ft is not None:
    pairs["score_fusion"] = (fv, ft)

Xv, Xt = gate_matrix(va, "full_gate", None), gate_matrix(te, "full_gate", None)
gv, gt = [], []
for s in SEEDS:
    m = fit_mlp(Xv, yv, seed=s)
    gv.append(apply_mlp(Xv, m))
    gt.append(apply_mlp(Xt, m))
pairs["full_gate"] = (np.mean(gv, 0), np.mean(gt, 0))
log("gate penuh dipasang, 5 seed")

if ft is not None:
    Xvf = np.stack([va[c] for c in EV6] + [fv], 1)
    Xtf = np.stack([te[c] for c in EV6] + [ft], 1)
    fgv, fgt = [], []
    for s in SEEDS:
        m = fit_mlp(Xvf, yv, seed=s)
        fgv.append(apply_mlp(Xvf, m))
        fgt.append(apply_mlp(Xtf, m))
    pairs["gate_on_fusion"] = (np.mean(fgv, 0), np.mean(fgt, 0))
    log("gate di atas skor fusi dipasang, 5 seed")

np.savez_compressed(CACHE, y_test=yt, **{f"test__{k}": v[1] for k, v in pairs.items()},
                    **{f"valid__{k}": v[0] for k, v in pairs.items()}, y_valid=yv)

res = {"pt_grid": PT, "prevalence_test": prev, "n_test": int(len(yt)),
       "rule": "DCA baku: pertahankan bila p_hat >= p_t (bukan kuantil)",
       "calibration": "penskalaan suhu dipasang pada OIA-DDR validasi",
       "v1_defect": ("v1 memakai ambang kuantil sehingga fraksi retensi terpaku "
                     "pada p_t dan NB terkunci <= p_t; kemenangan retain-semua "
                     "pada p_t <= 0.20 dipaksa oleh aritmetika"),
       "systems": {}, "curves": {}, "temperatures": {}}

res["curves"]["treat_all"] = [prev - (1 - prev) * (p / (1 - p)) for p in PT]
res["curves"]["treat_none"] = [0.0 for _ in PT]

for name, (sv, st) in pairs.items():
    T = fit_temperature(np.clip(sv, 1e-6, 1 - 1e-6), yv)
    ct = apply_temperature(np.clip(st, 1e-6, 1 - 1e-6), T)
    res["temperatures"][name] = float(T)
    res["systems"][name] = {
        "temperature": float(T),
        "ece_raw": ece(st, yt), "ece_cal": ece(ct, yt),
        "brier_raw": float(np.mean((st - yt) ** 2)),
        "brier_cal": float(np.mean((ct - yt) ** 2)),
        "mean_p_raw": float(st.mean()), "mean_p_cal": float(ct.mean()),
    }
    for tag, arr in (("raw", st), ("cal", ct)):
        rows = [net_benefit(arr, yt, p) for p in PT]
        res["curves"][f"{name}__{tag}"] = [r["nb"] for r in rows]
        res["systems"][name][f"retained_frac_{tag}"] = [r["retained_frac"] for r in rows]
    s = res["systems"][name]
    log(f"{name}: T={T:.2f} ECE {s['ece_raw']:.4f}->{s['ece_cal']:.4f} "
        f"Brier {s['brier_raw']:.4f}->{s['brier_cal']:.4f}")

for tag in ("raw", "cal"):
    best = {}
    for i, p in enumerate(PT):
        cand = {"treat_all": res["curves"]["treat_all"][i], "treat_none": 0.0}
        for name in pairs:
            cand[name] = res["curves"][f"{name}__{tag}"][i]
        best[str(p)] = max(cand, key=cand.get)
    res[f"best_system_per_pt_{tag}"] = best

OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")

for tag in ("raw", "cal"):
    log(f"--- {tag.upper()} ---")
    for i, p in enumerate(PT):
        if i % 2:
            continue
        row = "  ".join(
            f"{k}={res['curves'][f'{k}__{tag}'][i]:+.4f}"
            f"(r={res['systems'][k][f'retained_frac_{tag}'][i]:.2f})" for k in pairs)
        log(f"  pt={p:.2f} treat_all={res['curves']['treat_all'][i]:+.4f}  {row}"
            f"  -> {res[f'best_system_per_pt_{tag}'][str(p)]}")
log(f"selesai -> {OUT.name}")
