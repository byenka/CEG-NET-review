"""annotation_endpoint_experiment.py — validasi prior terhadap endpoint anotasi ahli.

Protokol dikunci di _NATIVE/PROTOCOL_LOCK_annotation_endpoint.md SEBELUM dijalankan.
Ringkasnya: outcome berasal dari anotasi ahli (kelas prediksi != kelas GT pada deteksi
terlokalisasi), prediktor hanya dari keluaran model dan prior. SDM tidak dipakai untuk
mendefinisikan outcome, sehingga uji ini tidak sirkular seperti Tabel 13.

Kriteria lolos: A1 AUROC SDCA > 0.5 (CI tidak memuat 0.5) di OIA-DDR test dan >=2/3 eksternal;
A2 SDCA > kontrol permutasi (CI berpasangan > 0) di OIA-DDR test; A3 AUROC(MARGIN+SDCA) >
AUROC(MARGIN) (CI berpasangan > 0) di OIA-DDR test dan >=2/3 eksternal.

CPU saja. Hanya membaca CSV dan graf di lokasi asli; menulis ke _NATIVE/results/.

Jalankan dari root repositori:
  python _NATIVE/scripts/annotation_endpoint_experiment.py
"""
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve()
NATIVE = HERE.parents[1]
ROOT = HERE.parents[2]

GRAPH = ROOT / "outputs/stage3_audit/kg_graph.json"
FIT = ("OIA-DDR valid", ROOT / "outputs/0506/evsds_v2sec/oiaddr/valid/governance_features_valid.csv")
EVAL = [
    ("OIA-DDR test", ROOT / "outputs/0506/evsds_v2sec/oiaddr/test/governance_features_test.csv"),
    ("IDRiD", ROOT / "outputs/0506/evsds_pss/idrid/test/governance_features_test.csv"),
    ("TJDR", ROOT / "outputs/0506/evsds_pss/tjdr/test/governance_features_test.csv"),
    ("FGADR", ROOT / "outputs/0506/evsds_pss/fgadr/train/governance_features_train.csv"),
]
LESIONS = ("MA", "HE", "Hard_EX", "Soft_EX")
PROB_COLS = ("prob_ma", "prob_he", "prob_hardex", "prob_softex")
N_BOOT, BOOT_SEED = 2000, 0
PERM_SEEDS = (101, 102, 103, 104, 105)
DELTA = 0.30
OUT = NATIVE / "results" / "annotation_endpoint.json"
T0 = time.perf_counter()


def log(m):
    print(f"[endpoint] {m} ({time.perf_counter() - T0:.0f}s)", flush=True)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


# ----------------------------------------------------------------------- SDM prior
def load_sdm():
    """Matriks jarak semantik dari field clinical_cost (BUKAN weight yang berlantai epsilon)."""
    g = json.loads(GRAPH.read_text(encoding="utf-8"))
    idx = {n: i for i, n in enumerate(LESIONS)}
    raw = np.zeros((4, 4))
    for e in g["edges"]["severity_transition"]:
        a, b = idx[e["src_name"]], idx[e["dst_name"]]
        raw[a, b] = raw[b, a] = float(e["clinical_cost"])
    return raw / raw.max()


def permuted_sdm(sdm, seed):
    """Permutasi 6 nilai off-diagonal antar pasangan; himpunan nilai dipertahankan."""
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    vals = np.random.default_rng(seed).permutation([sdm[i, j] for i, j in pairs])
    out = np.zeros((4, 4))
    for (i, j), v in zip(pairs, vals):
        out[i, j] = out[j, i] = v
    return out


# ---------------------------------------------------------------------------- data
def load_cohort(path):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    probs = np.array([[float(r[c]) for c in PROB_COLS] for r in rows], dtype=np.float64)
    probs /= np.clip(probs.sum(1, keepdims=True), 1e-12, None)
    order = np.argsort(-probs, axis=1)
    iou = np.array([float(r["best_iou"]) if r["best_iou"] not in ("", "nan") else 0.0 for r in rows])
    gt = np.array([r["best_gt_class"] for r in rows])
    keep = (iou >= 0.5) & (gt != "")
    idx = {n: i for i, n in enumerate(LESIONS)}
    d = {
        "top1": order[:, 0][keep], "top2": order[:, 1][keep],
        "pred_class": np.array([r["pred_class"] for r in rows])[keep],
        "majority_class": np.array([r["majority_class"] for r in rows])[keep],
        "gt_class": gt[keep],
        "image_id": np.array([r["image_id"] for r in rows])[keep],
        "sdca": np.array([float(r["ev_sdca"]) for r in rows])[keep],
        "margin": np.array([float(r["unc_margin"]) for r in rows])[keep],
        "entropy": np.array([float(r["unc_entconf"]) for r in rows])[keep],
        "msp": np.array([float(r["unc_msp"]) for r in rows])[keep],
        "n_all": len(rows),
    }
    d["mismatch"] = (d["pred_class"] != d["gt_class"]).astype(np.int64)
    d["mismatch_major"] = (d["majority_class"] != d["gt_class"]).astype(np.int64)
    d["gt_idx"] = np.array([idx.get(g, -1) for g in d["gt_class"]])
    return d


def sdca_from(sdm, d):
    return sdm[d["top1"], d["top2"]]


# ---------------------------------------------------------------------- statistics
def image_boot_idx(image_id, n_boot, seed):
    uniq, inv = np.unique(image_id, return_inverse=True)
    groups = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield np.concatenate([groups[i] for i in rng.integers(0, len(uniq), len(uniq))])


def auc_ci(y, s, image_id):
    """AUROC dengan CI bootstrap berklaster-citra; draw degenerate dilewati."""
    point = float(roc_auc_score(y, s))
    vals = []
    for idx in image_boot_idx(image_id, N_BOOT, BOOT_SEED):
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        vals.append(roc_auc_score(yb, s[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"auroc": point, "ci95": [float(lo), float(hi)],
            "excludes_half": bool(lo > 0.5 or hi < 0.5), "n_valid_draws": len(vals)}


def paired_auc_delta(y, s_a, s_b, image_id):
    point = float(roc_auc_score(y, s_a) - roc_auc_score(y, s_b))
    vals = []
    for idx in image_boot_idx(image_id, N_BOOT, BOOT_SEED):
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        vals.append(roc_auc_score(yb, s_a[idx]) - roc_auc_score(yb, s_b[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"delta_auroc": point, "ci95": [float(lo), float(hi)],
            "sig_positive": bool(lo > 0.0), "n_valid_draws": len(vals)}


def odds_ratio(y, flag, image_id):
    def _or(yy, ff):
        a = float(((ff == 1) & (yy == 1)).sum()) + 0.5
        b = float(((ff == 1) & (yy == 0)).sum()) + 0.5
        c = float(((ff == 0) & (yy == 1)).sum()) + 0.5
        dd = float(((ff == 0) & (yy == 0)).sum()) + 0.5
        return (a / b) / (c / dd)
    point = _or(y, flag)
    vals = [_or(y[i], flag[i]) for i in image_boot_idx(image_id, N_BOOT, BOOT_SEED)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"odds_ratio": float(point), "ci95": [float(lo), float(hi)],
            "excludes_one": bool(lo > 1.0 or hi < 1.0),
            "rate_high": float(y[flag == 1].mean()) if (flag == 1).any() else None,
            "rate_low": float(y[flag == 0].mean()) if (flag == 0).any() else None}


# --------------------------------------------------------------------------- main
def main():
    sdm = load_sdm()
    log(f"SDM_norm off-diagonal: {[round(sdm[i, j], 3) for i in range(4) for j in range(i + 1, 4)]}")
    fit_name, fit_path = FIT
    fit_d = load_cohort(fit_path)
    ev = {n: load_cohort(p) for n, p in EVAL}
    log(f"{fit_name}: {len(fit_d['mismatch'])} terlokalisasi, {int(fit_d['mismatch'].sum())} salah kelas")
    for n, d in ev.items():
        log(f"{n}: {len(d['mismatch'])} terlokalisasi, {int(d['mismatch'].sum())} salah kelas, "
            f"{len(np.unique(d['image_id']))} citra")

    # kontrol positif: ev_sdca tersimpan harus sama dengan SDM_norm[top1, top2]
    ctrl = {}
    for n, d in list(ev.items()) + [(fit_name, fit_d)]:
        recon = sdca_from(sdm, d)
        ctrl[n] = {"n": len(recon), "n_match": int(np.isclose(recon, d["sdca"], atol=1e-6).sum())}
        ctrl[n]["all_match"] = ctrl[n]["n"] == ctrl[n]["n_match"]
    harness_ok = all(v["all_match"] for v in ctrl.values())
    log("kontrol positif rekonstruksi SDCA: " + ("LOLOS" if harness_ok else "GAGAL"))

    # orientasi kontrol ambiguitas dikunci pada split fitting
    signs = {}
    for key in ("margin", "entropy", "msp"):
        a = roc_auc_score(fit_d["mismatch"], fit_d[key])
        signs[key] = 1.0 if a >= 0.5 else -1.0
        log(f"orientasi {key}: tanda {signs[key]:+.0f} (AUROC validasi {max(a, 1 - a):.4f})")

    # relasi empiris P(GT=b | pred=a) dari split fitting saja, smoothing add-1
    idx = {n: i for i, n in enumerate(LESIONS)}
    cnt = np.ones((4, 4))
    for pc, gi in zip(fit_d["pred_class"], fit_d["gt_idx"]):
        if pc in idx and gi >= 0:
            cnt[idx[pc], gi] += 1
    emp = cnt / cnt.sum(1, keepdims=True)
    log("relasi empiris (baris = kelas prediksi, kolom = kelas GT):\n    "
        + "\n    ".join(f"{LESIONS[i]:<9s}" + " ".join(f"{emp[i, j]:.3f}" for j in range(4))
                        for i in range(4)))

    def scores(d):
        pred_idx = np.array([idx.get(c, 0) for c in d["pred_class"]])
        s = {"SDCA": d["sdca"],
             "MARGIN": signs["margin"] * d["margin"],
             "ENTROPY": signs["entropy"] * d["entropy"],
             "MSP": signs["msp"] * d["msp"],
             "EMP": emp[pred_idx, d["top2"]]}
        s["PERM"] = np.mean([permuted_sdm(sdm, sd)[d["top1"], d["top2"]] for sd in PERM_SEEDS], 0)
        return s

    # model inkremental dipasang pada split fitting, diterapkan apa adanya
    fs = scores(fit_d)
    Xm = fs["MARGIN"].reshape(-1, 1)
    Xms = np.stack([fs["MARGIN"], fs["SDCA"]], 1)
    lr_m = LogisticRegression(max_iter=2000).fit(Xm, fit_d["mismatch"])
    lr_ms = LogisticRegression(max_iter=2000).fit(Xms, fit_d["mismatch"])
    log(f"koefisien SDCA pada model MARGIN+SDCA (fit validasi): {lr_ms.coef_[0][1]:+.4f}")

    results = {}
    for name, d in ev.items():
        s = scores(d)
        y = d["mismatch"]
        per = {k: {"auroc": float(roc_auc_score(y, v)), "auprc": float(average_precision_score(y, v))}
               for k, v in s.items()}
        pm = lr_m.predict_proba(s["MARGIN"].reshape(-1, 1))[:, 1]
        pms = lr_ms.predict_proba(np.stack([s["MARGIN"], s["SDCA"]], 1))[:, 1]
        per["MARGIN_model"] = {"auroc": float(roc_auc_score(y, pm))}
        per["MARGIN+SDCA_model"] = {"auroc": float(roc_auc_score(y, pms))}
        results[name] = {
            "n_localized": int(len(y)), "n_mismatch": int(y.sum()),
            "mismatch_rate": float(y.mean()), "n_images": int(len(np.unique(d["image_id"]))),
            "per_signal": per,
            "sdca_auc_ci": auc_ci(y, s["SDCA"], d["image_id"]),
            "sdca_vs_perm": paired_auc_delta(y, s["SDCA"], s["PERM"], d["image_id"]),
            "sdca_vs_margin": paired_auc_delta(y, s["SDCA"], s["MARGIN"], d["image_id"]),
            "incremental_over_margin": paired_auc_delta(y, pms, pm, d["image_id"]),
            "odds_ratio_delta030": odds_ratio(y, (s["SDCA"] >= DELTA).astype(int), d["image_id"]),
            "sensitivity_majority_outcome": {
                "sdca_auroc": float(roc_auc_score(d["mismatch_major"], s["SDCA"])),
                "n_mismatch": int(d["mismatch_major"].sum())},
        }
        r = results[name]
        log(f"{name}: SDCA AUROC {r['sdca_auc_ci']['auroc']:.4f} "
            f"CI[{r['sdca_auc_ci']['ci95'][0]:.4f},{r['sdca_auc_ci']['ci95'][1]:.4f}] | "
            f"MARGIN {per['MARGIN']['auroc']:.4f} EMP {per['EMP']['auroc']:.4f} "
            f"PERM {per['PERM']['auroc']:.4f} | inkremental "
            f"{r['incremental_over_margin']['delta_auroc']:+.4f} "
            f"CI[{r['incremental_over_margin']['ci95'][0]:+.4f},"
            f"{r['incremental_over_margin']['ci95'][1]:+.4f}] | OR "
            f"{r['odds_ratio_delta030']['odds_ratio']:.2f}")

    ext = [n for n, _ in EVAL if n != "OIA-DDR test"]
    main_c = "OIA-DDR test"
    a1_main = results[main_c]["sdca_auc_ci"]["excludes_half"] and results[main_c]["sdca_auc_ci"]["auroc"] > 0.5
    a1_ext = sum(results[n]["sdca_auc_ci"]["excludes_half"] and results[n]["sdca_auc_ci"]["auroc"] > 0.5
                 for n in ext)
    a1 = bool(a1_main and a1_ext >= 2)
    a2 = bool(results[main_c]["sdca_vs_perm"]["sig_positive"])
    a3_main = results[main_c]["incremental_over_margin"]["sig_positive"]
    a3_ext = sum(results[n]["incremental_over_margin"]["sig_positive"] for n in ext)
    a3 = bool(a3_main and a3_ext >= 2)
    emp_ge = [n for n in results
              if results[n]["per_signal"]["EMP"]["auroc"] >= results[n]["per_signal"]["SDCA"]["auroc"]]
    passed = bool(harness_ok and a1 and a2 and a3)
    verdict = {
        "harness_positive_control": {"passed": bool(harness_ok), "detail": ctrl},
        "A1_sdca_above_chance": {"main": bool(a1_main), "n_external": int(a1_ext), "passed": a1},
        "A2_sdca_beats_permuted": {"passed": a2,
                                   "delta": results[main_c]["sdca_vs_perm"]["delta_auroc"],
                                   "ci95": results[main_c]["sdca_vs_perm"]["ci95"]},
        "A3_incremental_over_margin": {"main": bool(a3_main), "n_external": int(a3_ext), "passed": a3},
        "PASSED": passed,
        "empirical_ge_sdca_cohorts": emp_ge,
        "interpretation": (
            "TIDAK LOLOS: tidak ada dasar mengklaim validitas kriteria retrospektif untuk prior"
            if not passed else
            ("LOLOS, tetapi relasi turunan anotasi sekurangnya setara pada >=2 kohort; "
             "relasi empiris lebih defensible daripada prior penulis"
             if len(emp_ge) >= 2 else
             "LOLOS: prior menunjukkan validitas kriteria retrospektif terhadap anotasi ahli")),
    }
    if not a3:
        verdict["note_a3"] = ("A3 gagal: yang terbukti hanya bahwa model yang ragu lebih sering "
                              "salah. Ini BUKAN validasi prior dan tidak boleh ditulis demikian.")

    out = {"protocol_lock": "_NATIVE/PROTOCOL_LOCK_annotation_endpoint.md",
           "fit_split": fit_name, "n_boot": N_BOOT, "boot_seed": BOOT_SEED, "delta": DELTA,
           "sdm_norm": sdm.tolist(), "empirical_relation": emp.tolist(),
           "control_orientation_signs": signs,
           "logreg_coef_margin_sdca": {"margin": float(lr_ms.coef_[0][0]),
                                       "sdca": float(lr_ms.coef_[0][1])},
           "results": results, "verdict": verdict,
           "provenance": {"graph_md5": md5(GRAPH), "script_md5": md5(HERE)},
           "elapsed_s": time.perf_counter() - T0}
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"VONIS: {verdict['interpretation']}")
    log(f"ditulis {OUT}")


if __name__ == "__main__":
    main()
