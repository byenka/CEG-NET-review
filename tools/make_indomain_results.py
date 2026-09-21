#!/usr/bin/env python
"""make_indomain_results.py — hasil in-domain OIA-DDR dengan SATU protokol (Opsi B).

Latar: Tabel 2 dan 3 lama berasal dari run Stage3D terkunci yang memasang gate pada split
TRAIN dengan sklearn MLP (32,16) dan LGG torch berlatih Gumbel-Softmax, dengan bootstrap
1.000 resampel, sementara naskah dan tabel lain memakai gate numpy yang dipasang pada
VALID. Skrip ini menghitung ulang hasil in-domain dengan protokol yang ditulis naskah.

Protokol (Eq. gate; Sec. 4.3-4.4; tabel hiperparameter):
  gate   : fit_mlp — satu lapis tersembunyi 16 tanh, GD full-batch 1.500 iterasi, lr 0.1,
           standardisasi statistik valid; dipasang pada OIA-DDR VALID
  tau*   : threshold_for_tp_ret(skor valid, 0.80)
  seed   : gate 0..4; ringkasan mean dan SD SAMPEL (ddof 1)
  CI     : bootstrap berpasangan image-clustered, 2.000 resampel, bootstrap_seed 0, untuk
           SETIAP seed gate; skema resampel identik dengan paired_bootstrap_auprc
  AUPRC  : average precision (sklearn average_precision_score) untuk SEMUA baris, selisih dan
           bootstrap. Alasan: auprc() benchmark memakai trapezoid atas argsort biasa, sehingga
           urutan di dalam kelompok skor kembar sembarang dan bergantung versi numpy/platform.
           Diagnostik lokal 2026-09-15, batas trapezoid bila di dalam kelompok kembar negatif
           atau positif didahulukan: evidence-only logreg (16 nilai unik) [0.424, 0.686];
           MSP [0.4719, 0.4751]; entropy [0.4743, 0.4747]; margin [0.4706, 0.4726];
           M0 [0.64855, 0.64868]. Average precision memperlakukan skor kembar sebagai satu
           ambang sehingga deterministik. Trapezoid tetap dihitung dan disimpan untuk
           cross-check dengan artefak server yang memakai auprc().
Konfigurasi: deployed (outputs/0506/evsds_v2sec/oiaddr), sama dengan T12, T13, T14, S2, S3,
tier dan sertifikat. Baris APS eksternal pada T3 mengganti kolom ev_aps dengan nilai model
anatomi eksternal dari outputs/0506/evsds_full/stage3c (baris sejajar; diperiksa di sini).

Cross-check (gagal = berhenti; semuanya memakai trapezoid seperti artefak server):
  * B4 dan B6 per seed vs outputs/0506/evsds_pss/domain_shift_5seed.json
  * seed 0 vs outputs/0506/evsds_v2sec/oiaddr_benchmark.json: selisih <= TOL_AUPRC, atau
    nilai server berada di dalam batas urutan kembar lokal
  * seed 0: CI bootstrap KG - M0 vs oiaddr_benchmark.json bootstrap_auprc.kg_minus_m0
  * SDCA konstan per seed vs varian 'uniform' pada outputs/0506/sdm_prior_sensitivity.json

Keluaran:
  outputs/0506/indomain_5seed/indomain_results.json
  _FINAL_Latex/tables/T2_governance_main.tex
  _FINAL_Latex/tables/T3_evidence_fidelity.tex
  _FINAL_Latex/tables/T8_fp_reduction_benchmark.tex

Jalankan dari root repo:
  python tools/make_indomain_results.py               # hitung semua lalu tulis tabel
  python tools/make_indomain_results.py --tables_only # tulis ulang tabel dari JSON
"""
from __future__ import annotations

import json
import sys
import time
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.eval_fp_reduction_benchmark import (  # noqa: E402
    apply_logreg, apply_mlp, apply_temperature, auprc, eval_at_threshold, fit_logreg,
    fit_mlp, fit_temperature, load_csv, paired_bootstrap_auprc, threshold_for_tp_ret,
)

POSITIVE = "governance_retain"
TP_RET = 0.80
SEEDS = (0, 1, 2, 3, 4)
N_BOOT = 2000
BOOT_SEED = 0
EV = ("ev_sas", "ev_sdca", "ev_aps", "ev_pss")
TOL_AUPRC = 5e-4
TOL_CI = 1e-3
TOL_TIE = 1e-6  # kelonggaran batas urutan kembar (derau floating point antar mesin)
EPS = 1e-6      # selisih di bawah ini dianggap nol (derau floating point)

BASE = ROOT / "outputs/0506"
V_DEP = BASE / "evsds_v2sec/oiaddr/valid/governance_features_valid.csv"
T_DEP = BASE / "evsds_v2sec/oiaddr/test/governance_features_test.csv"
V_EXT = BASE / "evsds_full/stage3c/valid/governance_features_valid.csv"
T_EXT = BASE / "evsds_full/stage3c/test/governance_features_test.csv"
REF_5SEED = BASE / "evsds_pss/domain_shift_5seed.json"
REF_BENCH = BASE / "evsds_v2sec/oiaddr_benchmark.json"
REF_S2 = BASE / "sdm_prior_sensitivity.json"
OUT = BASE / "indomain_5seed"
TAB = ROOT / "_FINAL_Latex/tables"

GATES = ("score_only", "evidence_only", "kg_gate", "full_gate",
         "kg_sas_const", "kg_sdca_const", "kg_pss_const", "kg_aps_external")
MAIN_GATES = ("score_only", "evidence_only", "kg_gate", "full_gate")
BOOT_PAIRS = (("kg_gate", "M0"), ("full_gate", "M0"), ("full_gate", "kg_gate"),
              ("evidence_only", "M0"), ("score_only", "M0"))


# ── utilitas ──────────────────────────────────────────────────────────────────

def rel(p: Path) -> str:
    return p.resolve().relative_to(ROOT).as_posix()


def q(v: float, nd: int = 3) -> str:
    return str(Decimal(repr(float(v))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def sgn(v: float, nd: int = 3) -> str:
    if abs(v) < EPS:
        return q(0.0, nd)
    return ("$+$" if v > 0 else "$-$") + q(abs(v), nd)


def summary(vals) -> dict:
    a = np.asarray(vals, dtype=np.float64)
    return {"per_seed": [float(x) for x in a], "mean": float(a.mean()),
            "sd": float(a.std(ddof=1)), "n_pos": int((a > EPS).sum()),
            "n_neg": int((a < -EPS).sum())}


def tie_bounds(score: np.ndarray, y: np.ndarray):
    """AUPRC trapezoid (rumus auprc()) bila di dalam kelompok kembar negatif / positif didahulukan."""
    def area(order):
        yy = y[order]
        tp, fp = np.cumsum(yy), np.cumsum(1 - yy)
        rec = tp / max(int(y.sum()), 1)
        prec = tp / np.maximum(tp + fp, 1)
        return float(np.trapz(np.r_[1.0, prec], np.r_[0.0, rec]))
    return area(np.lexsort((y, -score))), area(np.lexsort((-y, -score)))


def metric(score: np.ndarray, y: np.ndarray) -> dict:
    """auprc = average precision; trapezoid dan batas urutan kembar disimpan untuk audit."""
    lo, hi = tie_bounds(score, y)
    return {"auprc": float(average_precision_score(y, score)),
            "auprc_trapezoid": float(auprc(score, y)),
            "n_unique": int(np.unique(score).size),
            "tie_bounds": [lo, hi],
            "tz_determined_3dp": q(lo) == q(hi)}


def paired_bootstrap_ap(s_a, s_b, label, image_id, n_boot, seed=0) -> dict:
    """Skema resampel identik dengan paired_bootstrap_auprc, metrik average precision."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(image_id)
    groups = [np.where(image_id == im)[0] for im in uniq]
    n_img = len(groups)
    d = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([groups[k] for k in rng.integers(0, n_img, size=n_img)])
        d[b] = (average_precision_score(label[idx], s_a[idx])
                - average_precision_score(label[idx], s_b[idx]))
    lo, hi = (float(x) for x in np.percentile(d, [2.5, 97.5]))
    return {"gain_mean": float(d.mean()), "gain_ci95": [lo, hi],
            "p_gain_gt0": float((d > 0).mean()), "sig": bool(lo > EPS),
            "n_bootstrap": n_boot, "cluster": "image_id", "n_clusters": n_img,
            "estimator": "average_precision"}


def boot(sa: np.ndarray, sb: np.ndarray, yt: np.ndarray, img: np.ndarray) -> dict:
    bs = paired_bootstrap_ap(sa, sb, yt, img, N_BOOT, BOOT_SEED)
    bs["gain_point"] = float(average_precision_score(yt, sa) - average_precision_score(yt, sb))
    return bs


def gate_matrix(d: dict, name: str, aps_ext: np.ndarray) -> np.ndarray:
    """Kolom disusun persis seperti build_methods(): [SAS, SDCA, APS, PSS, skor, sds, nsrc]."""
    cols = {c: d[c] for c in EV}
    if name == "score_only":
        return d["pred_score"].reshape(-1, 1)
    if name == "kg_sas_const":
        cols["ev_sas"] = np.full_like(d["ev_sas"], 0.5)
    elif name == "kg_sdca_const":
        cols["ev_sdca"] = np.full_like(d["ev_sdca"], 0.5)
    elif name == "kg_pss_const":
        cols["ev_pss"] = np.full_like(d["ev_pss"], 0.5)
    elif name == "kg_aps_external":
        cols["ev_aps"] = aps_ext
    ev = np.stack([cols[c] for c in EV], 1)
    if name == "evidence_only":
        return ev
    x = np.concatenate([ev, d["pred_score"].reshape(-1, 1)], 1)
    if name == "full_gate":
        x = np.concatenate([x, d["ev_sds"].reshape(-1, 1), d["ev_sds_nsrc"].reshape(-1, 1)], 1)
    return x


def op_metrics(sv: np.ndarray, st: np.ndarray, yv: np.ndarray, yt: np.ndarray) -> dict:
    tau = threshold_for_tp_ret(sv, yv, TP_RET)
    m = eval_at_threshold(st, yt, tau)
    out = metric(st, yt)
    out.update({"tp_ret": m["tp_ret"], "fp_supp": m["fp_supp"],
                "precision": m["precision_after"], "tau": tau})
    return out


def s2_crosscheck(res: dict) -> dict:
    """SDCA konstan identik secara matematis dengan prior 'uniform' pada S2."""
    d = json.loads(REF_S2.read_text(encoding="utf-8"))
    uni = next(v for v in d["results"] if v["variant"] == "uniform")
    mine = [r["auprc_trapezoid"] for r in res["gates"]["kg_sdca_const"]]
    return {"max_abs": max(abs(a - b) for a, b in zip(uni["gate_auprc_per_seed"], mine))}


# ── komputasi ─────────────────────────────────────────────────────────────────

def main() -> int:
    if "--tables_only" in sys.argv:
        return tables_only()
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    va, te = load_csv(V_DEP, POSITIVE), load_csv(T_DEP, POSITIVE)
    vx, tx = load_csv(V_EXT, POSITIVE), load_csv(T_EXT, POSITIVE)
    for a, b, split in ((va, vx, "valid"), (te, tx, "test")):
        aligned = (np.array_equal(a["image_id"], b["image_id"])
                   and np.array_equal(a["label"], b["label"])
                   and np.array_equal(a["pred_score"], b["pred_score"])
                   and np.array_equal(a["ev_sdca"], b["ev_sdca"])
                   and np.array_equal(a["ev_pss"], b["ev_pss"]))
        if not aligned:
            raise SystemExit(f"[indomain] {split}: CSV deployed dan eksternal tidak sejajar")
    yv, yt, img = va["label"], te["label"], te["image_id"]
    print(f"[indomain] valid={len(yv)} test={len(yt)} test_pos={int(yt.sum())}", flush=True)

    # metode yang tidak bergantung seed
    fixed = {"M0": (va["pred_score"], te["pred_score"]),
             "MSP": (va["unc_msp"], te["unc_msp"]),
             "Entropy": (va["unc_entconf"], te["unc_entconf"]),
             "Margin": (va["unc_margin"], te["unc_margin"])}
    T = fit_temperature(va["pred_score"], yv)
    fixed["TempScale"] = (apply_temperature(va["pred_score"], T),
                          apply_temperature(te["pred_score"], T))
    Xv_e, Xt_e = gate_matrix(va, "evidence_only", vx["ev_aps"]), gate_matrix(te, "evidence_only", tx["ev_aps"])
    p_e = fit_logreg(Xv_e, yv.astype(np.float64))
    fixed["LogReg_evidence"] = (apply_logreg(Xv_e, p_e), apply_logreg(Xt_e, p_e))
    Xv_k, Xt_k = gate_matrix(va, "kg_gate", vx["ev_aps"]), gate_matrix(te, "kg_gate", tx["ev_aps"])
    p_k = fit_logreg(Xv_k, yv.astype(np.float64))
    fixed["LogReg_evidence_score"] = (apply_logreg(Xv_k, p_k), apply_logreg(Xt_k, p_k))
    res = {"fixed": {k: op_metrics(sv, st, yv, yt) for k, (sv, st) in fixed.items()},
           "temperature": T, "gates": {g: [] for g in GATES}, "bootstrap": {}}
    for k, v in res["fixed"].items():
        print(f"[indomain] {k:22s} AP={v['auprc']:.5f} TZ={v['auprc_trapezoid']:.5f} "
              f"u={v['n_unique']} TP={v['tp_ret']:.4f} FP={v['fp_supp']:.4f} "
              f"P={v['precision']:.4f}", flush=True)

    # gate per seed
    for seed in SEEDS:
        scores = {"M0": te["pred_score"]}
        for g in GATES:
            Xv = gate_matrix(va, g, vx["ev_aps"])
            Xt = gate_matrix(te, g, tx["ev_aps"])
            m = fit_mlp(Xv, yv.astype(np.float64), seed=seed)
            sv, st = apply_mlp(Xv, m), apply_mlp(Xt, m)
            r = op_metrics(sv, st, yv, yt)
            r["seed"] = seed
            res["gates"][g].append(r)
            if g in MAIN_GATES:
                scores[g] = st
            print(f"[indomain] seed {seed} {g:16s} AP={r['auprc']:.5f} TZ={r['auprc_trapezoid']:.5f} "
                  f"u={r['n_unique']} TP={r['tp_ret']:.4f} FP={r['fp_supp']:.4f} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
        for a, b in BOOT_PAIRS:
            bs = boot(scores[a], scores[b], yt, img)
            res["bootstrap"].setdefault(f"{a}_minus_{b}", []).append({"seed": seed, **bs})
            print(f"[indomain] seed {seed} boot {a}-{b}: {bs['gain_point']:+.4f} "
                  f"[{bs['gain_ci95'][0]:+.4f}, {bs['gain_ci95'][1]:+.4f}] "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
        if seed == 0:
            # bootstrap trapezoid hanya untuk cross-check dengan artefak server
            tzb = paired_bootstrap_auprc(scores["kg_gate"], scores["M0"], yt, img, N_BOOT, BOOT_SEED)
            res["crosscheck_tz_boot_seed0_kg_minus_M0"] = tzb

    res["summary"] = summarize(res)

    # cross-check (trapezoid, sama dengan artefak server)
    checks = {}
    ref5 = json.loads(REF_5SEED.read_text(encoding="utf-8"))
    oia = next(c for c in ref5["cohorts"] if c["name"] == "OIA-DDR")
    checks["domain_shift_5seed"] = {
        "m0": abs(oia["m0_auprc"] - res["fixed"]["M0"]["auprc_trapezoid"]),
        "b4_max": max(abs(a - r["auprc_trapezoid"]) for a, r in zip(oia["b4_auprc_per_seed"], res["gates"]["kg_gate"])),
        "b6_max": max(abs(a - r["auprc_trapezoid"]) for a, r in zip(oia["b6_auprc_per_seed"], res["gates"]["full_gate"])),
    }
    bench = json.loads(REF_BENCH.read_text(encoding="utf-8"))
    bmap = {r["method"]: r["auprc"] for r in bench["results"]}
    mine0 = {"M0_score": res["fixed"]["M0"], "MSP": res["fixed"]["MSP"],
             "Entropy": res["fixed"]["Entropy"], "Margin": res["fixed"]["Margin"],
             "Evidence-LogReg": res["fixed"]["LogReg_evidence"],
             "Score-MLP": res["gates"]["score_only"][0],
             "KG-gate-MLP (ours)": res["gates"]["kg_gate"][0],
             "KG-gate+SDS+corrob (B6)": res["gates"]["full_gate"][0]}
    t_name = next(k for k in bmap if k.startswith("TempScale"))
    mine0[t_name] = res["fixed"]["TempScale"]
    bchk, failures = {}, []
    for k, row in mine0.items():
        if k not in bmap:
            continue
        diff = abs(bmap[k] - row["auprc_trapezoid"])
        lo, hi = row["tie_bounds"]
        within = (lo - TOL_TIE) <= bmap[k] <= (hi + TOL_TIE)
        bchk[k] = {"abs_diff": diff, "n_unique": row["n_unique"], "tie_bounds": [lo, hi],
                   "server_within_tie_bounds": within}
        if diff > TOL_AUPRC and not within:
            failures.append(k)
    checks["benchmark_seed0"] = bchk
    ci_ref = bench["bootstrap_auprc"]["kg_minus_m0"]["gain_ci95"]
    ci_mine = res["crosscheck_tz_boot_seed0_kg_minus_M0"]["gain_ci95"]
    checks["benchmark_seed0_ci"] = {"lo": abs(ci_ref[0] - ci_mine[0]), "hi": abs(ci_ref[1] - ci_mine[1])}
    checks["s2_uniform"] = s2_crosscheck(res)
    res["cross_checks"] = checks
    print("[indomain] cross-check:", json.dumps(checks, indent=1), flush=True)
    worst = max(checks["domain_shift_5seed"].values())
    if (failures or worst > TOL_AUPRC or max(checks["benchmark_seed0_ci"].values()) > TOL_CI
            or checks["s2_uniform"]["max_abs"] > TOL_AUPRC):
        (OUT / "indomain_results_FAILED_CHECK.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        raise SystemExit(f"[indomain] CROSS-CHECK GAGAL: {failures} worst={worst:.2e}")

    res["provenance"] = {"tool": "tools/make_indomain_results.py", "positive": POSITIVE,
                         "tp_ret_target": TP_RET, "seeds": list(SEEDS), "n_boot": N_BOOT,
                         "boot_seed": BOOT_SEED, "sd": "sample (ddof=1)",
                         "auprc_estimator": "average_precision (sklearn)",
                         "crosscheck_estimator": "trapezoid (tools/eval_fp_reduction_benchmark.auprc)",
                         "eps": EPS, "valid_csv": rel(V_DEP), "test_csv": rel(T_DEP),
                         "aps_external_csv": [rel(V_EXT), rel(T_EXT)]}
    (OUT / "indomain_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    write_tables(res)
    print(f"[indomain] selesai {time.perf_counter() - t0:.0f}s -> {rel(OUT / 'indomain_results.json')}")
    return 0


def summarize(res: dict) -> dict:
    m0 = res["fixed"]["M0"]["auprc"]
    kg = [r["auprc"] for r in res["gates"]["kg_gate"]]
    summ = {}
    for g in GATES:
        rows = res["gates"][g]
        summ[g] = {
            "n_unique": [r["n_unique"] for r in rows],
            "auprc": summary([r["auprc"] for r in rows]),
            "tp_ret": summary([r["tp_ret"] for r in rows]),
            "fp_supp": summary([r["fp_supp"] for r in rows]),
            "precision": summary([r["precision"] for r in rows]),
            "delta_vs_m0": summary([r["auprc"] - m0 for r in rows]),
            "delta_vs_kg": summary([r["auprc"] - k for r, k in zip(rows, kg)]),
        }
    for key, lst in res["bootstrap"].items():
        summ["boot_" + key] = {
            "n_ci_above_0": int(sum(1 for b in lst if b["gain_ci95"][0] > EPS)),
            "n_ci_below_0": int(sum(1 for b in lst if b["gain_ci95"][1] < -EPS)),
            "ci_lo_min": float(min(b["gain_ci95"][0] for b in lst)),
            "ci_hi_max": float(max(b["gain_ci95"][1] for b in lst))}
    diag = {}
    for k, r in res["fixed"].items():
        diag[k] = {"max_abs_ap_minus_tz": abs(r["auprc"] - r["auprc_trapezoid"]),
                   "tz_tie_width_max": r["tie_bounds"][1] - r["tie_bounds"][0],
                   "tz_undetermined_3dp": not r["tz_determined_3dp"],
                   "n_unique_min": r["n_unique"]}
    for g in GATES:
        rows = res["gates"][g]
        diag[g] = {"max_abs_ap_minus_tz": max(abs(r["auprc"] - r["auprc_trapezoid"]) for r in rows),
                   "tz_tie_width_max": max(r["tie_bounds"][1] - r["tie_bounds"][0] for r in rows),
                   "tz_undetermined_3dp": any(not r["tz_determined_3dp"] for r in rows),
                   "n_unique_min": min(r["n_unique"] for r in rows)}
    summ["estimator_diagnostics"] = diag
    return summ


def tables_only() -> int:
    """Tulis ulang tabel dari JSON yang sudah ada, tanpa menghitung ulang model."""
    p = OUT / "indomain_results.json"
    res = json.loads(p.read_text(encoding="utf-8"))
    write_tables(res)
    print("[indomain] tabel T2, T3, T8 ditulis ulang dari JSON", flush=True)
    return 0


def write_tables(res: dict) -> None:
    TAB.joinpath("T2_governance_main.tex").write_text(build_t2(res), encoding="utf-8")
    TAB.joinpath("T3_evidence_fidelity.tex").write_text(build_t3(res), encoding="utf-8")
    TAB.joinpath("T8_fp_reduction_benchmark.tex").write_text(build_t8(res), encoding="utf-8")


# ── tabel ─────────────────────────────────────────────────────────────────────

HEADER = ("% {name} — DIBANGKITKAN tools/make_indomain_results.py dari\n"
          "% outputs/0506/indomain_5seed/indomain_results.json. JANGAN diedit tangan.\n"
          "% Protokol tunggal: gate numpy dipasang pada OIA-DDR valid, konfigurasi deployed,\n"
          "% lima seed gate (SD sampel), bootstrap image-clustered 2.000 per seed.\n"
          "% AUPRC = average precision (sklearn), tidak bergantung urutan skor kembar.\n")


def build_t2(res: dict) -> str:
    s = res["summary"]
    m0 = res["fixed"]["M0"]["auprc"]
    boot_key = {"score_only": "score_only_minus_M0", "evidence_only": "evidence_only_minus_M0",
                "kg_gate": "kg_gate_minus_M0", "full_gate": "full_gate_minus_M0"}
    rows = [rf"Detector score threshold ($M_0$) & score & {q(m0)} & -- & -- & -- \\"]
    spec = (("score_only", "Score-only gate", "score"),
            ("evidence_only", "Evidence-only gate", "SAS, SDCA, APS, PSS"),
            ("kg_gate", "Knowledge-graph gate (B4)", r"evidence $+$ score"),
            ("full_gate", "Full gate, CEG-Net (B6)", r"B4 $+$ multi-source"))
    for g, name, inputs in spec:
        a, dl, b = s[g]["auprc"], s[g]["delta_vs_m0"], s["boot_" + boot_key[g]]
        rows.append(rf"{name} & {inputs} & {q(a['mean'])} $\pm$ {q(a['sd'])} & "
                    rf"{sgn(dl['mean'])} $\pm$ {q(dl['sd'])} & {dl['n_pos']}/5 & {b['n_ci_above_0']}/5 \\")
    ms, mb = s["full_gate"]["delta_vs_kg"], s["boot_full_gate_minus_kg_gate"]
    tail = (rf"\multicolumn{{3}}{{@{{}}l}}{{Multi-source contribution (B6 $-$ B4)}} & "
            rf"{sgn(ms['mean'])} $\pm$ {q(ms['sd'])} & {ms['n_pos']}/5 & {mb['n_ci_above_0']}/5 \\")
    return HEADER.format(name="T2") + "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        (r"\caption{In-domain gating on OIA-DDR test (deployed configuration). Gates are "
         r"fitted on OIA-DDR validation with five initialization seeds; values are mean "
         r"$\pm$ SD over seeds, and AUPRC is computed as average precision. The last column "
         r"counts seeds whose 95\% image-clustered paired-bootstrap interval (2{,}000 "
         r"resamples) for the gain excludes zero.}"),
        r"\label{tab:governance}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}llcccc@{}}",
        r"\toprule",
        r"Model & Inputs & AUPRC & $\Delta$AUPRC vs.\ $M_0$ & Seeds $\Delta>0$ & Seeds CI $>0$ \\",
        r"\midrule",
        *rows,
        r"\midrule",
        tail,
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
        "",
    ])


def build_t3(res: dict) -> str:
    s = res["summary"]
    ref = s["kg_gate"]["auprc"]
    rows = [rf"All evidence (reference) & {q(ref['mean'])} $\pm$ {q(ref['sd'])} & -- & -- \\",
            r"\midrule"]
    spec = (("kg_sas_const", "SAS set to a constant"),
            ("kg_sdca_const", "SDCA set to a constant"),
            ("kg_pss_const", "PSS set to a constant"),
            ("kg_aps_external", "APS from the external anatomy model"))
    for g, name in spec:
        a, dl = s[g]["auprc"], s[g]["delta_vs_kg"]
        rows.append(rf"{name} & {q(a['mean'])} $\pm$ {q(a['sd'])} & "
                    rf"{sgn(dl['mean'])} $\pm$ {q(dl['sd'])} & {dl['n_neg']}/5 \\")
    return HEADER.format(name="T3") + "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        (r"\caption{Contribution of each evidence pillar to the knowledge-graph gate on OIA-DDR "
         r"test (deployed configuration, five gate seeds, AUPRC as average precision). A pillar "
         r"is set to a constant in both splits before fitting; the last row replaces the "
         r"constant APS of the geometric anatomy approximation with the external anatomy model. "
         r"$\Delta$ is the change from the reference within each seed, mean $\pm$ SD.}"),
        r"\label{tab:evidence}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        r"Evidence vector & AUPRC & $\Delta$AUPRC & Seeds $\Delta<0$ \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])


def fusion_rows() -> tuple[list[str], list[str]]:
    """Baris baseline fusi skor dan baris gate-di-atas-fusi untuk T8.

    Angka datang dari dua eksperimen terpisah yang memakai protokol sama dengan tabel ini
    (fit pada OIA-DDR valid, 5 seed, average precision, tau* dari valid pada retensi TP 0.80):
      _septem/scripts_optionB/baselines_nokg.json   baseline fusi + gate tanpa knowledge graph
      _septem/scripts_optionB/gate_on_fusion.json   gate yang membaca skor fusi sebagai masukan
    Penting: ketiga pass yang dirata-ratakan (primer, flip, detektor kedua) adalah pass yang
    SUDAH dikonsumsi gate penuh lewat SAS, PSS dan ev_sds, jadi baseline ini tidak lebih mahal.
    Mengembalikan ([], []) bila JSON belum ada, supaya generator tetap dapat dijalankan.
    """
    fus = ROOT / "_septem/scripts_optionB/baselines_nokg.json"
    gof = ROOT / "_septem/scripts_optionB/gate_on_fusion.json"
    if not (fus.is_file() and gof.is_file()):
        print("[indomain] JSON fusi belum ada; T8 ditulis tanpa baris fusi", flush=True)
        return [], []
    b = json.loads(fus.read_text(encoding="utf-8"))
    g = json.loads(gof.read_text(encoding="utf-8"))

    def frow(name: str, family: str, d: dict) -> str:
        return (rf"{name} & {family} & {q(d['ap'])} & {q(d['tp_ret'])} & {q(d['fp_supp'])} & "
                rf"{q(d['precision'])} &  \\")

    def mrow(name: str, family: str, d: dict, audit: str = "", bold: bool = False) -> str:
        au = q(d["ap_mean"])
        au = rf"\textbf{{{au}}}" if bold else au
        return (rf"{name}$^{{\dagger}}$ & {family} & {au} & {q(d['tp_ret_mean'])} & "
                rf"{q(d['fp_supp_mean'])} & {q(d['precision_mean'])} & {audit} \\")

    base = [frow("Flip TTA, score mean", "[ensembling]", b["fusion"]["tta_mean0"]),
            frow("Two-detector ensemble, score mean", "[ensembling]", b["fusion"]["ens2_mean0"]),
            frow("Three-pass score mean", "[ensembling]", b["fusion"]["ens3_mean0"]),
            # Nama baris harus menyebut "full gate": varian ini membuang SDCA dan APS tetapi
            # MEMPERTAHANKAN sinyal multi-source, jadi pembandingnya adalah full gate (0.699),
            # bukan knowledge-graph gate B4 (0.681) yang tidak punya multi-source. Tanpa kata
            # "full", pembaca menyimpulkan membuang knowledge graph menaikkan AUPRC.
            mrow("Full gate without knowledge graph", "(ablation)", b["gates"]["nokg_gate"])]
    ours = [mrow("Full gate on the three-pass mean", "this work", g["gates"]["gate_fused"],
                 r"\checkmark", bold=True)]
    return base, ours


def build_t8(res: dict) -> str:
    f, s = res["fixed"], res["summary"]

    def fixed_row(name, family, key, audit=""):
        r = f[key]
        return (rf"{name} & {family} & {q(r['auprc'])} & {q(r['tp_ret'])} & {q(r['fp_supp'])} & "
                rf"{q(r['precision'])} & {audit} \\")

    def gate_row(name, family, key, audit="", bold=False):
        g = s[key]
        au = q(g["auprc"]["mean"])
        au = rf"\textbf{{{au}}}" if bold else au
        return (rf"{name}$^{{\dagger}}$ & {family} & {au} & {q(g['tp_ret']['mean'])} & "
                rf"{q(g['fp_supp']['mean'])} & {q(g['precision']['mean'])} & {audit} \\")

    fus_base, fus_ours = fusion_rows()
    rows = [
        fixed_row(r"Score threshold ($M_0$)", "---", "M0"),
        fixed_row("Max-softmax prob.", r"[Hendrycks'17]", "MSP"),
        fixed_row("Predictive entropy", "[uncertainty]", "Entropy"),
        fixed_row(r"Top-1$-$top-2 margin", "[uncertainty]", "Margin"),
        fixed_row("Temperature scaling", r"[Guo'17]", "TempScale"),
        gate_row("Score-only gate", "(learned-score)", "score_only"),
        *fus_base,
        # Regresi logistik pada masukan yang SAMA dengan knowledge-graph gate adalah varian
        # linear dari gate evidence kita. Versi lama memberinya evidence TANPA skor lalu
        # menyimpulkan "nonlinearity contributes"; perbandingan itu terkonfound.
        fixed_row(r"Knowledge-graph gate, linear", "this work", "LogReg_evidence_score", r"\checkmark"),
        gate_row("Knowledge-graph gate, MLP", "this work", "kg_gate", r"\checkmark"),
        gate_row("Full gate, CEG-Net", "this work", "full_gate", r"\checkmark", bold=not fus_ours),
        *fus_ours,
    ]
    return HEADER.format(name="T8") + "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        (r"\caption{Comparison with false-positive-reduction methods on OIA-DDR test (deployed "
         r"configuration). All methods use the same frozen-detector boxes, and learned methods "
         r"are fitted on validation. AUPRC is computed as average precision. TP retention, FP "
         r"suppression and precision are taken at the threshold selected on validation for a "
         r"TP retention of 0.80 and are descriptive. The ensembling rows average detector scores "
         r"over the passes the gate already consumes, namely the primary detector, its "
         r"horizontally flipped pass and the secondary detector; a box that a pass does not "
         r"produce contributes a score of zero. The last row gives that three-pass mean to the "
         r"full gate as one additional input.}"),
        r"\label{tab:fp_benchmark}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Method & Family & AUPRC & $\mathrm{TP}_{\mathrm{ret}}$ & FP suppr. & Prec. & Audit \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}}",
        r"\\[2pt]",
        r"{\footnotesize $^{\dagger}$Mean over five gate-initialization seeds.}",
        r"\end{table}",
        "",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
