"""step5_fusion_external_coverage.py - operating point of score averaging on the external cohorts.

Protocol locked before the run: PROTOCOL_LOCK_step5_fusion_external_coverage.md
Output: results/conditional_coverage/step5_fusion_external_coverage.json

Repository copy. The executed script read the merged detector caches and two control files at local scratch
paths. Here only those paths are made repository-relative: the caches under
results/detector_outputs/unpacked/{primary,flip,v2root}/ (unzip primary_<cohort>.zip, primary_flip_<cohort>.zip
and secondary_<cohort>.zip into those three folders, or point CEG_DETECTOR_CACHE at another root), and the
control files under results/finalization/ and results/selective_conformal/. The evidence CSVs are read at the
workspace paths of tools/eval_selective_conformal_baselines.py (gzipped copies in results/evidence/). The
result file already exists, so the script refuses to run until it is moved. Nothing else differs.

Measures, on OIA-DDR test, IDRiD, TJDR and FGADR, the split-conformal coverage gap (alpha 0.20, calibrated on OIA-DDR
validation) and the AURC of the detector score (M0), the full gate (B6), the three-pass score mean (F, two-pass on
FGADR) and the gate that reads F (F+E). Reproduction controls run first and stop the run when any of them fails.
Read-only with respect to every existing artifact. Run from the repository root with the project environment.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.eval_fp_reduction_benchmark import apply_mlp, build_methods, fit_mlp, load_csv  # noqa: E402
from tools.eval_selective_conformal_baselines import (  # noqa: E402
    COHORTS, VALID, aurc, conformal_threshold, coverage_at,
)

LOCK = ROOT / "PROTOCOL_LOCK_step5_fusion_external_coverage.md"
OUT = ROOT / "results/conditional_coverage/step5_fusion_external_coverage.json"
BASELINES = ROOT / "results/selective_conformal/baselines.json"
HEADTOHEAD = ROOT / "results/selective_conformal/fusion_headtohead.json"
FUSION_MERGED = ROOT / "results/finalization/fusion_external.json"  # identical to the merged-cache fusion run
FGADR_TWO = ROOT / "results/finalization/fgadr_two_pass_results.json"
CACHE_ROOT = Path(os.environ.get("CEG_DETECTOR_CACHE", ROOT / "results/detector_outputs/unpacked"))
CACHE = {"primary": CACHE_ROOT / "primary", "flip": CACHE_ROOT / "flip", "v2": CACHE_ROOT / "v2root"}
KEY = {"OIA-DDR test": ("oiaddr", "test"), "IDRiD": ("idrid", "test"),
       "TJDR": ("tjdr", "test"), "FGADR": ("fgadr", "train")}
FUSED_NAME = {"OIA-DDR test": "OIA-DDR", "IDRiD": "IDRiD", "TJDR": "TJDR", "FGADR": "FGADR"}
EV = ("ev_sas", "ev_sdca", "ev_aps", "ev_pss", "ev_sds", "ev_sds_nsrc")
TASK = "governance_retain"
ALPHA = 0.20
NOMINAL = 1.0 - ALPHA
SEEDS = (0, 1, 2, 3, 4)
N_BOOT = 2000
BOOT_SEED = 0
IOU_MIN = 0.15
M0, B6 = "M0_score", "KG-gate+SDS+corrob (B6)"
TOL, TOL_FINE = 1e-4, 1e-5
T0 = time.perf_counter()


def log(msg: str) -> None:
    print(f"[step5] {msg} ({time.perf_counter() - T0:.0f}s)", flush=True)


# ---------------------------------------------------------------- score fusion (recipe of fusion_external.py)
def cache_entries(root: Path, split: str, image_id: str):
    f = root / split / f"{image_id}.json"
    if not f.is_file():
        return []
    out = []
    for e in json.loads(f.read_text(encoding="utf-8")):
        raw = e.get("detector_raw") or e.get("r2a_raw") or {}
        if "bbox" in raw and "score" in raw:
            out.append((e.get("bbox_id"), raw["bbox"], float(raw["score"])))
    return out


def best_match(p: np.ndarray, others):
    """Score of the best-IoU box of another pass for each primary box, 0 when IoU < IOU_MIN.
    Same arithmetic and tie-breaking (first maximum) as the scalar loop of fusion_external.py."""
    n = p.shape[0]
    if not others:
        return np.zeros(n), np.zeros(n, dtype=bool)
    q = np.array([bb for bb, _s in others], dtype=np.float64)
    sc = np.array([s for _bb, s in others], dtype=np.float64)
    iw = np.maximum(0.0, np.minimum(p[:, None, 2], q[None, :, 2]) - np.maximum(p[:, None, 0], q[None, :, 0]))
    ih = np.maximum(0.0, np.minimum(p[:, None, 3], q[None, :, 3]) - np.maximum(p[:, None, 1], q[None, :, 1]))
    inter = iw * ih
    area_p = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1])
    area_q = (q[:, 2] - q[:, 0]) * (q[:, 3] - q[:, 1])
    union = area_p[:, None] + area_q[None, :] - inter
    iou = np.where((inter > 0) & (union > 0), inter / np.where(union > 0, union, 1.0), 0.0)
    j = iou.argmax(axis=1)
    best = iou[np.arange(n), j]
    ok = best >= IOU_MIN
    return np.where(ok, sc[j], 0.0), ok


def fused_scores(csv_path: Path, key: str, split: str, n_pass: int, pred_score: np.ndarray, image_id: np.ndarray):
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = [(r["bbox_id"], r["image_id"]) for r in csv.DictReader(fh)]
    assert len(rows) == pred_score.size, "row count differs from load_csv"
    assert all(rows[i][1] == image_id[i] for i in range(len(rows))), "row order differs from load_csv"
    by_img = {}
    for i, (_b, img) in enumerate(rows):
        by_img.setdefault(img, []).append(i)
    s_f = np.zeros(len(rows))
    s_v = np.zeros(len(rows))
    m_f = np.zeros(len(rows), dtype=bool)
    m_v = np.zeros(len(rows), dtype=bool)
    no_prim = no_box = 0
    for img, idx in by_img.items():
        prim = {bid: bb for bid, bb, _s in cache_entries(CACHE["primary"] / key, split, img)}
        if not prim:
            no_prim += 1
            continue
        keep = [i for i in idx if rows[i][0] in prim]
        no_box += len(idx) - len(keep)
        if not keep:
            continue
        p = np.array([prim[rows[i][0]] for i in keep], dtype=np.float64)
        ki = np.array(keep)
        flip = [(bb, sc) for _i, bb, sc in cache_entries(CACHE["flip"] / key, split, img)]
        s, ok = best_match(p, flip)
        s_f[ki], m_f[ki] = s, ok
        if n_pass == 3:
            v2 = [(bb, sc) for _i, bb, sc in cache_entries(CACHE["v2"] / key, split, img)]
            s, ok = best_match(p, v2)
            s_v[ki], m_v[ki] = s, ok
    fused = (pred_score + s_f + s_v) / 3.0 if n_pass == 3 else (pred_score + s_f) / 2.0
    info = {"n_pass": n_pass, "flip_matched": float(m_f.mean()),
            "v2_matched": float(m_v.mean()) if n_pass == 3 else None,
            "images_without_primary_cache": no_prim, "rows_without_primary_box": no_box}
    return fused, info


# ---------------------------------------------------------------- helpers
def fe_matrix(d: dict, fused: np.ndarray) -> np.ndarray:
    return np.stack([d[c] for c in EV] + [fused], 1)


def summarize(test_scores, valid_scores, lab):
    """Mean over seeds of coverage, gap, retained fraction, AURC and AP, plus the thresholds."""
    thr = [conformal_threshold(sv, VALID_LABEL, ALPHA) for sv in valid_scores]
    cov, ret, a, ap = [], [], [], []
    for st, t in zip(test_scores, thr):
        c = coverage_at(st, lab, t)
        cov.append(c["empirical_coverage"])
        ret.append(c["retained_fraction"])
        a.append(aurc(st, lab.astype(np.float64)))
        ap.append(float(average_precision_score(lab, st)))
    return {"thresholds": thr, "coverage": float(np.mean(cov)), "gap": float(np.mean(cov)) - NOMINAL,
            "retained_fraction": float(np.mean(ret)), "aurc": float(np.mean(a)), "auprc": float(np.mean(ap)),
            "auprc_per_seed": ap, "n_seeds": len(test_scores)}


def ci(v):
    v = np.asarray(v, dtype=np.float64)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def close(a, b, tol):
    return abs(float(a) - float(b)) <= tol


# ---------------------------------------------------------------- main
def main() -> None:
    global VALID_LABEL
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite a completed run: {OUT}")
    lock_sha = hashlib.sha256(LOCK.read_bytes()).hexdigest()
    log(f"protocol lock sha256 {lock_sha}")
    base = json.loads(BASELINES.read_text(encoding="utf-8"))["cohorts"]
    h2h = json.loads(HEADTOHEAD.read_text(encoding="utf-8"))["methods"]
    fm = {c["name"]: c for c in json.loads(FUSION_MERGED.read_text(encoding="utf-8"))["cohorts"]}
    f2 = json.loads(FGADR_TWO.read_text(encoding="utf-8"))
    f2_ap = [r["ap"] for r in f2["per_seed"]]

    valid = load_csv(VALID, positive=TASK)
    VALID_LABEL = valid["label"]
    yv = valid["label"].astype(np.float64)
    fv3, info_v3 = fused_scores(VALID, "oiaddr", "valid", 3, valid["pred_score"], valid["image_id"])
    fv2, info_v2 = fused_scores(VALID, "oiaddr", "valid", 2, valid["pred_score"], valid["image_id"])
    log(f"validation fusion ready: flip {info_v3['flip_matched']:.3f}, v2 {info_v3['v2_matched']:.3f}")
    with threadpool_limits(limits=1):
        fe3 = [fit_mlp(fe_matrix(valid, fv3), yv, seed=s) for s in SEEDS]
        fe2 = [fit_mlp(fe_matrix(valid, fv2), yv, seed=s) for s in SEEDS]
    fe3_valid = [apply_mlp(fe_matrix(valid, fv3), m) for m in fe3]
    fe2_valid = [apply_mlp(fe_matrix(valid, fv2), m) for m in fe2]
    log("F+E gates fitted on OIA-DDR validation (three-pass and two-pass inputs, five seeds each)")

    report = {"step": 5, "protocol_lock": str(LOCK.relative_to(ROOT)), "protocol_lock_sha256": lock_sha,
              "task": TASK, "alpha": ALPHA, "seeds": list(SEEDS), "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
              "iou_min": IOU_MIN, "validation_fusion": {"three_pass": info_v3, "two_pass": info_v2},
              "controls": {}, "cohorts": {}}
    prepared = {}
    all_ok = True
    for coh, path in COHORTS.items():
        key, split = KEY[coh]
        test = load_csv(path, positive=TASK)
        lab = test["label"]
        n_pass = 2 if coh == "FGADR" else 3
        ft, info_t = fused_scores(path, key, split, n_pass, test["pred_score"], test["image_id"])
        log(f"{coh}: n={lab.size}, fusion {n_pass}-pass, flip {info_t['flip_matched']:.3f}")

        sc_m0 = None
        b6_v, b6_t = [], []
        for seed in SEEDS:
            for name, _fam, sv, st in build_methods(valid, test, mlp_seed=seed):
                if name == M0 and sc_m0 is None:
                    sc_m0 = (sv, st)
                elif name == B6:
                    b6_v.append(sv)
                    b6_t.append(st)
        fe_models, fe_valid, fv = (fe3, fe3_valid, fv3) if n_pass == 3 else (fe2, fe2_valid, fv2)
        fe_t = [apply_mlp(fe_matrix(test, ft), m) for m in fe_models]
        res = {"M0": summarize([sc_m0[1]], [sc_m0[0]], lab),
               "B6": summarize(b6_t, b6_v, lab),
               "F": summarize([ft], [fv], lab),
               "F+E": summarize(fe_t, fe_valid, lab)}

        # ---- reproduction controls
        bm = base[coh]["methods"]
        ctl = {
            "M0_gap": [res["M0"]["gap"], bm[M0]["conformal_coverage_gap"]],
            "M0_aurc": [res["M0"]["aurc"], bm[M0]["aurc"]],
            "B6_gap": [res["B6"]["gap"], bm[B6]["conformal_coverage_gap"]],
            "B6_aurc": [res["B6"]["aurc"], bm[B6]["aurc"]],
            "F_auprc": [res["F"]["auprc"], fm[FUSED_NAME[coh]]["fused_ap"]],
        }
        checks = {k: close(v[0], v[1], TOL) for k, v in ctl.items()}
        if coh == "OIA-DDR test":
            ctl["F_gap_headtohead"] = [res["F"]["gap"], h2h["Three-pass mean (F)"]["conformal_coverage_gap"]]
            ctl["F_aurc_headtohead"] = [res["F"]["aurc"], h2h["Three-pass mean (F)"]["aurc"]]
            checks["F_gap_headtohead"] = close(*ctl["F_gap_headtohead"], TOL)
            checks["F_aurc_headtohead"] = close(*ctl["F_aurc_headtohead"], TOL)
        if n_pass == 3:
            ctl["FE_auprc_mean"] = [res["F+E"]["auprc"], fm[FUSED_NAME[coh]]["gates"]["gate_fused"]["ap_mean"]]
            checks["FE_auprc_mean"] = close(*ctl["FE_auprc_mean"], TOL)
        else:
            ctl["FE_auprc_per_seed"] = [res["F+E"]["auprc_per_seed"], f2_ap]
            checks["FE_auprc_per_seed"] = all(close(a, b, TOL_FINE) for a, b in zip(res["F+E"]["auprc_per_seed"], f2_ap))
        report["controls"][coh] = {"values": ctl, "passed": checks}
        ok = all(checks.values())
        all_ok &= ok
        log(f"{coh}: controls {'PASS' if ok else 'FAIL'} {checks}")
        prepared[coh] = {"test": test, "lab": lab, "res": res, "info": info_t, "n_pass": n_pass,
                         "scores": {"M0": [sc_m0[1]], "B6": b6_t, "F": [ft], "F+E": fe_t}}

    if not all_ok:
        report["verdict"] = "STOPPED: a reproduction control failed; nothing is interpreted"
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        raise SystemExit("reproduction control failed, see " + str(OUT))

    # ---- paired image-clustered bootstrap of the absolute coverage gaps
    for coh, p in prepared.items():
        lab, res = p["lab"], p["res"]
        _, img = np.unique(p["test"]["image_id"], return_inverse=True)
        n_img = int(img.max()) + 1
        pos = lab == 1
        pos_img = np.bincount(img, weights=pos.astype(np.float64), minlength=n_img)
        neg_img = np.bincount(img, weights=(~pos).astype(np.float64), minlength=n_img)
        kept = {}
        for m in ("M0", "B6", "F", "F+E"):
            kept[m] = np.stack([np.bincount(img, weights=(pos & (st >= t)).astype(np.float64), minlength=n_img)
                                for st, t in zip(p["scores"][m], res[m]["thresholds"])], 1)
        rng = np.random.default_rng(BOOT_SEED)
        d_b6_f, d_fe_f, d_b6_m0 = [], [], []
        for _ in range(N_BOOT):
            mult = np.bincount(rng.integers(0, n_img, n_img), minlength=n_img).astype(np.float64)
            den, dneg = float(mult @ pos_img), float(mult @ neg_img)
            if den == 0 or dneg == 0:
                continue
            g = {m: abs(float(np.mean((mult @ kept[m]) / den)) - NOMINAL) for m in kept}
            d_b6_f.append(g["B6"] - g["F"])
            d_fe_f.append(g["F+E"] - g["F"])
            d_b6_m0.append(g["B6"] - g["M0"])
        entry = {"n": int(lab.size), "n_positive": int(pos.sum()), "fusion": p["info"],
                 "fusion_passes": p["n_pass"], "methods": {}}
        for m in ("M0", "B6", "F", "F+E"):
            r = dict(res[m])
            r.pop("thresholds")
            entry["methods"][m] = r
        entry["paired"] = {
            "absgap_B6_minus_F": {"point": abs(res["B6"]["gap"]) - abs(res["F"]["gap"]), "ci95": ci(d_b6_f)},
            "absgap_FE_minus_F": {"point": abs(res["F+E"]["gap"]) - abs(res["F"]["gap"]), "ci95": ci(d_fe_f)},
            "absgap_B6_minus_M0_sanity": {"point": abs(res["B6"]["gap"]) - abs(res["M0"]["gap"]), "ci95": ci(d_b6_m0)},
            "n_resamples_used": len(d_b6_f),
        }
        report["cohorts"][coh] = entry
        log(f"{coh}: gap M0 {res['M0']['gap']:+.4f} B6 {res['B6']['gap']:+.4f} F {res['F']['gap']:+.4f} "
            f"F+E {res['F+E']['gap']:+.4f} | |B6|-|F| {entry['paired']['absgap_B6_minus_F']['point']:+.4f} "
            f"CI {entry['paired']['absgap_B6_minus_F']['ci95']}")

    # ---- criteria, exactly as locked
    C = report["cohorts"]
    c1 = sum(abs(C[c]["methods"]["B6"]["gap"]) < abs(C[c]["methods"]["F"]["gap"]) for c in C)
    c2_below = sum(C[c]["paired"]["absgap_B6_minus_F"]["ci95"][1] < 0 for c in C)
    c2_above = sum(C[c]["paired"]["absgap_B6_minus_F"]["ci95"][0] > 0 for c in C)
    c3 = sum((abs(C[c]["methods"]["F"]["gap"]) < abs(C[c]["methods"]["M0"]["gap"]))
             and (C[c]["methods"]["F"]["aurc"] < C[c]["methods"]["M0"]["aurc"]) for c in C)
    c4 = sum(abs(C[c]["methods"]["F+E"]["gap"]) < abs(C[c]["methods"]["F"]["gap"]) for c in C)
    report["criteria"] = {
        "C1_gate_closer_than_F": {"cohorts": int(c1), "of": 4, "met": bool(c1 >= 3)},
        "C2_paired_ci": {"below_zero": int(c2_below), "above_zero": int(c2_above), "of": 4},
        "C3_F_improves_both_over_M0": {"cohorts": int(c3), "of": 4},
        "C4_FE_closer_than_F": {"cohorts": int(c4), "of": 4},
    }
    report["verdict"] = "C1 MET" if c1 >= 3 else "C1 NOT MET"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log(f"criteria {report['criteria']} -> {report['verdict']}; written {OUT}")


if __name__ == "__main__":
    main()
