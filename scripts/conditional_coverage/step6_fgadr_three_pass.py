"""step6_fgadr_three_pass.py - FGADR comparisons with the three-pass score mean.

Protocol locked before the run: PROTOCOL_LOCK_step6_fgadr_three_pass.md
Output: results/conditional_coverage/step6_fgadr_three_pass.json

Repository copy. The executed script read the merged detector caches and the reference results at local scratch
paths. Here only those paths are repository-relative: unzip primary_<cohort>.zip, primary_flip_<cohort>.zip and
secondary_<cohort>.zip of results/detector_outputs/ into results/detector_outputs/unpacked/{primary,flip,v2root}/
(or point CEG_DETECTOR_CACHE at another root). The evidence CSVs are read at the workspace paths below
(gzipped copies in results/evidence/). The result file already exists, so the script refuses to run until it is
moved. Nothing else differs from the executed script.

Recomputes every FGADR comparison against score fusion with the three-pass mean (primary, flip, secondary), now that
the complete FGADR secondary cache is used: fusion AUPRC (T12, T15), the fusion-input gate with paired intervals
(T15, Supplementary S4), the grade-shifting AUROC of fusion (Supplementary S1) and the operating point of fusion
(step 5, Section 5.1). The same code run with the two-pass mean must first reproduce the published two-pass numbers.
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
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.eval_fp_reduction_benchmark import apply_mlp, build_methods, fit_mlp, load_csv  # noqa: E402
from tools.eval_selective_conformal_baselines import (  # noqa: E402
    VALID, aurc, conformal_threshold, coverage_at,
)

csv.field_size_limit(10 ** 9)
LOCK = ROOT / "PROTOCOL_LOCK_step6_fgadr_three_pass.md"
OUT = ROOT / "results/conditional_coverage/step6_fgadr_three_pass.json"
FGADR_CSV = ROOT / "outputs/0506/evsds_pss/fgadr/train/governance_features_train.csv"
OIA_TEST_CSV = ROOT / "outputs/0506/evsds_v2sec/oiaddr/test/governance_features_test.csv"
MERGED = Path(os.environ.get("CEG_DETECTOR_CACHE", ROOT / "results/detector_outputs/unpacked"))
PRIMARY, FLIP = MERGED / "primary", MERGED / "flip"
V2_OIA, V2_FGADR = MERGED / "v2root", MERGED / "v2root"  # every secondary zip unpacks into v2root
REF_BASELINES = ROOT / "results/selective_conformal/baselines.json"
REF_FUSION = ROOT / "results/finalization/fusion_external.json"  # identical to the merged-cache fusion run
REF_TWO = ROOT / "results/finalization/fgadr_two_pass_results.json"
REF_CLIN = ROOT / "results/experiments/clinical_cost_external.json"
REF_STEP5 = ROOT / "results/conditional_coverage/step5_fusion_external_coverage.json"
EV = ("ev_sas", "ev_sdca", "ev_aps", "ev_pss", "ev_sds", "ev_sds_nsrc")
TASK = "governance_retain"
ALPHA, NOMINAL = 0.20, 0.80
SEEDS = (0, 1, 2, 3, 4)
N_BOOT, BOOT_SEED, IOU_MIN = 2000, 0, 0.15
M0, B6 = "M0_score", "KG-gate+SDS+corrob (B6)"
UNSAFE, CLOSE = "UNSAFE_MISMATCH", "CLINICALLY_CLOSE_MISMATCH"
TOL, TOL_FINE = 1e-4, 1e-5
T0 = time.perf_counter()


def log(msg):
    print(f"[step6] {msg} ({time.perf_counter() - T0:.0f}s)", flush=True)


# ---------------------------------------------------------------- fusion (recipe of fusion_external.py)
def cache_entries(root: Path, cohort: str, split: str, image_id: str):
    f = root / cohort / split / f"{image_id}.json"
    if not f.is_file():
        return None
    out = []
    for e in json.loads(f.read_text(encoding="utf-8")):
        raw = e.get("detector_raw") or e.get("r2a_raw") or {}
        if "bbox" in raw and "score" in raw:
            out.append((e.get("bbox_id"), raw["bbox"], float(raw["score"])))
    return out


def best_match(p: np.ndarray, others):
    n = p.shape[0]
    if not others:
        return np.zeros(n), np.zeros(n, dtype=bool)
    q = np.array([bb for _i, bb, _s in others], dtype=np.float64)
    sc = np.array([s for _i, _bb, s in others], dtype=np.float64)
    iw = np.maximum(0.0, np.minimum(p[:, None, 2], q[None, :, 2]) - np.maximum(p[:, None, 0], q[None, :, 0]))
    ih = np.maximum(0.0, np.minimum(p[:, None, 3], q[None, :, 3]) - np.maximum(p[:, None, 1], q[None, :, 1]))
    inter = iw * ih
    area_p = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1])
    area_q = (q[:, 2] - q[:, 0]) * (q[:, 3] - q[:, 1])
    union = area_p[:, None] + area_q[None, :] - inter
    iou = np.where((inter > 0) & (union > 0), inter / np.where(union > 0, union, 1.0), 0.0)
    j = iou.argmax(axis=1)
    ok = iou[np.arange(n), j] >= IOU_MIN
    return np.where(ok, sc[j], 0.0), ok


def fusion_components(csv_path: Path, cohort: str, split: str, v2_root, d: dict):
    """Primary, flip and secondary score per CSV row, plus the category column."""
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = [(r["bbox_id"], r["image_id"], r["category"]) for r in csv.DictReader(fh)]
    assert len(rows) == d["pred_score"].size and all(rows[i][1] == d["image_id"][i] for i in range(len(rows)))
    by_img = {}
    for i, (_b, img, _c) in enumerate(rows):
        by_img.setdefault(img, []).append(i)
    s_f, s_v = np.zeros(len(rows)), np.zeros(len(rows))
    m_f, m_v = np.zeros(len(rows), dtype=bool), np.zeros(len(rows), dtype=bool)
    info = {"images": len(by_img), "images_without_primary_cache": 0, "rows_without_primary_box": 0,
            "images_without_secondary_file": 0}
    for img, idx in by_img.items():
        prim_list = cache_entries(PRIMARY, cohort, split, img)
        prim = {b: bb for b, bb, _s in (prim_list or [])}
        if not prim:
            info["images_without_primary_cache"] += 1
            continue
        keep = [i for i in idx if rows[i][0] in prim]
        info["rows_without_primary_box"] += len(idx) - len(keep)
        if not keep:
            continue
        ki = np.array(keep)
        p = np.array([prim[rows[i][0]] for i in keep], dtype=np.float64)
        s, ok = best_match(p, cache_entries(FLIP, cohort, split, img) or [])
        s_f[ki], m_f[ki] = s, ok
        if v2_root is not None:
            sec = cache_entries(v2_root, cohort, split, img)
            if sec is None:
                info["images_without_secondary_file"] += 1
                sec = []
            s, ok = best_match(p, sec)
            s_v[ki], m_v[ki] = s, ok
    info["flip_matched"] = float(m_f.mean())
    info["secondary_matched"] = float(m_v.mean()) if v2_root is not None else None
    cats = np.array([r[2] for r in rows])
    return d["pred_score"], s_f, s_v, cats, info


# ---------------------------------------------------------------- weighted AP bootstrap (exact, fgadr_two_pass.py)
def prepare_ap(score, label, image_index):
    order = np.argsort(-score, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(score[order]) != 0), len(order) - 1]
    return label[order].astype(np.float64), image_index[order], ends


def weighted_ap(prepared, counts):
    y, group, ends = prepared
    w = counts[group]
    tp = np.cumsum(w * y)[ends]
    total = np.cumsum(w)[ends]
    return float(np.sum(np.diff(np.r_[0.0, tp]) * tp / np.maximum(total, 1)) / tp[-1])


def boot_auroc(score, y, img, n_boot=N_BOOT, seed=BOOT_SEED):
    """clinical_cost_external.py, unchanged."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(img)
    groups = [np.where(img == im)[0] for im in uniq]
    n = len(groups)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[k] for k in rng.integers(0, n, size=n)])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        vals.append(roc_auc_score(yy, score[idx]))
    lo, hi = (float(x) for x in np.percentile(vals, [2.5, 97.5]))
    return [lo, hi], len(vals)


def ci(v):
    v = np.asarray(v, dtype=np.float64)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def close(a, b, tol):
    return abs(float(a) - float(b)) <= tol


def op_summary(test_scores, valid_scores, valid_label, lab):
    thr = [conformal_threshold(sv, valid_label, ALPHA) for sv in valid_scores]
    cov, ret, a, ap = [], [], [], []
    for st, t in zip(test_scores, thr):
        c = coverage_at(st, lab, t)
        cov.append(c["empirical_coverage"])
        ret.append(c["retained_fraction"])
        a.append(aurc(st, lab.astype(np.float64)))
        ap.append(float(average_precision_score(lab, st)))
    return {"thresholds": thr, "coverage": float(np.mean(cov)), "gap": float(np.mean(cov)) - NOMINAL,
            "retained_fraction": float(np.mean(ret)), "aurc": float(np.mean(a)), "auprc": float(np.mean(ap)),
            "auprc_per_seed": ap}


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite a completed run: {OUT}")
    lock_sha = hashlib.sha256(LOCK.read_bytes()).hexdigest()
    log(f"protocol lock sha256 {lock_sha}")
    base = json.loads(REF_BASELINES.read_text(encoding="utf-8"))["cohorts"]["FGADR"]["methods"]
    fm = {c["name"]: c for c in json.loads(REF_FUSION.read_text(encoding="utf-8"))["cohorts"]}
    two = json.loads(REF_TWO.read_text(encoding="utf-8"))
    clin = next(c for c in json.loads(REF_CLIN.read_text(encoding="utf-8"))["cohorts"] if c["name"] == "FGADR")
    step5 = json.loads(REF_STEP5.read_text(encoding="utf-8"))
    report = {"step": 6, "protocol_lock": str(LOCK.relative_to(ROOT)), "protocol_lock_sha256": lock_sha,
              "task": TASK, "alpha": ALPHA, "seeds": list(SEEDS), "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
              "iou_min": IOU_MIN, "controls": {}, "fusion_info": {}}

    # ---- source split and gates
    valid = load_csv(VALID, positive=TASK)
    yv = valid["label"].astype(np.float64)
    p, f, v, _c, info_v = fusion_components(VALID, "oiaddr", "valid", V2_OIA, valid)
    fv3, fv2 = (p + f + v) / 3.0, (p + f) / 2.0
    report["fusion_info"]["oiaddr_valid"] = info_v
    xv3 = np.stack([valid[c] for c in EV] + [fv3], 1)
    xv2 = np.stack([valid[c] for c in EV] + [fv2], 1)
    with threadpool_limits(limits=1):
        fe3 = [fit_mlp(xv3, yv, seed=s) for s in SEEDS]
        fe2 = [fit_mlp(xv2, yv, seed=s) for s in SEEDS]
    fe3_valid = [apply_mlp(xv3, m) for m in fe3]
    fe2_valid = [apply_mlp(xv2, m) for m in fe2]
    log("fusion-input gates fitted on OIA-DDR validation (three-pass and two-pass, five seeds each)")

    # control 6: three-pass gates reproduce the published OIA-DDR fusion-input AUPRC
    oia = load_csv(OIA_TEST_CSV, positive=TASK)
    p, f, v, _c, _i = fusion_components(OIA_TEST_CSV, "oiaddr", "test", V2_OIA, oia)
    xo = np.stack([oia[c] for c in EV] + [(p + f + v) / 3.0], 1)
    oia_fe = float(np.mean([average_precision_score(oia["label"], apply_mlp(xo, m)) for m in fe3]))
    report["controls"]["6_oia_fusion_input_gate_auprc"] = {"value": oia_fe, "reference": fm["OIA-DDR"]["gates"]["gate_fused"]["ap_mean"],
                                                          "passed": close(oia_fe, fm["OIA-DDR"]["gates"]["gate_fused"]["ap_mean"], TOL)}
    del oia, xo

    # ---- FGADR
    te = load_csv(FGADR_CSV, positive=TASK)
    lab = te["label"]
    p, f, v, cats, info_t = fusion_components(FGADR_CSV, "fgadr", "train", V2_FGADR, te)
    ft3, ft2 = (p + f + v) / 3.0, (p + f) / 2.0
    report["fusion_info"]["fgadr_train"] = info_t
    report["controls"]["7_secondary_file_for_every_image"] = {
        "images": info_t["images"], "images_without_secondary_file": info_t["images_without_secondary_file"],
        "passed": info_t["images_without_secondary_file"] == 0 and info_t["images_without_primary_cache"] == 0}
    log(f"FGADR fusion ready: flip {info_t['flip_matched']:.3f}, secondary {info_t['secondary_matched']:.3f}")

    xt3 = np.stack([te[c] for c in EV] + [ft3], 1)
    xt2 = np.stack([te[c] for c in EV] + [ft2], 1)
    fe3_t = [apply_mlp(xt3, m) for m in fe3]
    fe2_t = [apply_mlp(xt2, m) for m in fe2]

    m0_pair, b6_v, b6_t = None, [], []
    for seed in SEEDS:
        for name, _fam, sv, st in build_methods(valid, te, mlp_seed=seed):
            if name == M0 and m0_pair is None:
                m0_pair = (sv, st)
            elif name == B6:
                b6_v.append(sv)
                b6_t.append(st)
    log("M0 and B6 built on FGADR")

    # ---- A and B: AUPRC and the weighted-AP paired bootstrap
    ap_f3, ap_f2 = float(average_precision_score(lab, ft3)), float(average_precision_score(lab, ft2))
    ap_fe3 = [float(average_precision_score(lab, s)) for s in fe3_t]
    ap_fe2 = [float(average_precision_score(lab, s)) for s in fe2_t]
    ap_b6 = float(np.mean([average_precision_score(lab, s) for s in b6_t]))
    ap_m0 = float(average_precision_score(lab, m0_pair[1]))
    uniq, img_index = np.unique(te["image_id"], return_inverse=True)
    rng = np.random.default_rng(BOOT_SEED)
    draws = np.array([np.bincount(rng.integers(0, len(uniq), size=len(uniq)), minlength=len(uniq))
                      for _ in range(N_BOOT)], dtype=np.int32)
    prep = {"F3": prepare_ap(ft3, lab, img_index), "F2": prepare_ap(ft2, lab, img_index)}
    for i in range(len(SEEDS)):
        prep[f"FE3_{i}"] = prepare_ap(fe3_t[i], lab, img_index)
        prep[f"FE2_{i}"] = prepare_ap(fe2_t[i], lab, img_index)
    for w in draws[:3]:  # exactness check against sklearn on the expanded resample
        sample = np.repeat(np.arange(lab.size), w[img_index])
        assert abs(weighted_ap(prep["F3"], w) - average_precision_score(lab[sample], ft3[sample])) < 1e-10
    boots = {k: np.array([weighted_ap(pr, w) for w in draws]) for k, pr in prep.items()}
    log("weighted-AP bootstrap done")

    def seed_rows(tag, ref_key, aps, ref_ap):
        rows = []
        for i, s in enumerate(SEEDS):
            lo, hi = ci(boots[f"{tag}_{i}"] - boots[ref_key])
            rows.append({"seed": s, "ap": aps[i], "gain": aps[i] - ref_ap, "ci95": [lo, hi]})
        return rows

    rows3 = seed_rows("FE3", "F3", ap_fe3, ap_f3)
    rows2 = seed_rows("FE2", "F2", ap_fe2, ap_f2)
    report["controls"]["1_two_pass_fusion_auprc"] = {"value": ap_f2, "reference": fm["FGADR"]["fused_ap"],
                                                     "passed": close(ap_f2, fm["FGADR"]["fused_ap"], TOL)}
    ref_rows = two["per_seed"]
    ok2 = all(close(r["ap"], q["ap"], TOL_FINE) and close(r["ci95"][0], q["ci95"][0], TOL_FINE)
              and close(r["ci95"][1], q["ci95"][1], TOL_FINE) for r, q in zip(rows2, ref_rows))
    report["controls"]["2_two_pass_fusion_input_gate"] = {"values": rows2, "reference": ref_rows, "passed": ok2}

    # ---- C: grade-shifting AUROC of fusion
    sel = np.isin(cats, [UNSAFE, CLOSE])
    ys = (cats[sel] == UNSAFE).astype(int)
    imgs = te["image_id"][sel]
    auc3 = float(roc_auc_score(ys, ft3[sel]))
    ci3, nv3 = boot_auroc(ft3[sel], ys, imgs)
    auc2 = float(roc_auc_score(ys, ft2[sel]))
    ci2, _nv2 = boot_auroc(ft2[sel], ys, imgs)
    ref2 = clin["signals"]["fused_2_pass"]
    report["controls"]["3_two_pass_fusion_grade_shifting_auroc"] = {
        "value": [auc2, ci2], "reference": [ref2["auroc"], ref2["ci95"]],
        "passed": close(auc2, ref2["auroc"], TOL) and close(ci2[0], ref2["ci95"][0], TOL) and close(ci2[1], ref2["ci95"][1], TOL)}
    log(f"grade-shifting AUROC three-pass {auc3:.4f} {ci3}")

    # ---- D: operating point
    ops = {"M0": op_summary([m0_pair[1]], [m0_pair[0]], valid["label"], lab),
           "B6": op_summary(b6_t, b6_v, valid["label"], lab),
           "F3": op_summary([ft3], [fv3], valid["label"], lab),
           "FE3": op_summary(fe3_t, fe3_valid, valid["label"], lab),
           "F2": op_summary([ft2], [fv2], valid["label"], lab)}
    pos = lab == 1
    n_img = len(uniq)
    pos_img = np.bincount(img_index, weights=pos.astype(np.float64), minlength=n_img)
    neg_img = np.bincount(img_index, weights=(~pos).astype(np.float64), minlength=n_img)
    score_lists = {"M0": [m0_pair[1]], "B6": b6_t, "F3": [ft3], "FE3": fe3_t, "F2": [ft2]}
    kept = {m: np.stack([np.bincount(img_index, weights=(pos & (st >= t)).astype(np.float64), minlength=n_img)
                         for st, t in zip(score_lists[m], ops[m]["thresholds"])], 1) for m in score_lists}
    rng = np.random.default_rng(BOOT_SEED)
    d = {"B6_F3": [], "FE3_F3": [], "B6_F2": []}
    for _ in range(N_BOOT):
        mult = np.bincount(rng.integers(0, n_img, n_img), minlength=n_img).astype(np.float64)
        den, dneg = float(mult @ pos_img), float(mult @ neg_img)
        if den == 0 or dneg == 0:
            continue
        g = {m: abs(float(np.mean((mult @ kept[m]) / den)) - NOMINAL) for m in kept}
        d["B6_F3"].append(g["B6"] - g["F3"])
        d["FE3_F3"].append(g["FE3"] - g["F3"])
        d["B6_F2"].append(g["B6"] - g["F2"])
    s5 = step5["cohorts"]["FGADR"]
    report["controls"]["4_two_pass_operating_point"] = {
        "value": [ops["F2"]["gap"], ci(d["B6_F2"])],
        "reference": [s5["methods"]["F"]["gap"], s5["paired"]["absgap_B6_minus_F"]["ci95"]],
        "passed": close(ops["F2"]["gap"], s5["methods"]["F"]["gap"], TOL)
        and all(close(a, b, TOL) for a, b in zip(ci(d["B6_F2"]), s5["paired"]["absgap_B6_minus_F"]["ci95"]))}
    report["controls"]["5_m0_b6_reproduce"] = {
        "values": {"M0_gap": ops["M0"]["gap"], "M0_aurc": ops["M0"]["aurc"], "B6_gap": ops["B6"]["gap"],
                   "B6_aurc": ops["B6"]["aurc"], "B6_auprc": ap_b6},
        "passed": close(ops["M0"]["gap"], base[M0]["conformal_coverage_gap"], TOL)
        and close(ops["M0"]["aurc"], base[M0]["aurc"], TOL) and close(ops["B6"]["gap"], base[B6]["conformal_coverage_gap"], TOL)
        and close(ops["B6"]["aurc"], base[B6]["aurc"], TOL) and close(ap_b6, fm["FGADR"]["gates"]["full_gate"]["ap_mean"], TOL)}
    all_ok = all(v["passed"] for v in report["controls"].values())
    for k, v in report["controls"].items():
        log(f"control {k}: {'PASS' if v['passed'] else 'FAIL'}")
    if not all_ok:
        report["verdict"] = "STOPPED: a reproduction control failed; nothing is interpreted"
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        raise SystemExit("reproduction control failed, see " + str(OUT))

    # ---- results
    fe_mean = float(np.mean(ap_fe3))
    report["fgadr_three_pass"] = {
        "A_fusion_auprc": ap_f3,
        "B_fusion_input_gate": {"per_seed": rows3, "ap_mean": fe_mean, "ap_sd": float(np.std(ap_fe3, ddof=1)),
                                "gain_vs_fused_mean": fe_mean - ap_f3,
                                "n_ci_above_0": sum(r["ci95"][0] > 0 for r in rows3),
                                "n_ci_below_0": sum(r["ci95"][1] < 0 for r in rows3),
                                "m0_ap": ap_m0, "full_gate_ap": ap_b6},
        "C_grade_shifting_auroc": {"auroc": auc3, "ci95": ci3, "n_boot_valid": nv3,
                                   "n_unsafe": int(ys.sum()), "n_close": int((1 - ys).sum())},
        "D_operating_point": {m: {k: v for k, v in ops[m].items() if k != "thresholds"} for m in ("M0", "B6", "F3", "FE3")},
        "D_paired": {"absgap_B6_minus_F": {"point": abs(ops["B6"]["gap"]) - abs(ops["F3"]["gap"]), "ci95": ci(d["B6_F3"])},
                     "absgap_FE_minus_F": {"point": abs(ops["FE3"]["gap"]) - abs(ops["F3"]["gap"]), "ci95": ci(d["FE3_F3"])},
                     "n_resamples_used": len(d["B6_F3"])},
    }
    # ---- criteria of step 5, recounted with FGADR three-pass
    coh = {c: step5["cohorts"][c] for c in ("OIA-DDR test", "IDRiD", "TJDR")}
    rec = {c: {"B6": e["methods"]["B6"], "F": e["methods"]["F"], "FE": e["methods"]["F+E"], "M0": e["methods"]["M0"],
               "ci": e["paired"]["absgap_B6_minus_F"]["ci95"]} for c, e in coh.items()}
    rec["FGADR"] = {"B6": ops["B6"], "F": ops["F3"], "FE": ops["FE3"], "M0": ops["M0"], "ci": ci(d["B6_F3"])}
    c1 = sum(abs(r["B6"]["gap"]) < abs(r["F"]["gap"]) for r in rec.values())
    c2b = sum(r["ci"][1] < 0 for r in rec.values())
    c2a = sum(r["ci"][0] > 0 for r in rec.values())
    c3 = sum(abs(r["F"]["gap"]) < abs(r["M0"]["gap"]) and r["F"]["aurc"] < r["M0"]["aurc"] for r in rec.values())
    c4 = sum(abs(r["FE"]["gap"]) < abs(r["F"]["gap"]) for r in rec.values())
    report["criteria_recounted"] = {"C1_gate_closer_than_F": {"cohorts": int(c1), "of": 4, "met": bool(c1 >= 3)},
                                    "C2_paired_ci": {"below_zero": int(c2b), "above_zero": int(c2a), "of": 4},
                                    "C3_F_improves_both_over_M0": {"cohorts": int(c3), "of": 4},
                                    "C4_FE_closer_than_F": {"cohorts": int(c4), "of": 4}}
    report["verdict"] = "C1 MET" if c1 >= 3 else "C1 NOT MET"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    r = report["fgadr_three_pass"]
    log(f"A fusion AUPRC {ap_f3:.4f} | B F+E {fe_mean:.4f}+-{r['B_fusion_input_gate']['ap_sd']:.4f} "
        f"gain {fe_mean - ap_f3:+.4f} CI>0 {r['B_fusion_input_gate']['n_ci_above_0']}/5 CI<0 {r['B_fusion_input_gate']['n_ci_below_0']}/5")
    log(f"C grade-shifting AUROC {auc3:.4f} {ci3}")
    log(f"D gap F {ops['F3']['gap']:+.4f} F+E {ops['FE3']['gap']:+.4f} B6 {ops['B6']['gap']:+.4f} | "
        f"|B6|-|F| {r['D_paired']['absgap_B6_minus_F']['point']:+.4f} {r['D_paired']['absgap_B6_minus_F']['ci95']}")
    log(f"criteria {report['criteria_recounted']} -> {report['verdict']}; written {OUT}")


if __name__ == "__main__":
    main()
