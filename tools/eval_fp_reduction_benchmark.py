#!/usr/bin/env python
"""eval_fp_reduction_benchmark.py — controlled FP-reduction benchmark (>=5 methods).

CAS Tier-1 rigour: compare the KG-evidence governance gate against several
RECOGNISED false-positive-reduction / confidence / uncertainty baselines, ALL on
the SAME frozen detector outputs, SAME per-box set, SAME splits. No new detector
runs — everything is derived from the Stage3C governance_features CSVs that
already contain per-box detector score, class-uncertainty signals (MSP, margin,
entropy-confidence), KG evidence, and the TP/FP label.

Protocol (matches the Mode-1 governance evaluation):
  - label = 1 : box localises a real lesion (IoU>=0.1 with GT)  -> keep (TP-side)
  - label = 0 : background false positive                       -> reject (FP-side)
  - For each method, the per-box "keep-score" s is higher for boxes to keep.
  - Operating point: choose the threshold on VALID that retains >= target TP
    fraction (default 0.80); apply that threshold to TEST.
  - Report on TEST: TP-retention, FP-suppression, precision-after, and the
    threshold-free AUPRC of the keep-score for TP-vs-FP separation.
  - Learned methods (temperature scaling, logistic regression) are fit on VALID
    only (no test leakage).

Methods (>=5):
  M0_score        : detector score                       (threshold baseline)
  MSP             : max-softmax prob (unc_msp)           [Hendrycks&Gimpel'17]
  Entropy         : entropy-confidence (unc_entconf)     [predictive entropy]
  Margin          : top1-top2 margin (unc_margin)        [margin confidence]
  TempScale       : temperature-scaled score             [Guo et al. 2017]
  Evidence-LogReg : logistic reg. on [SAS,SDCA,APS,PSS]  (learned-linear gate)
  KG-gate (ours)  : LGG governance gate                  (provide via --ours_*)

Usage:
  python tools/eval_fp_reduction_benchmark.py \
    --valid_csv <stage3c>/governance_features_valid.csv \
    --test_csv  <stage3c>/governance_features_test.csv \
    --tp_ret_target 0.80 \
    --output_json outputs/fp_reduction_benchmark.json \
    [--ours_fp_supp 0.5584 --ours_tp_ret 0.7856 --ours_r1_prec 0.1347 --ours_auprc 0.5314]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Sequence

import numpy as np


# ---------- data ----------

def load_csv(path: Path, positive: str = "tp_exact") -> Dict[str, np.ndarray]:
    """positive='tp_exact' -> label 1 iff category==TP_EXACT (clinically correct
    detection: correct class + IoU>=0.5); the clinical FP-reduction task aligned
    with the governance routing claim. positive='near_lesion' -> label 1 iff the
    CSV `label` column is 1 (IoU>=0.1 with any GT, looser)."""
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    def fcol(c):
        return np.array([float(r[c]) if r[c] not in ("", "nan") else np.nan for r in rows],
                        dtype=np.float64)
    def fcol_opt(c, default=0.0):  # backward-compatible: missing/empty -> default
        return np.array([float(r[c]) if (c in r and r[c] not in ("", "nan")) else default
                         for r in rows], dtype=np.float64)
    # governance_retain = the DEPLOYED triage target the LGG actually predicts and
    # that the Mode-1 B10 lock evaluates on (lgg_test_metrics B0..B4): RETAIN =
    # {TP_EXACT, CLINICALLY_CLOSE_MISMATCH, LOCALIZATION_WEAK} vs SUPPRESS =
    # {UNSAFE_MISMATCH, BACKGROUND_FP, LOW_CONF_CANDIDATE}. This is the canonical
    # governance task (vs the stricter tp_exact detection task).
    # governance_retain_clean drops the two SDCA-defined categories (CLOSE/UNSAFE)
    # so the target is purely IoU/score-defined -> zero label circularity.
    _RETAIN = {"TP_EXACT", "CLINICALLY_CLOSE_MISMATCH", "LOCALIZATION_WEAK"}
    _RETAIN_CLEAN = {"TP_EXACT", "LOCALIZATION_WEAK"}
    _DROP_CLEAN = {"CLINICALLY_CLOSE_MISMATCH", "UNSAFE_MISMATCH"}
    if positive == "tp_exact":
        lab = np.array([1 if r["category"] == "TP_EXACT" else 0 for r in rows], dtype=np.int64)
    elif positive == "governance_retain":
        lab = np.array([1 if r["category"] in _RETAIN else 0 for r in rows], dtype=np.int64)
    elif positive == "governance_retain_clean":
        keep = np.array([r["category"] not in _DROP_CLEAN for r in rows])
        lab = np.array([1 if r["category"] in _RETAIN_CLEAN else 0 for r in rows], dtype=np.int64)
        # mask out CLOSE/UNSAFE rows entirely (non-circular subset)
        rows = [r for r, k in zip(rows, keep) if k]
        lab = lab[keep]
    else:
        lab = np.array([int(r["label"]) for r in rows], dtype=np.int64)
    # image_id enables image-level (cluster) bootstrap; fall back to per-box id.
    ids = np.array([r.get("image_id") or r.get("bbox_id", str(i))
                    for i, r in enumerate(rows)])
    return {
        "label": lab,
        "image_id": ids,
        "pred_score": fcol("pred_score"),
        "ev_sas": fcol("ev_sas"),
        "ev_sdca": fcol("ev_sdca"),
        "ev_aps": fcol("ev_aps"),
        "ev_pss": fcol("ev_pss"),
        "unc_msp": fcol("unc_msp"),
        "unc_margin": fcol("unc_margin"),
        "unc_entconf": fcol("unc_entconf"),
        "ev_sds": fcol_opt("ev_sds"),          # multi-source disagreement (0 if absent)
        "ev_sds_nsrc": fcol_opt("ev_sds_nsrc"),  # cross-detector corroboration count
    }


# ---------- metrics ----------

def auprc(keep_score: np.ndarray, label: np.ndarray) -> float:
    """All-points AUPRC for keep-score separating label=1 (pos) from label=0."""
    order = np.argsort(-keep_score)
    y = label[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    n_pos = max(int(label.sum()), 1)
    recall = tp / n_pos
    precision = tp / np.maximum(tp + fp, 1)
    # integrate precision over recall
    recall = np.concatenate(([0.0], recall))
    precision = np.concatenate(([1.0], precision))
    return float(np.trapezoid(precision, recall))


def threshold_for_tp_ret(keep_score: np.ndarray, label: np.ndarray,
                         target: float) -> float:
    """Highest threshold t such that TP-retention(keep_score>=t) >= target."""
    pos = keep_score[label == 1]
    if pos.size == 0:
        return float(np.min(keep_score))
    # retain top (target) fraction of positives -> threshold = (1-target) quantile
    q = max(0.0, min(1.0, 1.0 - target))
    return float(np.quantile(pos, q))


def eval_at_threshold(keep_score: np.ndarray, label: np.ndarray,
                      thr: float) -> Dict[str, float]:
    keep = keep_score >= thr
    n_pos = max(int((label == 1).sum()), 1)
    n_neg = max(int((label == 0).sum()), 1)
    tp_kept = int(((label == 1) & keep).sum())
    fp_kept = int(((label == 0) & keep).sum())
    fp_rejected = int(((label == 0) & ~keep).sum())
    tp_ret = tp_kept / n_pos
    fp_supp = fp_rejected / n_neg
    prec = tp_kept / max(tp_kept + fp_kept, 1)
    return {"tp_ret": tp_ret, "fp_supp": fp_supp, "precision_after": prec,
            "threshold": thr}


# ---------- learned baselines ----------

def fit_temperature(score: np.ndarray, label: np.ndarray) -> float:
    """1-D temperature on logits of score to minimise NLL (grid search)."""
    eps = 1e-6
    p = np.clip(score, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    best_T, best_nll = 1.0, math.inf
    for T in [round(0.1 * i, 2) for i in range(1, 51)]:
        q = 1.0 / (1.0 + np.exp(-logit / T))
        q = np.clip(q, eps, 1 - eps)
        nll = -np.mean(label * np.log(q) + (1 - label) * np.log(1 - q))
        if nll < best_nll:
            best_nll, best_T = nll, T
    return best_T


def apply_temperature(score: np.ndarray, T: float) -> np.ndarray:
    eps = 1e-6
    p = np.clip(score, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    return 1.0 / (1.0 + np.exp(-logit / T))


def fit_logreg(X: np.ndarray, y: np.ndarray, iters: int = 500, lr: float = 0.5
               ) -> np.ndarray:
    """Tiny logistic regression (numpy GD), standardised features + bias."""
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xs = (X - mu) / sd
    Xb = np.concatenate([Xs, np.ones((Xs.shape[0], 1))], axis=1)
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        z = Xb @ w
        p = 1.0 / (1.0 + np.exp(-z))
        grad = Xb.T @ (p - y) / Xb.shape[0]
        w -= lr * grad
    return np.concatenate([mu, sd, w])  # pack params


def apply_logreg(X: np.ndarray, params: np.ndarray) -> np.ndarray:
    d = X.shape[1]
    mu, sd, w = params[:d], params[d:2*d], params[2*d:]
    Xs = (X - mu) / sd
    Xb = np.concatenate([Xs, np.ones((Xs.shape[0], 1))], axis=1)
    return 1.0 / (1.0 + np.exp(-(Xb @ w)))


def fit_mlp(X: np.ndarray, y: np.ndarray, hidden: int = 16, iters: int = 1500,
            lr: float = 0.1, seed: int = 0) -> dict:
    """Tiny 1-hidden-layer MLP (numpy, tanh + sigmoid), standardised features.
    Faithful in-protocol proxy of the nonlinear LGG governance gate."""
    rng = np.random.default_rng(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xs = (X - mu) / sd
    n, d = Xs.shape
    W1 = rng.normal(0, 0.5, (d, hidden)); b1 = np.zeros(hidden)
    W2 = rng.normal(0, 0.5, (hidden, 1)); b2 = np.zeros(1)
    for _ in range(iters):
        h = np.tanh(Xs @ W1 + b1)
        z = h @ W2 + b2
        p = 1.0 / (1.0 + np.exp(-z))
        g = (p - y[:, None]) / n
        gW2 = h.T @ g; gb2 = g.sum(0)
        gh = (g @ W2.T) * (1 - h ** 2)
        gW1 = Xs.T @ gh; gb1 = gh.sum(0)
        W1 -= lr * gW1; b1 -= lr * gb1; W2 -= lr * gW2; b2 -= lr * gb2
    return {"mu": mu, "sd": sd, "W1": W1, "b1": b1, "W2": W2, "b2": b2}


def apply_mlp(X: np.ndarray, m: dict) -> np.ndarray:
    Xs = (X - m["mu"]) / m["sd"]
    h = np.tanh(Xs @ m["W1"] + m["b1"])
    z = h @ m["W2"] + m["b2"]
    return (1.0 / (1.0 + np.exp(-z))).ravel()


# ---------- paired (image-clustered) bootstrap CI on the FP-suppression gain ----

def _fp_supp_idx(keep_score: np.ndarray, label: np.ndarray, thr: float,
                 idx: np.ndarray) -> float:
    lb = label[idx]
    keep = keep_score[idx] >= thr
    n_neg = max(int((lb == 0).sum()), 1)
    fp_rejected = int(((lb == 0) & ~keep).sum())
    return fp_rejected / n_neg


def paired_bootstrap_gain(m0_score: np.ndarray, kg_score: np.ndarray,
                          label: np.ndarray, image_id: np.ndarray,
                          thr_m0: float, thr_kg: float,
                          n_boot: int, seed: int = 0) -> dict:
    """Image-clustered paired bootstrap of FP-suppression gain (KG − M0) at FIXED
    thresholds (fit once on full valid). Resamples whole images with replacement so
    boxes correlated within an image are kept together (honest CI width)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(image_id)
    groups = [np.where(image_id == im)[0] for im in uniq]
    n_img = len(groups)
    gains = np.empty(n_boot); kgs = np.empty(n_boot); m0s = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n_img, size=n_img)
        idx = np.concatenate([groups[k] for k in pick])
        m0 = _fp_supp_idx(m0_score, label, thr_m0, idx)
        kg = _fp_supp_idx(kg_score, label, thr_kg, idx)
        gains[b] = kg - m0; kgs[b] = kg; m0s[b] = m0
    lo, hi = (float(x) for x in np.percentile(gains, [2.5, 97.5]))
    return {
        "metric": "fp_supp_gain_kg_minus_m0",
        "n_bootstrap": n_boot, "cluster": "image_id", "n_clusters": n_img,
        "gain_point": 0.0,  # overwritten by caller with measured KG-M0 point gain
        "gain_mean": float(gains.mean()),
        "gain_ci95": [lo, hi],
        "p_gain_gt0": float((gains > 0).mean()),
        "sig": bool(lo > 0.0),
        "kg_fp_supp_ci95": [float(np.percentile(kgs, 2.5)), float(np.percentile(kgs, 97.5))],
        "m0_fp_supp_ci95": [float(np.percentile(m0s, 2.5)), float(np.percentile(m0s, 97.5))],
    }


def paired_bootstrap_auprc(s_a: np.ndarray, s_b: np.ndarray, label: np.ndarray,
                           image_id: np.ndarray, n_boot: int, seed: int = 0) -> dict:
    """Image-clustered paired bootstrap of the AUPRC gain auprc(s_a) - auprc(s_b).
    Threshold-free (no operating point), so no thresholds needed."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(image_id)
    groups = [np.where(image_id == im)[0] for im in uniq]
    n_img = len(groups)
    d = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([groups[k] for k in rng.integers(0, n_img, size=n_img)])
        d[b] = auprc(s_a[idx], label[idx]) - auprc(s_b[idx], label[idx])
    lo, hi = (float(x) for x in np.percentile(d, [2.5, 97.5]))
    return {"gain_mean": float(d.mean()), "gain_ci95": [lo, hi],
            "p_gain_gt0": float((d > 0).mean()), "sig": bool(lo > 0.0),
            "n_bootstrap": n_boot, "cluster": "image_id", "n_clusters": n_img}


# ---------- method registry ----------

EVIDENCE_FEATURES: Tuple[str, ...] = ("ev_sas", "ev_sdca", "ev_aps", "ev_pss")


def build_methods(valid: Dict, test: Dict, mlp_seed: int = 0,
                  drop_features: Sequence[str] = ()
                  ) -> List[Tuple[str, str, np.ndarray, np.ndarray]]:
    """Return list of (name, family, keep_score_valid, keep_score_test).

    mlp_seed controls the gate (MLP) weight init -> multi-seed robustness checks.

    drop_features removes evidence pillars from the gate input. Motivation
    (2026-09-14): ev_pss is a real, varying signal on the OIA-DDR split the gate
    is fitted on (59% 'implemented') but a dead constant 0.5 on every external
    cohort (100% 'placeholder'). Fitting on a live feature and testing on a
    constant that lies outside the training range confounds the domain-shift
    comparison. Dropping ev_pss (and ev_aps, constant everywhere) makes the
    feature set identical across domains. Default () = unchanged behaviour.
    """
    ev_used = [f for f in EVIDENCE_FEATURES if f not in set(drop_features)]
    if not ev_used:
        raise SystemExit("[bench] --drop_features membuang SEMUA fitur evidence; "
                         "sisakan minimal satu.")
    methods = []
    # 1. detector score (M0)
    methods.append(("M0_score", "threshold", valid["pred_score"], test["pred_score"]))
    # 2. MSP
    methods.append(("MSP", "confidence", valid["unc_msp"], test["unc_msp"]))
    # 3. Entropy-confidence
    methods.append(("Entropy", "uncertainty", valid["unc_entconf"], test["unc_entconf"]))
    # 4. Margin
    methods.append(("Margin", "uncertainty", valid["unc_margin"], test["unc_margin"]))
    # 5. Temperature scaling (fit on valid)
    T = fit_temperature(valid["pred_score"], valid["label"])
    methods.append((f"TempScale(T={T:.2f})", "calibration",
                    apply_temperature(valid["pred_score"], T),
                    apply_temperature(test["pred_score"], T)))
    # 6. Logistic regression on evidence [SAS,SDCA,APS,PSS] (learned-linear)
    Xv = np.stack([valid[f] for f in ev_used], 1)
    Xt = np.stack([test[f] for f in ev_used], 1)
    params = fit_logreg(Xv, valid["label"].astype(np.float64))
    methods.append(("Evidence-LogReg", "learned-linear",
                    apply_logreg(Xv, params), apply_logreg(Xt, params)))
    # 6b. Score-MLP: SAME nonlinear model on the detector score ALONE. Comparing
    #     the full KG-gate against this isolates the KG-evidence contribution
    #     (KG-gate - Score-MLP), controlling for model capacity.
    Xv_s = valid["pred_score"].reshape(-1, 1)
    Xt_s = test["pred_score"].reshape(-1, 1)
    mlp_s = fit_mlp(Xv_s, valid["label"].astype(np.float64), seed=mlp_seed)
    methods.append(("Score-MLP", "learned-score",
                    apply_mlp(Xv_s, mlp_s), apply_mlp(Xt_s, mlp_s)))
    # 7. KG-gate (MLP, ours): nonlinear gate on the SAME KG-evidence vector,
    #    a faithful in-protocol proxy of the Learned Governance Gate (LGG).
    Xv2 = np.stack([valid[f] for f in ev_used] + [valid["pred_score"]], 1)
    Xt2 = np.stack([test[f] for f in ev_used] + [test["pred_score"]], 1)
    mlp = fit_mlp(Xv2, valid["label"].astype(np.float64), seed=mlp_seed)
    methods.append(("KG-gate-MLP (ours)", "governance",
                    apply_mlp(Xv2, mlp), apply_mlp(Xt2, mlp)))
    # 8. B5 = KG-gate + multi-source disagreement pillar (ev_sds). Ablation of the
    #    new evidence pillar; compared head-to-head against the B4 KG-gate above.
    Xv3 = np.concatenate([Xv2, valid["ev_sds"].reshape(-1, 1)], axis=1)
    Xt3 = np.concatenate([Xt2, test["ev_sds"].reshape(-1, 1)], axis=1)
    mlp3 = fit_mlp(Xv3, valid["label"].astype(np.float64), seed=mlp_seed)
    methods.append(("KG-gate+SDS (B5)", "governance",
                    apply_mlp(Xv3, mlp3), apply_mlp(Xt3, mlp3)))
    # 9. B6 = B5 + cross-detector EXISTENCE corroboration (ev_sds_nsrc): 1 = primary
    #    only (uncorroborated -> likely FP), 2 = a second independent detector also
    #    fired here (corroborated -> likely TP). 100% coverage; captures the
    #    strongest multi-source signal that JSD-on-co-detected boxes throws away.
    Xv4 = np.concatenate([Xv3, valid["ev_sds_nsrc"].reshape(-1, 1)], axis=1)
    Xt4 = np.concatenate([Xt3, test["ev_sds_nsrc"].reshape(-1, 1)], axis=1)
    mlp4 = fit_mlp(Xv4, valid["label"].astype(np.float64), seed=mlp_seed)
    methods.append(("KG-gate+SDS+corrob (B6)", "governance",
                    apply_mlp(Xv4, mlp4), apply_mlp(Xt4, mlp4)))
    return methods


def main() -> int:
    ap = argparse.ArgumentParser(description="Controlled FP-reduction benchmark (>=5 methods).")
    ap.add_argument("--valid_csv", required=True, type=Path)
    ap.add_argument("--test_csv", required=True, type=Path)
    ap.add_argument("--tp_ret_target", type=float, default=0.80)
    ap.add_argument("--positive",
                    choices=["tp_exact", "near_lesion", "governance_retain",
                             "governance_retain_clean"],
                    default="tp_exact",
                    help="evaluation task. tp_exact = strict detection (correct "
                         "class + IoU>=0.5). governance_retain = the DEPLOYED triage "
                         "target the LGG predicts and the B10 lock evaluates on "
                         "(RETAIN vs SUPPRESS) -> the canonical governance claim. "
                         "governance_retain_clean = same but drops SDCA-defined "
                         "categories (zero label circularity). near_lesion = IoU>=0.1.")
    ap.add_argument("--output_json", required=True, type=Path)
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="if >0, image-clustered paired bootstrap iters for a 95%% "
                         "CI on the FP-suppression gain (KG-gate - M0). 0 = off "
                         "(default; preserves prior behaviour).")
    ap.add_argument("--bootstrap_seed", type=int, default=0)
    ap.add_argument("--mlp_seed", type=int, default=0,
                    help="gate (MLP) weight-init seed -> vary for multi-seed robustness.")
    ap.add_argument("--drop_features", nargs="*", default=[],
                    choices=list(EVIDENCE_FEATURES),
                    help="drop evidence pillars from the gate input (default: none, "
                         "i.e. unchanged behaviour). Use --drop_features ev_pss ev_aps "
                         "to make the feature set identical across domains: ev_pss is "
                         "real on the OIA-DDR fitting split but a dead 0.5 constant on "
                         "every external cohort, which confounds domain-shift transfer.")
    args = ap.parse_args()

    valid = load_csv(args.valid_csv, args.positive)
    test = load_csv(args.test_csv, args.positive)
    methods = build_methods(valid, test, mlp_seed=args.mlp_seed,
                            drop_features=args.drop_features)

    results = []
    cap = {}  # method -> (threshold, test keep-score) for bootstrap
    for name, family, sv, st_ in methods:
        thr = threshold_for_tp_ret(sv, valid["label"], args.tp_ret_target)
        m = eval_at_threshold(st_, test["label"], thr)
        m["auprc"] = auprc(st_, test["label"])
        m.update({"method": name, "family": family})
        results.append(m)
        cap[name] = (thr, st_)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {"tool": "eval_fp_reduction_benchmark",
               "positive_label": args.positive,
               "tp_ret_target": args.tp_ret_target,
               "n_test_boxes": int(test["label"].size),
               "n_test_pos": int((test["label"] == 1).sum()),
               "n_test_neg": int((test["label"] == 0).sum()),
               "mlp_seed": int(args.mlp_seed),
               "evidence_features_used": [f for f in EVIDENCE_FEATURES
                                          if f not in set(args.drop_features)],
               "evidence_features_dropped": list(args.drop_features),
               "results": results}

    # G4: paired image-clustered bootstrap CI on the headline gain (KG - M0).
    if args.bootstrap > 0:
        ours = next((r["method"] for r in results if r["method"].startswith("KG-gate")), None)
        if ours and "M0_score" in cap and ours in cap:
            thr_m0, s_m0 = cap["M0_score"]
            thr_kg, s_kg = cap[ours]
            bs = paired_bootstrap_gain(s_m0, s_kg, test["label"], test["image_id"],
                                       thr_m0, thr_kg, args.bootstrap, args.bootstrap_seed)
            m0_fp = next(r["fp_supp"] for r in results if r["method"] == "M0_score")
            kg_fp = next(r["fp_supp"] for r in results if r["method"] == ours)
            bs["gain_point"] = float(kg_fp - m0_fp)
            payload["bootstrap"] = bs
            print(f"\n[bench] bootstrap({args.bootstrap}, cluster=image, n_img={bs['n_clusters']}): "
                  f"FP-suppr gain={bs['gain_point']:+.3f}  CI95=[{bs['gain_ci95'][0]:+.3f},"
                  f"{bs['gain_ci95'][1]:+.3f}]  P(>0)={bs['p_gain_gt0']:.3f}  "
                  f"sig={'YES' if bs['sig'] else 'NO'}")
            # Threshold-free AUPRC-gain CIs: KG vs M0 (governance over score) and
            # KG vs Score-MLP (isolated KG-evidence contribution).
            au = {}
            kg_auprc = next(r["auprc"] for r in results if r["method"] == ours)
            m0_auprc = next(r["auprc"] for r in results if r["method"] == "M0_score")
            bsa = paired_bootstrap_auprc(s_kg, s_m0, test["label"], test["image_id"],
                                         args.bootstrap, args.bootstrap_seed)
            bsa["gain_point"] = float(kg_auprc - m0_auprc)
            au["kg_minus_m0"] = bsa
            print(f"[bench] AUPRC gain (KG - M0)       ={bsa['gain_point']:+.3f}  "
                  f"CI95=[{bsa['gain_ci95'][0]:+.3f},{bsa['gain_ci95'][1]:+.3f}]  "
                  f"sig={'YES' if bsa['sig'] else 'NO'}")
            if "Score-MLP" in cap:
                _, s_sm = cap["Score-MLP"]
                sm_auprc = next(r["auprc"] for r in results if r["method"] == "Score-MLP")
                bss = paired_bootstrap_auprc(s_kg, s_sm, test["label"], test["image_id"],
                                             args.bootstrap, args.bootstrap_seed)
                bss["gain_point"] = float(kg_auprc - sm_auprc)
                au["kg_minus_scoremlp"] = bss
                print(f"[bench] AUPRC gain (KG - ScoreMLP) ={bss['gain_point']:+.3f}  "
                      f"CI95=[{bss['gain_ci95'][0]:+.3f},{bss['gain_ci95'][1]:+.3f}]  "
                      f"sig={'YES' if bss['sig'] else 'NO'}  (isolated KG contribution)")
            # ev_sds ablation: B5 (KG-gate + multi-source disagreement) vs B4 (KG-gate)
            b5 = next((r["method"] for r in results if r["method"].startswith("KG-gate+SDS")), None)
            if b5 and b5 in cap:
                _, s_b5 = cap[b5]
                b5_auprc = next(r["auprc"] for r in results if r["method"] == b5)
                bsd = paired_bootstrap_auprc(s_b5, s_kg, test["label"], test["image_id"],
                                             args.bootstrap, args.bootstrap_seed)
                bsd["gain_point"] = float(b5_auprc - kg_auprc)
                au["b5_minus_b4_evsds"] = bsd
                sds_mean = float(test["ev_sds"].mean()); sds_std = float(test["ev_sds"].std())
                sds_nz = float((test["ev_sds"] > 0).mean())
                au["ev_sds_stats"] = {"mean": sds_mean, "std": sds_std, "nonzero_frac": sds_nz}
                print(f"[bench] AUPRC gain (B5 - B4, +ev_sds)={bsd['gain_point']:+.4f}  "
                      f"CI95=[{bsd['gain_ci95'][0]:+.4f},{bsd['gain_ci95'][1]:+.4f}]  "
                      f"sig={'YES' if bsd['sig'] else 'NO'}  "
                      f"[ev_sds std={sds_std:.3f} cov={sds_nz:.0%}]  (multi-source disagreement)")
            # B6 = B5 + cross-detector existence corroboration (ev_sds_nsrc).
            # B6 - B4 = TOTAL multi-source contribution (disagreement + corroboration).
            b6 = next((r["method"] for r in results if r["method"].startswith("KG-gate+SDS+corrob")), None)
            if b6 and b6 in cap:
                _, s_b6 = cap[b6]
                b6_auprc = next(r["auprc"] for r in results if r["method"] == b6)
                bb4 = paired_bootstrap_auprc(s_b6, s_kg, test["label"], test["image_id"],
                                             args.bootstrap, args.bootstrap_seed)
                bb4["gain_point"] = float(b6_auprc - kg_auprc)
                au["b6_minus_b4_multisource"] = bb4
                nsrc2 = float((test["ev_sds_nsrc"] >= 2).mean())
                print(f"[bench] AUPRC gain (B6 - B4, multi-src)={bb4['gain_point']:+.4f}  "
                      f"CI95=[{bb4['gain_ci95'][0]:+.4f},{bb4['gain_ci95'][1]:+.4f}]  "
                      f"sig={'YES' if bb4['sig'] else 'NO'}  "
                      f"[corroborated={nsrc2:.0%}]  (ev_sds + existence corroboration)")
            payload["bootstrap_auprc"] = au
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # stdout table
    print(f"\n[bench] FP-reduction benchmark @ matched TP-retention={args.tp_ret_target}")
    print(f"[bench] test boxes={test['label'].size}  pos={int((test['label']==1).sum())}  "
          f"neg={int((test['label']==0).sum())}")
    print(f"\n{'method':<20}{'family':<16}{'TPret':>8}{'FPsupp':>9}{'Prec':>8}{'AUPRC':>8}")
    print("-" * 69)
    for r in results:
        fp = f"{r['fp_supp']:.3f}" if r['fp_supp'] is not None else "  -  "
        tr = f"{r['tp_ret']:.3f}" if r['tp_ret'] is not None else "  -  "
        pr = f"{r['precision_after']:.3f}" if r['precision_after'] is not None else "  -  "
        au = f"{r['auprc']:.3f}" if r['auprc'] is not None else "  -  "
        print(f"{r['method']:<20}{r['family']:<16}{tr:>8}{fp:>9}{pr:>8}{au:>8}")
    print(f"\n[bench] wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
