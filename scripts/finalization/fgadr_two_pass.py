"""Locked follow-up: source-matched two-pass fusion input on FGADR.

No detector training or tuning. Existing six evidence columns are retained,
including historical secondary evidence. Only the fusion input uses primary+flip
on BOTH source validation and target. This is not a two-pass compute-budget claim.
All new outputs stay beside this script; historical inputs are read-only.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys
from pathlib import Path
import json
import hashlib
import time
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from tools.eval_fp_reduction_benchmark import fit_mlp, apply_mlp

EV = ['ev_sas', 'ev_sdca', 'ev_aps', 'ev_pss', 'ev_sds', 'ev_sds_nsrc']
VALID = ROOT / 'outputs/0506/evsds_v2sec/oiaddr/valid/governance_features_valid.csv'
TARGET = ROOT / 'outputs/0506/evsds_pss/fgadr/train/governance_features_train.csv'
PROTOCOL = dict(fit='OIA-DDR valid only', target='FGADR train, held-out cohort',
                evidence=EV, fusion='(primary + matched flip)/2 on source and target',
                missing_match_score=0, iou_min=0.15, seeds=list(range(5)),
                hidden=16, iterations=1500, learning_rate=0.1,
                bootstrap_samples=2000, bootstrap_seed=0, cluster='image_id',
                estimator='average_precision', tuning='none',
                interpretation='Report all seeds and paired CIs regardless of sign; '
                'not pooled with the three-pass protocol and not clinical validation.',
                historical_secondary_evidence_retained=True)


def log(message):
    print(message, flush=True)


def entries(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    items = json.loads(path.read_text(encoding='utf-8'))
    out = []
    for item in items:
        raw = item.get('detector_raw') or item.get('r2a_raw') or {}
        if 'bbox' in raw and 'score' in raw:
            out.append((item.get('bbox_id'), raw['bbox'], float(raw['score'])))
    return out


def fused_two(frame, cohort, split):
    matched = np.zeros(len(frame), dtype=bool)
    flip = np.zeros(len(frame), dtype=float)
    for image_id, group in frame.groupby('image_id', sort=False):
        base = ROOT / 'outputs/detector_cache'
        primary = entries(base / 'primary' / cohort / split / f'{image_id}.json')
        secondary = entries(base / 'secondary' / cohort / split / f'{image_id}.json')
        boxes = {bid: box for bid, box, _ in primary}
        p = np.array([boxes[bid] for bid in group.bbox_id], dtype=float)
        if not secondary:
            continue
        q = np.array([box for _, box, _ in secondary], dtype=float)
        score = np.array([score for _, _, score in secondary])
        wh = np.maximum(0, np.minimum(p[:, None, 2:], q[None, :, 2:]) -
                        np.maximum(p[:, None, :2], q[None, :, :2]))
        inter = wh.prod(axis=2)
        union = ((p[:, 2:] - p[:, :2]).prod(axis=1)[:, None] +
                 (q[:, 2:] - q[:, :2]).prod(axis=1)[None, :] - inter)
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        ix = iou.argmax(axis=1)
        found = iou[np.arange(len(group)), ix] >= 0.15
        flip[group.index] = np.where(found, score[ix], 0.)
        matched[group.index] = found
    return (frame.pred_score.to_numpy() + flip) / 2., float(matched.mean())


def prepare_ap(score, label, image_index):
    order = np.argsort(-score, kind='stable')
    ends = np.r_[np.flatnonzero(np.diff(score[order]) != 0), len(order) - 1]
    return label[order], image_index[order], ends


def weighted_ap(prepared, counts):
    y, group, ends = prepared
    w = counts[group]
    tp = np.cumsum(w * y)[ends]
    total = np.cumsum(w)[ends]
    return float(np.sum(np.diff(np.r_[0., tp]) * tp / np.maximum(total, 1)) / tp[-1])


def main():
    output = HERE / 'fgadr_two_pass_results.json'
    if output.exists():
        raise RuntimeError('Refusing to overwrite a completed experiment')
    (HERE / 'fgadr_two_pass_protocol.json').write_text(json.dumps(PROTOCOL, indent=2), encoding='utf-8')
    t0 = time.perf_counter()
    types = {'image_id': str, 'bbox_id': str}
    va, te = pd.read_csv(VALID, dtype=types), pd.read_csv(TARGET, dtype=types)
    assert set(va.label.unique()) <= {0, 1} and set(te.label.unique()) <= {0, 1}
    fv, cv = fused_two(va, 'oiaddr', 'valid')
    ft, ct = fused_two(te, 'fgadr', 'train')
    yv, yt = va.label.to_numpy(), te.label.to_numpy()
    baseline = float(average_precision_score(yt, ft))
    previous = json.loads((ROOT / '_septem/scripts_optionB/fusion_external.json').read_text())
    expected = next(c['fused_ap'] for c in previous['cohorts'] if c['key'] == 'fgadr')
    assert abs(baseline - expected) < 1e-12, (baseline, expected)
    log(f'Fusion positive control passed: {baseline:.12f}; fitting five fixed seeds')
    xv = np.column_stack([va[EV].to_numpy(), fv])
    xt = np.column_stack([te[EV].to_numpy(), ft])
    unique, image_index = np.unique(te.image_id.to_numpy(), return_inverse=True)
    rng = np.random.default_rng(0)
    draws = np.array([np.bincount(rng.integers(0, len(unique), size=len(unique)),
                                 minlength=len(unique)) for _ in range(2000)], dtype=np.int32)
    base_prepared = prepare_ap(ft, yt, image_index)
    for weights in draws[:4]:
        sample = np.repeat(np.arange(len(yt)), weights[image_index])
        assert abs(weighted_ap(base_prepared, weights) - average_precision_score(yt[sample], ft[sample])) < 1e-12
    baseline_boot = np.array([weighted_ap(base_prepared, w) for w in draws])
    results = []
    with threadpool_limits(limits=1):
        for seed in range(5):
            model = fit_mlp(xv, yv, seed=seed)
            score = apply_mlp(xt, model)
            ap = float(average_precision_score(yt, score))
            prepared = prepare_ap(score, yt, image_index)
            for weights in draws[:4]:
                sample = np.repeat(np.arange(len(yt)), weights[image_index])
                assert abs(weighted_ap(prepared, weights) - average_precision_score(yt[sample], score[sample])) < 1e-12
            differences = np.array([weighted_ap(prepared, w) for w in draws]) - baseline_boot
            ci = np.percentile(differences, [2.5, 97.5]).tolist()
            results.append(dict(seed=seed, ap=ap, gain=ap-baseline, ci95=ci,
                                bootstrap_gain_mean=float(differences.mean())))
            log(f'seed {seed}: AP {ap:.6f}; gain {ap-baseline:+.6f}; CI {ci}; elapsed {time.perf_counter()-t0:.0f}s')
            np.savez_compressed(HERE / f'fgadr_two_pass_seed{seed}.npz', **model)
    aps = [r['ap'] for r in results]
    result = dict(protocol=PROTOCOL, n_valid=len(va), n_target=len(te), n_target_images=len(unique),
                  baseline_ap=baseline, prior_baseline_ap=expected, control_passed=True,
                  flip_coverage_valid=cv, flip_coverage_target=ct, per_seed=results,
                  mean_ap=float(np.mean(aps)), sd_ap=float(np.std(aps, ddof=1)),
                  mean_gain=float(np.mean(aps)-baseline),
                  positive_seed_count=sum(r['gain'] > 0 for r in results),
                  ci_above_zero_count=sum(r['ci95'][0] > 0 for r in results),
                  ci_below_zero_count=sum(r['ci95'][1] < 0 for r in results),
                  bootstrap_validation='Exact weighted image resampling checked against sklearn expanded samples, 4 draws per score',
                  source_sha256={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [VALID, TARGET, Path(__file__)]},
                  elapsed_seconds=time.perf_counter()-t0)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    log('DONE ' + str(output))


if __name__ == '__main__':
    main()
