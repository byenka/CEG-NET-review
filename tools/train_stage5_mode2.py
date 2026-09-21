#!/usr/bin/env python
"""
train_stage5_mode2.py
Phase 19N — Stage 5: Mode 2 / C3 / SDGL Governance-Centric Training.

Architecture spec (architecture-lgov-net-v3_realistic.txt, §C3):
  L_SDGL  = λ1·L_det^R1  +  λ2·L_soft^R2  +  λ3·Σ_j w_j·L_det^R3
  L_KG    = Σ_{(i,j) ∈ E_fixed} (SDM(c_i, c_j) − w_target)²
  L_total = L_SDGL + λ_gate·L_gate + λ_KG·L_KG

What this script trains:
  This script now supports two execution paths:
  1) Proxy path (default, Phase 1.5):
      - KG encoder + LGG + SDGL re-classifier over Stage 3C governance records.
      - Detector scores are frozen inputs.
  2) Full-detector path (--full_mode2_detector):
      - Loads the real Faster R-CNN checkpoint.
      - Replaces torchvision ROI classification loss with SDGL-shaped loss.
      - Fine-tunes detector parameters end-to-end on real detection batches.

Architecture note:
  The --full_mode2_detector path closes Mode 2 condition (i): SDGL loss is
  applied directly in detector training instead of proxy re-classification.
  The default proxy path remains available for continuity of existing Stage 5
  experiments and legacy ablation comparisons.

Mandatory ablations for C3 claim (architecture §C3 conditional rule):
  (i)  Standard run: full SDGL with SDM-temp, closed-loop KG, with L_KG.
  (ii) --uniform_temp: τ=1 for all classes (SDM-temp vs uniform-temp ablation).
  (iii) --freeze_kg: freeze KG encoder (closed-loop vs frozen-KG ablation).
  (iv) --lambda_kg 0.0: remove L_KG anchor (with vs without L_KG ablation).

Usage:
  # Smoke test
  python tools/train_stage5_mode2.py \\
    --detector_checkpoint outputs/detector_stage1/stage1c_fasterrcnn_r5b_ext25/fasterrcnn_best.pt \\
    --lgg_checkpoint outputs/stage3d_lgg/lgg_models.pkl \\
    --kg_dir outputs/stage3_audit \\
    --stage3c_dir outputs/stage3c_audit \\
    --data_root data/oiaddr \\
    --epochs 1 --smoke_test \\
    --output_dir outputs/mode2/smoke

  # Full standard run
  python tools/train_stage5_mode2.py \\
    --detector_checkpoint outputs/detector_stage1/stage1c_fasterrcnn_r5b_ext25/fasterrcnn_best.pt \\
    --lgg_checkpoint outputs/stage3d_lgg/lgg_models.pkl \\
    --kg_dir outputs/stage3_audit \\
    --stage3c_dir outputs/stage3c_audit \\
    --data_root data/oiaddr \\
    --epochs 20 --batch_size 128 --lr 5e-5 \\
    --lambda1 1.0 --lambda2 0.5 --lambda3 0.3 \\
    --lambda_kg 0.1 --lambda_gate 0.05 \\
    --tau_start 1.0 --tau_end 0.1 \\
    --seed 42 \\
    --output_dir outputs/mode2/standard

  # Ablation: uniform temperature (no SDM modulation)
  python tools/train_stage5_mode2.py ... --uniform_temp --output_dir outputs/mode2/ablation_uniform_temp

  # Ablation: frozen KG (no co-adaptation)
  python tools/train_stage5_mode2.py ... --freeze_kg --output_dir outputs/mode2/ablation_frozen_kg

  # Ablation: no L_KG anchor
  python tools/train_stage5_mode2.py ... --lambda_kg 0.0 --output_dir outputs/mode2/ablation_no_lkg
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, "..")
if _root not in sys.path:
    sys.path.insert(0, _root)

from lgovnet.src.governance.gate import LGG, concat_gate_input, build_pred_embedding
from lgovnet.src.kg.builder import build_graph_from_artifacts

try:
    from src.training.topk_checkpoints import TopKCheckpointManager
except ModuleNotFoundError:
    from lgovnet.src.training.topk_checkpoints import TopKCheckpointManager  # type: ignore
from lgovnet.src.kg.encoder import build_encoder_inputs
from lgovnet.src.kg.encoder import LGovNetEncoder
from lgovnet.src.losses.gate_loss import GateLoss
from lgovnet.src.losses.sdgl import (
    SDGLLoss,
    compute_l_kg,
    build_q_sdm,
    LESION_CLASSES,
)

try:
    from src.data.coco_adapter import DRDetectionDataset, collate_fn
    from src.models.fasterrcnn_roi_sdgl import (
        enable_roi_sdgl_loss,
        get_last_sdgl_roi_stats,
        set_roi_sdgl_dynamic_state,
        restore_default_roi_loss,
    )
    from src.losses.detector_anchor import (
        capture_roi_head_outputs,
        compute_anchor_loss,
    )
    from src.losses.online_evidence import (
        FRCNN_LABEL_TO_CLASS,
        build_proposal_records,
        evidence_fidelity_stats,
        load_compat_scores,
        load_secondary_detector_cache,
        records_to_tensors,
        summarise_records,
    )
    from src.governance.multiclass_inference import predict_with_class_probs
except ModuleNotFoundError:
    from lgovnet.src.data.coco_adapter import DRDetectionDataset, collate_fn
    from lgovnet.src.models.fasterrcnn_roi_sdgl import (
        enable_roi_sdgl_loss,
        get_last_sdgl_roi_stats,
        set_roi_sdgl_dynamic_state,
        restore_default_roi_loss,
    )
    from lgovnet.src.losses.detector_anchor import (
        capture_roi_head_outputs,
        compute_anchor_loss,
    )
    from lgovnet.src.losses.online_evidence import (
        FRCNN_LABEL_TO_CLASS,
        build_proposal_records,
        evidence_fidelity_stats,
        load_compat_scores,
        load_secondary_detector_cache,
        records_to_tensors,
        summarise_records,
    )
    from lgovnet.src.governance.multiclass_inference import predict_with_class_probs

try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    _TM_OK = True
except Exception:
    MeanAveragePrecision = None
    _TM_OK = False
    warnings.warn(
        "torchmetrics not available — full-mode checkpoint composite will "
        "fallback to manual AP50-only metrics."
    )

_TM_RUNTIME_WARNED = False

# Tier index mapping (matches stage3c tier_target column)
TIER_NAME_TO_IDX: Dict[str, int] = {"R1": 0, "R2": 1, "R3": 2, "R4": 3}
CLASS_IDS: Tuple[int, ...] = tuple(sorted(FRCNN_LABEL_TO_CLASS.keys()))
CLASS_NAME_TO_ID: Dict[str, int] = {
    class_name: class_id for class_id, class_name in FRCNN_LABEL_TO_CLASS.items()
}


def _to_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        return float(value.item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_float_list(
    raw: Optional[str],
    expected_len: int,
    arg_name: str,
) -> Optional[List[float]]:
    if raw is None:
        return None
    txt = str(raw).strip()
    if not txt:
        return None
    parts = [p.strip() for p in txt.split(",") if p.strip()]
    if len(parts) != expected_len:
        raise ValueError(
            f"{arg_name} must contain {expected_len} comma-separated floats; got {len(parts)}"
        )
    try:
        return [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"{arg_name} contains non-float value: {exc}") from exc


def _iou_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.shape[0] == 0 or b.shape[0] == 0:
        return torch.zeros(a.shape[0], b.shape[0], dtype=torch.float32)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0.0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-6)


def _compute_ap50_manual(preds: List[Dict], targets: List[Dict]) -> Dict[int, float]:
    ap: Dict[int, float] = {}
    for cls_id in CLASS_IDS:
        det_scores: List[float] = []
        det_tp: List[int] = []
        n_gt = 0
        for pred, tgt in zip(preds, targets):
            gt_m = tgt["labels"] == cls_id
            gt_b = tgt["boxes"][gt_m].cpu().float()
            n_gt += int(gt_m.sum())

            pd_m = pred["labels"] == cls_id
            if not pd_m.any():
                continue
            pb = pred["boxes"][pd_m].cpu().float()
            ps = pred["scores"][pd_m].cpu().float()

            if gt_b.shape[0] == 0:
                det_scores += ps.tolist()
                det_tp += [0] * pb.shape[0]
                continue

            iou = _iou_matrix(pb, gt_b)
            matched = torch.full((gt_b.shape[0],), False)
            for idx in ps.argsort(descending=True).tolist():
                best_iou, best_j = iou[idx].max(dim=0)
                det_scores.append(float(ps[idx]))
                if float(best_iou) >= 0.50 and not matched[int(best_j)]:
                    matched[int(best_j)] = True
                    det_tp.append(1)
                else:
                    det_tp.append(0)

        if n_gt == 0:
            ap[cls_id] = float("nan")
            continue

        order = sorted(range(len(det_scores)), key=lambda i: -det_scores[i])
        tp_cum = 0
        fp_cum = 0
        precs: List[float] = []
        recs: List[float] = []
        for i in order:
            if det_tp[i]:
                tp_cum += 1
            else:
                fp_cum += 1
            precs.append(tp_cum / max(tp_cum + fp_cum, 1))
            recs.append(tp_cum / max(n_gt, 1))

        ap_val = 0.0
        for t in [i / 10 for i in range(11)]:
            prec_at = [p for p, r in zip(precs, recs) if r >= t]
            ap_val += (max(prec_at) if prec_at else 0.0) / 11
        ap[cls_id] = ap_val
    return ap


def _count_targets_per_class(targets: List[Dict]) -> Dict[int, int]:
    counts = {c: 0 for c in CLASS_IDS}
    for tgt in targets:
        labels = tgt.get("labels", torch.zeros(0, dtype=torch.long))
        for cls_id in CLASS_IDS:
            counts[cls_id] += int((labels == cls_id).sum().item())
    return counts


def _compute_false_positive_pressure(
    preds: List[Dict],
    targets: List[Dict],
    iou_thr: float = 0.50,
) -> float:
    total_fp = 0
    for cls_id in CLASS_IDS:
        for pred, tgt in zip(preds, targets):
            gt_m = tgt["labels"] == cls_id
            gt_b = tgt["boxes"][gt_m].cpu().float()

            pd_m = pred["labels"] == cls_id
            if not pd_m.any():
                continue
            pb = pred["boxes"][pd_m].cpu().float()
            ps = pred["scores"][pd_m].cpu().float()

            if gt_b.shape[0] == 0:
                total_fp += int(pb.shape[0])
                continue

            iou = _iou_matrix(pb, gt_b)
            matched = torch.full((gt_b.shape[0],), False)
            for idx in ps.argsort(descending=True).tolist():
                best_iou, best_j = iou[idx].max(dim=0)
                if float(best_iou) >= iou_thr and not matched[int(best_j)]:
                    matched[int(best_j)] = True
                else:
                    total_fp += 1

    n_gt_per_class = _count_targets_per_class(targets)
    total_gt = sum(n_gt_per_class.values())
    return total_fp / max(total_gt, 1)


# ── Governance features loader ─────────────────────────────────────────────────


def load_governance_csv(csv_path: Path) -> List[Dict]:
    """Load Stage 3C governance_features CSV into list of row dicts."""
    rows = []
    if not csv_path.exists():
        raise FileNotFoundError(f"Stage 3C CSV not found: {csv_path}")
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"[Mode2] Loaded {len(rows)} governance records from {csv_path.name}")
    return rows


def _safe_float(row: Dict, key: str, default: float = 0.5) -> float:
    try:
        v = row.get(key, "")
        return float(v) if v not in ("", "None", "nan") else default
    except (ValueError, TypeError):
        return default


def parse_governance_row(row: Dict) -> Optional[Dict]:
    """Extract training-relevant fields from one Stage 3C CSV row.

    Returns None for R4 rows (no gradient contribution) or malformed rows.
    """
    tier_str = row.get("tier_target", "R4").strip()
    tier_idx = TIER_NAME_TO_IDX.get(tier_str, 3)

    # R4 — no gradient contribution; skip
    if tier_idx == 3:
        return None

    majority_class = row.get("majority_class", "MA").strip()
    if majority_class not in LESION_CLASSES:
        majority_class = "MA"

    # P0 leakage fix (2026-05-28): a supervised R1/R3 target requires a real
    # IoU>=0.5 GT match. Rows whose Stage 3C gt_class is unresolved ("") must NOT
    # fall back to majority_class (= detector argmax) — that would train the
    # re-classifier toward the detector's own prediction (circular leakage,
    # measured 95.8% on 2026-05-28). Exclude unresolved rows from supervision.
    gt_class = row.get("gt_class", "").strip()
    if gt_class not in LESION_CLASSES:
        return None
    gt_class_idx = LESION_CLASSES.index(gt_class)

    ev_sas = _safe_float(row, "ev_sas")
    ev_sdca = _safe_float(row, "ev_sdca")
    ev_aps = _safe_float(row, "ev_aps")
    ev_pss = _safe_float(row, "ev_pss")

    consensus = _safe_float(row, "consensus", 0.5)
    aps = ev_aps  # APS from evidence

    # Per-class probabilities: prefer explicit columns, fallback to uniform
    prob_dict = {}
    for cls in LESION_CLASSES:
        col = f"prob_{cls.lower().replace('_', '')}"
        if col not in row:
            col = f"prob_{cls}"
        prob_dict[cls] = _safe_float(row, col, 1.0 / len(LESION_CLASSES))

    # Normalise probabilities
    total = sum(prob_dict.values())
    if total > 0:
        prob_dict = {k: v / total for k, v in prob_dict.items()}

    return {
        "tier_idx": tier_idx,
        "majority_class": majority_class,
        "gt_class_idx": gt_class_idx,
        "evidence": torch.tensor(
            [ev_sas, ev_sdca, ev_aps, ev_pss], dtype=torch.float32
        ),
        "ensemble_probs": prob_dict,
        "consensus": consensus,
        "aps": aps,
    }


# ── KG artifact loader ─────────────────────────────────────────────────────────


def load_sdm_norm(kg_dir: Path) -> Dict:
    path = kg_dir / "sdm_norm_clinical_prior.json"
    if not path.exists():
        raise FileNotFoundError(f"SDM norm not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    result = {}
    for k, v in data.items():
        parts = k.split("__", 1)
        if len(parts) == 2:
            result[tuple(parts)] = v
    print(f"[Mode2] Loaded SDM norm: {len(result)} pairs from {path.name}")
    return result


def load_kg_graph(kg_dir: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load KG graph tensors for LGovNetEncoder (nodes, edges, edge_types)."""
    nodes_path = kg_dir / "nodes.json"
    edges_path = kg_dir / "edges.json"
    if not nodes_path.exists() or not edges_path.exists():
        raise FileNotFoundError(
            f"KG graph files not found in {kg_dir}. Run STEP 6 (build_static_kg.py) first."
        )
    with open(nodes_path, encoding="utf-8") as f:
        nodes_data = json.load(f)
    with open(edges_path, encoding="utf-8") as f:
        edges_data = json.load(f)

    # Stage 3A artifacts store node/edge metadata, not encoder-ready tensors.
    # Rebuild the canonical KG structure via the repo builder, then convert it
    # to padded encoder inputs with undirected edges mirrored for message passing.
    if (
        nodes_data
        and isinstance(nodes_data[0], dict)
        and "features" not in nodes_data[0]
    ) or (
        edges_data
        and isinstance(edges_data[0], dict)
        and "edge_type_idx" not in edges_data[0]
    ):
        kg = build_graph_from_artifacts(nodes_path, edges_path)
        enc_inputs = build_encoder_inputs(kg, node_feat_dim=17, device="cpu")
        node_feats = enc_inputs["x"].cpu()
        edge_index = enc_inputs["edge_index"].cpu()
        edge_types = enc_inputs["edge_type_ids"].cpu()
    else:
        # Backward-compatible path for older tensor-ready artifacts.
        node_feats = torch.tensor(
            [n["features"] for n in nodes_data], dtype=torch.float32
        )

        src_list, dst_list, etype_list = [], [], []
        for e in edges_data:
            src_list.append(e["src"])
            dst_list.append(e["dst"])
            etype_list.append(e["edge_type_idx"])
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_types = torch.tensor(etype_list, dtype=torch.long)

    print(
        f"[Mode2] KG graph: {node_feats.shape[0]} nodes, "
        f"{edge_index.shape[1]} edges, {edge_types.unique().numel()} edge types"
    )
    return node_feats, edge_index, edge_types


# ── LGG checkpoint loader ──────────────────────────────────────────────────────


def load_lgg_from_pkl(
    pkl_path: Path,
    device: torch.device,
    require_state_dict: bool = True,
) -> LGG:
    """Load LGG MLP weights from Stage 3D lgg_models.pkl.

    The pickle contains {"B2_MLP": {"weights": ..., "config": ...}} or similar.
    Initialises a fresh LGG and loads the best MLP variant weights.

    When require_state_dict=True (default), this loader enforces a real PyTorch
    LGG warm-start and will fail if no state_dict-backed key is available.
    """
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"LGG models not found: {pkl_path}. Run STEP 12 (train_stage3d_lgg.py) first."
        )
    with open(pkl_path, "rb") as f:
        models_dict = pickle.load(f)

    # Prefer real PyTorch LGG entries with an actual state_dict.
    preferred_with_state = [
        "B3_LGG_TORCH",
        "B2_LGG_TORCH",
        "B1_LGG_TORCH",
        "B2_MLP",
        "B3_MLP",
        "B1_MLP",
    ]

    key = None
    for cand in preferred_with_state:
        if cand not in models_dict:
            continue
        cand_obj = models_dict.get(cand, {})
        if isinstance(cand_obj.get("state_dict"), dict):
            key = cand
            break

    # Fallback to config-only warm-start is allowed only when explicitly requested.
    if key is None and not require_state_dict:
        key = next(
            (k for k in preferred_with_state if k in models_dict),
            next(iter(models_dict), None),
        )

    if key is None and require_state_dict:
        raise ValueError(
            "No state_dict-backed LGG key found in lgg_models.pkl. "
            "Run tools/train_stage3d_lgg.py with default settings "
            "(PyTorch LGG enabled) to generate B3_LGG_TORCH."
        )

    if key is None:
        raise ValueError("lgg_models.pkl is empty or has unexpected structure.")

    model_data = models_dict[key]
    config = model_data.get("config", {})

    evidence_dim = config.get("evidence_dim", 4)
    embedding_dim = config.get("embedding_dim", 64)
    hidden_dim = config.get("hidden_dim", 64)

    gate = LGG(
        evidence_dim=evidence_dim,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    if "state_dict" in model_data:
        gate.load_state_dict(model_data["state_dict"], strict=False)
        print(f"[Mode2] LGG loaded from pkl key='{key}' (state_dict)")
    elif require_state_dict:
        raise ValueError(
            f"Selected LGG key '{key}' has no state_dict but strict Mode2 "
            "loading is active. Re-run Stage3D mainline first."
        )
    else:
        print(f"[Mode2] LGG initialised fresh (pkl key='{key}' has no state_dict)")

    return gate


# ── B2 provenance helpers (Mode 2 contract) ────────────────────────────────────


def _probe_lgg_state_dict(pkl_path: Optional[Path]) -> Dict[str, object]:
    """Probe an LGG pkl WITHOUT instantiating the model.

    Used by B2 provenance contract to record whether ``B3_LGG_TORCH`` carries a
    real ``state_dict``, even in the full-detector path where LGG is not loaded
    into the training loop.
    """
    info: Dict[str, object] = {
        "pkl_path": str(pkl_path) if pkl_path else None,
        "exists": False,
        "key_used": None,
        "has_state_dict": False,
    }
    if pkl_path is None or not pkl_path.exists():
        return info
    info["exists"] = True
    try:
        with open(pkl_path, "rb") as f:
            models = pickle.load(f)
    except Exception:
        return info
    preferred = ["B3_LGG_TORCH", "B2_LGG_TORCH", "B1_LGG_TORCH"]
    for k in preferred:
        if k in models:
            obj = models.get(k) or {}
            sd = obj.get("state_dict") if isinstance(obj, dict) else None
            if isinstance(sd, dict) and sd:
                info["key_used"] = k
                info["has_state_dict"] = True
                return info
    for k in preferred:
        if k in models:
            info["key_used"] = k
            return info
    return info


def _resolve_baseline_metrics(
    baseline_audit_path: Optional[str],
) -> Dict[str, Optional[float]]:
    """Best-effort read of baseline detector metrics from stage1a_raw.json."""
    out: Dict[str, Optional[float]] = {
        "mAP_50": None,
        "mAP_medium": None,
        "false_positive_pressure": None,
        "MA_AP50": None,
    }
    if not baseline_audit_path:
        return out
    p = Path(baseline_audit_path)
    if not p.exists():
        return out
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return out

    mm = data.get("map_metrics") or {}
    out["mAP_50"] = _to_float(mm.get("mAP_50"), default=float("nan"))
    out["mAP_medium"] = _to_float(mm.get("mAP_medium"), default=float("nan"))
    out["false_positive_pressure"] = _to_float(
        data.get("false_positive_pressure"),
        default=float("nan"),
    )

    ap_per_class = data.get("ap_per_class") or {}
    ma_id = CLASS_NAME_TO_ID.get("MA", 1)
    ma_raw = ap_per_class.get("MA")
    if ma_raw is None:
        ma_raw = ap_per_class.get(str(ma_id))
    out["MA_AP50"] = _to_float(ma_raw, default=float("nan"))

    for k, v in list(out.items()):
        if v is not None and (math.isnan(v) or math.isinf(v)):
            out[k] = None
    return out


def _resolve_baseline_map50(baseline_audit_path: Optional[str]) -> Optional[float]:
    return _resolve_baseline_metrics(baseline_audit_path).get("mAP_50")


@torch.no_grad()
def _eval_detector_metrics_full_mode(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    detections_per_img: int,
) -> Dict[str, float]:
    """Evaluate detector metrics required by v2 composite checkpoint scoring."""
    was_training = model.training
    model.eval()

    preds: List[Dict] = []
    targets: List[Dict] = []
    for images, batch_targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        for pred, tgt in zip(outputs, batch_targets):
            preds.append(
                {
                    "boxes": pred.get("boxes", torch.zeros(0, 4)).detach().cpu(),
                    "labels": pred.get("labels", torch.zeros(0, dtype=torch.long))
                    .detach()
                    .cpu(),
                    "scores": pred.get("scores", torch.zeros(0)).detach().cpu(),
                }
            )
            targets.append(
                {
                    "boxes": tgt["boxes"].detach().cpu(),
                    "labels": tgt["labels"].detach().cpu(),
                }
            )

    if was_training:
        model.train()

    map_50 = 0.0
    map_medium = 0.0
    map_75 = 0.0
    if _TM_OK and MeanAveragePrecision is not None:
        try:
            metric = MeanAveragePrecision(
                box_format="xyxy",
                iou_type="bbox",
                max_detection_thresholds=[1, 10, max(100, int(detections_per_img))],
            )
            if hasattr(metric, "warn_on_many_detections"):
                metric.warn_on_many_detections = False
            metric.update(preds, targets)
            raw = metric.compute()
            map_50 = _to_float(raw.get("map_50"), default=0.0)
            map_75 = _to_float(raw.get("map_75"), default=0.0)
            map_medium = _to_float(raw.get("map_medium"), default=0.0)
        except Exception as exc:
            global _TM_RUNTIME_WARNED
            if not _TM_RUNTIME_WARNED:
                warnings.warn(
                    f"torchmetrics runtime failure ({exc}) — using manual AP50 fallback."
                )
                _TM_RUNTIME_WARNED = True

    ap_per_class = _compute_ap50_manual(preds, targets)
    valid_ap = [v for v in ap_per_class.values() if not math.isnan(v)]
    manual_map50 = sum(valid_ap) / max(len(valid_ap), 1)
    if map_50 <= 0.0:
        map_50 = manual_map50

    ma_cls_id = CLASS_NAME_TO_ID.get("MA", 1)
    ma_ap = ap_per_class.get(ma_cls_id, float("nan"))
    if math.isnan(ma_ap):
        ma_ap = 0.0

    fp_pressure = _compute_false_positive_pressure(preds, targets)

    return {
        "mAP_50": float(map_50),
        "mAP_75": float(map_75),
        "mAP_medium": float(map_medium),
        "MA_AP50": float(ma_ap),
        "false_positive_pressure": float(fp_pressure),
    }


def _compute_v2_composite_score(
    metrics: Dict[str, float],
    baseline_metrics: Dict[str, Optional[float]],
) -> float:
    """Lower-is-better v2 checkpoint score balancing detection and FP behavior."""

    def _norm_ratio(value: float, baseline: Optional[float], fallback: float) -> float:
        if baseline is not None and baseline > 0:
            denom = baseline
        else:
            denom = fallback if fallback > 0 else 1e-6
        return value / max(denom, 1e-6)

    map50_n = _norm_ratio(
        float(metrics.get("mAP_50", 0.0)),
        baseline_metrics.get("mAP_50"),
        float(metrics.get("mAP_50", 1e-6)),
    )
    mapmed_n = _norm_ratio(
        float(metrics.get("mAP_medium", 0.0)),
        baseline_metrics.get("mAP_medium"),
        max(float(metrics.get("mAP_medium", 0.0)), 1e-6),
    )
    fp_n = _norm_ratio(
        float(metrics.get("false_positive_pressure", 0.0)),
        baseline_metrics.get("false_positive_pressure"),
        max(float(metrics.get("false_positive_pressure", 1.0)), 1e-6),
    )
    ma_n = _norm_ratio(
        float(metrics.get("MA_AP50", 0.0)),
        baseline_metrics.get("MA_AP50"),
        max(float(metrics.get("MA_AP50", 0.0)), 1e-6),
    )

    # Baseline-centered composite: lower is better.
    return (-1.0 * map50_n) + (0.50 * fp_n) + (-0.30 * mapmed_n) + (-0.20 * ma_n)


# Full-mode scope labels:
# - _FULL_MODE2_STATIC_SCOPE: detached diagnostic path (ROI-SDGL + KG only)
# - _FULL_MODE2_CLAIM_SCOPE: closed-loop path (online evidence + LGG + GateLoss)
_FULL_MODE2_STATIC_SCOPE = "roi_sdgl_kg_encoder_only"
_FULL_MODE2_CLAIM_SCOPE = "full_detector_lgg_evidence_closed_loop"


class SDGLReClassifier(nn.Module):
    """Trainable proxy head that restores a real gradient path for L_SDGL."""

    def __init__(self, embedding_dim: int, hidden_dim: int = 64):
        super().__init__()
        input_dim = len(LESION_CLASSES) + embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(LESION_CLASSES)),
        )

    def forward(self, p_ens: torch.Tensor, e_pred: torch.Tensor) -> torch.Tensor:
        log_p_ens = torch.log(p_ens.clamp(min=1e-9))
        fused = torch.cat([log_p_ens, e_pred], dim=0)
        logits = self.mlp(fused)
        return F.log_softmax(logits, dim=0)


def _load_fasterrcnn_for_full_mode(
    ckpt_path: Path,
    device: torch.device,
    score_thresh: float,
    detections_per_img: int,
) -> Tuple[torch.nn.Module, Dict]:
    try:
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.rpn import AnchorGenerator

        try:
            from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

            _weights_v2 = True
        except ImportError:
            _weights_v2 = False
    except ImportError as exc:
        raise ImportError(f"torchvision is required for --full_mode2_detector: {exc}")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Detector checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" not in ckpt:
        raise ValueError(
            "Detector checkpoint missing 'model_state_dict'. "
            "Use Stage1 Faster R-CNN checkpoint (fasterrcnn_best.pt)."
        )

    cfg = ckpt.get("config", {})
    num_classes = int(cfg.get("num_classes", len(LESION_CLASSES) + 1))
    anchor_sizes = cfg.get("anchor_sizes", ((8,), (16,), (32,), (64,), (128,)))
    anchor_ratios = cfg.get("anchor_ratios", ((0.5, 1.0, 2.0),) * 5)

    model_kwargs = {
        "rpn_anchor_generator": AnchorGenerator(
            sizes=tuple(tuple(s) for s in anchor_sizes),
            aspect_ratios=tuple(tuple(r) for r in anchor_ratios),
        ),
        "box_score_thresh": float(score_thresh),
        "box_detections_per_img": int(detections_per_img),
        "rpn_pre_nms_top_n_train": int(cfg.get("rpn_pre_nms_top_n_train", 4000)),
        "rpn_post_nms_top_n_train": int(cfg.get("rpn_post_nms_top_n_train", 2000)),
        "rpn_pre_nms_top_n_test": int(cfg.get("rpn_pre_nms_top_n_test", 3000)),
        "rpn_post_nms_top_n_test": int(cfg.get("rpn_post_nms_top_n_test", 1500)),
    }

    if _weights_v2:
        # Intentionally load architecture without pretrained weights; checkpoint
        # state_dict defines the exact detector state for full-mode continuation.
        _ = FasterRCNN_ResNet50_FPN_Weights  # keep import for older torchvision guards
        model = fasterrcnn_resnet50_fpn(weights=None, **model_kwargs)
    else:
        model = fasterrcnn_resnet50_fpn(pretrained=False, **model_kwargs)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    model.roi_heads.score_thresh = float(score_thresh)
    model.roi_heads.detections_per_img = int(detections_per_img)
    model = model.to(device)

    print(
        f"[Mode2][Full] Loaded detector checkpoint: {ckpt_path.name} "
        f"(num_classes={num_classes}, detections_per_img={detections_per_img})"
    )
    return model, ckpt


def _set_detector_backbone_frozen(model: torch.nn.Module, frozen: bool) -> None:
    requires_grad = not frozen
    for param in model.backbone.body.parameters():
        param.requires_grad = requires_grad
    print("[Mode2][Full] Detector backbone " + ("FROZEN" if frozen else "TRAINABLE"))


def _train_detector_epoch_full_mode(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler],
    kg_encoder: Optional[LGovNetEncoder],
    node_feats: Optional[torch.Tensor],
    edge_index: Optional[torch.Tensor],
    edge_types: Optional[torch.Tensor],
    lambda_kg: float,
    freeze_kg: bool,
    anchor_teacher_model: Optional[torch.nn.Module],
    lambda_anchor: float,
    anchor_alpha: float,
    anchor_beta: float,
) -> Dict[str, float]:
    model.train()
    if kg_encoder is not None:
        if node_feats is None or edge_index is None or edge_types is None:
            raise ValueError("KG tensors are required when kg_encoder is enabled.")
        if freeze_kg:
            kg_encoder.eval()
        else:
            kg_encoder.train()

        node_x = node_feats.to(device)
        ei = edge_index.to(device)
        et = edge_types.to(device)
    else:
        node_x = ei = et = None

    total_loss = 0.0
    total_detector_loss = 0.0
    total_l_kg = 0.0
    total_l_anchor = 0.0
    total_l_anchor_cls = 0.0
    total_l_anchor_box = 0.0
    total_anchor_n_aligned = 0.0
    total_cls = 0.0
    total_box = 0.0
    total_obj = 0.0
    total_rpn = 0.0
    n_batches = 0
    roi_stats_sum = defaultdict(float)

    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [
            {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()}
            for t in targets
        ]

        optimizer.zero_grad()
        if (
            kg_encoder is not None
            and node_x is not None
            and ei is not None
            and et is not None
        ):
            if freeze_kg:
                with torch.no_grad():
                    all_embeddings = kg_encoder(node_x, ei, et)
                lesion_embeddings = all_embeddings[: len(LESION_CLASSES)].detach()
            else:
                all_embeddings = kg_encoder(node_x, ei, et)
                lesion_embeddings = all_embeddings[: len(LESION_CLASSES)]
            set_roi_sdgl_dynamic_state(lesion_embeddings)
            l_kg = compute_l_kg(lesion_embeddings)
        else:
            set_roi_sdgl_dynamic_state(None)
            l_kg = torch.tensor(0.0, device=device)

        with capture_roi_head_outputs(model) as student_anchor:
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    loss_dict = model(images, targets)
                    detector_loss = sum(loss_dict.values())
            else:
                loss_dict = model(images, targets)
                detector_loss = sum(loss_dict.values())

        anchor_out = {
            "l_anchor": torch.zeros((), device=device),
            "l_anchor_cls": torch.zeros((), device=device),
            "l_anchor_box": torch.zeros((), device=device),
            "n_aligned": torch.zeros((), device=device),
        }
        if anchor_teacher_model is not None and float(lambda_anchor) > 0.0:
            teacher_was_training = anchor_teacher_model.training
            anchor_teacher_model.train()
            with torch.no_grad():
                with capture_roi_head_outputs(anchor_teacher_model) as teacher_anchor:
                    _ = anchor_teacher_model(images, targets)
            anchor_teacher_model.train(teacher_was_training)
            anchor_out = compute_anchor_loss(
                student_cls_logits=student_anchor.get("cls_logits"),
                student_box_regression=student_anchor.get("box_regression"),
                teacher_cls_logits=teacher_anchor.get("cls_logits"),
                teacher_box_regression=teacher_anchor.get("box_regression"),
                alpha=float(anchor_alpha),
                beta=float(anchor_beta),
            )

        l_anchor = anchor_out["l_anchor"]
        loss = detector_loss + float(lambda_kg) * l_kg + float(lambda_anchor) * l_anchor

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_detector_loss += float(detector_loss.item())
        total_l_kg += float(l_kg.item())
        total_l_anchor += float(anchor_out["l_anchor"].item())
        total_l_anchor_cls += float(anchor_out["l_anchor_cls"].item())
        total_l_anchor_box += float(anchor_out["l_anchor_box"].item())
        total_anchor_n_aligned += float(anchor_out["n_aligned"].item())
        total_cls += float(loss_dict.get("loss_classifier", torch.tensor(0.0)).item())
        total_box += float(loss_dict.get("loss_box_reg", torch.tensor(0.0)).item())
        total_obj += float(loss_dict.get("loss_objectness", torch.tensor(0.0)).item())
        total_rpn += float(loss_dict.get("loss_rpn_box_reg", torch.tensor(0.0)).item())
        n_batches += 1

        batch_roi_stats = get_last_sdgl_roi_stats()
        for k, v in batch_roi_stats.items():
            roi_stats_sum[k] += float(v)

    set_roi_sdgl_dynamic_state(None)

    if n_batches == 0:
        return {
            "loss": 0.0,
            "loss_detector": 0.0,
            "l_kg": 0.0,
            "l_anchor": 0.0,
            "l_anchor_cls": 0.0,
            "l_anchor_box": 0.0,
            "anchor_n_aligned": 0.0,
            "loss_classifier": 0.0,
            "loss_box_reg": 0.0,
            "loss_objectness": 0.0,
            "loss_rpn_box_reg": 0.0,
            "roi_n_r1": 0.0,
            "roi_n_r2": 0.0,
            "roi_n_r3": 0.0,
            "roi_used_dynamic_sdm": 0.0,
        }

    return {
        "loss": total_loss / n_batches,
        "loss_detector": total_detector_loss / n_batches,
        "l_kg": total_l_kg / n_batches,
        "l_anchor": total_l_anchor / n_batches,
        "l_anchor_cls": total_l_anchor_cls / n_batches,
        "l_anchor_box": total_l_anchor_box / n_batches,
        "anchor_n_aligned": total_anchor_n_aligned / n_batches,
        "loss_classifier": total_cls / n_batches,
        "loss_box_reg": total_box / n_batches,
        "loss_objectness": total_obj / n_batches,
        "loss_rpn_box_reg": total_rpn / n_batches,
        "roi_n_r1": roi_stats_sum.get("n_r1", 0.0) / n_batches,
        "roi_n_r2": roi_stats_sum.get("n_r2", 0.0) / n_batches,
        "roi_n_r3": roi_stats_sum.get("n_r3", 0.0) / n_batches,
        "roi_used_fallback": roi_stats_sum.get("used_fallback", 0.0) / n_batches,
        "roi_used_dynamic_sdm": roi_stats_sum.get("used_dynamic_sdm", 0.0) / n_batches,
    }


# ─── B5: LGG warm-start from Mode 1 B3_LGG_TORCH ──────────────────────────────


def _load_lgg_b3_warmstart(
    pkl_path: Path,
    device: torch.device,
    embedding_dim: int = 64,
    hidden_dim: int = 64,
) -> Tuple[LGG, bool, Optional[str]]:
    """Build an LGG instance, warm-started from Mode 1 B3_LGG_TORCH where shape-compatible.

    Returns:
        (lgg, warmstart_loaded, reason)
        - warmstart_loaded=True iff state_dict actually loaded from B3_LGG_TORCH.
        - reason: 'mode1_b3_lgg_torch' on success, else a short failure reason
          ('missing_pkl' | 'missing_key' | 'no_state_dict' | 'shape_mismatch:...').
    """
    lgg = LGG(
        evidence_dim=4,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    if pkl_path is None or not pkl_path.exists():
        return lgg, False, "missing_pkl"
    try:
        with open(pkl_path, "rb") as f:
            models = pickle.load(f)
    except Exception as exc:
        return lgg, False, f"unreadable:{type(exc).__name__}"
    entry = models.get("B3_LGG_TORCH")
    if not isinstance(entry, dict):
        return lgg, False, "missing_key"
    sd = entry.get("state_dict")
    if not isinstance(sd, dict) or not sd:
        return lgg, False, "no_state_dict"
    try:
        missing, unexpected = lgg.load_state_dict(sd, strict=False)
    except Exception as exc:
        return lgg, False, f"shape_mismatch:{type(exc).__name__}"
    if missing:
        # Partial load; record as warning, still treat as warm-started if
        # at least one matched key was actually loaded.
        loaded = len(sd) - len(unexpected)
        if loaded <= 0:
            return lgg, False, f"shape_mismatch:loaded={loaded}"
    return lgg, True, "mode1_b3_lgg_torch"


# ─── B4/B5: closed-loop epoch trainer ─────────────────────────────────────────


def _train_detector_epoch_closed_loop(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler],
    kg_encoder: Optional[LGovNetEncoder],
    node_feats: Optional[torch.Tensor],
    edge_index: Optional[torch.Tensor],
    edge_types: Optional[torch.Tensor],
    lambda_kg: float,
    freeze_kg: bool,
    lgg: LGG,
    gate_loss_fn: GateLoss,
    sdm_norm: Dict,
    compat_scores: Dict,
    lambda_gate: float,
    gate_class_weights: Optional[torch.Tensor],
    anchor_teacher_model: Optional[torch.nn.Module],
    lambda_anchor: float,
    anchor_alpha: float,
    anchor_beta: float,
    *,
    use_real_class_probs_bridge: bool = True,
    secondary_cache: Optional[Dict[str, List[Dict]]] = None,
) -> Dict[str, float]:
    """Mode 2 closed-loop training step (B4 + B5 wired; REV-7 Sprint 1+2 patch).

    Per batch:
      1. KG encoder forward → lesion_embeddings (grad through KG if not frozen).
      2. Detector forward (train, grad) → loss_dict (incl. patched ROI SDGL).
      3. Detector forward (eval, no_grad) → final detections per image.
         REV-7: when ``use_real_class_probs_bridge`` is True, the eval pass uses
         ``predict_with_class_probs`` so per-box real ROI softmax flows into the
         online evidence builder; SDCA + soft e_pred become claim-valid
         (architecture §EVIDENCE-FIDELITY CONTRACT).
      4. Build per-detection online evidence + IoU-tier supervision target.
         REV-7: when ``secondary_cache`` is provided, per-image SAS uses real
         cross-detector agreement (Stage 5B prerequisite).
      5. LGG.forward(gate_input) → tier_logits; GateLoss vs tier_targets.
      6. L_total = sum(loss_dict) + λ_kg L_KG + λ_gate L_gate; backprop.

    Gradient pathways exercised:
      - Detector params (from loss_dict)
      - KG encoder (from L_KG anchor + via e_pred → gate_input → LGG forward)
      - LGG params (from L_gate)
    """
    model.train()
    lgg.train()
    if kg_encoder is not None:
        if node_feats is None or edge_index is None or edge_types is None:
            raise ValueError("KG tensors are required when kg_encoder is enabled.")
        if freeze_kg:
            kg_encoder.eval()
        else:
            kg_encoder.train()
        node_x = node_feats.to(device)
        ei = edge_index.to(device)
        et = edge_types.to(device)
    else:
        node_x = ei = et = None

    total_loss = 0.0
    total_detector_loss = 0.0
    total_l_kg = 0.0
    total_l_gate = 0.0
    total_l_anchor = 0.0
    total_l_anchor_cls = 0.0
    total_l_anchor_box = 0.0
    total_anchor_n_aligned = 0.0
    total_cls = 0.0
    total_box = 0.0
    total_obj = 0.0
    total_rpn = 0.0
    n_batches = 0
    lgg_forward_calls = 0
    gate_loss_max_seen = 0.0
    total_records = 0
    total_n_r1 = total_n_r2 = total_n_r3 = total_n_r4 = 0
    total_gate_tier_acc_r1 = 0.0
    total_gate_tier_acc_r2 = 0.0
    total_gate_tier_acc_r3 = 0.0
    total_gate_tier_acc_r4 = 0.0
    gate_batches = 0
    roi_stats_sum: Dict[str, float] = defaultdict(float)

    # REV-7 (Sprint 1+2) — evidence-fidelity telemetry accumulators (gate (0)).
    ev_fidelity_sum: Dict[str, float] = defaultdict(float)
    ev_fidelity_batches = 0
    ev_runnerup_pairs_seen: set = set()

    for images, targets in loader:
        images = [img.to(device) for img in images]
        # REV-7 (Sprint 2): extract image_ids BEFORE device move (image_id is a
        # str field added by DRDetectionDataset; keep it out of the device move).
        image_ids: List[Optional[str]] = [
            (t.get("image_id") if isinstance(t.get("image_id"), str) else None)
            for t in targets
        ]
        targets = [
            {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()}
            for t in targets
        ]

        optimizer.zero_grad()

        # ── 1. KG forward ────────────────────────────────────────────────────
        if (
            kg_encoder is not None
            and node_x is not None
            and ei is not None
            and et is not None
        ):
            if freeze_kg:
                with torch.no_grad():
                    all_embeddings = kg_encoder(node_x, ei, et)
                lesion_embeddings = all_embeddings[: len(LESION_CLASSES)].detach()
            else:
                all_embeddings = kg_encoder(node_x, ei, et)
                lesion_embeddings = all_embeddings[: len(LESION_CLASSES)]
            set_roi_sdgl_dynamic_state(lesion_embeddings)
            l_kg = compute_l_kg(lesion_embeddings)
        else:
            set_roi_sdgl_dynamic_state(None)
            lesion_embeddings = None
            l_kg = torch.tensor(0.0, device=device)

        # ── 2. Detector train-mode forward (grad): detector loss ─────────────
        with capture_roi_head_outputs(model) as student_anchor:
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    loss_dict = model(images, targets)
                    detector_loss = sum(loss_dict.values())
            else:
                loss_dict = model(images, targets)
                detector_loss = sum(loss_dict.values())

        anchor_out = {
            "l_anchor": torch.zeros((), device=device),
            "l_anchor_cls": torch.zeros((), device=device),
            "l_anchor_box": torch.zeros((), device=device),
            "n_aligned": torch.zeros((), device=device),
        }
        if anchor_teacher_model is not None and float(lambda_anchor) > 0.0:
            teacher_was_training = anchor_teacher_model.training
            anchor_teacher_model.train()
            with torch.no_grad():
                with capture_roi_head_outputs(anchor_teacher_model) as teacher_anchor:
                    _ = anchor_teacher_model(images, targets)
            anchor_teacher_model.train(teacher_was_training)
            anchor_out = compute_anchor_loss(
                student_cls_logits=student_anchor.get("cls_logits"),
                student_box_regression=student_anchor.get("box_regression"),
                teacher_cls_logits=teacher_anchor.get("cls_logits"),
                teacher_box_regression=teacher_anchor.get("box_regression"),
                alpha=float(anchor_alpha),
                beta=float(anchor_beta),
            )

        # ── 3. Detector eval-mode forward (no_grad): get detections ──────────
        # REV-7 (Sprint 1): use predict_with_class_probs() to retain real ROI
        # softmax per box; falls back to plain model(images) only when the
        # bridge is explicitly disabled for negative-ablation studies.
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                if use_real_class_probs_bridge:
                    detections = predict_with_class_probs(
                        model, images, device=device
                    )
                else:
                    detections = model(images)
        finally:
            model.train(was_training)

        # ── 4. Build per-detection online evidence + tier targets ────────────
        image_sizes = [(int(img.shape[-2]), int(img.shape[-1])) for img in images]
        records = build_proposal_records(
            detections=detections,
            targets=targets,
            image_sizes=image_sizes,
            sdm_norm=sdm_norm,
            compat_scores=compat_scores,
            label_to_class=FRCNN_LABEL_TO_CLASS,
            image_ids=image_ids,
            secondary_cache=secondary_cache,
        )
        rec_summary = summarise_records(records)

        # REV-7 (gate (0)): evidence-fidelity telemetry per batch.
        if records:
            batch_fidelity = evidence_fidelity_stats(records)
            for k, v in batch_fidelity.items():
                if k == "ev_n_records":
                    ev_fidelity_sum[k] += float(v)
                elif k in (
                    "ev_n_multi_source_agreed",
                    "ev_n_multi_source_conflict",
                ):
                    ev_fidelity_sum[k] += float(v)
                elif k.startswith("ev_") and not k.endswith("_distinct_values") \
                        and not k.endswith("_unique_pairs"):
                    # Average rate / std over batches.
                    ev_fidelity_sum[k] += float(v)
            ev_fidelity_batches += 1
            # Aggregate distinct-pair set across batches.
            for r in records:
                cv = r.get("class_probs_vec")
                if isinstance(cv, (list, tuple)) and len(cv) >= 2:
                    ranked = sorted(
                        range(len(cv)), key=lambda j: -float(cv[j])
                    )
                    if len(ranked) >= 2:
                        ev_runnerup_pairs_seen.add(
                            (LESION_CLASSES[ranked[0]], LESION_CLASSES[ranked[1]])
                        )

        # ── 5. LGG forward + GateLoss (closed-loop through KG via e_pred) ────
        l_gate = torch.tensor(0.0, device=device)
        if records and lesion_embeddings is not None:
            tensors = records_to_tensors(records, lesion_embeddings, device)
            if tensors is not None:
                sample_weights = None
                if gate_class_weights is not None:
                    class_w = gate_class_weights.to(
                        device=tensors["tier_targets"].device
                    )
                    sample_weights = class_w.index_select(0, tensors["majority_idx"])
                gate_out = lgg(tensors["gate_input"])
                tier_logits = gate_out["tier_logits"]
                gate_res = gate_loss_fn(
                    tier_logits,
                    tensors["tier_targets"],
                    sample_weights=sample_weights,
                )
                l_gate = gate_res.get("l_gate", l_gate)
                lgg_forward_calls += 1
                gate_loss_max_seen = max(gate_loss_max_seen, float(l_gate.item()))
                total_gate_tier_acc_r1 += _to_float(gate_res.get("tier_acc_r1"), 0.0)
                total_gate_tier_acc_r2 += _to_float(gate_res.get("tier_acc_r2"), 0.0)
                total_gate_tier_acc_r3 += _to_float(gate_res.get("tier_acc_r3"), 0.0)
                total_gate_tier_acc_r4 += _to_float(gate_res.get("tier_acc_r4"), 0.0)
                gate_batches += 1

        # ── 6. Total loss + backward ─────────────────────────────────────────
        loss = (
            detector_loss
            + float(lambda_kg) * l_kg
            + float(lambda_gate) * l_gate
            + float(lambda_anchor) * anchor_out["l_anchor"]
        )
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_detector_loss += float(detector_loss.item())
        total_l_kg += float(l_kg.item())
        total_l_gate += float(l_gate.item())
        total_l_anchor += float(anchor_out["l_anchor"].item())
        total_l_anchor_cls += float(anchor_out["l_anchor_cls"].item())
        total_l_anchor_box += float(anchor_out["l_anchor_box"].item())
        total_anchor_n_aligned += float(anchor_out["n_aligned"].item())
        total_cls += float(loss_dict.get("loss_classifier", torch.tensor(0.0)).item())
        total_box += float(loss_dict.get("loss_box_reg", torch.tensor(0.0)).item())
        total_obj += float(loss_dict.get("loss_objectness", torch.tensor(0.0)).item())
        total_rpn += float(loss_dict.get("loss_rpn_box_reg", torch.tensor(0.0)).item())
        total_records += rec_summary["n_total"]
        total_n_r1 += rec_summary["n_r1"]
        total_n_r2 += rec_summary["n_r2"]
        total_n_r3 += rec_summary["n_r3"]
        total_n_r4 += rec_summary["n_r4"]
        n_batches += 1

        batch_roi_stats = get_last_sdgl_roi_stats()
        for k, v in batch_roi_stats.items():
            roi_stats_sum[k] += float(v)

    set_roi_sdgl_dynamic_state(None)

    if n_batches == 0:
        return {
            "loss": 0.0,
            "loss_detector": 0.0,
            "l_kg": 0.0,
            "l_gate": 0.0,
            "l_anchor": 0.0,
            "l_anchor_cls": 0.0,
            "l_anchor_box": 0.0,
            "anchor_n_aligned": 0.0,
            "loss_classifier": 0.0,
            "loss_box_reg": 0.0,
            "loss_objectness": 0.0,
            "loss_rpn_box_reg": 0.0,
            "roi_n_r1": 0.0,
            "roi_n_r2": 0.0,
            "roi_n_r3": 0.0,
            "roi_used_fallback": 0.0,
            "roi_used_dynamic_sdm": 0.0,
            "lgg_forward_calls": 0,
            "gate_loss_max_seen": 0.0,
            "ev_n_records": 0,
            "ev_n_r1": 0,
            "ev_n_r2": 0,
            "ev_n_r3": 0,
            "ev_n_r4": 0,
            "gate_tier_acc_r1": 0.0,
            "gate_tier_acc_r2": 0.0,
            "gate_tier_acc_r3": 0.0,
            "gate_tier_acc_r4": 0.0,
        }

    # REV-7 (gate (0)): compute epoch-averaged evidence-fidelity stats.
    fid_batches = max(ev_fidelity_batches, 1)
    epoch_fidelity = {
        "ev_real_prob_coverage_full_mode": ev_fidelity_sum.get(
            "ev_real_prob_coverage", 0.0
        ) / fid_batches,
        "ev_soft_epred_coverage_full_mode": ev_fidelity_sum.get(
            "ev_soft_epred_coverage", 0.0
        ) / fid_batches,
        "ev_multi_source_sas_coverage_full_mode": ev_fidelity_sum.get(
            "ev_multi_source_sas_coverage", 0.0
        ) / fid_batches,
        "ev_n_multi_source_agreed_total": int(
            ev_fidelity_sum.get("ev_n_multi_source_agreed", 0.0)
        ),
        "ev_n_multi_source_conflict_total": int(
            ev_fidelity_sum.get("ev_n_multi_source_conflict", 0.0)
        ),
        "ev_sdca_std_online": ev_fidelity_sum.get("ev_sdca_std", 0.0) / fid_batches,
        "ev_sas_std_online": ev_fidelity_sum.get("ev_sas_std", 0.0) / fid_batches,
        "ev_aps_std_online": ev_fidelity_sum.get("ev_aps_std", 0.0) / fid_batches,
        "ev_pss_std_online": ev_fidelity_sum.get("ev_pss_std", 0.0) / fid_batches,
        # Epoch-wide unique pair count (set union across batches).
        "ev_runnerup_unique_pairs": int(len(ev_runnerup_pairs_seen)),
        # Used-real-bridge flag for downstream selection/gate audit.
        "ev_use_real_class_probs_bridge": bool(use_real_class_probs_bridge),
        "ev_secondary_cache_active": bool(
            secondary_cache is not None and len(secondary_cache) > 0
        ),
    }

    return {
        "loss": total_loss / n_batches,
        "loss_detector": total_detector_loss / n_batches,
        "l_kg": total_l_kg / n_batches,
        "l_gate": total_l_gate / n_batches,
        "l_anchor": total_l_anchor / n_batches,
        "l_anchor_cls": total_l_anchor_cls / n_batches,
        "l_anchor_box": total_l_anchor_box / n_batches,
        "anchor_n_aligned": total_anchor_n_aligned / n_batches,
        "loss_classifier": total_cls / n_batches,
        "loss_box_reg": total_box / n_batches,
        "loss_objectness": total_obj / n_batches,
        "loss_rpn_box_reg": total_rpn / n_batches,
        "roi_n_r1": roi_stats_sum.get("n_r1", 0.0) / n_batches,
        "roi_n_r2": roi_stats_sum.get("n_r2", 0.0) / n_batches,
        "roi_n_r3": roi_stats_sum.get("n_r3", 0.0) / n_batches,
        "roi_used_fallback": roi_stats_sum.get("used_fallback", 0.0) / n_batches,
        "roi_used_dynamic_sdm": roi_stats_sum.get("used_dynamic_sdm", 0.0) / n_batches,
        "lgg_forward_calls": lgg_forward_calls,
        "gate_loss_max_seen": gate_loss_max_seen,
        "ev_n_records": total_records,
        "ev_n_r1": total_n_r1,
        "ev_n_r2": total_n_r2,
        "ev_n_r3": total_n_r3,
        "ev_n_r4": total_n_r4,
        "gate_tier_acc_r1": total_gate_tier_acc_r1 / max(gate_batches, 1),
        "gate_tier_acc_r2": total_gate_tier_acc_r2 / max(gate_batches, 1),
        "gate_tier_acc_r3": total_gate_tier_acc_r3 / max(gate_batches, 1),
        "gate_tier_acc_r4": total_gate_tier_acc_r4 / max(gate_batches, 1),
        **epoch_fidelity,
    }


@torch.no_grad()
def _eval_detector_loss_full_mode(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler],
    kg_encoder: Optional[LGovNetEncoder],
    node_feats: Optional[torch.Tensor],
    edge_index: Optional[torch.Tensor],
    edge_types: Optional[torch.Tensor],
    lambda_kg: float,
    freeze_kg: bool,
) -> float:
    # torchvision detection losses are exposed in train() mode only.
    model.train()
    if kg_encoder is not None:
        if node_feats is None or edge_index is None or edge_types is None:
            raise ValueError("KG tensors are required when kg_encoder is enabled.")
        kg_encoder.eval()
        node_x = node_feats.to(device)
        ei = edge_index.to(device)
        et = edge_types.to(device)
    else:
        node_x = ei = et = None

    total = 0.0
    n_batches = 0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [
            {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()}
            for t in targets
        ]

        if (
            kg_encoder is not None
            and node_x is not None
            and ei is not None
            and et is not None
        ):
            all_embeddings = kg_encoder(node_x, ei, et)
            lesion_embeddings = all_embeddings[: len(LESION_CLASSES)]
            set_roi_sdgl_dynamic_state(lesion_embeddings.detach())
            l_kg = compute_l_kg(lesion_embeddings.detach())
        else:
            set_roi_sdgl_dynamic_state(None)
            l_kg = torch.tensor(0.0, device=device)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                loss_dict = model(images, targets)
                detector_loss = sum(loss_dict.values())
                loss = detector_loss + float(lambda_kg) * l_kg
        else:
            loss_dict = model(images, targets)
            detector_loss = sum(loss_dict.values())
            loss = detector_loss + float(lambda_kg) * l_kg
        total += float(loss.item())
        n_batches += 1
    set_roi_sdgl_dynamic_state(None)
    model.eval()
    return total / max(n_batches, 1)


def run_full_mode2_detector(
    args: argparse.Namespace,
    sdm_norm: Dict,
    out_dir: Path,
    device: torch.device,
) -> None:
    print("[Mode2][Full] Running full detector path (--full_mode2_detector)")
    print(
        "[Mode2][Full] KG mode="
        + ("frozen" if args.freeze_kg else "closed-loop")
        + f"  lambda_kg={float(args.lambda_kg):.4f}"
    )

    if args.smoke_test and args.max_images <= 0:
        args.max_images = 16
    if args.smoke_test:
        args.epochs = 1

    max_images = args.max_images if args.max_images > 0 else None
    train_ds = DRDetectionDataset(
        split="train",
        data_root=args.data_root,
        max_images=max_images,
    )
    val_ds = DRDetectionDataset(
        split="valid",
        data_root=args.data_root,
        max_images=max_images,
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(
            "Full mode requires non-empty train/valid detection datasets. "
            "Check --data_root and XML/image pairing."
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=max(1, int(args.batch_size)),
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=(device.type == "cuda"),
    )
    print(
        f"[Mode2][Full] Dataset sizes: train={len(train_ds)} valid={len(val_ds)} "
        f"batch_size={max(1, int(args.batch_size))}"
    )

    node_feats, edge_index, edge_types = load_kg_graph(Path(args.kg_dir))
    kg_encoder = LGovNetEncoder(
        node_feat_dim=int(node_feats.shape[1]),
        hidden_dim=64,
        embedding_dim=64,
        num_edge_types=5,
    ).to(device)
    if args.freeze_kg:
        kg_encoder.eval()
        for p in kg_encoder.parameters():
            p.requires_grad_(False)
        print("[Mode2][Full] KG encoder FROZEN (condition iii ablation).")
    else:
        kg_encoder.train()
        print("[Mode2][Full] KG encoder TRAINABLE (closed-loop Mode2).")

    model, detector_ckpt = _load_fasterrcnn_for_full_mode(
        ckpt_path=Path(args.detector_checkpoint),
        device=device,
        score_thresh=args.det_score_thresh,
        detections_per_img=args.det_detections_per_img,
    )

    # ── REV-7B (2026-06-02): Mode 2A REAL preserve-standard-CE branch ────────
    # Empirical evidence (smoke/1-06-log-sh[X].txt R2 winner): SDGL replacement
    # of ROI CE caused +0.44 ECE and +15% FPpr vs frozen baseline, violating
    # gate (v) non-regression. Architecture Mode 2A explicitly requires the
    # standard ROI CE to be PRESERVED. The flag below honors that requirement.
    preserve_roi_loss: bool = bool(
        getattr(args, "mode2a_preserve_standard_roi_loss", False)
    )
    if preserve_roi_loss:
        print(
            "[Mode2][Full] REV-7B Mode 2A REAL: --mode2a_preserve_standard_roi_loss "
            "set → SKIPPING enable_roi_sdgl_loss(). Detector trains with "
            "vanilla Faster R-CNN ROI CE; L_KG, L_gate, L_anchor remain as "
            "auxiliary losses only. This is the LITERAL Mode 2A path."
        )
    else:
        enable_roi_sdgl_loss(
            sdm_norm=sdm_norm,
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            lambda3=args.lambda3,
            eta=1.0,
            close_threshold=args.mode2_close_threshold,
            r1_conf=args.mode2_r1_conf,
            r2_conf=args.mode2_r2_conf,
            r3_min_weight=args.mode2_r3_min_weight,
            bg_weight=args.mode2_bg_weight,
            uniform_temp=args.uniform_temp,
        )
        print("[Mode2][Full] ROI classification loss patched to SDGL")

    # ── REV-7B (2026-06-02): freeze entire detector when requested ───────────
    # Jalur B (no online detector): backbone + RPN + ROI head fully frozen.
    # Only KG-GAT + LGG receive gradient updates. Useful for clean KG-
    # attribution and as a fallback if Mode 2A REAL still under-performs.
    freeze_full_detector: bool = bool(getattr(args, "freeze_detector", False))
    if freeze_full_detector:
        for _name, _p in model.named_parameters():
            _p.requires_grad = False
        # Sanity: keep BN/Dropout consistent during eval pass that builds
        # detections; model.train()/eval() toggles handle that elsewhere.
        n_frozen_params = sum(1 for _p in model.parameters() if not _p.requires_grad)
        print(
            "[Mode2][Full] REV-7B Jalur B: --freeze_detector set → ALL "
            f"detector params frozen ({n_frozen_params} tensors). Only "
            "KG-GAT + LGG will receive gradient updates. Expect detector "
            "mAP_50 to match baseline exactly."
        )

    # ── B4/B5: closed-loop wiring (LGG + GateLoss + online evidence) ──────
    use_closed_loop = (
        getattr(args, "_mode2_scope", _FULL_MODE2_STATIC_SCOPE)
        == _FULL_MODE2_CLAIM_SCOPE
    )
    gate_tier_weights = _parse_float_list(
        args.mode2_gate_tier_weights,
        expected_len=4,
        arg_name="--mode2_gate_tier_weights",
    )
    gate_class_weights = _parse_float_list(
        args.mode2_gate_class_weights,
        expected_len=4,
        arg_name="--mode2_gate_class_weights",
    )
    gate_class_weights_t = (
        torch.tensor(gate_class_weights, dtype=torch.float32, device=device)
        if gate_class_weights is not None
        else None
    )

    lgg_mod: Optional[LGG] = None
    gate_loss_fn: Optional[GateLoss] = None
    compat_scores: Dict = {}
    lgg_warmstart_loaded = False
    lgg_warmstart_reason: Optional[str] = None
    if use_closed_loop:
        compat_path = Path(args.kg_dir) / "compat_scores.json"
        if compat_path.exists():
            try:
                compat_scores = load_compat_scores(compat_path)
                print(
                    f"[Mode2][Full] Loaded compat_scores: {len(compat_scores)} pairs "
                    f"(for online APS)"
                )
            except Exception as exc:
                print(
                    f"[Mode2][Full] WARN: compat_scores load failed ({exc}); "
                    "APS will fall back to 0.5 per box."
                )
        else:
            print(
                f"[Mode2][Full] WARN: {compat_path} missing; "
                "online APS will fall back to 0.5 per box."
            )
        lgg_mod, lgg_warmstart_loaded, lgg_warmstart_reason = _load_lgg_b3_warmstart(
            pkl_path=Path(args.lgg_checkpoint),
            device=device,
            embedding_dim=64,
            hidden_dim=64,
        )
        gate_loss_fn = GateLoss(
            tier_weights=gate_tier_weights,
            focal_gamma=float(args.mode2_gate_focal_gamma),
        )
        print(
            "[Mode2][Full] Closed-loop wiring ACTIVE: scope="
            f"{_FULL_MODE2_CLAIM_SCOPE}; LGG warm-start "
            f"loaded={lgg_warmstart_loaded} reason={lgg_warmstart_reason!r}"
        )
        if gate_tier_weights is not None:
            print(f"[Mode2][Full] Gate tier weights: {gate_tier_weights}")
        if gate_class_weights is not None:
            print(
                f"[Mode2][Full] Gate class weights (MA,HE,Hard_EX,Soft_EX): {gate_class_weights}"
            )
        print(
            f"[Mode2][Full] Gate focal gamma: {float(args.mode2_gate_focal_gamma):.3f}"
        )
    else:
        print(
            "[Mode2][Full] Closed-loop wiring SKIPPED: scope="
            f"{_FULL_MODE2_STATIC_SCOPE} (--allow_detached_lgg)."
        )

    anchor_teacher_model: Optional[torch.nn.Module] = None
    anchor_teacher_ckpt = Path(
        args.anchor_teacher_checkpoint or args.detector_checkpoint
    )
    if float(args.lambda_anchor) > 0.0:
        anchor_teacher_model, _ = _load_fasterrcnn_for_full_mode(
            ckpt_path=anchor_teacher_ckpt,
            device=device,
            score_thresh=args.det_score_thresh,
            detections_per_img=args.det_detections_per_img,
        )
        anchor_teacher_model.eval()
        for p in anchor_teacher_model.parameters():
            p.requires_grad_(False)
        print(
            "[Mode2][Full] Anchor teacher loaded: "
            f"{anchor_teacher_ckpt} (lambda_anchor={float(args.lambda_anchor):.4f})"
        )

    if args.det_freeze_backbone_epochs > 0:
        _set_detector_backbone_frozen(model, frozen=True)
    else:
        _set_detector_backbone_frozen(model, frozen=False)

    def _build_full_mode_optimizer(cur_lr: float) -> torch.optim.Optimizer:
        trainable_params = list(model.parameters())
        if not args.freeze_kg:
            trainable_params += list(kg_encoder.parameters())
        if use_closed_loop and lgg_mod is not None:
            trainable_params += list(lgg_mod.parameters())
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, trainable_params),
            lr=cur_lr,
            weight_decay=1e-4,
        )

    optimizer = _build_full_mode_optimizer(args.lr)

    use_amp = device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda") if use_amp else None
    except TypeError:
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

    baseline_metrics = dict(getattr(args, "_baseline_metrics", {}) or {})
    selection_metric = str(getattr(args, "checkpoint_selection_metric", "val_loss"))

    log_rows: List[Dict] = []
    best_epoch = 0
    best_val_loss = float("inf")
    best_val_loss_observed = float("inf")
    best_checkpoint_score = float("inf")
    best_checkpoint_metric_value = float("inf")
    final_val_loss = 0.0
    # v2.1: top-k checkpoint manager (lazy-init inside loop on first save).
    _topk_manager: Optional[TopKCheckpointManager] = None
    lgg_forward_calls_total = 0
    gate_loss_max_seen_overall = 0.0
    ev_n_records_total = 0
    ev_n_r1_total = ev_n_r2_total = ev_n_r3_total = ev_n_r4_total = 0

    # REV-7 (Sprint 1+2): resolve bridge flag + load secondary detector cache
    # once before the epoch loop. The cache map is reused per-image inside the
    # closed-loop epoch function.
    use_real_class_probs_bridge_flag: bool = not bool(
        getattr(args, "disable_real_class_probs_bridge", False)
    )
    secondary_cache_train: Dict[str, List[Dict]] = {}
    _secondary_cache_path = getattr(args, "secondary_detector_cache_dir", None)
    if _secondary_cache_path is not None:
        secondary_cache_train = load_secondary_detector_cache(
            Path(_secondary_cache_path)
        )
        print(
            f"[Mode2][Full] REV-7 Sprint 2: loaded secondary detector cache "
            f"from {_secondary_cache_path}: {len(secondary_cache_train)} images"
        )
    print(
        f"[Mode2][Full] REV-7 Sprint 1: real_class_probs_bridge="
        f"{use_real_class_probs_bridge_flag}; secondary_cache_active="
        f"{bool(secondary_cache_train)}"
    )
    # Telemetry slot — populated from the last closed-loop epoch's epoch-averaged
    # fidelity numbers; written into mode2_summary.json by the outer writer.
    ev_fidelity_last_epoch: Dict[str, object] = {}

    # REV-7C: lambda warmup factor helper. epoch is 1-indexed in the loop;
    # at epoch 1 with warmup=2 → factor 0.5, epoch 2 → 1.0, epoch 3+ → 1.0.
    # warmup_epochs=0 (default) → always 1.0 (no warmup, REV-7B behavior).
    def _lambda_warmup_factor(epoch_1idx: int, warmup_epochs: int) -> float:
        if warmup_epochs <= 0:
            return 1.0
        return float(min(1.0, max(0.0, epoch_1idx / float(warmup_epochs))))

    _kg_warmup_eps = int(getattr(args, "lambda_kg_warmup_epochs", 0))
    _gate_warmup_eps = int(getattr(args, "lambda_gate_warmup_epochs", 0))
    if _kg_warmup_eps > 0 or _gate_warmup_eps > 0:
        print(
            f"[Mode2][Full] REV-7C lambda warmup active: "
            f"kg={_kg_warmup_eps} epochs (target={float(args.lambda_kg):.4f}), "
            f"gate={_gate_warmup_eps} epochs (target={float(args.lambda_gate):.4f})"
        )

    try:
        for epoch in range(1, args.epochs + 1):
            if (
                epoch == args.det_freeze_backbone_epochs + 1
                and args.det_freeze_backbone_epochs > 0
            ):
                # REV-7B: when --freeze_detector is set, the whole detector
                # stays frozen forever; do NOT unfreeze the backbone at warmup
                # boundary. The original line below only flips the backbone
                # subset; with freeze_full_detector all params (including
                # backbone) already have requires_grad=False, so we just keep
                # them that way.
                if freeze_full_detector:
                    print(
                        f"[Mode2][Full] REV-7B Jalur B: Epoch {epoch}: backbone "
                        "stays FROZEN (--freeze_detector active); no unfreeze."
                    )
                else:
                    _set_detector_backbone_frozen(model, frozen=False)
                    optimizer = _build_full_mode_optimizer(args.lr * 0.1)
                    print(
                        f"[Mode2][Full] Epoch {epoch}: backbone unfrozen, lr={args.lr * 0.1:.2e}"
                    )

            t0 = time.time()
            if use_closed_loop and lgg_mod is not None and gate_loss_fn is not None:
                # REV-7 (Sprint 1+2): pass through the bridge + secondary-cache
                # toggles. Defaults are set up earlier in run_full_mode2_detector
                # from the CLI flags.
                # REV-7C (post-RCA): compute effective lambdas with warmup ramp.
                _kg_factor = _lambda_warmup_factor(epoch, _kg_warmup_eps)
                _gate_factor = _lambda_warmup_factor(epoch, _gate_warmup_eps)
                _lambda_kg_eff = float(args.lambda_kg) * _kg_factor
                _lambda_gate_eff = float(args.lambda_gate) * _gate_factor
                if _kg_warmup_eps > 0 or _gate_warmup_eps > 0:
                    print(
                        f"[Mode2][Full] REV-7C epoch {epoch}: lambda_kg_eff="
                        f"{_lambda_kg_eff:.4f} (factor={_kg_factor:.3f}), "
                        f"lambda_gate_eff={_lambda_gate_eff:.4f} "
                        f"(factor={_gate_factor:.3f})"
                    )
                train_metrics = _train_detector_epoch_closed_loop(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    device=device,
                    scaler=scaler,
                    kg_encoder=kg_encoder,
                    node_feats=node_feats,
                    edge_index=edge_index,
                    edge_types=edge_types,
                    lambda_kg=_lambda_kg_eff,
                    freeze_kg=bool(args.freeze_kg),
                    lgg=lgg_mod,
                    gate_loss_fn=gate_loss_fn,
                    sdm_norm=sdm_norm,
                    compat_scores=compat_scores,
                    lambda_gate=_lambda_gate_eff,
                    gate_class_weights=gate_class_weights_t,
                    anchor_teacher_model=anchor_teacher_model,
                    lambda_anchor=float(args.lambda_anchor),
                    anchor_alpha=float(args.anchor_alpha),
                    anchor_beta=float(args.anchor_beta),
                    use_real_class_probs_bridge=use_real_class_probs_bridge_flag,
                    secondary_cache=secondary_cache_train,
                )
                lgg_forward_calls_total += int(
                    train_metrics.get("lgg_forward_calls", 0)
                )
                gate_loss_max_seen_overall = max(
                    gate_loss_max_seen_overall,
                    float(train_metrics.get("gate_loss_max_seen", 0.0)),
                )
                ev_n_records_total += int(train_metrics.get("ev_n_records", 0))
                ev_n_r1_total += int(train_metrics.get("ev_n_r1", 0))
                ev_n_r2_total += int(train_metrics.get("ev_n_r2", 0))
                ev_n_r3_total += int(train_metrics.get("ev_n_r3", 0))
                ev_n_r4_total += int(train_metrics.get("ev_n_r4", 0))
                # REV-7 (gate (0)): propagate epoch-averaged evidence-fidelity
                # stats so the outer summary writer can record the bridge proof.
                for _fk in (
                    "ev_real_prob_coverage_full_mode",
                    "ev_soft_epred_coverage_full_mode",
                    "ev_multi_source_sas_coverage_full_mode",
                    "ev_sdca_std_online",
                    "ev_sas_std_online",
                    "ev_aps_std_online",
                    "ev_pss_std_online",
                    "ev_runnerup_unique_pairs",
                    "ev_n_multi_source_agreed_total",
                    "ev_n_multi_source_conflict_total",
                    "ev_use_real_class_probs_bridge",
                    "ev_secondary_cache_active",
                ):
                    if _fk in train_metrics:
                        ev_fidelity_last_epoch[_fk] = train_metrics[_fk]
            else:
                train_metrics = _train_detector_epoch_full_mode(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    device=device,
                    scaler=scaler,
                    kg_encoder=kg_encoder,
                    node_feats=node_feats,
                    edge_index=edge_index,
                    edge_types=edge_types,
                    lambda_kg=float(args.lambda_kg),
                    freeze_kg=bool(args.freeze_kg),
                    anchor_teacher_model=anchor_teacher_model,
                    lambda_anchor=float(args.lambda_anchor),
                    anchor_alpha=float(args.anchor_alpha),
                    anchor_beta=float(args.anchor_beta),
                )
            val_loss = _eval_detector_loss_full_mode(
                model=model,
                loader=val_loader,
                device=device,
                scaler=scaler,
                kg_encoder=kg_encoder,
                node_feats=node_feats,
                edge_index=edge_index,
                edge_types=edge_types,
                lambda_kg=float(args.lambda_kg),
                freeze_kg=bool(args.freeze_kg),
            )
            val_det_metrics = _eval_detector_metrics_full_mode(
                model=model,
                loader=val_loader,
                device=device,
                detections_per_img=int(args.det_detections_per_img),
            )

            if selection_metric == "v2_composite":
                checkpoint_metric_value = _compute_v2_composite_score(
                    val_det_metrics,
                    baseline_metrics,
                )
            else:
                checkpoint_metric_value = float(val_loss)

            checkpoint_score = float(checkpoint_metric_value)
            final_val_loss = float(val_loss)
            best_val_loss_observed = min(best_val_loss_observed, float(val_loss))
            elapsed = time.time() - t0

            row = {
                "epoch": epoch,
                "train_loss_total": round(train_metrics["loss"], 6),
                "train_loss": round(train_metrics["loss"], 6),
                "train_loss_detector": round(train_metrics["loss_detector"], 6),
                "train_l_kg": round(train_metrics["l_kg"], 6),
                "train_l_anchor": round(train_metrics.get("l_anchor", 0.0), 6),
                "train_l_anchor_cls": round(train_metrics.get("l_anchor_cls", 0.0), 6),
                "train_l_anchor_box": round(train_metrics.get("l_anchor_box", 0.0), 6),
                "train_anchor_n_aligned": round(
                    train_metrics.get("anchor_n_aligned", 0.0),
                    3,
                ),
                "train_loss_classifier": round(train_metrics["loss_classifier"], 6),
                "train_loss_box_reg": round(train_metrics["loss_box_reg"], 6),
                "train_loss_objectness": round(train_metrics["loss_objectness"], 6),
                "train_loss_rpn_box_reg": round(train_metrics["loss_rpn_box_reg"], 6),
                "val_loss_total": round(val_loss, 6),
                "val_loss": round(val_loss, 6),
                "val_mAP_50": round(val_det_metrics.get("mAP_50", 0.0), 6),
                "val_mAP_75": round(val_det_metrics.get("mAP_75", 0.0), 6),
                "val_mAP_medium": round(val_det_metrics.get("mAP_medium", 0.0), 6),
                "val_MA_AP50": round(val_det_metrics.get("MA_AP50", 0.0), 6),
                "val_false_positive_pressure": round(
                    val_det_metrics.get("false_positive_pressure", 0.0),
                    6,
                ),
                "checkpoint_selection_metric": selection_metric,
                "checkpoint_metric_value": round(checkpoint_metric_value, 6),
                "roi_n_r1": round(train_metrics.get("roi_n_r1", 0.0), 3),
                "roi_n_r2": round(train_metrics.get("roi_n_r2", 0.0), 3),
                "roi_n_r3": round(train_metrics.get("roi_n_r3", 0.0), 3),
                "gate_tier_acc_r1": round(
                    train_metrics.get("gate_tier_acc_r1", 0.0),
                    6,
                ),
                "gate_tier_acc_r2": round(
                    train_metrics.get("gate_tier_acc_r2", 0.0),
                    6,
                ),
                "gate_tier_acc_r3": round(
                    train_metrics.get("gate_tier_acc_r3", 0.0),
                    6,
                ),
                "gate_tier_acc_r4": round(
                    train_metrics.get("gate_tier_acc_r4", 0.0),
                    6,
                ),
                "roi_used_fallback": round(
                    train_metrics.get("roi_used_fallback", 0.0), 6
                ),
                "roi_used_dynamic_sdm": round(
                    train_metrics.get("roi_used_dynamic_sdm", 0.0), 6
                ),
                "uniform_temp": bool(args.uniform_temp),
                "freeze_kg": bool(args.freeze_kg),
                "lambda_kg": float(args.lambda_kg),
                "lambda_anchor": float(args.lambda_anchor),
                "elapsed_sec": round(elapsed, 2),
            }
            log_rows.append(row)

            print(
                f"[Mode2][Full] Epoch {epoch:3d}/{args.epochs} "
                f"train_total={train_metrics['loss']:.4f} "
                f"train_det={train_metrics['loss_detector']:.4f} "
                f"train_lkg={train_metrics['l_kg']:.4f} "
                f"train_lanchor={train_metrics.get('l_anchor', 0.0):.4f} "
                f"val_total={val_loss:.4f} "
                f"val_mAP50={val_det_metrics.get('mAP_50', 0.0):.4f} "
                f"val_FPpr={val_det_metrics.get('false_positive_pressure', 0.0):.4f} "
                f"ckpt_score={checkpoint_score:.4f} "
                f"cls={train_metrics['loss_classifier']:.4f} "
                f"R1/R2/R3={train_metrics.get('roi_n_r1', 0.0):.1f}/"
                f"{train_metrics.get('roi_n_r2', 0.0):.1f}/"
                f"{train_metrics.get('roi_n_r3', 0.0):.1f} "
                f"{elapsed:.1f}s"
            )

            # v2.1: top-k checkpoint manager. Per-epoch, save best-by-X
            # candidates without forcing a single composite winner during
            # training. The audit-based Pareto selector picks the final
            # winner after training from external audit metrics, not from the
            # training-time val proxy. mode2_full_best.pt is preserved as a
            # backward-compatible alias for legacy tooling (still gated by
            # the composite criterion).
            def _build_ckpt_payload(
                _epoch: int = int(epoch),
                _val_loss: float = float(val_loss),
                _composite: float = float(checkpoint_metric_value),
            ) -> Dict[str, object]:
                return {
                    "epoch": _epoch,
                    "val_loss": _val_loss,
                    "checkpoint_selection_metric": selection_metric,
                    "checkpoint_metric_value": _composite,
                    "val_metrics": {
                        "mAP_50": float(val_det_metrics.get("mAP_50", 0.0)),
                        "mAP_75": float(val_det_metrics.get("mAP_75", 0.0)),
                        "mAP_medium": float(val_det_metrics.get("mAP_medium", 0.0)),
                        "MA_AP50": float(val_det_metrics.get("MA_AP50", 0.0)),
                        "false_positive_pressure": float(
                            val_det_metrics.get("false_positive_pressure", 0.0)
                        ),
                    },
                    "model_state_dict": model.state_dict(),
                    "kg_encoder_state_dict": kg_encoder.state_dict(),
                    "freeze_kg": bool(args.freeze_kg),
                    "lambda_kg": float(args.lambda_kg),
                    "lambda_anchor": float(args.lambda_anchor),
                    "config": detector_ckpt.get("config", {}),
                    "mode2_path": "full_mode2_detector",
                    "detector_checkpoint_in": str(args.detector_checkpoint),
                    "detector_config_in": detector_ckpt.get("config", {}),
                    "args": vars(args),
                }

            if getattr(args, "topk_checkpoints", False):
                if "_topk_manager" not in locals() or _topk_manager is None:  # type: ignore[has-type]
                    _topk_manager = TopKCheckpointManager(
                        out_dir=out_dir,
                        criteria=(
                            ("composite", "lower"),
                            ("map50", "higher"),
                            ("map_medium", "higher"),
                            ("ma_ap50", "higher"),
                        ),
                    )
                _topk_manager.update(
                    epoch=int(epoch),
                    metrics={
                        "composite": float(checkpoint_metric_value),
                        "map50": float(val_det_metrics.get("mAP_50", 0.0)),
                        "map_medium": float(val_det_metrics.get("mAP_medium", 0.0)),
                        "ma_ap50": float(val_det_metrics.get("MA_AP50", 0.0)),
                    },
                    payload_builder=_build_ckpt_payload,
                )
                _topk_manager.save_latest(payload_builder=_build_ckpt_payload)

            if checkpoint_score < best_checkpoint_score:
                best_checkpoint_score = float(checkpoint_score)
                best_checkpoint_metric_value = float(checkpoint_metric_value)
                best_val_loss = float(val_loss)
                best_epoch = epoch
                # Always also write mode2_full_best.pt (composite winner) for
                # backward compatibility with legacy audit tooling. When
                # --topk_checkpoints is on, this file mirrors best_by_composite.pt.
                torch.save(_build_ckpt_payload(), out_dir / "mode2_full_best.pt")

        torch.save(
            {
                "epoch": args.epochs,
                "val_loss": final_val_loss,
                "checkpoint_selection_metric": selection_metric,
                "model_state_dict": model.state_dict(),
                "kg_encoder_state_dict": kg_encoder.state_dict(),
                "freeze_kg": bool(args.freeze_kg),
                "lambda_kg": float(args.lambda_kg),
                "lambda_anchor": float(args.lambda_anchor),
                "config": detector_ckpt.get("config", {}),
                "mode2_path": "full_mode2_detector",
                "detector_checkpoint_in": str(args.detector_checkpoint),
                "detector_config_in": detector_ckpt.get("config", {}),
                "args": vars(args),
            },
            out_dir / "mode2_full_final.pt",
        )
    finally:
        restore_default_roi_loss()

    log_path = out_dir / "training_log_full_mode2_detector.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        if log_rows:
            writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            writer.writeheader()
            writer.writerows(log_rows)

    def _is_nonuniform_weight_vector(values: Optional[List[float]]) -> bool:
        if values is None:
            return False
        return any(abs(float(v) - 1.0) > 1e-9 for v in values)

    v2_features: List[str] = []
    if str(getattr(args, "mode2_version", "v1")) == "v2":
        if _is_nonuniform_weight_vector(gate_tier_weights):
            v2_features.append("tier_balanced_gate")
        if float(args.mode2_gate_focal_gamma) > 0.0:
            v2_features.append("focal_gate")
        if _is_nonuniform_weight_vector(gate_class_weights):
            v2_features.append("class_balanced_gate")
        if float(args.lambda_anchor) > 0.0:
            v2_features.append("L_anchor")
        if int(args.det_freeze_backbone_epochs) > 0:
            v2_features.append("head_only_warmup")
        if selection_metric == "v2_composite":
            v2_features.append("multi_metric_ckpt")

    summary = {
        "mode2_path": "full_mode2_detector",
        "mode2_version": str(getattr(args, "mode2_version", "v1")),
        "v2_features": v2_features,
        # B2 provenance contract (Mode2 checklist Mei 31_claude.md): the
        # canonical scope label is the SOLE truth for downstream C3-claim
        # eligibility. Closed-loop claim scope and detached diagnostic scope
        # are both supported in run_full_mode2_detector.
        "mode2_path_canonical": getattr(args, "_mode2_path", "full_detector"),
        "mode2_scope": getattr(args, "_mode2_scope", _FULL_MODE2_STATIC_SCOPE),
        "source_mode1_lock_file": args.mode1_lock_file,
        "source_mode1_run_root": args.mode1_run_root,
        "source_stage3c_dir": args.stage3c_dir,
        "source_lgg_checkpoint": args.lgg_checkpoint,
        "b3_lgg_torch_state_dict_loaded": bool(
            (getattr(args, "_lgg_probe", {}) or {}).get("has_state_dict")
            and (getattr(args, "_lgg_probe", {}) or {}).get("key_used")
            == "B3_LGG_TORCH"
        ),
        "baseline_audit": args.baseline_audit,
        "baseline_mAP_50": (
            args._baseline_map50_resolved
            if getattr(args, "_baseline_map50_resolved", None) is not None
            else args.expected_baseline_map50
        ),
        "baseline_mAP_medium": baseline_metrics.get("mAP_medium"),
        "baseline_false_positive_pressure": baseline_metrics.get(
            "false_positive_pressure"
        ),
        "baseline_MA_AP50": baseline_metrics.get("MA_AP50"),
        # B5: warm-start source for LGG in the closed-loop full path.
        "kg_warmstart_source": (
            "mode1_b3_lgg_torch" if lgg_warmstart_loaded else "random_init"
        ),
        "lgg_warmstart_loaded": bool(lgg_warmstart_loaded),
        "lgg_warmstart_reason": lgg_warmstart_reason,
        # B4 telemetry — proves online evidence + LGG forward + GateLoss actually
        # ran in the training loop. Counters > 0 are the runtime predicate for an
        # honest closed-loop scope claim.
        "lgg_forward_calls_total": int(lgg_forward_calls_total),
        "gate_loss_max_seen": float(gate_loss_max_seen_overall),
        "ev_n_records_total": int(ev_n_records_total),
        "ev_n_r1_total": int(ev_n_r1_total),
        "ev_n_r2_total": int(ev_n_r2_total),
        "ev_n_r3_total": int(ev_n_r3_total),
        "ev_n_r4_total": int(ev_n_r4_total),
        # REV-7 (Sprint 1+2) evidence-fidelity telemetry — gate (0) of the
        # CONDITIONAL CLAIM RULE (§C3). Populated from the last closed-loop
        # epoch's epoch-averaged statistics. Architecture thresholds:
        #   ev_real_prob_coverage_full_mode        >= 0.95
        #   ev_soft_epred_coverage_full_mode       >= 0.95
        #   ev_multi_source_sas_coverage_full_mode >= 0.80
        #   ev_runnerup_unique_pairs               >= 5
        #   ev_sdca_std_online                     >= 0.10
        #   ev_sas_distinct_values                 >= 20 (Stage 5B)
        "evidence_fidelity_last_epoch": dict(ev_fidelity_last_epoch),
        "ev_use_real_class_probs_bridge_flag": bool(
            use_real_class_probs_bridge_flag
        ),
        "ev_secondary_cache_active_flag": bool(secondary_cache_train),
        "ev_secondary_cache_n_images": int(len(secondary_cache_train)),
        "ev_secondary_cache_dir": (
            str(_secondary_cache_path) if _secondary_cache_path else None
        ),
        # REV-7B (2026-06-02) Mode 2A REAL + Jalur B flags.
        "mode2a_preserve_standard_roi_loss": bool(preserve_roi_loss),
        "freeze_full_detector": bool(freeze_full_detector),
        "lambda_gate": float(args.lambda_gate),
        "mode2_gate_tier_weights": gate_tier_weights,
        "mode2_gate_class_weights": gate_class_weights,
        "mode2_gate_focal_gamma": float(args.mode2_gate_focal_gamma),
        "allow_detached_lgg": bool(args.allow_detached_lgg),
        "full_mode2_detector": True,
        "detector_loss_replacement": "roi_sdgl",
        # HONEST c3 gating (Mode2_implementation_checklist Mei 31_claude.md §D8):
        # Only TRUE when (i) canonical scope is the claim variant AND (ii) LGG
        # was actually called >=1 time AND (iii) GateLoss produced non-zero
        # signal during training. Silent wiring failures will keep this FALSE
        # even if the scope label says claim.
        "c3_condition_i_met": (
            getattr(args, "_mode2_scope", _FULL_MODE2_STATIC_SCOPE)
            == _FULL_MODE2_CLAIM_SCOPE
            and int(lgg_forward_calls_total) > 0
            and float(gate_loss_max_seen_overall) > 0.0
        ),
        "freeze_kg_ignored": False,
        "lambda_kg_ignored": False,
        "freeze_kg": bool(args.freeze_kg),
        "lambda_kg": float(args.lambda_kg),
        "lambda_anchor": float(args.lambda_anchor),
        "anchor_alpha": float(args.anchor_alpha),
        "anchor_beta": float(args.anchor_beta),
        "anchor_teacher_checkpoint": str(anchor_teacher_ckpt),
        "kg_closed_loop": not bool(args.freeze_kg),
        "kg_encoder_trainable": not bool(args.freeze_kg),
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6) if best_epoch > 0 else None,
        "best_val_loss_observed": (
            round(best_val_loss_observed, 6)
            if best_val_loss_observed < float("inf")
            else None
        ),
        "checkpoint_selection_metric": selection_metric,
        "best_checkpoint_score": (
            round(best_checkpoint_score, 6)
            if best_checkpoint_score < float("inf")
            else None
        ),
        "best_checkpoint_metric_value": (
            round(best_checkpoint_metric_value, 6)
            if best_checkpoint_metric_value < float("inf")
            else None
        ),
        "final_val_loss": round(final_val_loss, 6) if log_rows else None,
        "epochs": args.epochs,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "warmup_epochs": int(args.det_freeze_backbone_epochs),
        "uniform_temp": bool(args.uniform_temp),
        "lambda1": float(args.lambda1),
        "lambda2": float(args.lambda2),
        "lambda3": float(args.lambda3),
        "mode2_close_threshold": float(args.mode2_close_threshold),
        "mode2_r1_conf": float(args.mode2_r1_conf),
        "mode2_r2_conf": float(args.mode2_r2_conf),
        "mode2_r3_min_weight": float(args.mode2_r3_min_weight),
        "mode2_bg_weight": float(args.mode2_bg_weight),
        "detector_checkpoint_in": str(args.detector_checkpoint),
        "data_root": str(args.data_root),
        "train_images": len(train_ds),
        "valid_images": len(val_ds),
        "smoke_test": bool(args.smoke_test),
        "final_val_mAP_50": (
            float(log_rows[-1].get("val_mAP_50", 0.0)) if log_rows else None
        ),
        "final_val_mAP_medium": (
            float(log_rows[-1].get("val_mAP_medium", 0.0)) if log_rows else None
        ),
        "final_val_MA_AP50": (
            float(log_rows[-1].get("val_MA_AP50", 0.0)) if log_rows else None
        ),
        "final_val_false_positive_pressure": (
            float(log_rows[-1].get("val_false_positive_pressure", 0.0))
            if log_rows
            else None
        ),
        "note": (
            "Detector head/backbone receives SDGL-shaped gradients via patched "
            "ROI classification loss. Dynamic SDM in ROI loss is sourced from "
            "current KG encoder embeddings; --freeze_kg and --lambda_kg are "
            "active in full-detector mode. Checkpoint selection follows "
            "checkpoint_selection_metric (val_loss or v2_composite with mAP/FP/MA). "
            "For publication claims, always confirm with "
            "tools/audit_detector_stage1a.py on mode2_full_best.pt."
        ),
    }
    with open(out_dir / "mode2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n[Mode2][Full] === Training complete ===")
    print(
        "[Mode2][Full] Best checkpoint: "
        f"metric={selection_metric} score={best_checkpoint_score:.4f} "
        f"(epoch={best_epoch}, val_loss={best_val_loss:.4f})"
    )
    print(
        f"[Mode2][Full] Checkpoints: {out_dir}/mode2_full_best.pt, mode2_full_final.pt"
    )
    print(f"[Mode2][Full] Log: {log_path}")


# ── Training loop ──────────────────────────────────────────────────────────────


def train_one_epoch(
    encoder: LGovNetEncoder,
    gate: LGG,
    reclassifier: SDGLReClassifier,
    sdgl_loss_fn: SDGLLoss,
    optimizer: torch.optim.Optimizer,
    train_records: List[Dict],
    node_feats: torch.Tensor,
    edge_index: torch.Tensor,
    edge_types: torch.Tensor,
    sdm_norm: Dict,
    device: torch.device,
    batch_size: int,
    lambda_gate: float,
    uniform_temp: bool,
    freeze_kg: bool,
    epoch: int,
    tau_gumbel: float,
) -> Dict[str, float]:
    encoder.train()
    gate.train()
    reclassifier.train()
    if freeze_kg:
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad_(False)

    total_loss = 0.0
    total_l_sdgl = 0.0
    total_l_kg = 0.0
    total_l_gate = 0.0
    n_batches = 0
    sdgl_has_grad = False
    reclassifier_has_grad = False
    n_records = len(train_records)

    # Shuffle records
    indices = list(range(n_records))
    random.shuffle(indices)

    for batch_start in range(0, n_records, batch_size):
        batch_indices = indices[batch_start : batch_start + batch_size]
        batch = [train_records[i] for i in batch_indices]
        if not batch:
            continue

        optimizer.zero_grad()

        # ── KG forward: get lesion embeddings ────────────────────────────────
        node_x = node_feats.to(device)
        ei = edge_index.to(device)
        et = edge_types.to(device)
        all_embeddings = encoder(node_x, ei, et)  # [N_nodes, d]
        # First 4 nodes are lesion classes (MA, HE, Hard_EX, Soft_EX per NODE_ORDER)
        lesion_embeddings = all_embeddings[:4]  # [4, d]

        # ── L_KG anchor loss ─────────────────────────────────────────────────
        l_kg = compute_l_kg(lesion_embeddings)

        # ── Per-record SDGL loss ─────────────────────────────────────────────
        l_sdgl_total = torch.zeros(1, device=device)
        l_gate_total = torch.zeros(1, device=device)
        n_active = 0

        for rec in batch:
            ev_vec = rec["evidence"].to(device)  # [4]
            tier_idx = rec["tier_idx"]
            gt_class_idx = rec["gt_class_idx"]
            ensemble_probs = rec["ensemble_probs"]
            consensus = rec["consensus"]
            aps = rec["aps"]

            # ── Frozen detector ensemble scores drive both gate input and the
            # trainable SDGL re-classifier. The gradient path enters through
            # reclassifier parameters and e_pred / lesion_embeddings.
            p_ens = torch.tensor(
                [ensemble_probs.get(c, 0.25) for c in LESION_CLASSES],
                dtype=torch.float32,
                device=device,
            ).clamp(min=1e-9)
            p_ens = p_ens / p_ens.sum()
            e_pred = (p_ens.unsqueeze(1) * lesion_embeddings).sum(0)  # [d]

            # ── LGG forward (Gumbel-Softmax for differentiable routing) ──────
            gate_input = torch.cat([ev_vec, e_pred], dim=0).unsqueeze(0)  # [1, 4+d]
            gate_out = gate.gumbel_forward(
                gate_input, temperature=tau_gumbel, hard=False
            )
            tier_probs = gate_out["tier_probs"].squeeze(0)  # [4]

            # ── L_gate: cross-entropy to GT tier ─────────────────────────────
            tier_target_t = torch.tensor([tier_idx], dtype=torch.long, device=device)
            l_gate_i = F.cross_entropy(tier_probs.unsqueeze(0), tier_target_t)
            l_gate_total = l_gate_total + l_gate_i

            # ── Trainable log_probs for SDGL (Option B) ──────────────────────
            log_probs = reclassifier(p_ens, e_pred)  # [C]

            # ── SDGL per-tier loss ────────────────────────────────────────────
            if tier_idx == 0:  # R1 — standard hard target
                l_i = sdgl_loss_fn.lambda1 * (-log_probs[gt_class_idx])

            elif tier_idx == 1:  # R2 — SDM-shaped soft target
                if uniform_temp:
                    # Uniform temperature ablation: use frozen ensemble scores
                    # directly as the detached soft target.
                    q_sdm = p_ens.detach()
                else:
                    q_sdm = build_q_sdm(
                        ensemble_probs, sdm_norm, eta=sdgl_loss_fn.eta
                    ).to(device)
                l_i = sdgl_loss_fn.lambda2 * (-(q_sdm * log_probs).sum())

            elif tier_idx == 2:  # R3 — weighted hard target
                w_j = float(consensus) * float(aps)
                l_i = sdgl_loss_fn.lambda3 * w_j * (-log_probs[gt_class_idx])

            else:
                continue  # R4 — no gradient (already filtered, but defensive)

            l_sdgl_total = l_sdgl_total + l_i
            n_active += 1

        if n_active == 0:
            continue

        # ── Total loss ────────────────────────────────────────────────────────
        l_sdgl_mean = l_sdgl_total / n_active
        l_gate_mean = l_gate_total / len(batch)
        sdgl_has_grad = sdgl_has_grad or bool(l_sdgl_mean.requires_grad)
        loss = l_sdgl_mean + lambda_gate * l_gate_mean + sdgl_loss_fn.lambda_kg * l_kg

        loss.backward()
        batch_reclassifier_has_grad = any(
            p.grad is not None and float(p.grad.detach().abs().sum().item()) > 0.0
            for p in reclassifier.parameters()
        )
        reclassifier_has_grad = reclassifier_has_grad or batch_reclassifier_has_grad
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters())
            + list(gate.parameters())
            + list(reclassifier.parameters()),
            max_norm=1.0,
        )
        optimizer.step()

        total_loss += float(loss.item())
        total_l_sdgl += float(l_sdgl_mean.item())
        total_l_kg += float(l_kg.item())
        total_l_gate += float(l_gate_mean.item())
        n_batches += 1

    if n_batches == 0:
        return {
            "loss": 0.0,
            "l_sdgl": 0.0,
            "l_kg": 0.0,
            "l_gate": 0.0,
            "sdgl_has_gradient": sdgl_has_grad,
            "reclassifier_has_gradient": reclassifier_has_grad,
        }

    return {
        "loss": total_loss / n_batches,
        "l_sdgl": total_l_sdgl / n_batches,
        "l_kg": total_l_kg / n_batches,
        "l_gate": total_l_gate / n_batches,
        "sdgl_has_gradient": sdgl_has_grad,
        "reclassifier_has_gradient": reclassifier_has_grad,
    }


@torch.no_grad()
def eval_gate_accuracy(
    encoder: LGovNetEncoder,
    gate: LGG,
    val_records: List[Dict],
    node_feats: torch.Tensor,
    edge_index: torch.Tensor,
    edge_types: torch.Tensor,
    device: torch.device,
) -> float:
    """Compute gate tier-prediction accuracy on val records."""
    encoder.eval()
    gate.eval()

    node_x = node_feats.to(device)
    ei = edge_index.to(device)
    et = edge_types.to(device)
    all_emb = encoder(node_x, ei, et)
    lesion_emb = all_emb[:4]

    correct = 0
    total = 0
    for rec in val_records:
        ev_vec = rec["evidence"].to(device)
        p_ens = torch.tensor(
            [rec["ensemble_probs"].get(c, 0.25) for c in LESION_CLASSES],
            dtype=torch.float32,
            device=device,
        )
        e_pred = (p_ens.unsqueeze(1) * lesion_emb).sum(0)
        gate_input = torch.cat([ev_vec, e_pred], dim=0).unsqueeze(0)
        gate_out = gate(gate_input)
        pred_tier = int(gate_out["tier_hard"].item())
        if pred_tier == rec["tier_idx"]:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


@torch.no_grad()
def eval_reclassifier(
    encoder: LGovNetEncoder,
    reclassifier: "SDGLReClassifier",
    val_records: List[Dict],
    node_feats: torch.Tensor,
    edge_index: torch.Tensor,
    edge_types: torch.Tensor,
    sdm_norm: Dict,
    device: torch.device,
) -> Dict[str, float]:
    """Measure what L_SDGL actually trains — the re-classifier head.

    Until this exists, val_gate_acc reflects only L_gate; the SDGL branch
    (re-classifier output) is a dead-end and the four C3 ablations cannot
    be distinguished. This evaluates the re-classifier directly on R1–R3
    val records:
      - reclass_acc: argmax-class prediction accuracy.
      - reclass_sdm_cost_mean: average clinical-SDM cost per record
        (0 for correct predictions; SDM(pred,gt) for misclassifications).
      - reclass_sdm_cost_misclass: SDM cost averaged over MISCLASSIFIED
        records only — interpretable as 'severity of typical error'.
        THIS is the H2-aligned proxy metric: SDM-temp / freeze_kg /
        lambda_kg ablations must be compared on this, NOT val_gate_acc.
      - reclass_per_class_acc: per-lesion-class accuracy.
    """
    encoder.eval()
    reclassifier.eval()

    node_x = node_feats.to(device)
    ei = edge_index.to(device)
    et = edge_types.to(device)
    all_emb = encoder(node_x, ei, et)
    lesion_emb = all_emb[:4]

    n_classes = len(LESION_CLASSES)
    lesion_idx_map = {c: i for i, c in enumerate(LESION_CLASSES)}
    correct = 0
    total = 0
    sdm_cost_sum = 0.0
    sdm_cost_misclass_sum = 0.0
    n_misclass = 0
    per_class_total = [0] * n_classes
    per_class_correct = [0] * n_classes

    for rec in val_records:
        p_ens = torch.tensor(
            [rec["ensemble_probs"].get(c, 0.25) for c in LESION_CLASSES],
            dtype=torch.float32,
            device=device,
        ).clamp(min=1e-9)
        p_ens = p_ens / p_ens.sum()
        e_pred = (p_ens.unsqueeze(1) * lesion_emb).sum(0)
        log_probs = reclassifier(p_ens, e_pred)
        pred_idx = int(torch.argmax(log_probs).item())
        gt_idx = int(rec["gt_class_idx"])

        total += 1
        per_class_total[gt_idx] += 1
        if pred_idx == gt_idx:
            correct += 1
            per_class_correct[gt_idx] += 1
            # SDM cost = 0 for correct prediction
        else:
            n_misclass += 1
            pred_name = LESION_CLASSES[pred_idx]
            gt_name = LESION_CLASSES[gt_idx]
            # Canonical SDM key: lower lesion index first
            if lesion_idx_map[pred_name] <= lesion_idx_map[gt_name]:
                key = (pred_name, gt_name)
            else:
                key = (gt_name, pred_name)
            entry = sdm_norm.get(key)
            sdm_val = float(entry["norm"]) if entry is not None else 0.0
            sdm_cost_sum += sdm_val
            sdm_cost_misclass_sum += sdm_val

    acc = correct / total if total > 0 else 0.0
    mean_sdm_cost = sdm_cost_sum / total if total > 0 else 0.0
    mean_sdm_cost_misclass = (
        sdm_cost_misclass_sum / n_misclass if n_misclass > 0 else 0.0
    )
    per_class_acc = [
        per_class_correct[i] / per_class_total[i] if per_class_total[i] > 0 else 0.0
        for i in range(n_classes)
    ]
    return {
        "reclass_acc": round(acc, 6),
        "reclass_sdm_cost_mean": round(mean_sdm_cost, 6),
        "reclass_sdm_cost_misclass": round(mean_sdm_cost_misclass, 6),
        "reclass_n_misclass": n_misclass,
        "reclass_per_class_acc": [round(v, 6) for v in per_class_acc],
    }


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 5: Mode 2 / C3 / SDGL - KG + LGG joint training."
    )
    parser.add_argument(
        "--detector_checkpoint",
        required=True,
        help="Path to r5b_ext25 Faster R-CNN checkpoint (for reference/audit only in Phase 1).",
    )
    parser.add_argument(
        "--lgg_checkpoint",
        required=True,
        help="Path to Stage 3D lgg_models.pkl.",
    )
    parser.add_argument(
        "--kg_dir",
        default="outputs/stage3_audit",
        help="Stage 3A KG artifacts dir (nodes.json, edges.json, sdm_norm_clinical_prior.json).",
    )
    parser.add_argument(
        "--stage3c_dir",
        default="outputs/stage3c_audit",
        help="Stage 3C governance_features CSV dir.",
    )
    parser.add_argument(
        "--data_root",
        default="data/oiaddr",
        help="Dataset root (for future Phase 2 integration).",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help=(
            "Batch size. Proxy path: governance records per step. "
            "--full_mode2_detector: images per step."
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="DataLoader workers for --full_mode2_detector.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="Optional cap per split for --full_mode2_detector (0=no cap).",
    )
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lambda1", type=float, default=1.0, help="SDGL R1 weight.")
    parser.add_argument("--lambda2", type=float, default=0.5, help="SDGL R2 weight.")
    parser.add_argument("--lambda3", type=float, default=0.3, help="SDGL R3 weight.")
    parser.add_argument("--lambda_kg", type=float, default=0.1, help="L_KG weight.")
    parser.add_argument(
        "--lambda_gate", type=float, default=0.05, help="L_gate weight."
    )
    parser.add_argument(
        "--mode2_version",
        choices=["v1", "v2"],
        default="v1",
        help="Mode 2 recipe version. v2 enables Phase-A defaults.",
    )
    parser.add_argument(
        "--checkpoint_selection_metric",
        choices=["auto", "val_loss", "v2_composite"],
        default="auto",
        help=(
            "Checkpoint selection metric for full-mode runs. "
            "auto=val_loss for v1, v2_composite for v2."
        ),
    )
    parser.add_argument(
        "--lambda_anchor",
        type=float,
        default=None,
        help="Weight for detector anchor distillation regularizer.",
    )
    parser.add_argument(
        "--anchor_alpha",
        type=float,
        default=1.0,
        help="Classification KL weight inside L_anchor.",
    )
    parser.add_argument(
        "--anchor_beta",
        type=float,
        default=0.5,
        help="Box SmoothL1 weight inside L_anchor.",
    )
    parser.add_argument(
        "--anchor_teacher_checkpoint",
        default=None,
        help="Optional teacher checkpoint path for L_anchor (defaults to detector_checkpoint).",
    )
    parser.add_argument(
        "--mode2_gate_tier_weights",
        default=None,
        help="Optional comma-separated 4-float weights for GateLoss tiers R1,R2,R3,R4.",
    )
    parser.add_argument(
        "--mode2_gate_class_weights",
        default=None,
        help=(
            "Optional comma-separated 4-float sample weights for gate supervision "
            "by majority class (MA,HE,Hard_EX,Soft_EX)."
        ),
    )
    parser.add_argument(
        "--mode2_gate_focal_gamma",
        type=float,
        default=None,
        help="Focal gamma for GateLoss (0 disables focal scaling).",
    )
    parser.add_argument(
        "--tau_start",
        type=float,
        default=1.0,
        help="Gumbel-Softmax temperature start (annealed to tau_end).",
    )
    parser.add_argument(
        "--tau_end",
        type=float,
        default=0.1,
        help="Gumbel-Softmax temperature end.",
    )
    parser.add_argument(
        "--uniform_temp",
        action="store_true",
        help="C3 ablation: use uniform temperature (tau=1) instead of SDM-guided tau_SDM.",
    )
    parser.add_argument(
        "--freeze_kg",
        action="store_true",
        help="C3 ablation: freeze KG encoder - no closed-loop KG co-adaptation.",
    )
    parser.add_argument(
        "--full_mode2_detector",
        action="store_true",
        help=(
            "Run full detector path: patch Faster R-CNN ROI classification loss "
            "with SDGL and fine-tune detector end-to-end."
        ),
    )
    parser.add_argument(
        "--det_score_thresh",
        type=float,
        default=0.01,
        help="Detector score threshold for --full_mode2_detector.",
    )
    parser.add_argument(
        "--det_detections_per_img",
        type=int,
        default=300,
        help="ROI detections per image cap for --full_mode2_detector.",
    )
    parser.add_argument(
        "--det_freeze_backbone_epochs",
        type=int,
        default=None,
        help="Freeze backbone for first N epochs in --full_mode2_detector.",
    )
    parser.add_argument(
        "--mode2_close_threshold",
        type=float,
        default=0.30,
        help="SDM(norm) threshold for clinically-close class pairs in ROI SDGL.",
    )
    parser.add_argument(
        "--mode2_r1_conf",
        type=float,
        default=0.70,
        help="Confidence threshold for R1 assignment in ROI SDGL.",
    )
    parser.add_argument(
        "--mode2_r2_conf",
        type=float,
        default=0.40,
        help="Confidence threshold for R2 assignment in ROI SDGL.",
    )
    parser.add_argument(
        "--mode2_r3_min_weight",
        type=float,
        default=0.10,
        help="Minimum weight for R3 hard-loss term in ROI SDGL.",
    )
    parser.add_argument(
        "--mode2_bg_weight",
        type=float,
        default=0.20,
        help="Background supervision weight in ROI SDGL.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Short run with 64 records max, 1 epoch.",
    )
    parser.add_argument(
        "--allow_proxy_lgg_warmstart",
        action="store_true",
        help=(
            "Allow legacy config-only LGG warm-start (no state_dict). "
            "Use only for controlled proxy comparisons; not recommended "
            "for Mode B mainline."
        ),
    )

    # ── B2 provenance + scope contract (Mode 2 checklist Mei 31_claude.md) ──
    parser.add_argument(
        "--mode1_lock_file",
        default=None,
        help="Path to Mode 1 final LOCK markdown (provenance for mode2_summary).",
    )
    parser.add_argument(
        "--mode1_run_root",
        default=None,
        help="Path to Mode 1 final run root (provenance for mode2_summary).",
    )
    parser.add_argument(
        "--baseline_audit",
        default=None,
        help="Path to baseline detector stage1a_raw.json (e.g. r5b baseline).",
    )
    parser.add_argument(
        "--expected_baseline_map50",
        type=float,
        default=0.241129,
        help="Expected baseline mAP_50 for hard-gate reporting.",
    )
    parser.add_argument(
        "--require_cuda",
        action="store_true",
        help=(
            "Hard-fail if torch.cuda.is_available() is False. Recommended on "
            "server runs (v2.1+); CPU fallback is too slow to be admissible."
        ),
    )
    parser.add_argument(
        "--topk_checkpoints",
        action="store_true",
        help=(
            "Mode 2 v2.1: save up to 5 candidate checkpoints per run "
            "(best_by_composite/map50/map_medium/ma_ap50 + latest) instead of "
            "the legacy single mode2_full_best.pt. mode2_full_best.pt is still "
            "written as a backward-compatible alias of best_by_composite.pt."
        ),
    )
    parser.add_argument(
        "--allow_detached_lgg",
        action="store_true",
        help=(
            "Force detached diagnostic scope for --full_mode2_detector: "
            "'roi_sdgl_kg_encoder_only' (LGG/evidence/GateLoss skipped in loop). "
            "Use ONLY for diagnostic smoke; this flag does NOT authorise a C3 claim."
        ),
    )
    # ── REV-7 (Sprint 1): real-class-probabilities bridge ────────────────────
    parser.add_argument(
        "--disable_real_class_probs_bridge",
        action="store_true",
        help=(
            "REV-7 Sprint 1: when set, the Mode 2 closed-loop reverts to "
            "model(images) + _synthetic_class_probs() for detector-side "
            "ambiguity (4-deterministic-value SDCA). Default is OFF, i.e. the "
            "real-prob bridge is ACTIVE and predict_with_class_probs() feeds "
            "the online evidence builder with real ROI softmax. Use this flag "
            "ONLY for negative-ablation runs that establish the synthetic "
            "baseline; positive C3 claims require the bridge ACTIVE per "
            "architecture §EVIDENCE-FIDELITY CONTRACT and §C3 gate (0)."
        ),
    )
    # ── REV-7 (Sprint 2): multi-source SAS ───────────────────────────────────
    parser.add_argument(
        "--secondary_detector_cache_dir",
        type=Path,
        default=None,
        help=(
            "REV-7 Sprint 2: path to a per-image secondary-detector cache "
            "(e.g. RetinaNet r1a JSON files written by "
            "tools/export_detector_predictions.py). When provided, the Mode 2 "
            "closed-loop computes cross-detector agreement SAS at training "
            "time (real multi-source) instead of the single-source max-softmax "
            "proxy. Required to satisfy architecture §C3 gate (0) SAS clause "
            "(multi_source_sas_coverage >= 0.80 and sas_distinct_values >= 20). "
            "None = single-source SAS proxy (negative-ablation only)."
        ),
    )
    # ── REV-7B (post-Sprint-2 outcome fix, 2026-06-02): Mode 2A REAL ──────────
    # Empirical finding: bridge fidelity (Sprint 1+2) PASS but detector ECE
    # collapsed (+0.44 vs baseline) and FPpr +15% because enable_roi_sdgl_loss()
    # is called unconditionally and REPLACES standard ROI CE with SDGL.
    # Architecture §EXECUTION CONTRACT Mode 2A explicitly requires the standard
    # ROI CE to be PRESERVED as the stability anchor — but the code violates
    # this. The flag below brings code into line with architecture.
    parser.add_argument(
        "--mode2a_preserve_standard_roi_loss",
        action="store_true",
        help=(
            "REV-7B (Mode 2A REAL): SKIP enable_roi_sdgl_loss() so that the "
            "detector trains with vanilla Faster R-CNN ROI CE (stability "
            "anchor). L_KG, L_gate, L_anchor remain as auxiliary losses but no "
            "longer dominate the classifier output distribution. This is the "
            "LITERAL implementation of architecture Mode 2A and is required "
            "for any positive C3 claim under the evidence-faithful interface."
        ),
    )
    parser.add_argument(
        "--freeze_detector",
        action="store_true",
        help=(
            "REV-7B (Mode 2A frozen ablation, Jalur B): freeze ALL detector "
            "parameters (backbone + RPN + ROI head). Only KG-GAT + LGG receive "
            "gradient updates. The detector forward still runs to produce "
            "detections + class_probs for online evidence, but its weights do "
            "not move. mAP_50 should equal baseline EXACTLY; KG contribution "
            "is measured purely at the LGG routing layer. Useful for clean "
            "attribution test and as a fallback if --mode2a_preserve_standard_"
            "roi_loss still under-performs."
        ),
    )
    # ── REV-7C (2026-06-02 post-RCA): lambda warmup for KG and gate losses ──
    # RCA suspect #3: KG-GAT random-init causes large L_KG (≈7.97) at epoch 1
    # which over-perturbs detector classifier. R3 epoch 1 mAP dipped to 0.2222
    # vs no-KG ablation epoch 1 = 0.2406 — a 0.018 absolute gap that mostly
    # came from initial KG noise. Ramping lambda_kg/lambda_gate from 0 to
    # target over a few warmup epochs prevents this initial perturbation while
    # preserving the steady-state KG signal that helps from epoch 2 onward.
    parser.add_argument(
        "--lambda_kg_warmup_epochs",
        type=int,
        default=0,
        help=(
            "REV-7C: linear ramp of lambda_kg from 0 to target over this many "
            "initial epochs. Default 0 = no warmup (preserve REV-7B behavior). "
            "Recommended 2 for full Mode 2A REAL runs to mitigate KG-GAT "
            "random-init perturbation observed at epoch 1 of REV-7B R3 smoke."
        ),
    )
    parser.add_argument(
        "--lambda_gate_warmup_epochs",
        type=int,
        default=0,
        help=(
            "REV-7C: linear ramp of lambda_gate from 0 to target over this "
            "many initial epochs. Default 0 = no warmup. Recommended 1 to "
            "give LGG time to stabilize its routing distribution before "
            "GateLoss starts pulling the KG embeddings via e_pred."
        ),
    )

    parser.add_argument("--output_dir", default="outputs/mode2/standard")
    args = parser.parse_args()

    if args.mode2_version == "v2":
        if args.det_freeze_backbone_epochs is None:
            args.det_freeze_backbone_epochs = 5
        if args.mode2_gate_tier_weights is None:
            args.mode2_gate_tier_weights = "1.0,1.2,1.3,0.7"
        if args.mode2_gate_class_weights is None:
            args.mode2_gate_class_weights = "1.25,1.0,1.0,1.25"
        if args.mode2_gate_focal_gamma is None:
            args.mode2_gate_focal_gamma = 2.0
        if args.lambda_anchor is None:
            args.lambda_anchor = 0.1
    else:
        if args.det_freeze_backbone_epochs is None:
            args.det_freeze_backbone_epochs = 0
        if args.mode2_gate_focal_gamma is None:
            args.mode2_gate_focal_gamma = 0.0
        if args.lambda_anchor is None:
            args.lambda_anchor = 0.0

    if args.checkpoint_selection_metric == "auto":
        args.checkpoint_selection_metric = (
            "v2_composite" if args.mode2_version == "v2" else "val_loss"
        )

    # ── B2/B5 scope assignment (Mode2_implementation_checklist Mei 31_claude.md) ─
    # Two valid scopes for the full-detector path:
    #   - roi_sdgl_kg_encoder_only       : diagnostic baseline (Opsi 3a). Selected
    #     by --allow_detached_lgg. LGG / online evidence / GateLoss are NOT in
    #     the training loop; only ROI SDGL + KG encoder. NEVER a C3 claim.
    #   - full_detector_lgg_evidence_closed_loop : claim scope (Opsi 3b, default
    #     when --full_mode2_detector). B4-B5 wiring active: online evidence per
    #     detection, LGG.forward in loop, GateLoss in L_total. KG encoder + LGG
    #     warm-start from B3_LGG_TORCH where compatible.
    if args.full_mode2_detector:
        args._mode2_path = "full_detector"
        args._mode2_scope = (
            _FULL_MODE2_STATIC_SCOPE
            if args.allow_detached_lgg
            else _FULL_MODE2_CLAIM_SCOPE
        )
    else:
        args._mode2_path = "proxy"
        args._mode2_scope = "proxy_reclassifier"

    args._lgg_probe = _probe_lgg_state_dict(Path(args.lgg_checkpoint))
    args._baseline_metrics = _resolve_baseline_metrics(args.baseline_audit)
    args._baseline_map50_resolved = (args._baseline_metrics or {}).get("mAP_50")

    # ── Setup ─────────────────────────────────────────────────────────────────
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # v2.1: hard-fail when CUDA is required but unavailable. The closed-loop
    # full-detector path runs Faster R-CNN + KG encoder + LGG + teacher anchor
    # forwards per batch; CPU fallback yields unusable throughput. The flag is
    # opt-in to preserve compatibility with diagnostic CPU smoke; runbook v2.1
    # always sets it on server.
    if getattr(args, "require_cuda", False) and not torch.cuda.is_available():
        print(
            "[Mode2] FATAL: --require_cuda set but torch.cuda.is_available() "
            "is False on this host. Aborting before any compute starts."
        )
        sys.exit(3)
    if getattr(args, "require_cuda", False):
        try:
            print(
                f"[Mode2] CUDA device verified: {torch.cuda.get_device_name(0)} "
                f"(CUDA={torch.version.cuda}, torch={torch.__version__})"
            )
        except Exception:
            pass

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kg_dir = Path(args.kg_dir)
    stage3c_dir = Path(args.stage3c_dir)

    ablation_label = (
        ("uniform_temp" if args.uniform_temp else "sdm_temp")
        + ("_frozen_kg" if args.freeze_kg else "_live_kg")
        + (f"_lkg{args.lambda_kg:.2f}")
    )
    print(f"[Mode2] === Stage 5 Mode 2 / C3 / SDGL ===")
    print(f"[Mode2] device={device}  ablation={ablation_label}")
    print(f"[Mode2] output_dir={out_dir}")
    print(
        f"[Mode2] mode2_version={args.mode2_version}  "
        f"checkpoint_selection_metric={args.checkpoint_selection_metric}"
    )
    print(
        "[Mode2] LGG warm-start mode="
        + (
            "proxy-allowed (--allow_proxy_lgg_warmstart)"
            if args.allow_proxy_lgg_warmstart
            else "strict-state_dict (default)"
        )
    )

    if args.full_mode2_detector:
        print(
            "[Mode2][Full] Mainline detector path active: image batches + Faster R-CNN "
            "fine-tuning run on this device."
        )
        print(
            f"[Mode2][Full] warmup_epochs={int(args.det_freeze_backbone_epochs)}  "
            f"lambda_anchor={float(args.lambda_anchor):.4f}"
        )
    else:
        print(
            "[Mode2][Proxy] Legacy proxy path active: training runs on precomputed "
            "governance CSV rows, not detector image batches."
        )
        print(
            "[Mode2][Proxy] CPU-dominant by design; detector_checkpoint/data_root are "
            "metadata only in this path and do not trigger full detector training."
        )
        if device.type == "cuda":
            print(
                "[Mode2][Proxy] Low GPU utilization is expected here even when "
                "device=cuda."
            )

    # SDM prior is required in both proxy and full-detector paths.
    sdm_norm = load_sdm_norm(kg_dir)

    if args.full_mode2_detector:
        run_full_mode2_detector(
            args=args,
            sdm_norm=sdm_norm,
            out_dir=out_dir,
            device=device,
        )
        return

    # ── Load KG graph (proxy path only) ──────────────────────────────────────
    node_feats, edge_index, edge_types = load_kg_graph(kg_dir)

    # ── Build KG encoder ──────────────────────────────────────────────────────
    node_feat_dim = node_feats.shape[1]
    encoder_hidden_dim = 64
    embedding_dim = 64
    encoder = LGovNetEncoder(
        node_feat_dim=node_feat_dim,
        hidden_dim=encoder_hidden_dim,
        embedding_dim=embedding_dim,
        num_edge_types=5,
    ).to(device)
    print(f"[Mode2] KG encoder built: node_feat_dim={node_feat_dim}")

    reclassifier = SDGLReClassifier(
        embedding_dim=embedding_dim,
        hidden_dim=64,
    ).to(device)
    print(
        f"[Mode2] SDGL re-classifier built: input_dim={len(LESION_CLASSES) + embedding_dim}"
    )

    # ── Load LGG ─────────────────────────────────────────────────────────────
    gate = load_lgg_from_pkl(
        Path(args.lgg_checkpoint),
        device,
        require_state_dict=not args.allow_proxy_lgg_warmstart,
    )

    # ── SDGL loss ─────────────────────────────────────────────────────────────
    sdgl_loss_fn = SDGLLoss(
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        lambda3=args.lambda3,
        lambda_kg=args.lambda_kg,
        eta=1.0,
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    trainable_params = list(gate.parameters()) + list(reclassifier.parameters())
    if not args.freeze_kg:
        trainable_params += list(encoder.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    optimizer_param_ids = {
        id(param) for group in optimizer.param_groups for param in group["params"]
    }
    missing_reclassifier = [
        name
        for name, param in reclassifier.named_parameters()
        if id(param) not in optimizer_param_ids
    ]
    if missing_reclassifier:
        raise RuntimeError(
            "Re-classifier parameters missing from optimizer: "
            + ", ".join(missing_reclassifier)
        )

    # ── Load governance features ───────────────────────────────────────────────
    train_csv = stage3c_dir / "governance_features_train.csv"
    val_csv = stage3c_dir / "governance_features_valid.csv"

    raw_train = load_governance_csv(train_csv)
    raw_val = load_governance_csv(val_csv)

    train_records = [
        r for row in raw_train if (r := parse_governance_row(row)) is not None
    ]
    val_records = [r for row in raw_val if (r := parse_governance_row(row)) is not None]

    print(
        f"[Mode2] Train records (R1–R3): {len(train_records)} | Val: {len(val_records)}"
    )

    if args.smoke_test:
        train_records = train_records[:64]
        val_records = val_records[:32]
        args.epochs = 1
        print("[Mode2] SMOKE TEST: 64 train / 32 val / 1 epoch")

    # ── Gumbel temperature schedule ────────────────────────────────────────────
    tau_values = [
        args.tau_start
        + (args.tau_end - args.tau_start) * (epoch / max(args.epochs - 1, 1))
        for epoch in range(args.epochs)
    ]

    # ── Training ───────────────────────────────────────────────────────────────
    log_rows = []
    best_val_acc = -1.0
    best_epoch = 0
    # Phase-1.5 proxy C3 metric: lower SDM-weighted misclass cost = better.
    # Tracked independently of best_val_acc so existing C2/gate selection
    # behavior is unchanged.
    best_reclass_sdm_misclass = float("inf")
    best_reclass_epoch = 0
    metrics = {}  # holds last-epoch train metrics (incl. sdgl_has_gradient guard)
    reclass_metrics: Dict = {  # holds last-epoch reclassifier metrics
        "reclass_acc": 0.0,
        "reclass_sdm_cost_mean": 0.0,
        "reclass_sdm_cost_misclass": 0.0,
        "reclass_n_misclass": 0,
        "reclass_per_class_acc": [],
    }

    for epoch in range(args.epochs):
        tau = tau_values[epoch]
        t0 = time.time()

        metrics = train_one_epoch(
            encoder=encoder,
            gate=gate,
            reclassifier=reclassifier,
            sdgl_loss_fn=sdgl_loss_fn,
            optimizer=optimizer,
            train_records=train_records,
            node_feats=node_feats,
            edge_index=edge_index,
            edge_types=edge_types,
            sdm_norm=sdm_norm,
            device=device,
            batch_size=args.batch_size,
            lambda_gate=args.lambda_gate,
            uniform_temp=args.uniform_temp,
            freeze_kg=args.freeze_kg,
            epoch=epoch,
            tau_gumbel=tau,
        )

        val_acc = eval_gate_accuracy(
            encoder, gate, val_records, node_feats, edge_index, edge_types, device
        )
        reclass_metrics = eval_reclassifier(
            encoder,
            reclassifier,
            val_records,
            node_feats,
            edge_index,
            edge_types,
            sdm_norm,
            device,
        )

        elapsed = time.time() - t0
        print(
            f"[Mode2] Epoch {epoch + 1:3d}/{args.epochs}"
            f"  loss={metrics['loss']:.4f}"
            f"  l_sdgl={metrics['l_sdgl']:.4f}"
            f"  l_kg={metrics['l_kg']:.4f}"
            f"  l_gate={metrics['l_gate']:.4f}"
            f"  val_gate_acc={val_acc:.3f}"
            f"  rcl_acc={reclass_metrics['reclass_acc']:.3f}"
            f"  rcl_sdm_misc={reclass_metrics['reclass_sdm_cost_misclass']:.4f}"
            f"  τ={tau:.3f}"
            f"  {elapsed:.1f}s"
        )

        row = {
            "epoch": epoch + 1,
            "loss": round(metrics["loss"], 6),
            "l_sdgl": round(metrics["l_sdgl"], 6),
            "l_kg": round(metrics["l_kg"], 6),
            "l_gate": round(metrics["l_gate"], 6),
            "val_gate_acc": round(val_acc, 4),
            "reclass_acc": reclass_metrics["reclass_acc"],
            "reclass_sdm_cost_mean": reclass_metrics["reclass_sdm_cost_mean"],
            "reclass_sdm_cost_misclass": reclass_metrics["reclass_sdm_cost_misclass"],
            "reclass_n_misclass": reclass_metrics["reclass_n_misclass"],
            "tau_gumbel": round(tau, 4),
            "sdgl_has_gradient": bool(metrics["sdgl_has_gradient"]),
            "reclassifier_has_gradient": bool(metrics["reclassifier_has_gradient"]),
            "ablation": ablation_label,
        }
        log_rows.append(row)
        # Track best Phase-1.5 proxy C3 checkpoint (lower misclass cost is better)
        cur_misclass = reclass_metrics["reclass_sdm_cost_misclass"]
        if (
            reclass_metrics["reclass_n_misclass"] > 0
            and cur_misclass < best_reclass_sdm_misclass
        ):
            best_reclass_sdm_misclass = cur_misclass
            best_reclass_epoch = epoch + 1

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(
                {
                    "epoch": epoch + 1,
                    "encoder_state_dict": encoder.state_dict(),
                    "gate_state_dict": gate.state_dict(),
                    "reclassifier_state_dict": reclassifier.state_dict(),
                    "val_gate_acc": val_acc,
                    "ablation": ablation_label,
                    "args": vars(args),
                },
                out_dir / "mode2_best.pt",
            )

    # ── Save final checkpoint ─────────────────────────────────────────────────
    torch.save(
        {
            "epoch": args.epochs,
            "encoder_state_dict": encoder.state_dict(),
            "gate_state_dict": gate.state_dict(),
            "reclassifier_state_dict": reclassifier.state_dict(),
            "val_gate_acc": val_acc,
            "ablation": ablation_label,
            "args": vars(args),
        },
        out_dir / "mode2_final.pt",
    )

    # ── Save training log ─────────────────────────────────────────────────────
    log_path = out_dir / "training_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        if log_rows:
            writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            writer.writeheader()
            writer.writerows(log_rows)

    # ── SDGL gradient-flow guard (Phase-1 limitation check) ───────────────────
    sdgl_has_grad = bool(metrics.get("sdgl_has_gradient", False))
    reclassifier_has_grad = bool(metrics.get("reclassifier_has_gradient", False))
    if not sdgl_has_grad:
        print(
            "\n[Mode2] ==================== SDGL GRADIENT WARNING ====================\n"
            "[Mode2]  L_SDGL still has NO gradient path after Option B wiring.\n"
            "[Mode2]  This run is invalid for C3 because the SDGL branch is not\n"
            "[Mode2]  reaching any trainable parameters. Check re-classifier usage\n"
            "[Mode2]  and any accidental detach in the log_probs path.\n"
            "[Mode2] ==============================================================="
        )
    if sdgl_has_grad and not reclassifier_has_grad:
        print(
            "\n[Mode2] ================= RECLASSIFIER GRAD WARNING =================\n"
            "[Mode2]  L_SDGL carries a gradient, but the re-classifier never saw\n"
            "[Mode2]  a non-zero parameter gradient in this epoch. The optimizer\n"
            "[Mode2]  wiring or forward path is still suspect. Do not trust this\n"
            "[Mode2]  run for Option B ablation until this is resolved.\n"
            "[Mode2] ==============================================================="
        )

    # ── Save config + summary ─────────────────────────────────────────────────
    summary = {
        # Phase-1.5 PROXY path: SDGL trains a re-classifier head; detector frozen.
        # This run DOES NOT satisfy C3 conditional rule (i) — that requires
        # --full_mode2_detector (see run_full_mode2_detector). The explicit
        # mode2_path field lets downstream audits separate proxy vs full runs
        # without inferring from absence of keys.
        "mode2_path": "proxy_reclassifier",
        # B2 provenance contract (Mode2 checklist Mei 31_claude.md): canonical
        # scope label distinct from legacy mode2_path, source provenance to
        # MODE1 LOCKED, and baseline reference. Downstream audits MUST read
        # these to determine claim eligibility, NOT the legacy mode2_path.
        "mode2_path_canonical": getattr(args, "_mode2_path", "proxy"),
        "mode2_scope": getattr(args, "_mode2_scope", "proxy_reclassifier"),
        "source_mode1_lock_file": args.mode1_lock_file,
        "source_mode1_run_root": args.mode1_run_root,
        "source_stage3c_dir": args.stage3c_dir,
        "source_lgg_checkpoint": args.lgg_checkpoint,
        "b3_lgg_torch_state_dict_loaded": bool(
            (getattr(args, "_lgg_probe", {}) or {}).get("has_state_dict")
            and (getattr(args, "_lgg_probe", {}) or {}).get("key_used")
            == "B3_LGG_TORCH"
        ),
        "baseline_audit": args.baseline_audit,
        "baseline_mAP_50": (
            args._baseline_map50_resolved
            if getattr(args, "_baseline_map50_resolved", None) is not None
            else args.expected_baseline_map50
        ),
        "kg_warmstart_source": "random_init",
        "allow_detached_lgg": bool(args.allow_detached_lgg),
        "full_mode2_detector": False,
        "c3_condition_i_met": False,
        "ablation": ablation_label,
        "best_epoch": best_epoch,
        "best_val_gate_acc": round(best_val_acc, 4),
        "final_val_gate_acc": round(val_acc, 4) if log_rows else 0.0,
        # Phase-1.5 proxy C3 metrics (H2-aligned; compare ablations HERE,
        # NOT on val_gate_acc which sees only the gate, not the SDGL branch).
        # gt_class leakage fix (2026-05-28): parse_governance_row now EXCLUDES
        # rows without a valid IoU>=0.5 gt_class instead of falling back to
        # majority_class, and Stage 3C writes gt_class="" for unresolved boxes.
        # reclass_acc is therefore measured only on genuinely GT-matched rows.
        # Still regenerate governance_features_*.csv with the fixed Stage 3C
        # (audit_stage3c_governance_targets.py) before trusting these numbers.
        "best_reclass_epoch": best_reclass_epoch,
        "best_reclass_sdm_cost_misclass": (
            round(best_reclass_sdm_misclass, 6)
            if best_reclass_sdm_misclass != float("inf")
            else None
        ),
        "final_reclass_acc": reclass_metrics["reclass_acc"] if log_rows else 0.0,
        "final_reclass_sdm_cost_mean": (
            reclass_metrics["reclass_sdm_cost_mean"] if log_rows else 0.0
        ),
        "final_reclass_sdm_cost_misclass": (
            reclass_metrics["reclass_sdm_cost_misclass"] if log_rows else 0.0
        ),
        "final_reclass_per_class_acc": reclass_metrics.get("reclass_per_class_acc", []),
        "epochs": args.epochs,
        "uniform_temp": args.uniform_temp,
        "freeze_kg": args.freeze_kg,
        "lambda_kg": args.lambda_kg,
        "lambda1": args.lambda1,
        "lambda2": args.lambda2,
        "lambda3": args.lambda3,
        "lambda_gate": args.lambda_gate,
        "lr": args.lr,
        "seed": args.seed,
        "train_records": len(train_records),
        "val_records": len(val_records),
        "sdgl_has_gradient": sdgl_has_grad,
        "reclassifier_has_gradient": reclassifier_has_grad,
        "c3_claim_eligible": (
            best_val_acc > 0.5
            and not args.smoke_test
            and sdgl_has_grad
            and reclassifier_has_grad
        ),
        # Phase-1.5 proxy validation: SDGL trains the re-classifier AND its
        # output is measurable on an H2-aligned metric. This is the gate
        # for "C3 proxy claim" — full C3 still requires Phase-2 detector.
        "c3_proxy_metric_available": (
            sdgl_has_grad
            and reclassifier_has_grad
            and best_reclass_sdm_misclass != float("inf")
        ),
    }
    with open(out_dir / "mode2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Mode2] === Training complete ===")
    print(f"[Mode2] Best val_gate_acc={best_val_acc:.4f} at epoch {best_epoch}")
    if best_reclass_sdm_misclass != float("inf"):
        print(
            f"[Mode2] Best reclass_sdm_cost_misclass={best_reclass_sdm_misclass:.4f}"
            f" at epoch {best_reclass_epoch}  (lower = better; H2-aligned proxy)"
        )
    else:
        print(
            "[Mode2] reclass_sdm_cost_misclass: never improved (no misclassifications, or 0 epochs)."
        )
    print(f"[Mode2] Ablation: {ablation_label}")
    print(f"[Mode2] C3 claim eligible: {summary['c3_claim_eligible']}")
    print(f"[Mode2] C3 proxy metric available: {summary['c3_proxy_metric_available']}")
    print(f"[Mode2] Checkpoints: {out_dir}/mode2_best.pt, mode2_final.pt")
    print(f"[Mode2] Log: {log_path}")
    print(
        f"\n[Mode2] C3 conditional claim rule (§C3 architecture spec):\n"
        f"  (i)  This run: ablation={ablation_label}  ← log saved\n"
        f"  (ii) --uniform_temp ablation required for full C3 claim\n"
        f"  (iii) --freeze_kg ablation required for full C3 claim\n"
        f"  (iv) --lambda_kg 0.0 ablation required for full C3 claim"
    )


if __name__ == "__main__":
    main()
