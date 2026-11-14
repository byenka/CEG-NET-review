#!/usr/bin/env python


from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List

import torch

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, "..")
if _root not in sys.path:
    sys.path.insert(0, _root)


def _make_synthetic_val_records(
    n: int = 32, n_classes: int = 4, seed: int = 0
) -> List[Dict]:

    from lgovnet.src.losses.sdgl import LESION_CLASSES

    g = torch.Generator().manual_seed(seed)
    records: List[Dict] = []
    for i in range(n):
        gt_idx = int(torch.randint(0, n_classes, (1,), generator=g).item())
        # Random-ish probs, NOT peaked on gt — guarantees some misclassifications.
        raw = torch.rand(n_classes, generator=g) + 0.05
        raw = raw / raw.sum()
        prob_dict = {c: float(raw[k].item()) for k, c in enumerate(LESION_CLASSES)}
        records.append(
            {
                "tier_idx": 0,  # R1; tier doesn't matter for reclassifier eval
                "majority_class": LESION_CLASSES[gt_idx],
                "gt_class_idx": gt_idx,
                "evidence": torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32),
                "ensemble_probs": prob_dict,
                "consensus": 0.5,
                "aps": 0.5,
            }
        )
    return records


def main() -> int:
    out_dir = Path(_root) / "outputs" / "smoke_reclassifier_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "smoke_report.json"

    report: Dict = {
        "test": "smoke_eval_reclassifier",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stages": {},
    }
    overall_pass = True

    # ── Stage 1: import ───────────────────────────────────────────────────────
    try:
        from tools.train_stage5_mode2 import (  # noqa: F401
            SDGLReClassifier,
            eval_reclassifier,
            load_kg_graph,
            load_sdm_norm,
        )
        from lgovnet.src.kg.encoder import LGovNetEncoder
        from lgovnet.src.losses.sdgl import LESION_CLASSES

        report["stages"]["import"] = {"status": "pass"}
    except Exception as e:  # noqa: BLE001
        report["stages"]["import"] = {
            "status": "fail",
            "error": repr(e),
            "trace": traceback.format_exc(),
        }
        overall_pass = False
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[smoke] FAIL at import. Report -> {log_path}")
        return 1

    device = torch.device("cpu")  # smoke runs on CPU; deterministic, no VRAM
    embedding_dim = 64

    # ── Stage 2: SDGLReClassifier forward shape check ─────────────────────────
    try:
        recl = SDGLReClassifier(embedding_dim=embedding_dim, hidden_dim=64).to(device)
        recl.eval()
        n_classes = len(LESION_CLASSES)
        p_ens = torch.tensor([0.25] * n_classes, dtype=torch.float32, device=device)
        e_pred = torch.zeros(embedding_dim, dtype=torch.float32, device=device)
        with torch.no_grad():
            out = recl(p_ens, e_pred)
        assert out.shape == (n_classes,), f"bad shape: {tuple(out.shape)}"
        assert torch.allclose(
            torch.exp(out).sum(), torch.tensor(1.0), atol=1e-4
        ), "log_softmax output does not sum to 1"
        report["stages"]["reclassifier_forward"] = {
            "status": "pass",
            "output_shape": list(out.shape),
            "sum_exp": float(torch.exp(out).sum().item()),
        }
    except Exception as e:  # noqa: BLE001
        report["stages"]["reclassifier_forward"] = {
            "status": "fail",
            "error": repr(e),
            "trace": traceback.format_exc(),
        }
        overall_pass = False

    # ── Stage 3: eval_reclassifier end-to-end with real encoder ───────────────
    try:
        kg_dir = Path(_root) / "outputs" / "stage3_audit"
        if not (kg_dir / "nodes.json").exists():
            raise FileNotFoundError(
                f"KG audit missing: {kg_dir}/nodes.json. Run STEP 6 first."
            )
        node_feats, edge_index, edge_types = load_kg_graph(kg_dir)
        sdm_norm = load_sdm_norm(kg_dir)

        encoder = LGovNetEncoder(
            node_feat_dim=node_feats.shape[1],
            hidden_dim=64,
            embedding_dim=embedding_dim,
            num_edge_types=5,
        ).to(device)
        encoder.eval()
        # Reuse Stage 2 reclassifier (untrained) — smoke is structural, not perf.
        val_records = _make_synthetic_val_records(n=32, n_classes=n_classes, seed=42)

        metrics = eval_reclassifier(
            encoder=encoder,
            reclassifier=recl,
            val_records=val_records,
            node_feats=node_feats,
            edge_index=edge_index,
            edge_types=edge_types,
            sdm_norm=sdm_norm,
            device=device,
        )

        expected_keys = {
            "reclass_acc",
            "reclass_sdm_cost_mean",
            "reclass_sdm_cost_misclass",
            "reclass_n_misclass",
            "reclass_per_class_acc",
        }
        missing = expected_keys - set(metrics.keys())
        if missing:
            raise AssertionError(f"missing metric keys: {missing}")
        assert 0.0 <= metrics["reclass_acc"] <= 1.0, "reclass_acc out of [0,1]"
        assert metrics["reclass_sdm_cost_mean"] >= 0.0, "sdm_cost_mean < 0"
        assert metrics["reclass_sdm_cost_misclass"] >= 0.0, "sdm_cost_misclass < 0"
        assert 0 <= metrics["reclass_n_misclass"] <= len(val_records), (
            "reclass_n_misclass out of [0, n_val]"
        )
        assert len(metrics["reclass_per_class_acc"]) == n_classes, (
            f"per_class_acc len != {n_classes}"
        )

        # Soft check: with untrained reclassifier + random probs, we expect at
        # least some misclassifications. If 0, the smoke's negative path isn't
        # exercised — flag as warning, not fail (untrained MLP can collapse).
        sdm_path_exercised = metrics["reclass_n_misclass"] > 0

        report["stages"]["eval_reclassifier_end_to_end"] = {
            "status": "pass",
            "n_val": len(val_records),
            "metrics": metrics,
            "sdm_cost_path_exercised": sdm_path_exercised,
        }
    except Exception as e:  # noqa: BLE001
        report["stages"]["eval_reclassifier_end_to_end"] = {
            "status": "fail",
            "error": repr(e),
            "trace": traceback.format_exc(),
        }
        overall_pass = False

    # ── Stage 4: c3_proxy_metric_available logic ──────────────────────────────
    try:
        # Mirrors main()'s summary gate without running training.
        sdgl_has_grad = True
        reclassifier_has_grad = True
        stage3 = report["stages"].get("eval_reclassifier_end_to_end", {})
        if stage3.get("status") != "pass":
            raise RuntimeError("eval_reclassifier stage did not pass; cannot gate")
        best_reclass_sdm_misclass = stage3["metrics"]["reclass_sdm_cost_misclass"]
        c3_proxy_metric_available = (
            sdgl_has_grad
            and reclassifier_has_grad
            and best_reclass_sdm_misclass != float("inf")
        )
        report["stages"]["c3_proxy_metric_gate"] = {
            "status": "pass",
            "c3_proxy_metric_available": bool(c3_proxy_metric_available),
        }
    except Exception as e:  # noqa: BLE001
        report["stages"]["c3_proxy_metric_gate"] = {
            "status": "fail",
            "error": repr(e),
            "trace": traceback.format_exc(),
        }
        overall_pass = False

    report["overall"] = "pass" if overall_pass else "fail"
    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[smoke] overall={report['overall']}  report -> {log_path}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())