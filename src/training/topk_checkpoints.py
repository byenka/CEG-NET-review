"""topk_checkpoints.py — Mode 2 v2.1 top-k checkpoint manager.

Replaces the legacy "best by single composite score" + "final" pattern with a
small set of best-by-X candidates plus an always-overwritten latest.pt for
recovery. Rationale (from 1-Juni audit): the val_loss / single-composite
criterion can pick a checkpoint that is dominated on clinical-relevant axes
(mAP_medium, MA_AP50) by an earlier or later epoch. Preserving top-k
candidates lets the audit-based selector (tools/select_final_by_audit.py)
pick the final winner from full external audit, not from training-time val.

Contract:
    manager = TopKCheckpointManager(
        out_dir,
        criteria=[
            ("composite",  "lower"),    # baseline-centered composite score
            ("map50",      "higher"),   # val mAP_50
            ("map_medium", "higher"),   # val mAP_medium
            ("ma_ap50",    "higher"),   # val MA AP_50 (per-class)
        ],
    )

    # per epoch, after computing val metrics + composite:
    manager.update(
        epoch=epoch,
        metrics={
            "composite":  composite_score,
            "map50":      val_metrics["mAP_50"],
            "map_medium": val_metrics["mAP_medium"],
            "ma_ap50":    val_metrics["MA_AP50"],
        },
        payload_builder=lambda: {  # built only when actually saving
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            ...
        },
    )

    # at the end (every epoch is fine; cheap):
    manager.save_latest(payload_builder=...)

    # produces:
    #   out_dir/best_by_composite.pt
    #   out_dir/best_by_map50.pt
    #   out_dir/best_by_map_medium.pt
    #   out_dir/best_by_ma_ap50.pt
    #   out_dir/latest.pt
    #   out_dir/topk_candidates_index.json   (which epoch is in which file)
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch

# (metric_name, direction) where direction in {"lower", "higher"}.
Criterion = Tuple[str, str]


def _better(direction: str, new: float, current: float) -> bool:
    if direction == "lower":
        return new < current
    if direction == "higher":
        return new > current
    raise ValueError(f"Unknown direction: {direction!r}")


def _worst(direction: str) -> float:
    return math.inf if direction == "lower" else -math.inf


class TopKCheckpointManager:
    """Maintain best-by-X checkpoints + latest.pt for Mode 2 training runs.

    The manager does NOT build the payload itself — it asks the caller for it
    only when actually saving, so per-epoch metric updates that do not improve
    any criterion are O(1) (no state_dict copy).

    Attributes:
        out_dir: directory under which all candidate files are written.
        criteria: ordered list of ``(name, direction)`` to track.
    """

    def __init__(
        self,
        out_dir: Path,
        criteria: Iterable[Criterion] = (
            ("composite", "lower"),
            ("map50", "higher"),
            ("map_medium", "higher"),
            ("ma_ap50", "higher"),
        ),
        latest_filename: str = "latest.pt",
        index_filename: str = "topk_candidates_index.json",
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.criteria: List[Criterion] = [(str(n), str(d)) for n, d in criteria]
        self.latest_filename = latest_filename
        self.index_filename = index_filename
        self._best: Dict[str, float] = {
            n: _worst(d) for n, d in self.criteria
        }
        self._best_epoch: Dict[str, Optional[int]] = {n: None for n, _ in self.criteria}
        self._best_metrics: Dict[str, Dict[str, float]] = {n: {} for n, _ in self.criteria}

    # ------------------------------------------------------------------
    # Filename helpers
    # ------------------------------------------------------------------
    def path_for(self, criterion_name: str) -> Path:
        return self.out_dir / f"best_by_{criterion_name}.pt"

    @property
    def latest_path(self) -> Path:
        return self.out_dir / self.latest_filename

    @property
    def index_path(self) -> Path:
        return self.out_dir / self.index_filename

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------
    def update(
        self,
        epoch: int,
        metrics: Dict[str, float],
        payload_builder: Callable[[], Dict[str, Any]],
    ) -> List[str]:
        """Check each criterion; save candidates that improved.

        Returns the list of criteria whose best was updated in this call.
        """
        updated: List[str] = []
        cached_payload: Optional[Dict[str, Any]] = None
        for name, direction in self.criteria:
            if name not in metrics:
                continue
            val = float(metrics[name])
            if not math.isfinite(val):
                continue
            if _better(direction, val, self._best[name]):
                if cached_payload is None:
                    cached_payload = dict(payload_builder())
                    # tag the payload with the metric snapshot to ease audit later.
                    cached_payload.setdefault("epoch", int(epoch))
                    cached_payload.setdefault("topk_metrics_snapshot", {
                        k: float(v) for k, v in metrics.items()
                        if isinstance(v, (int, float)) and math.isfinite(float(v))
                    })
                self._best[name] = val
                self._best_epoch[name] = int(epoch)
                self._best_metrics[name] = dict(cached_payload.get("topk_metrics_snapshot", {}))
                _safe_save(cached_payload, self.path_for(name))
                updated.append(name)
        self._write_index()
        return updated

    def save_latest(
        self,
        payload_builder: Callable[[], Dict[str, Any]],
    ) -> None:
        """Always-overwrite latest.pt for recovery."""
        payload = dict(payload_builder())
        _safe_save(payload, self.latest_path)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def candidates(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for name, direction in self.criteria:
            p = self.path_for(name)
            if p.exists():
                out.append({
                    "criterion": name,
                    "direction": direction,
                    "path": str(p),
                    "best_value": self._best[name],
                    "best_epoch": self._best_epoch[name],
                    "snapshot_metrics": self._best_metrics[name],
                })
        if self.latest_path.exists():
            out.append({
                "criterion": "latest",
                "direction": "n/a",
                "path": str(self.latest_path),
                "best_value": None,
                "best_epoch": None,
                "snapshot_metrics": {},
            })
        return out

    def _write_index(self) -> None:
        index = {
            "out_dir": str(self.out_dir),
            "criteria": [
                {
                    "name": n,
                    "direction": d,
                    "best_value": self._best[n],
                    "best_epoch": self._best_epoch[n],
                    "snapshot_metrics": self._best_metrics[n],
                    "path": str(self.path_for(n)),
                    "exists": self.path_for(n).exists(),
                }
                for n, d in self.criteria
            ],
            "latest": {
                "path": str(self.latest_path),
                "exists": self.latest_path.exists(),
            },
        }
        self.index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Disk pruning (called after audit-based selection picks the winner)
    # ------------------------------------------------------------------
    def prune_losers(self, keep_paths: Iterable[Path]) -> List[str]:
        """Delete candidates not in ``keep_paths``. latest.pt is always kept.

        Returns list of deleted absolute paths (as strings).
        """
        keep_resolved = {str(Path(p).resolve()) for p in keep_paths}
        keep_resolved.add(str(self.latest_path.resolve()))
        deleted: List[str] = []
        for name, _ in self.criteria:
            p = self.path_for(name)
            if not p.exists():
                continue
            if str(p.resolve()) in keep_resolved:
                continue
            try:
                p.unlink()
                deleted.append(str(p.resolve()))
            except OSError:
                continue
        self._write_index()
        return deleted


def _safe_save(payload: Dict[str, Any], path: Path) -> None:
    """torch.save with atomic-ish replace to avoid corrupt files on interruption."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
