

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

# ── Project root guard ────────────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, "..")
if _root not in sys.path:
    sys.path.insert(0, _root)

# ── Constants ──────────────────────────────────────────────────────────────────

#: All valid detector mode strings.
DETECTOR_MODES: frozenset = frozenset(
    {
        "stub",
        "single_head_real",
        "multi_head_sequential",
        "cached_predictions",
    }
)

#: Default mode — always safe, no GPU required.
DEFAULT_MODE: str = "stub"

#: Modes that require a real backbone checkpoint (not yet implemented).
_REAL_BACKBONE_MODES: frozenset = frozenset({"single_head_real"})

#: Modes that need a cache directory.
_CACHE_REQUIRED_MODES: frozenset = frozenset({"cached_predictions"})


# ══════════════════════════════════════════════════════════════════════════════
# Mode resolver
# ══════════════════════════════════════════════════════════════════════════════


def get_detector_mode(cfg: Optional[dict] = None) -> str:

    mode: str = DEFAULT_MODE

    # Environment variable override
    env_mode = os.environ.get("LGOVNET_DETECTOR_MODE", "").strip()
    if env_mode:
        mode = env_mode

    # Config dict override (highest priority)
    if cfg:
        cfg_mode = str(cfg.get("detector_mode", "")).strip()
        if cfg_mode:
            mode = cfg_mode

    if mode not in DETECTOR_MODES:
        raise ValueError(
            f"Unknown detector_mode: {mode!r}. "
            f"Valid modes: {sorted(DETECTOR_MODES)}"
        )
    return mode


# ══════════════════════════════════════════════════════════════════════════════
# Unified dispatcher
# ══════════════════════════════════════════════════════════════════════════════


def get_detector_outputs(
    image_ids: List[str],
    mode: Optional[str] = None,
    cfg: Optional[dict] = None,
    bboxes_per_image: int = 4,
    seed: int = 42,
    cache_dir: Optional[str] = None,
    use_amp: bool = False,
    micro_batch_size: int = 2,
    sequential_heads: bool = True,
) -> Dict[str, List[dict]]:

    resolved_mode = mode or get_detector_mode(cfg)

    if resolved_mode == "stub":
        return run_detector_stub(
            image_ids, bboxes_per_image=bboxes_per_image, seed=seed
        )

    if resolved_mode == "single_head_real":
        return run_single_head_real(image_ids, cfg=cfg, use_amp=use_amp)

    if resolved_mode == "multi_head_sequential":
        return run_multi_head_sequential(
            image_ids,
            cfg=cfg,
            use_amp=use_amp,
            micro_batch_size=micro_batch_size,
            sequential_heads=sequential_heads,
        )

    if resolved_mode == "cached_predictions":
        return load_cached_predictions(image_ids, cache_dir=cache_dir)

    raise ValueError(f"Unhandled detector_mode: {resolved_mode!r}")


# ══════════════════════════════════════════════════════════════════════════════
# Backend implementations
# ══════════════════════════════════════════════════════════════════════════════


def run_detector_stub(
    image_ids: List[str],
    bboxes_per_image: int = 4,
    seed: int = 42,
) -> Dict[str, List[dict]]:

    from lgovnet.src.train_step import detector_stub

    return detector_stub(image_ids, bboxes_per_image=bboxes_per_image, seed=seed)


def run_single_head_real(
    image_ids: List[str],
    cfg: Optional[dict] = None,
    use_amp: bool = False,
) -> Dict[str, List[dict]]:

    raise NotImplementedError(
        "single_head_real backend is not yet implemented. "
        "This is Phase 15 scope: integrate a trained single-class detector checkpoint. "
        "Use detector_mode=stub for smoke tests, or "
        "detector_mode=cached_predictions if pre-computed predictions are available."
    )


def run_multi_head_sequential(
    image_ids: List[str],
    cfg: Optional[dict] = None,
    use_amp: bool = False,
    micro_batch_size: int = 2,
    sequential_heads: bool = True,
) -> Dict[str, List[dict]]:

    raise NotImplementedError(
        "multi_head_sequential backend is not yet implemented. "
        "This is Phase 15/19 scope: integrate trained per-head checkpoints. "
        "RTX 4090 safety design is documented in this function's docstring. "
        "Use detector_mode=stub or detector_mode=cached_predictions for now."
    )


def load_cached_predictions(
    image_ids: List[str],
    cache_dir: Optional[str] = None,
) -> Dict[str, List[dict]]:

    if cache_dir is None:
        raise FileNotFoundError(
            "cache_dir is required for cached_predictions mode. "
            "Set --cache_dir or add 'cache_dir' to experiment config."
        )

    cache_path = Path(cache_dir)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache directory not found: {cache_path}. "
            f"Generate predictions first and save to this directory."
        )

    results: Dict[str, List[dict]] = {}
    n_loaded = 0
    n_missing = 0

    for image_id in image_ids:
        # Support both bare stems and path-like IDs
        stem = Path(image_id).stem
        json_path = cache_path / f"{stem}.json"

        if not json_path.exists():
            warnings.warn(
                f"[cached_predictions] Cache file not found: {json_path}. "
                f"Image {image_id} will have empty detections.",
                UserWarning,
                stacklevel=2,
            )
            results[image_id] = []
            n_missing += 1
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                boxes = json.load(f)
            if not isinstance(boxes, list):
                raise ValueError(
                    f"Expected list of DetectorBox dicts, got {type(boxes)}"
                )
            results[image_id] = boxes
            n_loaded += 1
        except (json.JSONDecodeError, ValueError) as exc:
            warnings.warn(
                f"[cached_predictions] Failed to load {json_path}: {exc}. "
                f"Image {image_id} will have empty detections.",
                UserWarning,
                stacklevel=2,
            )
            results[image_id] = []
            n_missing += 1

    print(
        f"  [detector_backend] cached_predictions: "
        f"{n_loaded} loaded, {n_missing} missing/malformed"
    )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Introspection helper — used by run_experiment.py for status reporting
# ══════════════════════════════════════════════════════════════════════════════


def describe_detector_mode(mode: str) -> dict:

    _desc = {
        "stub": {
            "implemented": True,
            "gpu_required": False,
            "safe_for_rtx4090": True,
            "description": (
                "Synthetic Dirichlet class_probs, fixed-position bboxes. "
                "No backbone. All detections → R4 when real GT present. "
                "Use for smoke tests and ablation."
            ),
        },
        "single_head_real": {
            "implemented": False,
            "gpu_required": True,
            "safe_for_rtx4090": True,
            "description": (
                "One trained single-class detector head. "
                "~2–4 GB VRAM with fp16. Phase 15 scope."
            ),
        },
        "multi_head_sequential": {
            "implemented": False,
            "gpu_required": True,
            "safe_for_rtx4090": True,  # ONLY when sequential_heads=True
            "description": (
                "Four heads run sequentially (one at a time) to respect VRAM limits. "
                "Recommended over full-parallel for RTX 4090 safety. Phase 15/19 scope."
            ),
        },
        "cached_predictions": {
            "implemented": True,
            "gpu_required": False,
            "safe_for_rtx4090": True,
            "description": (
                "Reads pre-computed DetectorBox JSON files from cache_dir. "
                "Skips backbone; only governance pipeline runs. "
                "Requires prior prediction generation step."
            ),
        },
    }
    info = _desc.get(mode, {})
    return {"mode": mode, **info}
