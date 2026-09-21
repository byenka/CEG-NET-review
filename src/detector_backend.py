"""
detector_backend.py
Detector backend interface for LGov-Net — Phase 17R.

Provides a single entry point  get_detector_outputs()  that selects the
appropriate backend implementation based on  detector_mode.

Four detector modes are defined:

  stub  (default / smoke-test)
  ────────────────────────────
  Uses detector_stub() from train_step.py.
  Generates synthetic DetectorBox dicts with random Dirichlet class_probs.
  Bboxes are fixed-position (not spatially anchored to image content).
  Routing will assign all boxes R4 because IoU with real GT = 0.
  CPU only, no GPU required.
  USE FOR: smoke tests, ablation, architecture validation.

  single_head_real  (Phase 15 placeholder)
  ─────────────────────────────────────────
  Reserved for a real single-class detector backbone (e.g. Faster R-CNN
  trained on one lesion class).  Currently raises NotImplementedError with
  a clear Phase 15 scope message.
  USE FOR: when a single trained head is available for integration testing.

  multi_head_sequential  (RTX 4090 safe)
  ───────────────────────────────────────
  Intended for four separate single-class detector heads run one at a time.
  Each head processes the image batch, results are merged on CPU.
  This avoids forcing all four heads into VRAM simultaneously.
  On RTX 4090 (24 GB) this is safe even for large images.
  Sequential mode is the FIRST-CLASS recommended mode for real-data runs
  until ablation validates full parallel loading is safe.
  Currently: loads predictions from per-head cache files if available,
  otherwise falls back to stub output per head.
  USE FOR: real-data experiments when backbone checkpoints exist.

  cached_predictions  (inference-only / governance-only)
  ───────────────────────────────────────────────────────
  Loads pre-computed detector predictions from a JSON/CSV cache directory.
  Skips all backbone computation.  The governance pipeline (evidence +
  routing + certificates) runs on the loaded predictions.
  USE FOR: re-running governance analysis without re-running the detector;
  debugging tier routing on a fixed set of predictions.

RTX 4090 24 GB compatibility notes
────────────────────────────────────
  - Full parallel multi-head (4 × ResNet-50 backbone) may approach VRAM
    limit for batch_size > 4 and image_size > 1024×1024.
  - Recommended safe config: micro_batch_size=2, sequential_heads=True.
  - AMP (fp16/bf16) reduces VRAM by ~40%; enable via use_amp=True in config.
  - grad_accum_steps allows effective larger batches without VRAM pressure.

WARNING: single_head_real and multi_head_sequential are NOT IMPLEMENTED in
this phase.  They will raise NotImplementedError when called.  This module
defines the *contract* so run_experiment.py can select a mode without
crashing at import time.  Implementation belongs in Phase 15/19.

Public API:
  DETECTOR_MODES            frozenset of valid mode strings
  get_detector_mode(cfg)    -> str  (resolve mode from config)
  get_detector_outputs(     -> Dict[str, List[dict]]
      image_ids, mode, ...)
  run_detector_stub(...)    -> Dict[str, List[dict]]
  run_single_head_real(...) -> Dict[str, List[dict]]  [NotImplemented]
  run_multi_head_sequential(...) -> Dict[str, List[dict]]  [NotImplemented]
  load_cached_predictions(...) -> Dict[str, List[dict]]
"""

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
    """Resolve the detector mode from a config dict.

    Priority:
      1. ``cfg["detector_mode"]`` if present and valid
      2. Environment variable  ``LGOVNET_DETECTOR_MODE`` if set
      3. DEFAULT_MODE ("stub")

    Args:
        cfg: Experiment or step config dict.  May be None.

    Returns:
        One of DETECTOR_MODES strings.

    Raises:
        ValueError: if the resolved mode is not in DETECTOR_MODES.
    """
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
    """Dispatch detector execution to the selected backend.

    Args:
        image_ids:        List of image identifier strings (stems or file paths).
        mode:             Detector mode string.  If None, resolved via get_detector_mode(cfg).
        cfg:              Experiment config dict for auto-mode resolution.
        bboxes_per_image: Number of detections per image (stub mode only).
        seed:             Random seed (stub mode only).
        cache_dir:        Path to prediction cache directory (cached_predictions mode).
        use_amp:          Use automatic mixed precision (multi_head_sequential only,
                          passed through to backbone if/when implemented).
        micro_batch_size: Per-head GPU batch size (multi_head_sequential, not yet live).
        sequential_heads: Run heads sequentially rather than in parallel
                          (multi_head_sequential, recommended for RTX 4090 safety).

    Returns:
        {image_id: [DetectorBox, ...]}  — one list per image_id.

    Raises:
        ValueError:          if mode is invalid.
        NotImplementedError: if mode requires a backbone not yet integrated.
        FileNotFoundError:   if cache_dir is required but missing.
    """
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
    """Backend: stub detector — synthetic random DetectorBox dicts.

    Delegates to detector_stub() from train_step.py.  All class_probs
    are drawn from Dirichlet(1,1,1,1) for maximal diversity.

    NOTE: Stuart bboxes have fixed positions that do NOT match XML GT
    annotations → all detections are routed R4 (Reject) when GT is present.
    This is CORRECT behaviour for the stub mode; it exercises the full
    pipeline without a trained backbone.

    Args:
        image_ids:        Image identifier strings.
        bboxes_per_image: Number of boxes per image.
        seed:             Reproducibility seed.

    Returns:
        {image_id: [DetectorBox, ...]}
    """
    from lgovnet.src.train_step import detector_stub

    return detector_stub(image_ids, bboxes_per_image=bboxes_per_image, seed=seed)


def run_single_head_real(
    image_ids: List[str],
    cfg: Optional[dict] = None,
    use_amp: bool = False,
) -> Dict[str, List[dict]]:
    """Backend: single real detector head for one lesion class.

    PHASE 15 SCOPE — NOT YET IMPLEMENTED.

    This function will:
      1. Load a checkpoint for one trained single-class detector
         (e.g. Faster R-CNN or YOLOv8 trained on MA class only).
      2. Run inference on the provided images.
      3. Return DetectorBox dicts with real bbox coordinates and class_probs.

    GPU memory note:  A single-head backbone (ResNet-50) uses ~2–4 GB VRAM
    with fp16.  Well within RTX 4090 24 GB budget.

    Args:
        image_ids: Image identifier strings.
        cfg:       Config dict; expected keys:
                   ``single_head_checkpoint`` — path to model checkpoint
                   ``single_head_class``      — which lesion class this head detects
                   ``image_dir``              — directory containing image files
        use_amp:   Enable fp16 inference for lower VRAM usage.

    Raises:
        NotImplementedError: Always, until Phase 15 implementation.
    """
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
    """Backend: four detector heads run sequentially for RTX 4090 safety.

    PHASE 15/19 SCOPE — NOT YET IMPLEMENTED.

    Design (when implemented):
      1. For each of the 4 lesion heads (MA, HE, Hard_EX, Soft_EX):
         a. Load head-specific checkpoint onto GPU.
         b. Run inference in micro-batches of ``micro_batch_size``.
         c. Move predictions to CPU and release GPU memory.
      2. Merge all four head outputs into DetectorBox format.
      3. Compute ensemble_probs by averaging head class_probs.

    RTX 4090 safety strategy (sequential_heads=True):
      - Only one head occupies GPU at a time → max VRAM ≈ single backbone.
      - With fp16 + batch_size=2: ~4–6 GB VRAM per head.
      - Inference time increases 4× vs parallel, but memory is safe.

    Full-parallel mode (sequential_heads=False):
      - All 4 heads on GPU simultaneously → ~16–24 GB VRAM.
      - Only attempt this on RTX 4090 with fp16 and batch_size=1.
      - NOT recommended unless sequential mode has been profiled first.

    CAUTION: Do NOT enable sequential_heads=False without profiling.
    Set sequential_heads: true in configs/experiment.yaml as the safe default.

    Args:
        image_ids:         Image identifier strings.
        cfg:               Config dict; expected keys:
                           ``head_checkpoints``   — {class: checkpoint_path}
                           ``image_dir``          — directory containing image files
                           ``sequential_heads``   — override (True = safe)
        use_amp:           Enable fp16 for lower VRAM.
        micro_batch_size:  Images per GPU forward pass per head.
        sequential_heads:  True (recommended) = run heads one at a time.

    Raises:
        NotImplementedError: Always, until Phase 15/19 implementation.
    """
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
    """Backend: load pre-computed detector predictions from disk.

    Reads one JSON file per image from cache_dir.  Each file must conform
    to the DetectorBox contract (bbox_id, ensemble_probs, head_predictions,
    assigned_zone).  Missing images or malformed files fall back to an empty
    box list with a warning.

    Expected file layout:
        <cache_dir>/
            <image_id>.json   — one DetectorBox list per image

    DetectorBox JSON format (one file = list of boxes):
        [
          {
            "bbox_id": "img001_b00",
            "assigned_zone": "Foveal",
            "ensemble_probs": {"MA": 0.6, "HE": 0.2, "Hard_EX": 0.1, "Soft_EX": 0.1},
            "head_predictions": [
              {"head": "MA", "bbox": [10, 10, 50, 50],
               "class_probs": {...}, "score": 0.9},
              ...
            ]
          },
          ...
        ]

    Args:
        image_ids: Image identifier strings (stems; .json extension is added).
        cache_dir: Directory containing per-image JSON prediction files.
                   If None, raises FileNotFoundError.

    Returns:
        {image_id: [DetectorBox, ...]}  — empty list if file missing/malformed.

    Raises:
        FileNotFoundError: if cache_dir is None or does not exist.
    """
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
    """Return a summary dict describing a detector mode for logging/reporting.

    Args:
        mode: One of DETECTOR_MODES.

    Returns:
        {
          "mode":         str,
          "implemented":  bool,
          "gpu_required": bool,
          "description":  str,
          "safe_for_rtx4090": bool,
        }
    """
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
