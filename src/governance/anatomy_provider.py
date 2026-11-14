

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# ── Constants ──────────────────────────────────────────────────────────────────

#: Valid confidence levels for od_confidence field.
OD_CONFIDENCE_LEVELS = ("high", "moderate", "low", "n/a")

#: Valid eye laterality values.
EYE_SIDE_VALUES = ("left", "right", "unknown")

#: When od_confidence is LOW or N/A, APS must be set to this neutral value.
#: Architecture v3: "APS = neutral (0.5) when AnatomyContext.od_confidence == low"
APS_NEUTRAL_ON_LOW_CONFIDENCE: float = 0.5

#: od_confidence levels that trigger the APS fail-safe.
_LOW_CONFIDENCE_LEVELS = frozenset({"low", "n/a"})


# ── AnatomyContext dataclass ───────────────────────────────────────────────────


@dataclass
class AnatomyContext:


    od_center: Optional[Tuple[float, float]]
    od_radius: Optional[float]
    od_confidence: str  # one of OD_CONFIDENCE_LEVELS
    fovea_center: Optional[Tuple[float, float]] = None
    vessel_map: Optional[Any] = None  # np.ndarray | None
    etdrs_zone_map: Optional[Any] = None  # np.ndarray | None
    eye_side: str = "unknown"
    provider_name: str = "unknown"
    provider_validation: Dict[str, Any] = field(default_factory=dict)
    image_id: str = ""

    def __post_init__(self) -> None:
        if self.od_confidence not in OD_CONFIDENCE_LEVELS:
            raise ValueError(
                f"od_confidence must be one of {OD_CONFIDENCE_LEVELS}, "
                f"got: {self.od_confidence!r}"
            )
        if self.eye_side not in EYE_SIDE_VALUES:
            raise ValueError(
                f"eye_side must be one of {EYE_SIDE_VALUES}, " f"got: {self.eye_side!r}"
            )

    @property
    def aps_should_be_neutral(self) -> bool:

        return self.od_confidence in _LOW_CONFIDENCE_LEVELS

    def to_dict(self) -> dict:

        return {
            "od_center": list(self.od_center) if self.od_center else None,
            "od_radius": self.od_radius,
            "od_confidence": self.od_confidence,
            "fovea_center": list(self.fovea_center) if self.fovea_center else None,
            "eye_side": self.eye_side,
            "provider_name": self.provider_name,
            "provider_validation": self.provider_validation,
            "image_id": self.image_id,
        }


# ── Abstract base ──────────────────────────────────────────────────────────────


class AnatomyProvider(ABC):


    @abstractmethod
    def __call__(self, image: Any, image_id: str = "") -> AnatomyContext:

        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:

        ...


# ── Concrete provider: Approximate (placeholder) ───────────────────────────────


class ApproximateAnatomyProvider(AnatomyProvider):


    _NAME = "ApproximateAnatomyProvider (geometry, od_confidence=low)"

    def __init__(
        self,
        dd_px: float = 45.0,
        disc_offset_dd: float = 3.5,
        auto_scale_dd: bool = True,
    ) -> None:
        self._dd_px = dd_px
        self._disc_offset_dd = disc_offset_dd
        self._auto_scale_dd = auto_scale_dd

    @property
    def provider_name(self) -> str:
        return self._NAME

    def __call__(self, image: Any, image_id: str = "") -> AnatomyContext:

        W, H = self._get_image_dims(image)

        dd = self._dd_px
        if self._auto_scale_dd and W > 0:
            # Scale calibrated 45px/512px ratio linearly to actual image width
            dd = self._dd_px * (W / 512.0)

        cx, cy = W / 2.0, H / 2.0
        disc_x = cx + self._disc_offset_dd * dd

        return AnatomyContext(
            od_center=(disc_x, cy),
            od_radius=dd / 2.0,
            od_confidence="low",
            fovea_center=(cx, cy),
            eye_side="unknown",
            provider_name=self._NAME,
            provider_validation={
                "method": "geometry_only",
                "dd_px_used": round(dd, 1),
                "auto_scale_dd": self._auto_scale_dd,
                "disc_offset_dd": self._disc_offset_dd,
                "note": (
                    "Right-eye assumption. Eye laterality ignored. "
                    "od_confidence=low forces APS=0.5 neutral in evidence computation."
                ),
            },
            image_id=image_id,
        )

    @staticmethod
    def _get_image_dims(image: Any) -> Tuple[int, int]:

        try:
            # PIL Image
            if hasattr(image, "size") and hasattr(image, "mode"):
                return int(image.size[0]), int(image.size[1])
        except Exception:
            pass
        try:
            import torch

            if isinstance(image, torch.Tensor):
                if image.ndim == 3:  # [C, H, W]
                    return int(image.shape[2]), int(image.shape[1])
                if image.ndim == 4:  # [B, C, H, W]
                    return int(image.shape[3]), int(image.shape[2])
        except ImportError:
            pass
        try:
            import numpy as np

            if isinstance(image, np.ndarray):
                if image.ndim == 3:  # [H, W, C]
                    return int(image.shape[1]), int(image.shape[0])
                if image.ndim == 2:  # [H, W]
                    return int(image.shape[1]), int(image.shape[0])
        except ImportError:
            pass
        warnings.warn(
            "[ApproximateAnatomyProvider] Could not determine image dimensions "
            "from image type. Falling back to 512×512.",
            UserWarning,
            stacklevel=3,
        )
        return 512, 512


# ── Concrete provider: Radial-Zone (Pilar 3, gap G3) ───────────────────────────


class RadialZoneAnatomyProvider(AnatomyProvider):


    _NAME = "RadialZoneAnatomyProvider (centre-based, od_confidence=moderate)"

    _BIN_RFOVEAL = 0.20
    _BIN_RPARAFOVEAL = 0.40
    _BIN_RVASCULAR = 0.65

    @property
    def provider_name(self) -> str:
        return self._NAME

    def __call__(self, image: Any, image_id: str = "") -> AnatomyContext:
        W, H = ApproximateAnatomyProvider._get_image_dims(image)
        cx, cy = W / 2.0, H / 2.0
        disc_x = cx + 0.30 * min(W, H) / 2.0  # right-eye heuristic placeholder
        return AnatomyContext(
            od_center=(disc_x, cy),
            od_radius=min(W, H) / 14.0,
            od_confidence="moderate",
            fovea_center=(cx, cy),
            eye_side="unknown",
            provider_name=self._NAME,
            provider_validation={
                "method": "radial_zone_no_od_detector",
                "image_w": W,
                "image_h": H,
                "note": (
                    "Centre-of-image proxy for macular region. "
                    "od_confidence='moderate' permits APS=Compat(class, radial_zone)."
                ),
            },
            image_id=image_id,
        )

    @classmethod
    def assign_radial_zone(
        cls, box_cx: float, box_cy: float, image_w: int, image_h: int
    ) -> str:

        cx, cy = image_w / 2.0, image_h / 2.0
        dx = (box_cx - cx) / (image_w / 2.0)
        dy = (box_cy - cy) / (image_h / 2.0)
        r = math.sqrt(dx * dx + dy * dy)
        if r < cls._BIN_RFOVEAL:
            return "Foveal"
        if r < cls._BIN_RPARAFOVEAL:
            return "Parafoveal"
        if r < cls._BIN_RVASCULAR:
            return "Vascular_Arcades"
        return "Peripheral"


# ── Concrete provider: External pretrained ─────────────────────────────────────


class ExternalPretrainedProvider(AnatomyProvider):


    _NAME = "ExternalPretrainedProvider"

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        input_size: int = 512,
        mask_threshold: float = 0.35,
        moderate_confidence_threshold: float = 0.55,
        high_confidence_threshold: float = 0.80,
        min_mask_area: int = 64,
    ) -> None:
        if input_size <= 0:
            raise ValueError(f"input_size must be > 0, got {input_size}")
        if not (0.0 < mask_threshold < 1.0):
            raise ValueError(f"mask_threshold must be in (0,1), got {mask_threshold}")
        if not (0.0 < moderate_confidence_threshold <= 1.0):
            raise ValueError(
                "moderate_confidence_threshold must be in (0,1], "
                f"got {moderate_confidence_threshold}"
            )
        if not (0.0 < high_confidence_threshold <= 1.0):
            raise ValueError(
                "high_confidence_threshold must be in (0,1], "
                f"got {high_confidence_threshold}"
            )
        if high_confidence_threshold < moderate_confidence_threshold:
            raise ValueError(
                "high_confidence_threshold must be >= moderate_confidence_threshold"
            )
        if min_mask_area <= 0:
            raise ValueError(f"min_mask_area must be > 0, got {min_mask_area}")

        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"External anatomy checkpoint not found: {ckpt}")

        self._checkpoint_path = ckpt
        self._device = device
        self._input_size = int(input_size)
        self._mask_threshold = float(mask_threshold)
        self._moderate_thr = float(moderate_confidence_threshold)
        self._high_thr = float(high_confidence_threshold)
        self._min_mask_area = int(min_mask_area)

        self._runtime_ready = False
        self._backend = "uninitialized"
        self._model = None
        self._onnx_session = None
        self._onnx_input_name = None

    @property
    def provider_name(self) -> str:
        if self._runtime_ready:
            return (
                f"{self._NAME}({self._backend}, " f"ckpt={self._checkpoint_path.name})"
            )
        return f"{self._NAME}(ckpt={self._checkpoint_path.name})"

    def __call__(self, image: Any, image_id: str = "") -> AnatomyContext:
        try:
            np_img, image_w, image_h = self._to_numpy_rgb(image)
        except Exception as exc:
            return self._fallback_context(
                image=image,
                image_id=image_id,
                reason=f"image_decode_failed: {exc}",
            )

        try:
            self._ensure_runtime()
            prob_map = self._predict_mask_probability(np_img)

            od_center, od_radius, conf_score, area_px = self._extract_disc_geometry(
                prob_map,
                image_w=image_w,
                image_h=image_h,
            )
            od_confidence = self._confidence_tier(conf_score, area_px)
            fovea_center = self._estimate_fovea_center(
                od_center,
                od_radius,
                image_w=image_w,
                image_h=image_h,
            )

            return AnatomyContext(
                od_center=od_center,
                od_radius=od_radius,
                od_confidence=od_confidence,
                fovea_center=fovea_center,
                eye_side="unknown",
                provider_name=self.provider_name,
                provider_validation={
                    "method": "external_pretrained_segmentation",
                    "checkpoint": str(self._checkpoint_path),
                    "backend": self._backend,
                    "session_providers": (
                        self._onnx_session.get_providers()
                        if self._onnx_session is not None
                        else None
                    ),
                    "input_size": self._input_size,
                    "mask_threshold": self._mask_threshold,
                    "mask_area_px": int(area_px),
                    "mask_confidence_mean": round(float(conf_score), 4),
                    "confidence_thresholds": {
                        "moderate": self._moderate_thr,
                        "high": self._high_thr,
                    },
                },
                image_id=image_id,
            )
        except Exception as exc:
            return self._fallback_context(
                image=image,
                image_id=image_id,
                reason=f"inference_failed: {exc}",
            )

    def _fallback_context(
        self, image: Any, image_id: str, reason: str
    ) -> AnatomyContext:
        warnings.warn(
            "[ExternalPretrainedProvider] Falling back to "
            f"ApproximateAnatomyProvider due to: {reason}",
            UserWarning,
            stacklevel=3,
        )
        fallback = ApproximateAnatomyProvider(auto_scale_dd=True)
        ctx = fallback(image, image_id=image_id)
        ctx.provider_name = f"{self._NAME}(fallback=approximate)"
        ctx.provider_validation.update(
            {
                "method": "external_pretrained_fallback",
                "checkpoint": str(self._checkpoint_path),
                "backend": self._backend,
                "reason": reason,
            }
        )
        return ctx

    def _ensure_runtime(self) -> None:
        if self._runtime_ready:
            return

        ckpt_str = str(self._checkpoint_path)
        suffix = self._checkpoint_path.suffix.lower()

        if suffix == ".onnx":
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError(
                    "ONNX checkpoint selected but onnxruntime is not installed"
                ) from exc

            providers = ["CPUExecutionProvider"]
            if self._device.startswith("cuda"):
                preload_dlls = getattr(ort, "preload_dlls", None)
                if callable(preload_dlls):
                    try:
                        # ORT GPU wheels can load CUDA/cuDNN from NVIDIA site-packages.
                        preload_dlls(directory="")
                    except TypeError:
                        preload_dlls()
                    except Exception as exc:
                        warnings.warn(
                            "[ExternalPretrainedProvider] onnxruntime.preload_dlls "
                            f"failed before CUDA session creation: {exc}",
                            UserWarning,
                            stacklevel=2,
                        )
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

            self._onnx_session = ort.InferenceSession(ckpt_str, providers=providers)
            session_providers = self._onnx_session.get_providers()
            if (
                self._device.startswith("cuda")
                and "CUDAExecutionProvider" not in session_providers
            ):
                raise RuntimeError(
                    "CUDAExecutionProvider requested but ONNX session providers are "
                    f"{session_providers}"
                )
            self._onnx_input_name = self._onnx_session.get_inputs()[0].name
            self._backend = "onnxruntime"
            self._runtime_ready = True
            return

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required for non-ONNX external anatomy checkpoints"
            ) from exc

        try:
            model = torch.jit.load(ckpt_str, map_location=self._device)
            model.eval()
            self._model = model
            self._backend = "torchscript"
            self._runtime_ready = True
            return
        except Exception:
            pass

        payload = torch.load(ckpt_str, map_location=self._device)
        if isinstance(payload, torch.nn.Module):
            payload.eval()
            self._model = payload
            self._backend = "torch_module"
            self._runtime_ready = True
            return

        if isinstance(payload, dict) and isinstance(
            payload.get("model"), torch.nn.Module
        ):
            model = payload["model"]
            model.eval()
            self._model = model
            self._backend = "torch_module_dict"
            self._runtime_ready = True
            return

        raise RuntimeError(
            "Unsupported checkpoint format for ExternalPretrainedProvider. "
            "Use .onnx, TorchScript, or a serialized torch.nn.Module."
        )

    def _predict_mask_probability(self, image_rgb: Any) -> Any:
        import numpy as np

        inp = self._preprocess(image_rgb)

        if self._backend == "onnxruntime":
            if self._onnx_session is None or self._onnx_input_name is None:
                raise RuntimeError("ONNX runtime not initialized")
            raw_out = self._onnx_session.run(None, {self._onnx_input_name: inp})[0]
            return self._decode_probability_map(raw_out)

        if self._model is None:
            raise RuntimeError("Torch model runtime not initialized")

        import torch

        with torch.no_grad():
            x = torch.from_numpy(inp).to(self._device)
            out = self._model(x)
        out = self._unwrap_model_output(out)
        if isinstance(out, torch.Tensor):
            out = out.detach().cpu().numpy()
        return self._decode_probability_map(out)

    def _preprocess(self, image_rgb: Any) -> Any:
        import numpy as np
        from PIL import Image as PILImage

        if image_rgb.dtype != np.uint8:
            arr = image_rgb.astype(np.float32)
            if arr.max() <= 1.0:
                arr = arr * 255.0
            image_rgb = np.clip(arr, 0.0, 255.0).astype(np.uint8)

        resized = PILImage.fromarray(image_rgb).resize(
            (self._input_size, self._input_size),
            resample=PILImage.BILINEAR,
        )
        x = np.asarray(resized).astype(np.float32) / 255.0
        if x.ndim == 2:
            x = np.stack([x, x, x], axis=-1)
        x = x.transpose(2, 0, 1)[None, ...]
        return x

    @staticmethod
    def _unwrap_model_output(out: Any) -> Any:
        if isinstance(out, dict):
            if "out" in out:
                return out["out"]
            if "mask" in out:
                return out["mask"]
            return next(iter(out.values()))
        if isinstance(out, (list, tuple)):
            if not out:
                raise RuntimeError("Model returned an empty output list")
            return out[0]
        return out

    @staticmethod
    def _decode_probability_map(raw_out: Any) -> Any:
        import numpy as np

        arr = np.asarray(raw_out)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]

        if arr.ndim == 3:
            if arr.shape[0] == 1:
                arr = arr[0]
            else:
                logits = arr.astype(np.float32)
                logits = logits - logits.max(axis=0, keepdims=True)
                exp_logits = np.exp(logits)
                denom = np.maximum(exp_logits.sum(axis=0, keepdims=True), 1e-8)
                probs = exp_logits / denom
                arr = probs[1] if probs.shape[0] > 1 else probs[0]

        if arr.ndim != 2:
            raise RuntimeError(f"Unsupported OD mask shape: {arr.shape}")

        arr = arr.astype(np.float32)
        if float(arr.min()) < 0.0 or float(arr.max()) > 1.0:
            arr = 1.0 / (1.0 + np.exp(-arr))
        return arr

    def _extract_disc_geometry(
        self,
        prob_map: Any,
        image_w: int,
        image_h: int,
    ) -> Tuple[Optional[Tuple[float, float]], Optional[float], float, int]:
        import numpy as np

        binary = prob_map >= self._mask_threshold
        component = self._largest_component(binary)
        if component is None or len(component) == 0:
            return None, None, 0.0, 0

        ys = component[:, 0]
        xs = component[:, 1]
        area_px = int(len(component))
        conf_score = float(prob_map[ys, xs].mean())

        h_mask, w_mask = prob_map.shape
        sx = float(image_w) / float(w_mask)
        sy = float(image_h) / float(h_mask)

        cx = (float(xs.mean()) + 0.5) * sx
        cy = (float(ys.mean()) + 0.5) * sy

        radius_small = math.sqrt(area_px / math.pi)
        radius = radius_small * ((sx + sy) / 2.0)
        return (cx, cy), float(radius), conf_score, area_px

    def _confidence_tier(self, confidence_score: float, area_px: int) -> str:
        if area_px < self._min_mask_area:
            return "low"
        if confidence_score >= self._high_thr:
            return "high"
        if confidence_score >= self._moderate_thr:
            return "moderate"
        return "low"

    @staticmethod
    def _estimate_fovea_center(
        od_center: Optional[Tuple[float, float]],
        od_radius: Optional[float],
        image_w: int,
        image_h: int,
    ) -> Optional[Tuple[float, float]]:
        if od_center is None or od_radius is None:
            return None

        dd = max(2.0 * od_radius, 1.0)
        if od_center[0] >= image_w / 2.0:
            fx = od_center[0] - 3.5 * dd
        else:
            fx = od_center[0] + 3.5 * dd

        fx = max(0.0, min(float(image_w - 1), fx))
        fy = max(0.0, min(float(image_h - 1), od_center[1]))
        return (fx, fy)

    @staticmethod
    def _largest_component(binary_mask: Any) -> Optional[Any]:
        import numpy as np

        coords = np.argwhere(binary_mask)
        if coords.size == 0:
            return None

        h, w = binary_mask.shape
        visited = np.zeros((h, w), dtype=bool)
        best_component = []

        for y, x in coords:
            y_i = int(y)
            x_i = int(x)
            if visited[y_i, x_i]:
                continue

            stack = [(y_i, x_i)]
            visited[y_i, x_i] = True
            component = []

            while stack:
                cy, cx = stack.pop()
                component.append((cy, cx))

                for ny, nx in (
                    (cy - 1, cx),
                    (cy + 1, cx),
                    (cy, cx - 1),
                    (cy, cx + 1),
                ):
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    if visited[ny, nx] or not binary_mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))

            if len(component) > len(best_component):
                best_component = component

        if not best_component:
            return None
        return np.asarray(best_component, dtype=np.int32)

    @staticmethod
    def _to_numpy_rgb(image: Any) -> Tuple[Any, int, int]:
        import numpy as np

        if image is None:
            raise ValueError("image is None")

        arr = None

        # PIL image-like object
        if hasattr(image, "convert") and hasattr(image, "size"):
            arr = np.asarray(image.convert("RGB"))

        # Torch tensor
        if arr is None:
            try:
                import torch

                if isinstance(image, torch.Tensor):
                    t = image.detach().cpu()
                    if t.ndim == 4:
                        t = t[0]
                    if t.ndim != 3:
                        raise ValueError(
                            f"Unsupported tensor shape for anatomy image: {t.shape}"
                        )
                    if t.shape[0] in (1, 3):
                        t = t.permute(1, 2, 0)
                    arr = t.numpy()
            except ImportError:
                pass

        # NumPy ndarray
        if arr is None and hasattr(image, "shape"):
            arr = np.asarray(image)

        if arr is None:
            raise TypeError(
                f"Unsupported image type for anatomy provider: {type(image)}"
            )

        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.ndim != 3:
            raise ValueError(
                f"Unsupported image shape for anatomy provider: {arr.shape}"
            )

        if arr.shape[2] > 3:
            arr = arr[:, :, :3]

        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            if float(arr.max()) <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)

        h, w = int(arr.shape[0]), int(arr.shape[1])
        return arr, w, h


# ── Helper: build AnatomyContext from dict (for cached / manual landmark data) ─


def make_anatomy_context(
    od_center: Optional[Tuple[float, float]],
    od_radius: Optional[float],
    od_confidence: str,
    fovea_center: Optional[Tuple[float, float]] = None,
    eye_side: str = "unknown",
    provider_name: str = "manual",
    provider_validation: Optional[Dict[str, Any]] = None,
    image_id: str = "",
) -> AnatomyContext:

    return AnatomyContext(
        od_center=od_center,
        od_radius=od_radius,
        od_confidence=od_confidence,
        fovea_center=fovea_center,
        eye_side=eye_side,
        provider_name=provider_name,
        provider_validation=provider_validation or {},
        image_id=image_id,
    )
