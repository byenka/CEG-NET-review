#!/usr/bin/env python
"""Prepare a REFUGE SegFormer optic-disc checkpoint for LGovNet B10.

The Hugging Face model predicts three semantic classes:
background, optic disc, and optic cup.  LGovNet's ExternalPretrainedProvider
expects a single optic-disc probability mask.  This exporter wraps the
SegFormer model so the ONNX output is:

    input:  RGB tensor [1, 3, 512, 512] in [0, 1]
    output: optic-disc probability [1, 1, 512, 512]

The exported probability is P(Optic disc) + P(Optic cup), because the cup is
an anatomical subregion inside the disc and should contribute to OD geometry.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _require_imports() -> None:
    missing = []
    for module_name in ("torch", "transformers", "onnx", "onnxruntime", "PIL", "numpy"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        print("[SegFormerOD] Missing dependency:", ", ".join(missing), file=sys.stderr)
        print(
            "[SegFormerOD] Install on the target environment, for example:\n"
            "  pip install transformers onnx onnxruntime safetensors",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_label(config: Dict[str, Any], label: str, fallback: int) -> int:
    label2id = config.get("label2id") or {}
    if label in label2id:
        return int(label2id[label])
    return int(fallback)


def _write_metadata(
    output_path: Path,
    hf_model_dir: Path,
    config: Dict[str, Any],
    preprocessor: Dict[str, Any],
    disc_id: int,
    cup_id: Optional[int],
    input_size: int,
) -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "prepare_segformer_refuge_od_onnx",
        "source_model": str(hf_model_dir.resolve()),
        "source_name_or_path": config.get("_name_or_path"),
        "source_model_type": config.get("model_type"),
        "source_architectures": config.get("architectures"),
        "source_id2label": config.get("id2label"),
        "source_label2id": config.get("label2id"),
        "input_contract": {
            "name": "input",
            "shape": [1, 3, input_size, input_size],
            "range": "[0, 1]",
            "color_order": "RGB",
        },
        "output_contract": {
            "name": "od_probability",
            "shape": [1, 1, input_size, input_size],
            "range": "[0, 1]",
            "semantics": "P(Optic disc) + P(Optic cup)",
            "disc_label_id": disc_id,
            "cup_label_id": cup_id,
        },
        "preprocess_embedded_in_onnx": {
            "do_normalize": bool(preprocessor.get("do_normalize", True)),
            "image_mean": preprocessor.get("image_mean", [0.485, 0.456, 0.406]),
            "image_std": preprocessor.get("image_std", [0.229, 0.224, 0.225]),
            "provider_should_pass": "ExternalPretrainedProvider._preprocess output [0,1] RGB",
        },
        "recommended_lgovnet_env": {
            "ANATOMY_PROVIDER": "external_pretrained",
            "ANATOMY_DEVICE": "cuda",
            "OD_CKPT": str(output_path.as_posix()),
            "anatomy_input_size": input_size,
            "anatomy_quality_mode": "strict",
        },
    }
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[SegFormerOD] metadata written: {meta_path}")


def _smoke_provider(output_path: Path, image_path: Optional[Path], device: str) -> None:
    if image_path is None:
        return
    if not image_path.exists():
        raise FileNotFoundError(f"Smoke image not found: {image_path}")

    from PIL import Image

    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.governance.anatomy_provider import ExternalPretrainedProvider

    provider = ExternalPretrainedProvider(
        checkpoint_path=str(output_path),
        device=device,
        input_size=512,
    )
    ctx = provider(Image.open(image_path), image_id=image_path.stem)
    print("[SegFormerOD] smoke context:")
    print(json.dumps(ctx.to_dict(), indent=2, ensure_ascii=False))
    if "fallback=approximate" in ctx.provider_name:
        raise RuntimeError("Provider fell back to approximate during smoke test")
    if ctx.od_confidence in ("low", "n/a"):
        raise RuntimeError(f"Provider returned low-confidence OD: {ctx.od_confidence}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export HF SegFormer REFUGE optic-disc/cup model to LGovNet OD ONNX."
    )
    parser.add_argument(
        "--hf_model_dir",
        default="checkpoints/external_anatomy/segformer_refuge_hf",
        help="Downloaded Hugging Face model directory.",
    )
    parser.add_argument(
        "--output_onnx",
        default="checkpoints/external_anatomy/od_fovea_pretrained.onnx",
        help="Output ONNX path consumed by ExternalPretrainedProvider.",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=512,
        help="Fixed input/output size for the exported ONNX wrapper.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device used during export and optional smoke test.",
    )
    parser.add_argument(
        "--smoke_image",
        default="",
        help="Optional fundus image path to validate the exported provider.",
    )
    args = parser.parse_args()

    _require_imports()

    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import SegformerForSemanticSegmentation

    hf_model_dir = Path(args.hf_model_dir)
    output_path = Path(args.output_onnx)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config_path = hf_model_dir / "config.json"
    preprocessor_path = hf_model_dir / "preprocessor_config.json"
    if not config_path.exists() or not preprocessor_path.exists():
        raise FileNotFoundError(
            f"Missing HF config/preprocessor under {hf_model_dir}. "
            "Download the model snapshot first."
        )

    config = _load_json(config_path)
    preprocessor = _load_json(preprocessor_path)
    disc_id = _resolve_label(config, "Optic disc", fallback=1)
    cup_id = _resolve_label(config, "Optic cup", fallback=2)

    mean = torch.tensor(
        preprocessor.get("image_mean", [0.485, 0.456, 0.406]),
        dtype=torch.float32,
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        preprocessor.get("image_std", [0.229, 0.224, 0.225]),
        dtype=torch.float32,
    ).view(1, 3, 1, 1)

    model = SegformerForSemanticSegmentation.from_pretrained(
        str(hf_model_dir),
        local_files_only=True,
    )
    model.eval()

    class LGovNetODWrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = (x - self.mean) / self.std
            logits = self.model(pixel_values=x).logits
            logits = F.interpolate(
                logits,
                size=(args.input_size, args.input_size),
                mode="bilinear",
                align_corners=False,
            )
            probs = torch.softmax(logits, dim=1)
            od = probs[:, disc_id : disc_id + 1] + probs[:, cup_id : cup_id + 1]
            return torch.clamp(od, 0.0, 1.0)

    wrapper = LGovNetODWrapper().to(args.device).eval()
    dummy = torch.zeros(1, 3, args.input_size, args.input_size, device=args.device)

    print(f"[SegFormerOD] exporting ONNX: {output_path}")
    torch.onnx.export(
        wrapper,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["od_probability"],
        opset_version=17,
        do_constant_folding=True,
    )

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    out = session.run(None, {"input": np.zeros((1, 3, args.input_size, args.input_size), dtype=np.float32)})[0]
    if out.shape != (1, 1, args.input_size, args.input_size):
        raise RuntimeError(f"Unexpected ONNX output shape: {out.shape}")
    if float(out.min()) < -1e-5 or float(out.max()) > 1.00001:
        raise RuntimeError(f"Unexpected ONNX output range: min={out.min()} max={out.max()}")

    print(f"[SegFormerOD] ONNX validation PASS shape={out.shape}")
    _write_metadata(
        output_path=output_path,
        hf_model_dir=hf_model_dir,
        config=config,
        preprocessor=preprocessor,
        disc_id=disc_id,
        cup_id=cup_id,
        input_size=args.input_size,
    )

    smoke_image = Path(args.smoke_image) if args.smoke_image else None
    _smoke_provider(output_path, smoke_image, args.device)
    print("[SegFormerOD] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
