"""_safe_output.py — shared no-overwrite guard for comparison tooling.

Goal (user constraint 2026-06-04): "pastikan setiap hasilnya tidak menimpa
hasil-hasil sebelumnya dari kita" — every comparison run must NEVER clobber a
previous result. These helpers refuse to write into a directory that already
holds result artifacts unless ``--overwrite`` is explicitly passed.

Used by: tools/train_detector_compare.py, tools/convert_voc_to_yolo.py,
tools/train_yolo_compare.py, tools/aggregate_detector_compare.py.
Pure-stdlib; imports nothing from Mode 1.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Iterable, Sequence

# Files whose presence means "this dir already contains a finished/started run".
RESULT_SENTINELS: Sequence[str] = (
    "compare_results.json",
    "best.pt",
    "data.yaml",
    "aggregate_summary.json",
)


def timestamp() -> str:
    """UTC-ish local timestamp safe for directory names."""
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _existing_sentinels(output_dir: Path, sentinels: Iterable[str]) -> list[Path]:
    out = []
    for name in sentinels:
        p = output_dir / name
        if p.exists():
            out.append(p)
    return out


def guard_output_dir(
    output_dir: Path,
    overwrite: bool = False,
    sentinels: Iterable[str] = RESULT_SENTINELS,
    tag: str = "[safe_output]",
) -> Path:
    """Ensure ``output_dir`` is safe to write into.

    If it already contains any result sentinel and ``overwrite`` is False, print
    a clear message and raise SystemExit(3) — protecting prior results. Returns
    the (created) ``output_dir`` on success.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        clash = _existing_sentinels(output_dir, sentinels)
        if clash and not overwrite:
            print(f"{tag} REFUSING to overwrite existing results in:")
            print(f"{tag}   {output_dir}")
            for p in clash:
                print(f"{tag}   - found {p.name}")
            print(f"{tag} Use a DIFFERENT --output_dir, or pass --overwrite to")
            print(f"{tag} intentionally replace these files.")
            raise SystemExit(3)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def guard_file(path: Path, overwrite: bool = False, tag: str = "[safe_output]") -> Path:
    """Refuse to clobber a single existing file unless ``overwrite`` is set."""
    path = Path(path)
    if path.exists() and not overwrite:
        print(f"{tag} REFUSING to overwrite existing file: {path}")
        print(f"{tag} Use a different path or pass --overwrite.")
        raise SystemExit(3)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def fresh_timestamped_dir(parent: Path, prefix: str = "run") -> Path:
    """Create and return a brand-new timestamped subdir under ``parent``.

    Guarantees no overwrite by construction (timestamp + numeric suffix on
    the rare collision). Use for aggregations that should accumulate, not
    replace.
    """
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    base = f"{prefix}_{timestamp()}"
    cand = parent / base
    i = 1
    while cand.exists():
        cand = parent / f"{base}_{i}"
        i += 1
    cand.mkdir(parents=True)
    return cand
