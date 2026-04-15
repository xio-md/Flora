from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def default_output_dir(repo_root: Path, sample_name: str) -> Path:
    return repo_root / ".temp" / "demo_outputs" / sample_name


def frame_output_path(
    output_dir: Path,
    stem: str,
    frame_index: int,
    *,
    suffix: str = ".ppm",
    digits: int = 3,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{stem}_{frame_index:0{digits}d}{suffix}"


def write_rgb_ppm(path: Path, rgb: np.ndarray) -> None:
    array = np.asarray(rgb, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("RGB images must have shape (H, W, 3).")

    path.parent.mkdir(parents=True, exist_ok=True)
    height, width, _channels = array.shape
    with path.open("wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        f.write(np.ascontiguousarray(array).tobytes())


def write_rgba_bytes_ppm(path: Path, rgba: bytes, width: int, height: int) -> None:
    expected_bytes = int(width) * int(height) * 4
    if len(rgba) != expected_bytes:
        raise ValueError(f"Expected {expected_bytes} RGBA bytes, got {len(rgba)}.")

    image = np.frombuffer(rgba, dtype=np.uint8).reshape(int(height), int(width), 4)
    write_rgb_ppm(path, image[:, :, :3])


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
