from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np


SENSOR_PRODUCTS = ("color", "depth", "normal", "instance", "semantic")


def normalize_sensor_products(products: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for product in products:
        name = str(product).lower()
        if name not in SENSOR_PRODUCTS:
            raise ValueError(f"Unknown sensor product: {product!r}")
        if name not in normalized:
            normalized.append(name)
    if not normalized:
        raise ValueError("At least one sensor product must be requested.")
    return tuple(normalized)


@dataclass(frozen=True)
class SensorFrame:
    """Aligned observations from one camera and one scene frame.

    Depth is optical-axis distance in meters with 0 for background. Normals are
    world-space unit vectors with (0, 0, 0) for background. Instance and
    semantic images are uint32 and reserve 0 for background/unknown.
    """

    width: int
    height: int
    color: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    normal: Optional[np.ndarray] = None
    instance: Optional[np.ndarray] = None
    semantic: Optional[np.ndarray] = None

    def products(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in SENSOR_PRODUCTS
            if getattr(self, name) is not None
        )


def _decode_array(
    raw: Mapping[str, object],
    name: str,
    dtype: np.dtype,
    shape: tuple[int, ...],
) -> Optional[np.ndarray]:
    payload = raw.get(name)
    if payload is None:
        return None
    expected = int(np.prod(shape, dtype=np.int64))
    expected_bytes = expected * dtype.itemsize
    try:
        payload_bytes = memoryview(payload).nbytes
    except TypeError as exc:
        raise RuntimeError(
            f"Sensor product {name!r} does not expose a byte buffer."
        ) from exc
    if payload_bytes != expected_bytes:
        raise RuntimeError(
            f"Sensor product {name!r} has {payload_bytes} bytes; "
            f"expected {expected_bytes}."
        )
    array = np.frombuffer(payload, dtype=dtype, count=expected)
    return array.reshape(shape).copy(order="C")


def decode_sensor_frame(raw: Mapping[str, object]) -> SensorFrame:
    width = int(raw["width"])
    height = int(raw["height"])
    if width <= 0 or height <= 0:
        raise RuntimeError("Native sensor frame has an invalid resolution.")
    return SensorFrame(
        width=width,
        height=height,
        color=_decode_array(raw, "color", np.dtype(np.uint8), (height, width, 4)),
        depth=_decode_array(raw, "depth", np.dtype(np.float32), (height, width)),
        normal=_decode_array(
            raw, "normal", np.dtype(np.float32), (height, width, 3)
        ),
        instance=_decode_array(
            raw, "instance", np.dtype(np.uint32), (height, width)
        ),
        semantic=_decode_array(
            raw, "semantic", np.dtype(np.uint32), (height, width)
        ),
    )


def decode_sensor_frames(
    frames: Sequence[Mapping[str, object]],
) -> tuple[SensorFrame, ...]:
    return tuple(decode_sensor_frame(frame) for frame in frames)
