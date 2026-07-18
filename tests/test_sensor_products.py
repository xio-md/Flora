from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from rtxns_genesis_style.sensor import (
    SENSOR_PRODUCTS,
    decode_sensor_frame,
    normalize_sensor_products,
)


class SensorProductTests(unittest.TestCase):
    def test_product_names_are_validated_and_deduplicated(self) -> None:
        self.assertEqual(
            normalize_sensor_products(("depth", "color", "depth")),
            ("depth", "color"),
        )
        self.assertEqual(len(SENSOR_PRODUCTS), 5)
        with self.assertRaises(ValueError):
            normalize_sensor_products(())
        with self.assertRaises(ValueError):
            normalize_sensor_products(("optical_flow",))

    def test_native_payload_decodes_to_aligned_owned_arrays(self) -> None:
        width, height = 3, 2
        color = np.arange(width * height * 4, dtype=np.uint8).reshape(
            height, width, 4
        )
        depth = np.arange(width * height, dtype=np.float32).reshape(height, width)
        normal = np.zeros((height, width, 3), dtype=np.float32)
        normal[:, :, 1] = 1.0
        instance = np.arange(width * height, dtype=np.uint32).reshape(height, width)
        semantic = instance + 10
        frame = decode_sensor_frame(
            {
                "width": width,
                "height": height,
                "color": color.tobytes(),
                "depth": depth.tobytes(),
                "normal": normal.tobytes(),
                "instance": instance.tobytes(),
                "semantic": semantic.tobytes(),
            }
        )
        self.assertEqual(frame.products(), SENSOR_PRODUCTS)
        self.assertEqual(frame.color.shape, (height, width, 4))
        self.assertEqual(frame.depth.shape, (height, width))
        self.assertEqual(frame.normal.shape, (height, width, 3))
        self.assertEqual(frame.instance.dtype, np.uint32)
        self.assertTrue(frame.color.flags.c_contiguous)
        self.assertTrue(frame.color.flags.owndata)
        np.testing.assert_array_equal(frame.semantic, semantic)

    def test_partial_payload_and_invalid_size(self) -> None:
        frame = decode_sensor_frame(
            {
                "width": 2,
                "height": 1,
                "depth": np.asarray([1.0, 2.0], dtype=np.float32).tobytes(),
            }
        )
        self.assertEqual(frame.products(), ("depth",))
        self.assertIsNone(frame.color)
        with self.assertRaises(RuntimeError):
            decode_sensor_frame(
                {"width": 2, "height": 2, "instance": b"too short"}
            )


if __name__ == "__main__":
    unittest.main()
