from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from donut_render_py import PoseDesc, ReplicaCADManifest, compose_transform_matrix


DATASET_ROOT = REPO_ROOT / "ReplicaCAD"


class ReplicaCADTransformTests(unittest.TestCase):
    def assertMatrixAlmostEqual(self, actual, expected) -> None:
        for actual_row, expected_row in zip(actual, expected):
            for actual_value, expected_value in zip(actual_row, expected_row):
                self.assertAlmostEqual(actual_value, expected_value, places=6)

    def test_identity_pose_uses_row_major_translation_column(self) -> None:
        matrix = compose_transform_matrix((1.0, 2.0, 3.0), (1.0, 0.0, 0.0, 0.0))
        self.assertEqual(
            matrix,
            (
                (1.0, 0.0, 0.0, 1.0),
                (0.0, 1.0, 0.0, 2.0),
                (0.0, 0.0, 1.0, 3.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
        )

    def test_wxyz_y_rotation_golden_matrix(self) -> None:
        half = math.sqrt(0.5)
        matrix = PoseDesc(rotation_wxyz=(half, 0.0, half, 0.0)).matrix_row_major
        self.assertMatrixAlmostEqual(
            matrix,
            (
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (-1.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
        )

    def test_quaternion_is_normalized(self) -> None:
        pose = PoseDesc(rotation_wxyz=(2.0, 0.0, 0.0, 0.0))
        self.assertEqual(pose.rotation_wxyz, (1.0, 0.0, 0.0, 0.0))

    def test_zero_quaternion_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PoseDesc(rotation_wxyz=(0.0, 0.0, 0.0, 0.0))

    @unittest.skipUnless(DATASET_ROOT.is_dir(), "ReplicaCAD dataset is not available")
    def test_apt_0_first_object_preserves_wxyz_and_translation(self) -> None:
        scene = ReplicaCADManifest.from_dataset_root(DATASET_ROOT).parse_scene("apt_0")
        instance = scene.objects[0]
        self.assertEqual(instance.template_name, "frl_apartment_basket")
        self.assertAlmostEqual(instance.pose.rotation_wxyz[0], 0.9846951961517334)
        self.assertAlmostEqual(instance.pose.translation[0], -1.9956579525706273)
        self.assertAlmostEqual(
            instance.pose.matrix_row_major[0][3], instance.pose.translation[0]
        )


if __name__ == "__main__":
    unittest.main()
