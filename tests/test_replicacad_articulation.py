from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from donut_render_py import ReplicaCADManifest, compile_donut_scene


DATASET_ROOT = REPO_ROOT / "ReplicaCAD"


def flatten_graph(nodes: list[dict[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for node in nodes:
        result.append(node)
        children = node.get("children", [])
        if isinstance(children, list):
            result.extend(flatten_graph(children))
    return result


@unittest.skipUnless(DATASET_ROOT.is_dir(), "ReplicaCAD dataset is not available")
class ReplicaCADArticulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        manifest = ReplicaCADManifest.from_dataset_root(DATASET_ROOT)
        cls.compiled = compile_donut_scene(manifest.parse_scene("apt_0"))
        cls.by_template = {
            articulation.template_name: articulation
            for articulation in cls.compiled.articulations
        }

    def test_graph_contains_root_link_visual_hierarchy(self) -> None:
        payload = self.compiled.scene_payload(REPO_ROOT / "output" / "synthetic.scene.json")
        graph = payload["graph"]
        self.assertIsInstance(graph, list)
        self.assertEqual(len(graph), 120)
        flattened = flatten_graph(graph)
        self.assertEqual(len(flattened), self.compiled.graph_node_count)
        self.assertEqual(len({node["name"] for node in flattened}), len(flattened))

        fridge = self.by_template["fridge"]
        root = next(node for node in graph if node["name"] == fridge.root_node_name)
        root_link = root["children"][0]
        self.assertIn("_link_000_root", root_link["name"])
        body_link = root_link["children"][0]
        self.assertIn("_link_001_body", body_link["name"])
        self.assertTrue(any("model" in child for child in body_link["children"]))

    def test_control_nodes_have_stable_contiguous_logical_handles(self) -> None:
        names = self.compiled.control_node_names
        self.assertEqual(len(names), len(set(names)))
        for index, name in enumerate(names):
            self.assertEqual(self.compiled.logical_node_handle(name), index)

    def test_sensor_labels_are_stable_and_reserve_zero_for_background(self) -> None:
        labels = self.compiled.sensor_labels
        self.assertEqual(len(labels), 120)
        self.assertEqual(
            {label.instance_id for label in labels},
            set(range(1, 121)),
        )
        self.assertEqual(labels[0].kind, "stage")
        self.assertEqual(labels[0].semantic_id, 0)
        ordinary = next(label for label in labels if label.kind == "object")
        self.assertGreater(ordinary.semantic_id, 0)
        self.assertTrue(
            all(label.semantic_id == 0 for label in labels if label.kind == "articulation")
        )

    def test_prismatic_joint_update_uses_joint_local_axis(self) -> None:
        cabinet = self.by_template["cabinet"]
        names, matrices = cabinet.joint_transform_updates({"left_slide": 0.4})
        self.assertEqual(len(names), 1)
        self.assertIn("left_door", names[0])
        matrix = matrices[0]
        self.assertAlmostEqual(matrix[0][3], -0.43573 + 0.4, places=6)
        self.assertAlmostEqual(matrix[1][3], 0.46828, places=6)
        self.assertAlmostEqual(matrix[2][3], 0.18, places=6)

    def test_revolute_joint_update_preserves_hinge_translation(self) -> None:
        fridge = self.by_template["fridge"]
        names, matrices = fridge.joint_transform_updates({"top_door_hinge": 1.0})
        self.assertEqual(len(names), 1)
        matrix = matrices[0]
        self.assertEqual(
            tuple(round(matrix[index][3], 6) for index in range(3)),
            (0.322, 0.353, -0.385),
        )
        self.assertAlmostEqual(matrix[0][0], math.cos(1.0), places=6)
        self.assertAlmostEqual(matrix[1][0], math.sin(1.0), places=6)

    def test_joint_limits_reject_or_clamp(self) -> None:
        cabinet = self.by_template["cabinet"]
        with self.assertRaises(ValueError):
            cabinet.joint_transform_updates({"left_slide": 2.0})
        _, matrices = cabinet.joint_transform_updates(
            {"left_slide": 2.0}, clamp_limits=True
        )
        self.assertAlmostEqual(matrices[0][0][3], -0.43573 + 0.8, places=6)

    def test_all_dataset_articulations_compile(self) -> None:
        manifest = ReplicaCADManifest.from_dataset_root(DATASET_ROOT)
        compiled = [compile_donut_scene(scene) for scene in manifest.parse_all_scenes()]
        self.assertEqual(sum(scene.articulated_instance_count for scene in compiled), 540)
        self.assertEqual(sum(scene.articulated_link_count for scene in compiled), 3330)
        self.assertEqual(sum(scene.articulated_visual_count for scene in compiled), 2790)
        self.assertTrue(all(scene.omitted_articulated_instances == 0 for scene in compiled))


if __name__ == "__main__":
    unittest.main()
