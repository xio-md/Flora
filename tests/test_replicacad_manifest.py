from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from donut_render_py import ReplicaCADManifest, ReplicaCADParseError


DATASET_ROOT = REPO_ROOT / "ReplicaCAD"


@unittest.skipUnless(DATASET_ROOT.is_dir(), "ReplicaCAD dataset is not available")
class ReplicaCADManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = ReplicaCADManifest.from_dataset_root(DATASET_ROOT)
        cls.scenes = cls.manifest.parse_all_scenes()
        cls.scene_by_name = {scene.name: scene for scene in cls.scenes}

    def test_registry_coverage(self) -> None:
        self.assertEqual(len(self.manifest.scene_handles), 91)
        self.assertEqual(len(self.manifest.stage_templates), 5)
        self.assertEqual(len(self.manifest.object_templates), 92)
        self.assertEqual(len(self.manifest.articulation_templates), 12)

    def test_all_scene_instances_parse(self) -> None:
        self.assertEqual(len(self.scenes), 91)
        self.assertEqual(sum(len(scene.objects) for scene in self.scenes), 2293)
        self.assertEqual(sum(len(scene.articulated) for scene in self.scenes), 540)
        self.assertFalse(any(scene.warnings for scene in self.scenes))

    def test_apt_0_contents_and_stable_ids(self) -> None:
        scene = self.scene_by_name["apt_0"]
        self.assertEqual(scene.stage.template_name, "frl_apartment_stage")
        self.assertEqual(scene.stage.instance_id, 0)
        self.assertEqual(len(scene.objects), 113)
        self.assertEqual(len(scene.articulated), 6)
        self.assertEqual([instance.instance_id for instance in scene.objects], list(range(1, 114)))
        self.assertEqual(
            [instance.instance_id for instance in scene.articulated], list(range(114, 120))
        )
        self.assertEqual(
            [instance.template_name for instance in scene.articulated],
            [
                "fridge",
                "kitchen_counter",
                "kitchenCupboard_01",
                "chestOfDrawers_01",
                "cabinet",
                "door2",
            ],
        )

    def test_scene_aliases_share_cached_description(self) -> None:
        by_handle = self.manifest.parse_scene("apt_0")
        by_filename = self.manifest.parse_scene("apt_0.scene_instance.json")
        by_path = self.manifest.parse_scene(
            DATASET_ROOT / "configs" / "scenes" / "apt_0.scene_instance.json"
        )
        self.assertIs(by_handle, by_filename)
        self.assertIs(by_handle, by_path)

        source = json.loads(by_handle.source_path.read_text(encoding="utf-8"))
        source["object_instances"] = []
        source["articulated_object_instances"] = []
        with tempfile.TemporaryDirectory() as directory:
            external_path = Path(directory) / "apt_0.scene_instance.json"
            external_path.write_text(json.dumps(source), encoding="utf-8")
            external = self.manifest.parse_scene(external_path)
            self.assertEqual(len(external.objects), 0)
            self.assertEqual(len(external.articulated), 0)

        self.assertIs(self.manifest.parse_scene("apt_0"), by_handle)

    def test_unknown_scene_is_rejected(self) -> None:
        with self.assertRaises(ReplicaCADParseError):
            self.manifest.parse_scene("does_not_exist")

    def test_all_required_visual_assets_exist(self) -> None:
        for template in self.manifest.stage_templates.values():
            self.assertTrue(template.visual_asset.source_path.is_absolute())
            self.assertTrue(template.visual_asset.source_path.is_file())
        for template in self.manifest.object_templates.values():
            self.assertTrue(template.visual_asset.source_path.is_absolute())
            self.assertTrue(template.visual_asset.source_path.is_file())
            collision = template.visual_asset.collision_source_path
            self.assertTrue(collision is None or collision.is_file())
        for template in self.manifest.articulation_templates.values():
            self.assertTrue(template.urdf_path.is_absolute())
            self.assertTrue(template.urdf_path.is_file())
            for visual in template.visuals:
                self.assertTrue(visual.mesh_path.is_absolute())
                self.assertTrue(visual.mesh_path.is_file())

    def test_registry_warnings_are_structured_and_non_required(self) -> None:
        codes = Counter(warning.code for warning in self.manifest.warnings)
        self.assertEqual(codes["missing_registry_path"], 1)
        self.assertEqual(codes["missing_navmesh_asset"], 21)
        self.assertEqual(sum(codes.values()), 22)

    def test_report_is_deterministic(self) -> None:
        first = self.manifest.build_report(self.scenes)
        second = self.manifest.build_report(self.manifest.parse_all_scenes())
        self.assertEqual(first["determinism_digest"], second["determinism_digest"])
        self.assertEqual(
            first["determinism_digest"],
            "85952eb72e072f834a603eb89b8602ef94ee2e58e905a5ccb3013ead5b754882",
        )
        self.assertEqual(first["summary"], second["summary"])
        self.assertEqual(first["summary"]["missing_required_visual_assets"], 0)


if __name__ == "__main__":
    unittest.main()
