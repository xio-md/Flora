from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from donut_render_py import (
    InstanceDesc,
    PoseDesc,
    ReplicaCADManifest,
    VisualAssetDesc,
    compile_donut_scene,
    compose_instance_asset_matrix,
)


DATASET_ROOT = REPO_ROOT / "ReplicaCAD"


@unittest.skipUnless(DATASET_ROOT.is_dir(), "ReplicaCAD dataset is not available")
class ReplicaCADAssemblyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = ReplicaCADManifest.from_dataset_root(DATASET_ROOT)
        cls.scenes = cls.manifest.parse_all_scenes()
        cls.apt_0 = next(scene for scene in cls.scenes if scene.name == "apt_0")
        cls.compiled = compile_donut_scene(cls.apt_0)

    def test_apt_0_compiles_complete_static_visual_scene(self) -> None:
        self.assertEqual(self.compiled.instance_count, 114)
        self.assertEqual(self.compiled.instances[0].kind, "stage")
        self.assertEqual(
            sum(instance.kind == "object" for instance in self.compiled.instances),
            113,
        )
        self.assertEqual(self.compiled.articulated_instance_count, 6)
        self.assertEqual(self.compiled.articulated_link_count, 37)
        self.assertEqual(self.compiled.articulated_visual_count, 31)
        self.assertEqual(self.compiled.render_instance_count, 145)
        self.assertEqual(self.compiled.graph_node_count, 188)
        self.assertEqual(len(self.compiled.control_node_names), 157)
        self.assertEqual(self.compiled.omitted_articulated_instances, 0)

    def test_models_are_deduplicated_by_resolved_asset_path(self) -> None:
        expected = {
            instance.visual_asset.source_path.resolve()
            for instance in (self.apt_0.stage, *self.apt_0.objects)
        }
        expected.update(
            visual.mesh_path.resolve()
            for articulation in self.apt_0.articulated
            for visual in articulation.visuals
        )
        self.assertEqual(set(self.compiled.models), expected)
        self.assertEqual(self.compiled.model_count, len(expected))
        self.assertLess(self.compiled.model_count, self.compiled.instance_count)

    def test_donut_quaternion_is_xyzw_and_ids_are_stable(self) -> None:
        source = self.apt_0.objects[0]
        compiled = self.compiled.instances[1]
        w, x, y, z = source.pose.rotation_wxyz
        self.assertEqual(compiled.rotation_xyzw, (x, y, z, w))
        self.assertEqual(compiled.instance_id, 1)
        self.assertEqual(compiled.node_name, "replicacad_object_000001")

    def test_com_translation_compensation_respects_rotation_and_scale(self) -> None:
        half = math.sqrt(0.5)
        asset = VisualAssetDesc(
            source_path=self.apt_0.objects[0].visual_asset.source_path,
            scale=(2.0, 3.0, 4.0),
        )
        instance = InstanceDesc(
            name="synthetic",
            template_name="synthetic",
            visual_asset=asset,
            pose=PoseDesc(
                translation=(10.0, 20.0, 30.0),
                rotation_wxyz=(half, 0.0, 0.0, half),
            ),
            motion_type="STATIC",
            semantic_id=7,
            instance_id=1,
            com=(1.0, 0.0, 0.0),
            translation_origin="COM",
        )
        matrix = compose_instance_asset_matrix(instance)
        self.assertAlmostEqual(matrix[0][3], 10.0)
        self.assertAlmostEqual(matrix[1][3], 18.0)
        self.assertAlmostEqual(matrix[2][3], 30.0)

        asset_local = replace(instance, translation_origin="ASSET_LOCAL")
        local_matrix = compose_instance_asset_matrix(asset_local)
        self.assertEqual(
            tuple(local_matrix[index][3] for index in range(3)),
            (10.0, 20.0, 30.0),
        )

    def test_all_registered_scenes_compile_without_asset_duplication(self) -> None:
        for scene in self.scenes:
            compiled = compile_donut_scene(scene)
            self.assertEqual(compiled.instance_count, 1 + len(scene.objects))
            self.assertEqual(compiled.articulated_instance_count, len(scene.articulated))
            self.assertEqual(compiled.omitted_articulated_instances, 0)
            self.assertEqual(len(compiled.models), len(set(compiled.models)))
            self.assertTrue(all(path.is_file() for path in compiled.models))

    def test_written_scene_is_portable_and_byte_deterministic(self) -> None:
        output_root = REPO_ROOT / "output"
        output_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output_root) as directory:
            output_path = Path(directory) / "apt_0.donut_scene.json"
            first = self.compiled.write(output_path)
            first_scene_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
            first_metadata_hash = hashlib.sha256(first.metadata_path.read_bytes()).hexdigest()
            second = self.compiled.write(output_path)
            self.assertEqual(
                hashlib.sha256(output_path.read_bytes()).hexdigest(), first_scene_hash
            )
            self.assertEqual(
                hashlib.sha256(second.metadata_path.read_bytes()).hexdigest(),
                first_metadata_hash,
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["models"]), self.compiled.model_count)
            self.assertTrue(all("\\" not in path for path in payload["models"]))
            self.assertTrue(all(":" not in path for path in payload["models"]))
            self.assertEqual(first.determinism_digest, second.determinism_digest)
