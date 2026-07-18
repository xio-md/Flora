from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from .scene_desc import InstanceDesc, Matrix4, SceneDesc, compose_transform_matrix


def _document_path(path: Path, document_path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        value = os.path.relpath(resolved, document_path.parent)
    except ValueError:
        value = str(resolved)
    return value.replace("\\", "/")


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _asset_origin_translation(instance: InstanceDesc) -> tuple[float, float, float]:
    translation = instance.pose.translation
    if instance.translation_origin == "ASSET_LOCAL":
        return translation

    scale = instance.visual_asset.scale
    scaled_com = tuple(instance.com[index] * scale[index] for index in range(3))
    rotation = instance.pose.matrix_row_major
    rotated_com = tuple(
        sum(rotation[row][column] * scaled_com[column] for column in range(3))
        for row in range(3)
    )
    return tuple(translation[index] - rotated_com[index] for index in range(3))


def compose_instance_asset_matrix(instance: InstanceDesc) -> Matrix4:
    translation = _asset_origin_translation(instance)
    matrix = compose_transform_matrix(translation, instance.pose.rotation_wxyz)
    sx, sy, sz = instance.visual_asset.scale
    return (
        (matrix[0][0] * sx, matrix[0][1] * sy, matrix[0][2] * sz, matrix[0][3]),
        (matrix[1][0] * sx, matrix[1][1] * sy, matrix[1][2] * sz, matrix[1][3]),
        (matrix[2][0] * sx, matrix[2][1] * sy, matrix[2][2] * sz, matrix[2][3]),
        (0.0, 0.0, 0.0, 1.0),
    )


@dataclass(frozen=True)
class CompiledInstanceDesc:
    node_name: str
    kind: str
    instance_id: int
    template_name: str
    model_index: int
    source_path: Path
    translation: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]
    scale: tuple[float, float, float]
    motion_type: str
    semantic_id: int
    translation_origin: str
    asset_matrix_row_major: Matrix4

    def graph_node(self) -> dict[str, object]:
        return {
            "name": self.node_name,
            "model": self.model_index,
            "translation": self.translation,
            "rotation": self.rotation_xyzw,
            "scaling": self.scale,
        }


@dataclass(frozen=True)
class DonutSceneArtifact:
    scene_path: Path
    metadata_path: Path
    determinism_digest: str


@dataclass(frozen=True)
class CompiledDonutScene:
    scene_name: str
    source_path: Path
    models: tuple[Path, ...]
    instances: tuple[CompiledInstanceDesc, ...]
    omitted_articulated_instances: int

    @property
    def model_count(self) -> int:
        return len(self.models)

    @property
    def instance_count(self) -> int:
        return len(self.instances)

    def scene_payload(self, output_path: Path) -> dict[str, object]:
        output_path = Path(output_path).resolve()
        return {
            "models": [_document_path(path, output_path) for path in self.models],
            "graph": [instance.graph_node() for instance in self.instances],
            "animations": [],
        }

    def metadata_payload(
        self, output_path: Path, scene_payload: dict[str, object]
    ) -> dict[str, object]:
        output_path = Path(output_path).resolve()
        semantic_payload = {
            "scene_name": self.scene_name,
            "source_path": _document_path(self.source_path, output_path),
            "models": scene_payload["models"],
            "instances": [
                {
                    "node_name": instance.node_name,
                    "kind": instance.kind,
                    "instance_id": instance.instance_id,
                    "template_name": instance.template_name,
                    "model_index": instance.model_index,
                    "source_path": _document_path(instance.source_path, output_path),
                    "motion_type": instance.motion_type,
                    "semantic_id": instance.semantic_id,
                    "translation_origin": instance.translation_origin,
                    "asset_matrix_row_major": instance.asset_matrix_row_major,
                }
                for instance in self.instances
            ],
            "omitted_articulated_instances": self.omitted_articulated_instances,
        }
        digest = hashlib.sha256(
            json.dumps(
                semantic_payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        return {
            "schema_version": 1,
            **semantic_payload,
            "summary": {
                "unique_models": self.model_count,
                "render_instances": self.instance_count,
                "stage_instances": sum(
                    instance.kind == "stage" for instance in self.instances
                ),
                "object_instances": sum(
                    instance.kind == "object" for instance in self.instances
                ),
                "omitted_articulated_instances": self.omitted_articulated_instances,
            },
            "determinism_digest": digest,
        }

    def write(self, output_path: str | Path) -> DonutSceneArtifact:
        scene_path = Path(output_path).resolve()
        if scene_path.suffix.lower() in {".gltf", ".glb"}:
            raise ValueError("Donut scene descriptions cannot use .gltf or .glb suffixes.")
        metadata_path = scene_path.with_suffix(".manifest.json")
        scene_payload = self.scene_payload(scene_path)
        metadata_payload = self.metadata_payload(metadata_path, scene_payload)
        _write_json_atomic(scene_path, scene_payload)
        _write_json_atomic(metadata_path, metadata_payload)
        return DonutSceneArtifact(
            scene_path=scene_path,
            metadata_path=metadata_path,
            determinism_digest=str(metadata_payload["determinism_digest"]),
        )


def _compile_instance(
    instance: InstanceDesc,
    kind: str,
    model_index: int,
) -> CompiledInstanceDesc:
    w, x, y, z = instance.pose.rotation_wxyz
    return CompiledInstanceDesc(
        node_name=f"replicacad_{kind}_{instance.instance_id:06d}",
        kind=kind,
        instance_id=instance.instance_id,
        template_name=instance.template_name,
        model_index=model_index,
        source_path=instance.visual_asset.source_path,
        translation=_asset_origin_translation(instance),
        rotation_xyzw=(x, y, z, w),
        scale=instance.visual_asset.scale,
        motion_type=instance.motion_type,
        semantic_id=instance.semantic_id,
        translation_origin=instance.translation_origin,
        asset_matrix_row_major=compose_instance_asset_matrix(instance),
    )


def compile_donut_scene(scene: SceneDesc) -> CompiledDonutScene:
    models: list[Path] = []
    model_indices: dict[Path, int] = {}
    instances: list[CompiledInstanceDesc] = []

    def append(instance: InstanceDesc, kind: str) -> None:
        source_path = instance.visual_asset.source_path.resolve()
        model_index = model_indices.get(source_path)
        if model_index is None:
            model_index = len(models)
            model_indices[source_path] = model_index
            models.append(source_path)
        instances.append(_compile_instance(instance, kind, model_index))

    append(scene.stage, "stage")
    for instance in scene.objects:
        append(instance, "object")

    return CompiledDonutScene(
        scene_name=scene.name,
        source_path=scene.source_path,
        models=tuple(models),
        instances=tuple(instances),
        omitted_articulated_instances=len(scene.articulated),
    )
