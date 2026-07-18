from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional

from .scene_desc import (
    ArticulationDesc,
    InstanceDesc,
    Matrix4,
    SceneDesc,
    UrdfJointDesc,
    compose_transform_matrix,
    normalize_quaternion_wxyz,
)


Vec3 = tuple[float, float, float]
QuatWxyz = tuple[float, float, float, float]
QuatXyzw = tuple[float, float, float, float]


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


def _asset_origin_translation(instance: InstanceDesc) -> Vec3:
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


def _compose_trs_matrix(
    translation: Vec3,
    rotation_wxyz: QuatWxyz,
    scale: Vec3 = (1.0, 1.0, 1.0),
) -> Matrix4:
    matrix = compose_transform_matrix(translation, rotation_wxyz)
    sx, sy, sz = scale
    return (
        (matrix[0][0] * sx, matrix[0][1] * sy, matrix[0][2] * sz, matrix[0][3]),
        (matrix[1][0] * sx, matrix[1][1] * sy, matrix[1][2] * sz, matrix[1][3]),
        (matrix[2][0] * sx, matrix[2][1] * sy, matrix[2][2] * sz, matrix[2][3]),
        (0.0, 0.0, 0.0, 1.0),
    )


def compose_instance_asset_matrix(instance: InstanceDesc) -> Matrix4:
    return _compose_trs_matrix(
        _asset_origin_translation(instance),
        instance.pose.rotation_wxyz,
        instance.visual_asset.scale,
    )


def _quaternion_xyzw(rotation_wxyz: QuatWxyz) -> QuatXyzw:
    w, x, y, z = rotation_wxyz
    return x, y, z, w


def _quaternion_multiply(left: QuatWxyz, right: QuatWxyz) -> QuatWxyz:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return normalize_quaternion_wxyz(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        )
    )


def _quaternion_from_rpy(rpy: Vec3) -> QuatWxyz:
    roll, pitch, yaw = rpy
    sr, cr = math.sin(roll * 0.5), math.cos(roll * 0.5)
    sp, cp = math.sin(pitch * 0.5), math.cos(pitch * 0.5)
    sy, cy = math.sin(yaw * 0.5), math.cos(yaw * 0.5)
    return normalize_quaternion_wxyz(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
    )


def _normalized_axis(axis: Vec3) -> Vec3:
    length = math.sqrt(sum(value * value for value in axis))
    if length <= 1.0e-12:
        raise ValueError("URDF joint axis must be non-zero.")
    return tuple(value / length for value in axis)


def _quaternion_from_axis_angle(axis: Vec3, angle: float) -> QuatWxyz:
    nx, ny, nz = _normalized_axis(axis)
    half = angle * 0.5
    scale = math.sin(half)
    return math.cos(half), nx * scale, ny * scale, nz * scale


def _rotate_vector(rotation_wxyz: QuatWxyz, vector: Vec3) -> Vec3:
    matrix = compose_transform_matrix((0.0, 0.0, 0.0), rotation_wxyz)
    return tuple(
        sum(matrix[row][column] * vector[column] for column in range(3))
        for row in range(3)
    )


def _node_component(name: str) -> str:
    component = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
    return component or "unnamed"


def _joint_position(joint: UrdfJointDesc, value: float, clamp: bool) -> float:
    position = float(value)
    if not math.isfinite(position):
        raise ValueError(f"Joint {joint.name!r} position must be finite.")
    if clamp:
        if joint.limit_lower is not None:
            position = max(position, joint.limit_lower)
        if joint.limit_upper is not None:
            position = min(position, joint.limit_upper)
        return position
    if joint.limit_lower is not None and position < joint.limit_lower - 1.0e-7:
        raise ValueError(
            f"Joint {joint.name!r} position {position} is below {joint.limit_lower}."
        )
    if joint.limit_upper is not None and position > joint.limit_upper + 1.0e-7:
        raise ValueError(
            f"Joint {joint.name!r} position {position} is above {joint.limit_upper}."
        )
    return position


def compose_urdf_joint_matrix(
    joint: UrdfJointDesc,
    position: float = 0.0,
    *,
    clamp_limits: bool = False,
) -> Matrix4:
    joint_type = joint.joint_type.lower()
    q = _joint_position(joint, position, clamp_limits)
    origin_rotation = _quaternion_from_rpy(joint.origin_rpy)
    translation = joint.origin_xyz
    rotation = origin_rotation

    if joint_type == "fixed":
        if abs(q) > 1.0e-7:
            raise ValueError(f"Fixed joint {joint.name!r} cannot have a non-zero position.")
    elif joint_type in {"revolute", "continuous"}:
        rotation = _quaternion_multiply(
            origin_rotation,
            _quaternion_from_axis_angle(joint.axis_xyz, q),
        )
    elif joint_type == "prismatic":
        local_offset = tuple(value * q for value in _normalized_axis(joint.axis_xyz))
        rotated_offset = _rotate_vector(origin_rotation, local_offset)
        translation = tuple(
            joint.origin_xyz[index] + rotated_offset[index] for index in range(3)
        )
    else:
        raise ValueError(
            f"Unsupported URDF joint type {joint.joint_type!r} on {joint.name!r}."
        )

    return _compose_trs_matrix(translation, rotation)


@dataclass(frozen=True)
class CompiledInstanceDesc:
    node_name: str
    kind: str
    instance_id: int
    template_name: str
    model_index: int
    source_path: Path
    translation: Vec3
    rotation_xyzw: QuatXyzw
    scale: Vec3
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
class CompiledUrdfVisualDesc:
    node_name: str
    link_name: str
    visual_index: int
    model_index: int
    source_path: Path
    translation: Vec3
    rotation_xyzw: QuatXyzw
    scale: Vec3
    local_matrix_row_major: Matrix4

    def graph_node(self) -> dict[str, object]:
        return {
            "name": self.node_name,
            "model": self.model_index,
            "translation": self.translation,
            "rotation": self.rotation_xyzw,
            "scaling": self.scale,
        }


@dataclass(frozen=True)
class CompiledUrdfLinkDesc:
    node_name: str
    link_name: str
    parent_link_name: Optional[str]
    joint: Optional[UrdfJointDesc]
    translation: Vec3
    rotation_xyzw: QuatXyzw
    local_matrix_row_major: Matrix4
    visuals: tuple[CompiledUrdfVisualDesc, ...]

    @property
    def joint_name(self) -> Optional[str]:
        return None if self.joint is None else self.joint.name

    def graph_node(self, children: list[dict[str, object]]) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.node_name,
            "translation": self.translation,
            "rotation": self.rotation_xyzw,
        }
        if children:
            payload["children"] = children
        return payload


@dataclass(frozen=True)
class CompiledArticulationDesc:
    root_node_name: str
    instance_id: int
    template_name: str
    urdf_path: Path
    translation: Vec3
    rotation_xyzw: QuatXyzw
    uniform_scale: float
    motion_type: str
    fixed_base: bool
    translation_origin: str
    auto_clamp_joint_limits: bool
    links: tuple[CompiledUrdfLinkDesc, ...]

    @property
    def visual_count(self) -> int:
        return sum(len(link.visuals) for link in self.links)

    @property
    def joint_count(self) -> int:
        return sum(link.joint is not None for link in self.links)

    @property
    def root_link_names(self) -> tuple[str, ...]:
        return tuple(link.link_name for link in self.links if link.parent_link_name is None)

    @property
    def control_node_names(self) -> tuple[str, ...]:
        return (self.root_node_name, *(link.node_name for link in self.links))

    def graph_node(self) -> dict[str, object]:
        links_by_name = {link.link_name: link for link in self.links}
        children_by_parent: dict[str, list[str]] = {link.link_name: [] for link in self.links}
        for link in self.links:
            if link.parent_link_name is not None:
                children_by_parent[link.parent_link_name].append(link.link_name)

        def build_link(link_name: str) -> dict[str, object]:
            link = links_by_name[link_name]
            children = [visual.graph_node() for visual in link.visuals]
            children.extend(build_link(child) for child in children_by_parent[link_name])
            return link.graph_node(children)

        root_children = [build_link(name) for name in self.root_link_names]
        payload: dict[str, object] = {
            "name": self.root_node_name,
            "translation": self.translation,
            "rotation": self.rotation_xyzw,
            "scaling": (self.uniform_scale,) * 3,
        }
        if root_children:
            payload["children"] = root_children
        return payload

    def joint_transform_updates(
        self,
        joint_positions: Mapping[str, float],
        *,
        clamp_limits: Optional[bool] = None,
    ) -> tuple[tuple[str, ...], tuple[Matrix4, ...]]:
        requested = {str(name): float(value) for name, value in joint_positions.items()}
        links_by_joint = {
            link.joint.name: link for link in self.links if link.joint is not None
        }
        unknown = sorted(set(requested) - set(links_by_joint))
        if unknown:
            raise KeyError(
                f"Unknown joints for articulation {self.template_name!r}: {unknown}"
            )
        clamp = self.auto_clamp_joint_limits if clamp_limits is None else clamp_limits
        names: list[str] = []
        matrices: list[Matrix4] = []
        for link in self.links:
            if link.joint is None or link.joint.name not in requested:
                continue
            names.append(link.node_name)
            matrices.append(
                compose_urdf_joint_matrix(
                    link.joint,
                    requested[link.joint.name],
                    clamp_limits=bool(clamp),
                )
            )
        return tuple(names), tuple(matrices)


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
    articulations: tuple[CompiledArticulationDesc, ...]
    omitted_articulated_instances: int = 0

    @property
    def model_count(self) -> int:
        return len(self.models)

    @property
    def instance_count(self) -> int:
        return len(self.instances)

    @property
    def articulated_instance_count(self) -> int:
        return len(self.articulations)

    @property
    def articulated_link_count(self) -> int:
        return sum(len(articulation.links) for articulation in self.articulations)

    @property
    def articulated_visual_count(self) -> int:
        return sum(articulation.visual_count for articulation in self.articulations)

    @property
    def render_instance_count(self) -> int:
        return self.instance_count + self.articulated_visual_count

    @property
    def graph_node_count(self) -> int:
        return (
            self.instance_count
            + self.articulated_instance_count
            + self.articulated_link_count
            + self.articulated_visual_count
        )

    @property
    def control_node_names(self) -> tuple[str, ...]:
        names = [instance.node_name for instance in self.instances]
        for articulation in self.articulations:
            names.extend(articulation.control_node_names)
        return tuple(names)

    def logical_node_handle(self, node_name: str) -> int:
        try:
            return self.control_node_names.index(str(node_name))
        except ValueError as exc:
            raise KeyError(f"Unknown compiled control node: {node_name}") from exc

    def articulation_by_id(self, instance_id: int) -> CompiledArticulationDesc:
        for articulation in self.articulations:
            if articulation.instance_id == instance_id:
                return articulation
        raise KeyError(f"Unknown articulation instance id: {instance_id}")

    def joint_transform_updates(
        self,
        positions_by_instance: Mapping[int, Mapping[str, float]],
        *,
        clamp_limits: Optional[bool] = None,
    ) -> tuple[tuple[str, ...], tuple[Matrix4, ...]]:
        names: list[str] = []
        matrices: list[Matrix4] = []
        for instance_id, joint_positions in positions_by_instance.items():
            articulation = self.articulation_by_id(int(instance_id))
            update_names, update_matrices = articulation.joint_transform_updates(
                joint_positions,
                clamp_limits=clamp_limits,
            )
            names.extend(update_names)
            matrices.extend(update_matrices)
        return tuple(names), tuple(matrices)

    def scene_payload(self, output_path: Path) -> dict[str, object]:
        output_path = Path(output_path).resolve()
        graph = [instance.graph_node() for instance in self.instances]
        graph.extend(articulation.graph_node() for articulation in self.articulations)
        return {
            "models": [_document_path(path, output_path) for path in self.models],
            "graph": graph,
            "animations": [],
        }

    def metadata_payload(
        self, output_path: Path, scene_payload: dict[str, object]
    ) -> dict[str, object]:
        output_path = Path(output_path).resolve()
        control_nodes: list[dict[str, object]] = []
        for instance in self.instances:
            control_nodes.append(
                {
                    "logical_handle": len(control_nodes),
                    "node_name": instance.node_name,
                    "kind": instance.kind,
                    "instance_id": instance.instance_id,
                    "link_name": None,
                }
            )
        for articulation in self.articulations:
            control_nodes.append(
                {
                    "logical_handle": len(control_nodes),
                    "node_name": articulation.root_node_name,
                    "kind": "articulation_root",
                    "instance_id": articulation.instance_id,
                    "link_name": None,
                }
            )
            for link in articulation.links:
                control_nodes.append(
                    {
                        "logical_handle": len(control_nodes),
                        "node_name": link.node_name,
                        "kind": "urdf_link",
                        "instance_id": articulation.instance_id,
                        "link_name": link.link_name,
                    }
                )

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
            "articulations": [
                {
                    "root_node_name": articulation.root_node_name,
                    "instance_id": articulation.instance_id,
                    "template_name": articulation.template_name,
                    "urdf_path": _document_path(articulation.urdf_path, output_path),
                    "motion_type": articulation.motion_type,
                    "fixed_base": articulation.fixed_base,
                    "translation_origin": articulation.translation_origin,
                    "uniform_scale": articulation.uniform_scale,
                    "links": [
                        {
                            "node_name": link.node_name,
                            "link_name": link.link_name,
                            "parent_link_name": link.parent_link_name,
                            "joint_name": link.joint_name,
                            "joint_type": None if link.joint is None else link.joint.joint_type,
                            "axis_xyz": None if link.joint is None else link.joint.axis_xyz,
                            "limit_lower": None if link.joint is None else link.joint.limit_lower,
                            "limit_upper": None if link.joint is None else link.joint.limit_upper,
                            "local_matrix_row_major": link.local_matrix_row_major,
                            "visuals": [
                                {
                                    "node_name": visual.node_name,
                                    "model_index": visual.model_index,
                                    "source_path": _document_path(
                                        visual.source_path, output_path
                                    ),
                                    "local_matrix_row_major": visual.local_matrix_row_major,
                                }
                                for visual in link.visuals
                            ],
                        }
                        for link in articulation.links
                    ],
                }
                for articulation in self.articulations
            ],
            "control_nodes": control_nodes,
            "omitted_articulated_instances": self.omitted_articulated_instances,
        }
        digest = hashlib.sha256(
            json.dumps(
                semantic_payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        return {
            "schema_version": 2,
            **semantic_payload,
            "summary": {
                "unique_models": self.model_count,
                "render_instances": self.render_instance_count,
                "graph_nodes": self.graph_node_count,
                "control_nodes": len(control_nodes),
                "stage_instances": sum(
                    instance.kind == "stage" for instance in self.instances
                ),
                "object_instances": sum(
                    instance.kind == "object" for instance in self.instances
                ),
                "articulated_instances": self.articulated_instance_count,
                "articulated_links": self.articulated_link_count,
                "articulated_visuals": self.articulated_visual_count,
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
    return CompiledInstanceDesc(
        node_name=f"replicacad_{kind}_{instance.instance_id:06d}",
        kind=kind,
        instance_id=instance.instance_id,
        template_name=instance.template_name,
        model_index=model_index,
        source_path=instance.visual_asset.source_path,
        translation=_asset_origin_translation(instance),
        rotation_xyzw=_quaternion_xyzw(instance.pose.rotation_wxyz),
        scale=instance.visual_asset.scale,
        motion_type=instance.motion_type,
        semantic_id=instance.semantic_id,
        translation_origin=instance.translation_origin,
        asset_matrix_row_major=compose_instance_asset_matrix(instance),
    )


def _compile_articulation(
    articulation: ArticulationDesc,
    model_index_for: Callable[[Path], int],
) -> CompiledArticulationDesc:
    link_names = set(articulation.links)
    joint_by_child: dict[str, UrdfJointDesc] = {}
    for joint in articulation.joints:
        if joint.parent_link not in link_names or joint.child_link not in link_names:
            raise ValueError(
                f"URDF joint {joint.name!r} references an unknown link in "
                f"{articulation.template_name!r}."
            )
        if joint.child_link in joint_by_child:
            raise ValueError(
                f"URDF link {joint.child_link!r} has multiple parent joints in "
                f"{articulation.template_name!r}."
            )
        joint_by_child[joint.child_link] = joint

    root_links = [name for name in articulation.links if name not in joint_by_child]
    if not root_links:
        raise ValueError(f"URDF {articulation.template_name!r} has no root link.")

    root_node_name = f"replicacad_articulation_{articulation.instance_id:06d}"
    link_node_names = {
        name: (
            f"{root_node_name}_link_{index:03d}_{_node_component(name)}"
        )
        for index, name in enumerate(articulation.links)
    }
    visuals_by_link: dict[str, list[CompiledUrdfVisualDesc]] = {
        name: [] for name in articulation.links
    }
    for visual_index, visual in enumerate(articulation.visuals):
        if visual.link_name not in link_names:
            raise ValueError(
                f"URDF visual references unknown link {visual.link_name!r} in "
                f"{articulation.template_name!r}."
            )
        model_index = model_index_for(visual.mesh_path)
        rotation = _quaternion_from_rpy(visual.origin_rpy)
        visuals_by_link[visual.link_name].append(
            CompiledUrdfVisualDesc(
                node_name=(
                    f"{link_node_names[visual.link_name]}_visual_{visual_index:03d}"
                ),
                link_name=visual.link_name,
                visual_index=visual_index,
                model_index=model_index,
                source_path=visual.mesh_path,
                translation=visual.origin_xyz,
                rotation_xyzw=_quaternion_xyzw(rotation),
                scale=visual.scale,
                local_matrix_row_major=_compose_trs_matrix(
                    visual.origin_xyz, rotation, visual.scale
                ),
            )
        )

    compiled_links: list[CompiledUrdfLinkDesc] = []
    for link_name in articulation.links:
        joint = joint_by_child.get(link_name)
        if joint is None:
            translation = (0.0, 0.0, 0.0)
            rotation = (1.0, 0.0, 0.0, 0.0)
            local_matrix = _compose_trs_matrix(translation, rotation)
            parent_link_name = None
        else:
            translation = joint.origin_xyz
            rotation = _quaternion_from_rpy(joint.origin_rpy)
            local_matrix = compose_urdf_joint_matrix(joint)
            parent_link_name = joint.parent_link
        compiled_links.append(
            CompiledUrdfLinkDesc(
                node_name=link_node_names[link_name],
                link_name=link_name,
                parent_link_name=parent_link_name,
                joint=joint,
                translation=translation,
                rotation_xyzw=_quaternion_xyzw(rotation),
                local_matrix_row_major=local_matrix,
                visuals=tuple(visuals_by_link[link_name]),
            )
        )

    children: dict[str, list[str]] = {name: [] for name in articulation.links}
    for link in compiled_links:
        if link.parent_link_name is not None:
            children[link.parent_link_name].append(link.link_name)
    visited: set[str] = set()

    def visit(name: str, active: set[str]) -> None:
        if name in active:
            raise ValueError(f"URDF {articulation.template_name!r} contains a link cycle.")
        if name in visited:
            return
        active.add(name)
        for child in children[name]:
            visit(child, active)
        active.remove(name)
        visited.add(name)

    for root_link in root_links:
        visit(root_link, set())
    if visited != link_names:
        missing = sorted(link_names - visited)
        raise ValueError(
            f"URDF {articulation.template_name!r} has unreachable links: {missing}"
        )

    return CompiledArticulationDesc(
        root_node_name=root_node_name,
        instance_id=articulation.instance_id,
        template_name=articulation.template_name,
        urdf_path=articulation.urdf_path,
        translation=articulation.pose.translation,
        rotation_xyzw=_quaternion_xyzw(articulation.pose.rotation_wxyz),
        uniform_scale=articulation.uniform_scale,
        motion_type=articulation.motion_type,
        fixed_base=articulation.fixed_base,
        translation_origin=articulation.translation_origin,
        auto_clamp_joint_limits=articulation.auto_clamp_joint_limits,
        links=tuple(compiled_links),
    )


def compile_donut_scene(scene: SceneDesc) -> CompiledDonutScene:
    models: list[Path] = []
    model_indices: dict[Path, int] = {}
    instances: list[CompiledInstanceDesc] = []

    def model_index_for(path: Path) -> int:
        source_path = Path(path).resolve()
        model_index = model_indices.get(source_path)
        if model_index is None:
            model_index = len(models)
            model_indices[source_path] = model_index
            models.append(source_path)
        return model_index

    def append(instance: InstanceDesc, kind: str) -> None:
        instances.append(
            _compile_instance(
                instance,
                kind,
                model_index_for(instance.visual_asset.source_path),
            )
        )

    append(scene.stage, "stage")
    for instance in scene.objects:
        append(instance, "object")

    articulations = tuple(
        _compile_articulation(articulation, model_index_for)
        for articulation in scene.articulated
    )

    return CompiledDonutScene(
        scene_name=scene.name,
        source_path=scene.source_path,
        models=tuple(models),
        instances=tuple(instances),
        articulations=articulations,
        omitted_articulated_instances=0,
    )
