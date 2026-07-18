from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Optional

from .scene_desc import (
    ArticulationDesc,
    ArticulationTemplateDesc,
    InstanceDesc,
    ManifestWarning,
    ObjectTemplateDesc,
    PoseDesc,
    SceneDesc,
    StageTemplateDesc,
    VisualAssetDesc,
    vector3,
)
from .urdf import parse_urdf_manifest


DATASET_CONFIG_NAME = "replicaCAD.scene_dataset_config.json"


class ReplicaCADParseError(RuntimeError):
    pass


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplicaCADParseError(f"Failed to read ReplicaCAD JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ReplicaCADParseError(f"Expected a JSON object in {path}.")
    return value


def _strip_suffix(name: str, suffix: str) -> str:
    return name[: -len(suffix)] if name.endswith(suffix) else name


def _normalize_handle(value: str) -> str:
    return str(value).strip().replace("\\", "/").rstrip("/")


def _scale3(value: object, name: str) -> tuple[float, float, float]:
    if value is None:
        return 1.0, 1.0, 1.0
    if isinstance(value, (int, float)):
        scale = float(value)
        return scale, scale, scale
    if isinstance(value, (list, tuple)):
        return vector3(value, name)
    raise ReplicaCADParseError(f"{name} must be a number or a three-element sequence.")


def _multiply_scale(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> tuple[float, float, float]:
    return left[0] * right[0], left[1] * right[1], left[2] * right[2]


def _pose(value: Mapping[str, object]) -> PoseDesc:
    translation = value.get("translation", (0.0, 0.0, 0.0))
    rotation = value.get("rotation", (1.0, 0.0, 0.0, 0.0))
    if not isinstance(translation, (list, tuple)) or not isinstance(rotation, (list, tuple)):
        raise ReplicaCADParseError("Instance translation and rotation must be arrays.")
    return PoseDesc(translation=vector3(translation, "translation"), rotation_wxyz=tuple(rotation))


class ReplicaCADManifest:
    def __init__(self, dataset_config_path: Path) -> None:
        self.dataset_config_path = Path(dataset_config_path).resolve()
        if not self.dataset_config_path.is_file():
            raise FileNotFoundError(f"ReplicaCAD dataset config not found: {self.dataset_config_path}")
        self.dataset_root = self.dataset_config_path.parent
        self._config = _read_json(self.dataset_config_path)
        self._warnings: list[ManifestWarning] = []
        self._scene_cache: dict[str, SceneDesc] = {}

        self._stage_templates: dict[str, StageTemplateDesc] = {}
        self._object_templates: dict[str, ObjectTemplateDesc] = {}
        self._articulation_templates: dict[str, ArticulationTemplateDesc] = {}
        self._scene_paths: dict[str, Path] = {}
        self._lighting_paths: dict[str, Path] = {}
        self._navmesh_paths: dict[str, Path] = {}

        self._stage_aliases: dict[str, str] = {}
        self._object_aliases: dict[str, str] = {}
        self._articulation_aliases: dict[str, str] = {}
        self._scene_aliases: dict[str, str] = {}
        self._lighting_aliases: dict[str, str] = {}

        self._load_stage_templates()
        self._load_object_templates()
        self._load_articulation_templates()
        self._load_scene_registry()
        self._load_lighting_registry()
        self._load_navmesh_registry()

    @classmethod
    def from_dataset_root(cls, dataset_root: Path) -> "ReplicaCADManifest":
        return cls(Path(dataset_root) / DATASET_CONFIG_NAME)

    @property
    def warnings(self) -> tuple[ManifestWarning, ...]:
        template_warnings = tuple(
            warning
            for template in self._articulation_templates.values()
            for warning in template.warnings
        )
        return tuple(self._warnings) + template_warnings

    @property
    def scene_handles(self) -> tuple[str, ...]:
        return tuple(sorted(self._scene_paths))

    @property
    def stage_templates(self) -> Mapping[str, StageTemplateDesc]:
        return dict(self._stage_templates)

    @property
    def object_templates(self) -> Mapping[str, ObjectTemplateDesc]:
        return dict(self._object_templates)

    @property
    def articulation_templates(self) -> Mapping[str, ArticulationTemplateDesc]:
        return dict(self._articulation_templates)

    def _warning(
        self,
        code: str,
        message: str,
        source_path: Optional[Path] = None,
        **context: object,
    ) -> None:
        self._warnings.append(
            ManifestWarning(
                code=code,
                message=message,
                source_path=source_path,
                context=tuple((key, str(value)) for key, value in context.items()),
            )
        )

    def _registry_files(self, section_name: str, extension: str) -> tuple[Path, ...]:
        section = self._config.get(section_name, {})
        paths = section.get("paths", {}) if isinstance(section, dict) else {}
        expressions = paths.get(extension, []) if isinstance(paths, dict) else []
        files: set[Path] = set()
        for raw_expression in expressions:
            expression = str(raw_expression).replace("\\", "/").rstrip("/")
            has_wildcard = any(character in expression for character in "*?[")
            matches = (
                tuple(self.dataset_root.glob(expression))
                if has_wildcard
                else ((self.dataset_root / expression).resolve(),)
            )
            existing_match = False
            for match in matches:
                candidate = Path(match).resolve()
                if candidate.is_file():
                    existing_match = True
                    if candidate.suffix.lower() == extension:
                        files.add(candidate)
                elif candidate.is_dir():
                    existing_match = True
                    files.update(path.resolve() for path in candidate.glob(f"*{extension}"))
            if not existing_match:
                self._warning(
                    "missing_registry_path",
                    "Dataset registry path does not exist.",
                    (self.dataset_root / expression).resolve(),
                    section=section_name,
                    expression=raw_expression,
                )
        return tuple(sorted(files, key=lambda path: path.as_posix().lower()))

    @staticmethod
    def _register_alias(
        aliases: dict[str, str], alias: str, canonical: str, registry_name: str
    ) -> None:
        normalized = _normalize_handle(alias)
        previous = aliases.get(normalized)
        if previous is not None and previous != canonical:
            raise ReplicaCADParseError(
                f"Ambiguous {registry_name} alias {normalized!r}: {previous!r} vs {canonical!r}."
            )
        aliases[normalized] = canonical

    def _register_template_aliases(
        self,
        aliases: dict[str, str],
        canonical: str,
        prefix: str,
        filename: str,
        registry_name: str,
    ) -> None:
        for alias in (canonical, f"{prefix}/{canonical}", filename):
            self._register_alias(aliases, alias, canonical, registry_name)

    def _asset_path(self, config_path: Path, raw_path: object, role: str) -> Path:
        if not isinstance(raw_path, str) or not raw_path:
            raise ReplicaCADParseError(f"Missing {role} in {config_path}.")
        result = (config_path.parent / raw_path).resolve()
        if not result.is_file():
            raise ReplicaCADParseError(f"Missing {role}: {result}")
        return result

    def _load_stage_templates(self) -> None:
        suffix = ".stage_config.json"
        for path in self._registry_files("stages", ".json"):
            if not path.name.endswith(suffix):
                continue
            data = _read_json(path)
            handle = _strip_suffix(path.name, suffix)
            render_asset = self._asset_path(path, data.get("render_asset"), "stage render asset")
            scale = _scale3(data.get("scale", data.get("uniform_scale")), "stage scale")
            template = StageTemplateDesc(
                handle=handle,
                config_path=path,
                visual_asset=VisualAssetDesc(
                    source_path=render_asset,
                    scale=scale,
                    template_path=path,
                ),
                up=vector3(data.get("up", (0.0, 1.0, 0.0)), "stage up"),
                front=vector3(data.get("front", (0.0, 0.0, -1.0)), "stage front"),
                requires_lighting=bool(data.get("requires_lighting", True)),
            )
            self._stage_templates[handle] = template
            self._register_template_aliases(
                self._stage_aliases, handle, "stages", path.name, "stage"
            )

    def _load_object_templates(self) -> None:
        suffix = ".object_config.json"
        for path in self._registry_files("objects", ".json"):
            if not path.name.endswith(suffix):
                continue
            data = _read_json(path)
            handle = _strip_suffix(path.name, suffix)
            render_asset = self._asset_path(path, data.get("render_asset"), "object render asset")
            collision_asset = None
            if data.get("collision_asset"):
                collision_asset = self._asset_path(
                    path, data.get("collision_asset"), "object collision asset"
                )
            scale = _scale3(data.get("scale", data.get("uniform_scale")), "object scale")
            mass = None if data.get("mass") is None else float(data["mass"])
            template = ObjectTemplateDesc(
                handle=handle,
                config_path=path,
                visual_asset=VisualAssetDesc(
                    source_path=render_asset,
                    scale=scale,
                    template_path=path,
                    collision_source_path=collision_asset,
                ),
                mass=mass,
                com=vector3(data.get("COM", (0.0, 0.0, 0.0)), "object COM"),
                semantic_id=int(data.get("semantic_id", -1)),
            )
            self._object_templates[handle] = template
            self._register_template_aliases(
                self._object_aliases, handle, "objects", path.name, "object"
            )

    def _load_articulation_templates(self) -> None:
        paths = self._registry_files("articulated_objects", ".urdf")
        for path in paths:
            handle = path.stem
            template = parse_urdf_manifest(path, handle=handle)
            self._articulation_templates[handle] = template
            self._register_alias(
                self._articulation_aliases, handle, handle, "articulation"
            )
            self._register_alias(
                self._articulation_aliases, f"urdf/{path.parent.name}/{handle}", handle, "articulation"
            )
            if path.parent.name == handle:
                self._register_alias(
                    self._articulation_aliases, path.parent.name, handle, "articulation"
                )

    def _load_scene_registry(self) -> None:
        suffix = ".scene_instance.json"
        for path in self._registry_files("scene_instances", ".json"):
            if not path.name.endswith(suffix):
                continue
            handle = _strip_suffix(path.name, suffix)
            self._scene_paths[handle] = path
            for alias in (handle, f"scenes/{handle}", path.name, str(path)):
                self._register_alias(self._scene_aliases, alias, handle, "scene")

    def _load_lighting_registry(self) -> None:
        suffix = ".lighting_config.json"
        for path in self._registry_files("light_setups", ".json"):
            if not path.name.endswith(suffix):
                continue
            handle = _strip_suffix(path.name, suffix)
            self._lighting_paths[handle] = path
            for alias in (handle, f"lighting/{handle}", path.name):
                self._register_alias(self._lighting_aliases, alias, handle, "lighting")

    def _load_navmesh_registry(self) -> None:
        registry = self._config.get("navmesh_instances", {})
        if not isinstance(registry, dict):
            return
        for handle, raw_path in registry.items():
            path = (self.dataset_root / str(raw_path)).resolve()
            if path.is_file():
                self._navmesh_paths[str(handle)] = path
            else:
                self._warning(
                    "missing_navmesh_asset",
                    "Registered navmesh file does not exist.",
                    path,
                    handle=handle,
                )

    @staticmethod
    def _resolve_alias(
        value: object,
        aliases: Mapping[str, str],
        registry: Mapping[str, object],
        kind: str,
    ) -> str:
        raw = _normalize_handle(str(value))
        candidates = (raw, Path(raw).name)
        for candidate in candidates:
            canonical = aliases.get(candidate)
            if canonical is not None and canonical in registry:
                return canonical
        raise ReplicaCADParseError(f"Unknown ReplicaCAD {kind} template: {value!r}.")

    def _resolve_scene(self, value: str | Path) -> tuple[str, Path]:
        candidate_path = Path(value)
        if candidate_path.is_file():
            path = candidate_path.resolve()
            handle = _strip_suffix(path.name, ".scene_instance.json")
            return handle, path
        raw = _normalize_handle(str(value))
        for candidate in (raw, Path(raw).name):
            canonical = self._scene_aliases.get(candidate)
            if canonical is not None:
                return canonical, self._scene_paths[canonical]
        raise ReplicaCADParseError(f"Unknown ReplicaCAD scene instance: {value!r}.")

    def _instance_visual(
        self, template: VisualAssetDesc, instance_data: Mapping[str, object]
    ) -> VisualAssetDesc:
        instance_scale = _scale3(
            instance_data.get("scale", instance_data.get("uniform_scale")), "instance scale"
        )
        return VisualAssetDesc(
            source_path=template.source_path,
            scale=_multiply_scale(template.scale, instance_scale),
            template_path=template.template_path,
            collision_source_path=template.collision_source_path,
        )

    def _lighting_path(
        self, raw_handle: object, source_path: Path, warnings: list[ManifestWarning]
    ) -> Optional[Path]:
        if raw_handle in (None, "", "no_lights"):
            return None
        normalized = _normalize_handle(str(raw_handle))
        for candidate in (normalized, Path(normalized).name):
            canonical = self._lighting_aliases.get(candidate)
            if canonical is not None:
                return self._lighting_paths[canonical]
        warnings.append(
            ManifestWarning(
                code="unknown_lighting_template",
                message="Scene lighting template is not registered.",
                source_path=source_path,
                context=(("template_name", normalized),),
            )
        )
        return None

    def _navmesh_path(
        self, raw_handle: object, source_path: Path, warnings: list[ManifestWarning]
    ) -> Optional[Path]:
        if raw_handle in (None, ""):
            return None
        handle = str(raw_handle)
        candidates = [handle]
        if handle.endswith("_navmesh"):
            candidates.append(handle[: -len("_navmesh")])
        for candidate in candidates:
            path = self._navmesh_paths.get(candidate)
            if path is not None:
                return path
        warnings.append(
            ManifestWarning(
                code="unknown_navmesh_instance",
                message="Scene navmesh instance is not registered.",
                source_path=source_path,
                context=(("navmesh_instance", handle),),
            )
        )
        return None

    def parse_scene(self, scene: str | Path) -> SceneDesc:
        scene_name, scene_path = self._resolve_scene(scene)
        cached = self._scene_cache.get(scene_name)
        if cached is not None and cached.source_path == scene_path:
            return cached

        data = _read_json(scene_path)
        warnings: list[ManifestWarning] = []
        stage_data = data.get("stage_instance")
        if not isinstance(stage_data, dict):
            raise ReplicaCADParseError(f"Scene has no stage_instance object: {scene_path}")
        stage_handle = self._resolve_alias(
            stage_data.get("template_name"),
            self._stage_aliases,
            self._stage_templates,
            "stage",
        )
        stage_template = self._stage_templates[stage_handle]
        stage = InstanceDesc(
            name=f"{scene_name}/stage",
            template_name=stage_handle,
            visual_asset=self._instance_visual(stage_template.visual_asset, stage_data),
            pose=_pose(stage_data),
            motion_type="STATIC",
            semantic_id=-1,
            instance_id=0,
            translation_origin=str(stage_data.get("translation_origin", "ASSET_LOCAL")),
        )

        objects: list[InstanceDesc] = []
        raw_objects = data.get("object_instances", [])
        if not isinstance(raw_objects, list):
            raise ReplicaCADParseError(f"object_instances must be an array: {scene_path}")
        for source_index, instance_data in enumerate(raw_objects):
            if not isinstance(instance_data, dict):
                raise ReplicaCADParseError(
                    f"object_instances[{source_index}] must be an object: {scene_path}"
                )
            handle = self._resolve_alias(
                instance_data.get("template_name"),
                self._object_aliases,
                self._object_templates,
                "object",
            )
            template = self._object_templates[handle]
            objects.append(
                InstanceDesc(
                    name=f"{scene_name}/object/{source_index:04d}",
                    template_name=handle,
                    visual_asset=self._instance_visual(template.visual_asset, instance_data),
                    pose=_pose(instance_data),
                    motion_type=str(instance_data.get("motion_type", "STATIC")),
                    semantic_id=template.semantic_id,
                    instance_id=source_index + 1,
                    com=template.com,
                    mass=template.mass,
                    translation_origin=str(
                        instance_data.get("translation_origin", "ASSET_LOCAL")
                    ),
                )
            )

        articulated: list[ArticulationDesc] = []
        raw_articulated = data.get("articulated_object_instances", [])
        if not isinstance(raw_articulated, list):
            raise ReplicaCADParseError(
                f"articulated_object_instances must be an array: {scene_path}"
            )
        first_articulation_id = len(objects) + 1
        for source_index, instance_data in enumerate(raw_articulated):
            if not isinstance(instance_data, dict):
                raise ReplicaCADParseError(
                    f"articulated_object_instances[{source_index}] must be an object: {scene_path}"
                )
            handle = self._resolve_alias(
                instance_data.get("template_name"),
                self._articulation_aliases,
                self._articulation_templates,
                "articulation",
            )
            template = self._articulation_templates[handle]
            articulated.append(
                ArticulationDesc(
                    name=f"{scene_name}/articulation/{source_index:04d}",
                    template_name=handle,
                    urdf_path=template.urdf_path,
                    pose=_pose(instance_data),
                    motion_type=str(instance_data.get("motion_type", "DYNAMIC")),
                    instance_id=first_articulation_id + source_index,
                    fixed_base=bool(instance_data.get("fixed_base", False)),
                    uniform_scale=float(instance_data.get("uniform_scale", 1.0)),
                    translation_origin=str(
                        instance_data.get("translation_origin", "ASSET_LOCAL")
                    ),
                    auto_clamp_joint_limits=bool(
                        instance_data.get("auto_clamp_joint_limits", False)
                    ),
                    visuals=template.visuals,
                    joints=template.joints,
                )
            )

        result = SceneDesc(
            name=scene_name,
            source_path=scene_path,
            stage=stage,
            objects=tuple(objects),
            articulated=tuple(articulated),
            lighting_path=self._lighting_path(data.get("default_lighting"), scene_path, warnings),
            navmesh_path=self._navmesh_path(data.get("navmesh_instance"), scene_path, warnings),
            warnings=tuple(warnings),
        )
        if self._scene_paths.get(scene_name) == scene_path:
            self._scene_cache[scene_name] = result
        return result

    def parse_all_scenes(self) -> tuple[SceneDesc, ...]:
        return tuple(self.parse_scene(handle) for handle in self.scene_handles)

    def _portable_path(self, path: Optional[Path]) -> Optional[str]:
        if path is None:
            return None
        resolved = Path(path).resolve()
        try:
            return resolved.relative_to(self.dataset_root).as_posix()
        except ValueError:
            return os.path.relpath(resolved, self.dataset_root).replace("\\", "/")

    def _warning_dict(self, warning: ManifestWarning) -> dict[str, object]:
        return {
            "code": warning.code,
            "message": warning.message,
            "source_path": self._portable_path(warning.source_path),
            "context": dict(warning.context),
        }

    def _scene_digest_payload(self, scene: SceneDesc) -> dict[str, object]:
        def visual_asset_payload(asset: VisualAssetDesc) -> dict[str, object]:
            return {
                "source_path": self._portable_path(asset.source_path),
                "collision_source_path": self._portable_path(asset.collision_source_path),
                "template_path": self._portable_path(asset.template_path),
                "scale": asset.scale,
            }

        def instance_payload(instance: InstanceDesc) -> dict[str, object]:
            return {
                "id": instance.instance_id,
                "name": instance.name,
                "template": instance.template_name,
                "translation": instance.pose.translation,
                "rotation_wxyz": instance.pose.rotation_wxyz,
                "motion_type": instance.motion_type,
                "semantic_id": instance.semantic_id,
                "translation_origin": instance.translation_origin,
                "com": instance.com,
                "mass": instance.mass,
                "visual_asset": visual_asset_payload(instance.visual_asset),
            }

        def articulation_payload(instance: ArticulationDesc) -> dict[str, object]:
            return {
                "id": instance.instance_id,
                "name": instance.name,
                "template": instance.template_name,
                "urdf_path": self._portable_path(instance.urdf_path),
                "translation": instance.pose.translation,
                "rotation_wxyz": instance.pose.rotation_wxyz,
                "motion_type": instance.motion_type,
                "fixed_base": instance.fixed_base,
                "uniform_scale": instance.uniform_scale,
                "translation_origin": instance.translation_origin,
                "auto_clamp_joint_limits": instance.auto_clamp_joint_limits,
                "visuals": [
                    {
                        "link": visual.link_name,
                        "mesh_path": self._portable_path(visual.mesh_path),
                        "origin_xyz": visual.origin_xyz,
                        "origin_rpy": visual.origin_rpy,
                        "scale": visual.scale,
                    }
                    for visual in instance.visuals
                ],
                "joints": [
                    {
                        "name": joint.name,
                        "type": joint.joint_type,
                        "parent": joint.parent_link,
                        "child": joint.child_link,
                        "origin_xyz": joint.origin_xyz,
                        "origin_rpy": joint.origin_rpy,
                        "axis_xyz": joint.axis_xyz,
                        "limit_lower": joint.limit_lower,
                        "limit_upper": joint.limit_upper,
                    }
                    for joint in instance.joints
                ],
            }

        return {
            "name": scene.name,
            "source_path": self._portable_path(scene.source_path),
            "lighting_path": self._portable_path(scene.lighting_path),
            "navmesh_path": self._portable_path(scene.navmesh_path),
            "stage": instance_payload(scene.stage),
            "objects": [instance_payload(instance) for instance in scene.objects],
            "articulated": [
                articulation_payload(instance) for instance in scene.articulated
            ],
            "warnings": [self._warning_dict(warning) for warning in scene.warnings],
        }

    def build_report(self, scenes: Optional[Iterable[SceneDesc]] = None) -> dict[str, object]:
        parsed = tuple(self.parse_all_scenes() if scenes is None else scenes)
        all_warnings = list(self.warnings)
        for scene in parsed:
            all_warnings.extend(scene.warnings)
        warning_keys: set[tuple[object, ...]] = set()
        unique_warnings: list[ManifestWarning] = []
        for warning in all_warnings:
            key = (
                warning.code,
                warning.message,
                warning.source_path,
                warning.context,
            )
            if key not in warning_keys:
                warning_keys.add(key)
                unique_warnings.append(warning)

        digest_source = [self._scene_digest_payload(scene) for scene in parsed]
        digest = hashlib.sha256(
            json.dumps(digest_source, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        scene_stats = [
            {
                "name": scene.name,
                "source_path": self._portable_path(scene.source_path),
                "stage_template": scene.stage.template_name,
                "object_instances": len(scene.objects),
                "articulated_instances": len(scene.articulated),
                "warning_count": len(scene.warnings),
            }
            for scene in parsed
        ]
        missing_registered_resources = sum(
            1 for warning in self._warnings if warning.code.startswith("missing_")
        )
        used_articulation_templates = {
            instance.template_name for scene in parsed for instance in scene.articulated
        }
        missing_required_visual_assets = sum(
            1
            for handle in used_articulation_templates
            for warning in self._articulation_templates[handle].warnings
            if warning.code == "missing_urdf_visual_asset"
        )
        return {
            "schema_version": 1,
            "dataset_config": self._portable_path(self.dataset_config_path),
            "summary": {
                "registered_scenes": len(self._scene_paths),
                "parsed_scenes": len(parsed),
                "stage_templates": len(self._stage_templates),
                "object_templates": len(self._object_templates),
                "urdf_templates": len(self._articulation_templates),
                "object_instances": sum(len(scene.objects) for scene in parsed),
                "articulated_instances": sum(len(scene.articulated) for scene in parsed),
                "urdf_links": sum(
                    len(template.links) for template in self._articulation_templates.values()
                ),
                "urdf_visuals": sum(
                    len(template.visuals) for template in self._articulation_templates.values()
                ),
                "warnings": len(unique_warnings),
                "registry_warnings": len(self._warnings),
                "scene_warnings": sum(len(scene.warnings) for scene in parsed),
                "missing_registered_resources": missing_registered_resources,
                "missing_required_visual_assets": missing_required_visual_assets,
            },
            "determinism_digest": digest,
            "warnings": [self._warning_dict(warning) for warning in unique_warnings],
            "scenes": scene_stats,
        }


def load_replicacad_manifest(path: str | Path) -> ReplicaCADManifest:
    candidate = Path(path)
    if candidate.is_dir():
        return ReplicaCADManifest.from_dataset_root(candidate)
    return ReplicaCADManifest(candidate)
