from __future__ import annotations

import hashlib
import os
import time as _time
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from rtxns_genesis_style import CameraDesc as _BackendCameraDesc
from rtxns_genesis_style import GenesisStyleRenderer
from rtxns_genesis_style import SurfaceDesc as _BackendSurfaceDesc

from .errors import (
    InvalidStateError,
    RuntimeNotInitializedError,
    SceneDestroyedError,
    SceneNotInitializedError,
    UnsupportedFeatureError,
)
from .objects import (
    Camera,
    ColorTexture,
    DeformableShape,
    DisneySurface,
    Environment,
    GlassSurface,
    ImageTexture,
    Light,
    MatrixTransform,
    MetalSurface,
    ParticlesShape,
    PinholeCamera,
    PlasticSurface,
    Render,
    RigidShape,
    Shape,
    Surface,
    Texture,
    ThinLensCamera,
    UniformSubsurface,
    _OwnedObject,
    _OwnershipState,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_module_dir() -> Path:
    platform_dir = "windows-x64" if os.name == "nt" else "linux-x64"
    return _repo_root() / "bin" / platform_dir


@dataclass(frozen=True)
class RuntimeOptions:
    context_path: Path
    context_id: str
    backend: str
    device_index: int
    log_level: object
    enable_debug: bool
    module_dir: Path
    runtime_dir: Path


@dataclass(frozen=True)
class _UpdateOperation:
    name: str
    scope: str
    execution_layer: str
    status: str
    current_path: str
    next_target: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "scope": self.scope,
            "execution_layer": self.execution_layer,
            "status": self.status,
            "current_path": self.current_path,
            "next_target": self.next_target,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class _SceneUpdatePlan:
    mode: str
    time: float
    dirty_categories: tuple[str, ...]
    dirty_sources: dict[str, tuple[str, ...]]
    force_render: bool
    backend_rebuilt: bool
    environment_applied: bool
    operations: tuple[_UpdateOperation, ...]
    blockers: tuple[str, ...]
    cxx_candidates: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "time": float(self.time),
            "dirty_categories": tuple(self.dirty_categories),
            "dirty_sources": {
                category: tuple(sources)
                for category, sources in self.dirty_sources.items()
            },
            "force_render": bool(self.force_render),
            "backend_rebuilt": bool(self.backend_rebuilt),
            "environment_applied": bool(self.environment_applied),
            "operations": [operation.to_dict() for operation in self.operations],
            "blockers": tuple(self.blockers),
            "cxx_candidates": tuple(self.cxx_candidates),
        }


def _copy_update_plan_dict(plan: dict[str, object]) -> dict[str, object]:
    return {
        "mode": str(plan.get("mode", "not_run")),
        "time": float(plan.get("time", 0.0)),
        "dirty_categories": tuple(plan.get("dirty_categories", ())),
        "dirty_sources": {
            str(category): tuple(sources)
            for category, sources in dict(plan.get("dirty_sources", {})).items()
        },
        "force_render": bool(plan.get("force_render", False)),
        "backend_rebuilt": bool(plan.get("backend_rebuilt", False)),
        "environment_applied": bool(plan.get("environment_applied", False)),
        "operations": [dict(operation) for operation in list(plan.get("operations", ()))],
        "blockers": tuple(plan.get("blockers", ())),
        "cxx_candidates": tuple(plan.get("cxx_candidates", ())),
    }


def _copy_update_report(report: dict[str, object]) -> dict[str, object]:
    copied = {
        "mode": str(report.get("mode", "not_run")),
        "dirty_categories": tuple(report.get("dirty_categories", ())),
        "dirty_sources": {
            str(category): tuple(sources)
            for category, sources in dict(report.get("dirty_sources", {})).items()
        },
        "duration_ms": float(report.get("duration_ms", 0.0)),
        "time": float(report.get("time", 0.0)),
        "backend_rebuilt": bool(report.get("backend_rebuilt", False)),
        "environment_applied": bool(report.get("environment_applied", False)),
    }
    copied["plan"] = _copy_update_plan_dict(dict(report.get("plan", {})))
    return copied


class _ModuleRuntime:
    def __init__(self) -> None:
        self.options: Optional[RuntimeOptions] = None
        self.scenes: weakref.WeakSet[Scene] = weakref.WeakSet()
        self.generation = 0
        self._next_scene_index = 0

    def ensure_initialized(self) -> RuntimeOptions:
        if self.options is None:
            raise RuntimeNotInitializedError("Call DonutRenderPy.init(...) before create_scene().")
        return self.options

    def register_scene(self, scene: "Scene") -> None:
        self.scenes.add(scene)

    def allocate_scene_token(self) -> str:
        self._next_scene_index += 1
        return f"scene-{self.generation}-{self._next_scene_index}"

    def reset(self) -> None:
        self.options = None
        self.scenes = weakref.WeakSet()
        self.generation += 1
        self._next_scene_index = 0


_RUNTIME = _ModuleRuntime()


def _texture_to_values(texture: Optional[Texture], channels: int, default: tuple[float, ...]) -> np.ndarray:
    if texture is None:
        return np.asarray(default, dtype=np.float32)

    if isinstance(texture, ColorTexture):
        values = np.asarray(texture.color, dtype=np.float32)
    elif isinstance(texture, ImageTexture):
        values = np.asarray(default if texture.scale is None else texture.scale, dtype=np.float32)
    elif np.isscalar(texture):
        values = np.asarray([texture], dtype=np.float32)
    else:
        raise UnsupportedFeatureError(f"Unsupported texture object: {type(texture).__name__}")

    if values.size == 1 and channels > 1:
        values = np.repeat(values, channels)
    elif values.size < channels:
        padding = np.asarray(default, dtype=np.float32)
        values = np.concatenate([values, padding[values.size : channels]])
    else:
        values = values[:channels]
    return np.clip(values.astype(np.float32), 0.0, 1.0)


def _texture_to_scalar(texture: Optional[Texture], default: float) -> float:
    return float(_texture_to_values(texture, 1, (default,))[0])


def _light_to_emissive(light: Optional[Light]) -> np.ndarray:
    if light is None:
        return np.zeros((3,), dtype=np.float32)
    return _texture_to_values(light.emission, 3, (0.0, 0.0, 0.0)) * float(light.intensity)


def _surface_to_backend_desc(surface: Optional[Surface], light: Optional[Light]) -> _BackendSurfaceDesc:
    if surface is None:
        base = (0.8, 0.8, 0.8, 1.0)
        roughness = 1.0
        metallic = 0.0
        double_sided = False
    elif isinstance(surface, PlasticSurface):
        base = _texture_to_values(surface.kd, 4, (0.8, 0.8, 0.8, 1.0))
        base[3] = _texture_to_scalar(surface.opacity, base[3])
        roughness = _texture_to_scalar(surface.roughness, 1.0)
        metallic = 0.0
        double_sided = surface.double_sided
    elif isinstance(surface, MetalSurface):
        base = _texture_to_values(surface.kd, 4, (0.85, 0.85, 0.85, 1.0))
        base[3] = _texture_to_scalar(surface.opacity, base[3])
        roughness = _texture_to_scalar(surface.roughness, 0.2)
        metallic = 1.0
        double_sided = surface.double_sided
    elif isinstance(surface, GlassSurface):
        base = _texture_to_values(surface.ks, 4, (0.9, 0.9, 0.95, 0.08))
        base[3] = _texture_to_scalar(surface.opacity, base[3])
        roughness = _texture_to_scalar(surface.roughness, 0.02)
        metallic = 0.0
        double_sided = surface.double_sided
    else:
        base = _texture_to_values(getattr(surface, "kd", None), 4, (0.8, 0.8, 0.8, 1.0))
        base[3] = _texture_to_scalar(getattr(surface, "opacity", None), base[3])
        roughness = _texture_to_scalar(getattr(surface, "roughness", None), 1.0)
        metallic = _texture_to_scalar(getattr(surface, "metallic", None), 0.0)
        double_sided = bool(getattr(surface, "double_sided", False))

    emissive = _light_to_emissive(light)
    return _BackendSurfaceDesc(
        base_color=tuple(float(v) for v in base),
        roughness=float(np.clip(roughness, 0.0, 1.0)),
        metallic=float(np.clip(metallic, 0.0, 1.0)),
        emissive=tuple(float(v) for v in emissive),
        double_sided=bool(double_sided),
    )


def _resolve_transform_matrix(transform: Optional[MatrixTransform]) -> np.ndarray:
    if transform is None:
        return np.eye(4, dtype=np.float32)
    if not isinstance(transform, MatrixTransform):
        raise UnsupportedFeatureError(
            f"{type(transform).__name__} is not supported in the Week 2 adapter. Use MatrixTransform."
        )
    return transform.matrix


def _camera_to_backend_desc(camera: Camera) -> _BackendCameraDesc:
    if isinstance(camera, ThinLensCamera):
        raise UnsupportedFeatureError("ThinLensCamera is part of the Week 2 API draft but not executable yet.")
    if not isinstance(camera, PinholeCamera):
        raise UnsupportedFeatureError(f"Unsupported camera type: {type(camera).__name__}")

    pose = _resolve_transform_matrix(camera.pose)
    position = tuple(float(v) for v in pose[:3, 3])
    lookat = tuple(float(v) for v in pose[:3, 3] - pose[:3, 2])
    up = tuple(float(v) for v in pose[:3, 1])
    return _BackendCameraDesc(
        uid=camera.name,
        pos=position,
        lookat=lookat,
        up=up,
        res=camera.film.resolution,
        fov=float(camera.fov),
        near=0.1,
        far=1000.0,
        model="pinhole",
    )


def _array_digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(str(array.dtype).encode("ascii"))
    hasher.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    hasher.update(array.view(np.uint8))
    return hasher.hexdigest()


def _transform_fingerprint(transform: object) -> object:
    if transform is None:
        return None
    if isinstance(transform, MatrixTransform):
        return ("MatrixTransform", _array_digest(transform.matrix))
    return (type(transform).__name__, repr(transform))


def _texture_fingerprint(texture: Optional[Texture]) -> object:
    if texture is None:
        return None
    if isinstance(texture, ColorTexture):
        return ("ColorTexture", tuple(float(v) for v in texture.color))
    if isinstance(texture, ImageTexture):
        image_bytes = bytes(texture.image_data)
        image_digest = hashlib.blake2b(image_bytes, digest_size=12).hexdigest()
        return (
            "ImageTexture",
            texture.file,
            texture.width,
            texture.height,
            texture.channel,
            None if texture.scale is None else tuple(float(v) for v in texture.scale),
            texture.encoding,
            image_digest,
        )
    return (type(texture).__name__, repr(texture))


def _light_fingerprint(light: Optional[Light]) -> object:
    if light is None:
        return None
    return (
        type(light).__name__,
        light.name,
        _texture_fingerprint(light.emission),
        float(light.intensity),
        bool(light.two_sided),
        float(light.beam_angle),
    )


def _subsurface_fingerprint(subsurface: Optional[UniformSubsurface]) -> object:
    if subsurface is None:
        return None
    return (
        type(subsurface).__name__,
        subsurface.name,
        _texture_fingerprint(subsurface.thickness),
    )


def _surface_fingerprint(surface: Optional[Surface]) -> object:
    if surface is None:
        return None

    return (
        type(surface).__name__,
        surface.name,
        bool(surface.double_sided),
        _texture_fingerprint(getattr(surface, "roughness", None)),
        _texture_fingerprint(getattr(surface, "opacity", None)),
        _texture_fingerprint(getattr(surface, "normal_map", None)),
        _texture_fingerprint(getattr(surface, "kd", None)),
        _texture_fingerprint(getattr(surface, "ks", None)),
        _texture_fingerprint(getattr(surface, "kt", None)),
        _texture_fingerprint(getattr(surface, "eta", None)),
        _texture_fingerprint(getattr(surface, "metallic", None)),
        _texture_fingerprint(getattr(surface, "specular_tint", None)),
        _texture_fingerprint(getattr(surface, "specular_trans", None)),
        _texture_fingerprint(getattr(surface, "diffuse_trans", None)),
    )


def _environment_fingerprint(environment: Optional[Environment]) -> object:
    if environment is None:
        return None
    return (
        type(environment).__name__,
        environment.name,
        _texture_fingerprint(environment.emission),
        _transform_fingerprint(environment.transform),
    )


def _camera_fingerprint(camera: Camera) -> object:
    return (
        type(camera).__name__,
        camera.name,
        _transform_fingerprint(camera.pose),
        tuple(int(v) for v in camera.film.resolution),
        float(camera.filter.radius),
        int(camera.spp),
        float(getattr(camera, "fov", 0.0)),
        float(getattr(camera, "aperture", 0.0)),
        float(getattr(camera, "focal_len", 0.0)),
        float(getattr(camera, "focus_dis", 0.0)),
    )


def _object_reference_fingerprint(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, Surface):
        return ("SurfaceObject", _surface_fingerprint(value))
    if isinstance(value, Light):
        return ("LightObject", _light_fingerprint(value))
    if isinstance(value, UniformSubsurface):
        return ("SubsurfaceObject", _subsurface_fingerprint(value))
    return ("Reference", str(value))


def _shape_state(shape: Shape) -> dict[str, object]:
    binding = (
        _object_reference_fingerprint(shape.surface),
        _object_reference_fingerprint(shape.emission),
        _object_reference_fingerprint(shape.subsurface),
        float(shape.clamp_normal),
    )

    if isinstance(shape, RigidShape):
        return {
            "binding": binding,
            "geometry": (
                type(shape).__name__,
                shape.name,
                str(shape.obj_path),
                _array_digest(shape.vertices),
                _array_digest(shape.triangles),
                _array_digest(shape.normals),
                _array_digest(shape.uvs),
            ),
            "transform": _transform_fingerprint(shape.transform),
        }

    if isinstance(shape, DeformableShape):
        return {
            "binding": binding,
            "geometry": (
                type(shape).__name__,
                shape.name,
                _array_digest(shape.vertices),
                _array_digest(shape.triangles),
                _array_digest(shape.normals),
                _array_digest(shape.uvs),
            ),
            "transform": None,
        }

    if isinstance(shape, ParticlesShape):
        return {
            "binding": binding,
            "geometry": (
                type(shape).__name__,
                shape.name,
                int(shape.subdivision),
                _array_digest(shape.centers),
                _array_digest(shape.radii),
            ),
            "transform": None,
        }

    return {
        "binding": binding,
        "geometry": (type(shape).__name__, shape.name, repr(shape)),
        "transform": None,
    }


class Scene:
    def __init__(self, options: RuntimeOptions, runtime_generation: int, scene_token: str) -> None:
        self._options = options
        self._runtime_generation = int(runtime_generation)
        self._scene_token = str(scene_token)
        self._render: Optional[Render] = None
        self._backend: Optional[GenesisStyleRenderer] = None
        self._initialized = False
        self._destroyed = False
        self._time = 0.0

        self._environments: dict[str, Environment] = {}
        self._emissions: dict[str, Light] = {}
        self._subsurfaces: dict[str, UniformSubsurface] = {}
        self._surfaces: dict[str, Surface] = {}
        self._shapes: dict[str, Shape] = {}
        self._cameras: dict[str, tuple[Camera, bool]] = {}
        self._dirty_flags: set[str] = set()
        self._dirty_sources: dict[str, set[str]] = {}
        self._owned_objects: weakref.WeakSet[_OwnedObject] = weakref.WeakSet()
        self._environment_snapshots: dict[str, object] = {}
        self._emission_snapshots: dict[str, object] = {}
        self._subsurface_snapshots: dict[str, object] = {}
        self._surface_snapshots: dict[str, object] = {}
        self._shape_snapshots: dict[str, dict[str, object]] = {}
        self._camera_snapshots: dict[str, object] = {}
        initial_plan = _SceneUpdatePlan(
            mode="not_run",
            time=0.0,
            dirty_categories=(),
            dirty_sources={},
            force_render=False,
            backend_rebuilt=False,
            environment_applied=False,
            operations=(),
            blockers=(),
            cxx_candidates=(),
        ).to_dict()
        self._last_update_plan: dict[str, object] = _copy_update_plan_dict(initial_plan)
        self._last_update_report: dict[str, object] = {
            "mode": "not_run",
            "dirty_categories": (),
            "dirty_sources": {},
            "duration_ms": 0.0,
            "time": 0.0,
            "backend_rebuilt": False,
            "environment_applied": False,
            "plan": _copy_update_plan_dict(initial_plan),
        }
        self._update_history: list[dict[str, object]] = []

    def _check_alive(self) -> None:
        if self._destroyed:
            raise SceneDestroyedError("This Scene has been destroyed.")

    def _check_ready(self) -> None:
        self._check_alive()
        if not self._initialized:
            raise SceneNotInitializedError("Call Scene.init(render) before Scene.update_*() or render_frame().")

    def _object_name(self, obj: object) -> str:
        if hasattr(obj, "name"):
            return str(getattr(obj, "name"))
        if hasattr(obj, "file"):
            path = str(getattr(obj, "file"))
            return path if path else "<unnamed>"
        return "<unnamed>"

    def _describe_object(self, obj: object) -> str:
        return f"{type(obj).__name__} '{self._object_name(obj)}'"

    def _mark_dirty(self, category: str, source: str) -> None:
        self._dirty_flags.add(category)
        self._dirty_sources.setdefault(category, set()).add(source)

    def _dirty_source_snapshot(self) -> dict[str, tuple[str, ...]]:
        return {
            category: tuple(sorted(sources))
            for category, sources in sorted(self._dirty_sources.items())
        }

    def _clear_dirty_state(self) -> None:
        self._dirty_flags.clear()
        self._dirty_sources.clear()

    def _make_operation(
        self,
        *,
        name: str,
        scope: str,
        execution_layer: str,
        status: str,
        current_path: str,
        next_target: str,
        reason: str,
    ) -> _UpdateOperation:
        return _UpdateOperation(
            name=name,
            scope=scope,
            execution_layer=execution_layer,
            status=status,
            current_path=current_path,
            next_target=next_target,
            reason=reason,
        )

    def _dirty_camera_names(self, dirty_sources: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
        result: list[str] = []
        for source in dirty_sources.get("camera", ()):
            if source.startswith("camera:"):
                result.append(source.split(":", 1)[1])
        return tuple(result)

    def _dirty_transform_shape_names(self, dirty_sources: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
        result: list[str] = []
        for source in dirty_sources.get("transform", ()):
            if not source.startswith("shape:"):
                continue
            payload = source.split(":", 1)[1]
            if "." not in payload:
                continue
            shape_name, suffix = payload.rsplit(".", 1)
            if suffix == "transform":
                result.append(shape_name)
        return tuple(result)

    def _queue_incremental_backend_updates(self, plan: dict[str, object]) -> None:
        if self._backend is None:
            raise InvalidStateError("Scene backend is unavailable while queuing incremental updates.")

        dirty_sources = {
            str(category): tuple(sources)
            for category, sources in dict(plan.get("dirty_sources", {})).items()
        }
        for camera_name in self._dirty_camera_names(dirty_sources):
            if camera_name not in self._cameras:
                continue
            camera, _denoise = self._cameras[camera_name]
            self._backend.update_camera(_camera_to_backend_desc(camera))

        for shape_name in self._dirty_transform_shape_names(dirty_sources):
            shape = self._shapes.get(shape_name)
            if not isinstance(shape, RigidShape):
                continue
            self._backend.update_rigid(shape_name, _resolve_transform_matrix(shape.transform))

    def _build_update_plan(self, *, time_value: float) -> dict[str, object]:
        dirty_categories = tuple(sorted(self._dirty_flags))
        dirty_sources = self._dirty_source_snapshot()
        operations: list[_UpdateOperation] = []
        blockers: list[str] = []
        cxx_candidates: list[str] = []

        def add_candidate(candidate: str) -> None:
            if candidate not in cxx_candidates:
                cxx_candidates.append(candidate)

        mode = "no_change"
        force_render = False
        backend_rebuilt = False
        environment_applied = False

        if self._backend is None:
            mode = "full_rebuild"
            force_render = True
            backend_rebuilt = True
            blockers.append("backend scene is not created yet, so the first update must replay the full Scene state.")
            operations.append(
                self._make_operation(
                    name="bootstrap_backend",
                    scope="scene",
                    execution_layer="python_adapter",
                    status="current",
                    current_path="Scene._rebuild_backend()",
                    next_target="native Scene resource handles",
                    reason="The adapter has no live backend scene yet, so environment, camera, and shape state must be rebuilt.",
                )
            )
            operations.append(
                self._make_operation(
                    name="advance_scene",
                    scope="scene",
                    execution_layer="backend_wrapper",
                    status="current",
                    current_path="GenesisStyleRenderer.update_scene(force_render=True)",
                    next_target="native Scene.update_scene(...)",
                    reason="The initial build still enters the GLB-backed load_scene() path in the current backend.",
                )
            )
            return _SceneUpdatePlan(
                mode=mode,
                time=float(time_value),
                dirty_categories=dirty_categories,
                dirty_sources=dirty_sources,
                force_render=force_render,
                backend_rebuilt=backend_rebuilt,
                environment_applied=environment_applied,
                operations=tuple(operations),
                blockers=tuple(blockers),
                cxx_candidates=tuple(cxx_candidates),
            ).to_dict()

        if not dirty_categories:
            operations.append(
                self._make_operation(
                    name="advance_scene",
                    scope="scene",
                    execution_layer="backend_wrapper",
                    status="current",
                    current_path="GenesisStyleRenderer.update_scene(force_render=False)",
                    next_target="native Scene.update_scene(...)",
                    reason="No dirty-state is pending, so we only advance the existing backend scene.",
                )
            )
            return _SceneUpdatePlan(
                mode=mode,
                time=float(time_value),
                dirty_categories=dirty_categories,
                dirty_sources=dirty_sources,
                force_render=force_render,
                backend_rebuilt=backend_rebuilt,
                environment_applied=environment_applied,
                operations=tuple(operations),
                blockers=tuple(blockers),
                cxx_candidates=tuple(cxx_candidates),
            ).to_dict()

        rebuild_categories = tuple(category for category in dirty_categories if category not in {"camera", "environment", "transform"})
        if rebuild_categories:
            mode = "full_rebuild"
            force_render = True
            backend_rebuilt = True

            if "camera" in dirty_categories:
                operations.append(
                    self._make_operation(
                        name="apply_camera",
                        scope="camera",
                        execution_layer="backend_wrapper",
                        status="current",
                        current_path="Scene._queue_incremental_backend_updates() -> GenesisStyleRenderer.update_camera()",
                        next_target="HeadlessPbrScene.set_camera(...)",
                        reason="Camera state can now be pushed through the backend without forcing a Scene rebuild.",
                    )
                )

            if "surface" in rebuild_categories:
                add_candidate("update_surface")
                blockers.append("surface dirty-state still falls back to full rebuild because material handles only exist in the GLB snapshot.")
                operations.append(
                    self._make_operation(
                        name="surface_incremental_candidate",
                        scope="surface.binding",
                        execution_layer="cxx_boundary",
                        status="planned_native",
                        current_path="full Scene rebuild via Scene._rebuild_backend()",
                        next_target="native surface/material update",
                        reason="The adapter can classify material changes, but the native backend cannot patch them in place yet.",
                    )
                )

            if "geometry" in rebuild_categories:
                add_candidate("update_shape")
                blockers.append("geometry dirty-state still falls back to full rebuild because mesh handles are not exposed incrementally.")
                operations.append(
                    self._make_operation(
                        name="geometry_incremental_candidate",
                        scope="shape.geometry",
                        execution_layer="cxx_boundary",
                        status="planned_native",
                        current_path="full Scene rebuild via Scene._rebuild_backend()",
                        next_target="native deformable/particles/mesh update",
                        reason="Shape add/update helpers already exist in Python, but the native scene only accepts a rebuilt GLB payload today.",
                    )
                )

            operations.append(
                self._make_operation(
                    name="rebuild_backend",
                    scope="scene",
                    execution_layer="python_adapter",
                    status="current",
                    current_path="Scene._rebuild_backend()",
                    next_target="incremental Scene.update_* handles",
                    reason=f"Dirty categories {', '.join(rebuild_categories)} still require a full backend replay in the current adapter.",
                )
            )
            operations.append(
                self._make_operation(
                    name="advance_scene",
                    scope="scene",
                    execution_layer="backend_wrapper",
                    status="current",
                    current_path="GenesisStyleRenderer.update_scene(force_render=True)",
                    next_target="native Scene.update_scene(...)",
                    reason="The rebuild path still refreshes the GLB snapshot before rendering can continue.",
                )
            )
            return _SceneUpdatePlan(
                mode=mode,
                time=float(time_value),
                dirty_categories=dirty_categories,
                dirty_sources=dirty_sources,
                force_render=force_render,
                backend_rebuilt=backend_rebuilt,
                environment_applied=environment_applied,
                operations=tuple(operations),
                blockers=tuple(blockers),
                cxx_candidates=tuple(cxx_candidates),
            ).to_dict()

        if "transform" in dirty_categories:
            mode = "incremental_camera_transform_environment"
        else:
            mode = "incremental_camera_environment"

        if "environment" in dirty_categories:
            environment_applied = True
            operations.append(
                self._make_operation(
                    name="apply_environment",
                    scope="environment",
                    execution_layer="backend_wrapper",
                    status="current",
                    current_path="Scene._apply_environment() -> GenesisStyleRenderer.set_ambient()/set_default_light()",
                    next_target="HeadlessPbrScene.set_ambient()/set_default_light()",
                    reason="Ambient lighting already updates the live backend scene without rebuilding geometry.",
                )
            )

        if "camera" in dirty_categories:
            operations.append(
                self._make_operation(
                    name="apply_camera",
                    scope="camera",
                    execution_layer="backend_wrapper",
                    status="current",
                    current_path="Scene._queue_incremental_backend_updates() -> GenesisStyleRenderer.update_camera()",
                    next_target="HeadlessPbrScene.set_camera(...)",
                    reason="Camera-only updates now stay on the live backend path instead of requiring a rebuild.",
                )
            )
        if "transform" in dirty_categories:
            operations.append(
                self._make_operation(
                    name="apply_rigid_transforms",
                    scope="shape.transform",
                    execution_layer="backend_wrapper",
                    status="current",
                    current_path="Scene._queue_incremental_backend_updates() -> GenesisStyleRenderer.update_rigid()",
                    next_target="HeadlessPbrScene.update_node_transform(...)",
                    reason="Rigid shape transforms now update the existing SceneGraph nodes without reloading GLB geometry.",
                )
            )

        operations.append(
            self._make_operation(
                name="advance_scene",
                scope="scene",
                execution_layer="backend_wrapper",
                status="current",
                current_path="GenesisStyleRenderer.update_scene(force_render=False)",
                next_target="native Scene.update_scene(...)",
                reason="Camera, transform, and environment changes can reuse the existing backend scene without a full reload.",
            )
        )
        return _SceneUpdatePlan(
            mode=mode,
            time=float(time_value),
            dirty_categories=dirty_categories,
            dirty_sources=dirty_sources,
            force_render=force_render,
            backend_rebuilt=backend_rebuilt,
            environment_applied=environment_applied,
            operations=tuple(operations),
            blockers=tuple(blockers),
            cxx_candidates=tuple(cxx_candidates),
        ).to_dict()

    def _execute_update_plan(self, plan: dict[str, object]) -> None:
        if bool(plan.get("backend_rebuilt", False)):
            self._rebuild_backend()
        elif bool(plan.get("environment_applied", False)):
            self._apply_environment()

        if self._backend is None:
            raise InvalidStateError("Scene backend is unavailable after executing the update plan.")
        if not bool(plan.get("backend_rebuilt", False)):
            self._queue_incremental_backend_updates(plan)
        self._backend.update_scene(
            force_render=bool(plan.get("force_render", False)),
            time=self._time,
        )

    def _record_update_report(
        self,
        *,
        plan: dict[str, object],
        duration_ms: float,
    ) -> None:
        copied_plan = _copy_update_plan_dict(plan)
        report = {
            "mode": str(copied_plan["mode"]),
            "dirty_categories": tuple(copied_plan["dirty_categories"]),
            "dirty_sources": {
                category: tuple(sources)
                for category, sources in dict(copied_plan["dirty_sources"]).items()
            },
            "duration_ms": float(duration_ms),
            "time": float(self._time),
            "backend_rebuilt": bool(copied_plan["backend_rebuilt"]),
            "environment_applied": bool(copied_plan["environment_applied"]),
            "plan": copied_plan,
        }
        self._last_update_plan = _copy_update_plan_dict(copied_plan)
        self._last_update_report = _copy_update_report(report)
        self._update_history.append(_copy_update_report(report))
        if len(self._update_history) > 64:
            self._update_history.pop(0)

    def get_dirty_state(self) -> dict[str, object]:
        self._check_alive()
        return {
            "categories": tuple(sorted(self._dirty_flags)),
            "sources": self._dirty_source_snapshot(),
        }

    def get_update_stats(self) -> dict[str, object]:
        self._check_alive()
        return _copy_update_report(self._last_update_report)

    def get_update_history(self) -> list[dict[str, object]]:
        self._check_alive()
        return [_copy_update_report(report) for report in self._update_history]

    def get_last_update_plan(self) -> dict[str, object]:
        self._check_alive()
        return _copy_update_plan_dict(self._last_update_plan)

    def preview_update_plan(self, time: Optional[float] = None) -> dict[str, object]:
        self._check_ready()
        time_value = self._time if time is None else float(time)
        return self._build_update_plan(time_value=time_value)

    def _iter_owned_dependencies(self, obj: _OwnedObject):
        if isinstance(obj, Environment):
            if isinstance(obj.emission, _OwnedObject):
                yield obj.emission
            return

        if isinstance(obj, Light):
            if isinstance(obj.emission, _OwnedObject):
                yield obj.emission
            return

        if isinstance(obj, UniformSubsurface):
            if isinstance(obj.thickness, _OwnedObject):
                yield obj.thickness
            return

        if isinstance(obj, (PlasticSurface, DisneySurface, MetalSurface, GlassSurface)):
            texture_fields = (
                "roughness",
                "opacity",
                "normal_map",
                "kd",
                "ks",
                "kt",
                "eta",
                "metallic",
                "specular_tint",
                "specular_trans",
                "diffuse_trans",
            )
            for field_name in texture_fields:
                value = getattr(obj, field_name, None)
                if isinstance(value, _OwnedObject):
                    yield value
            return

        if isinstance(obj, Shape):
            for value in (obj.surface, obj.emission, obj.subsurface):
                if isinstance(value, _OwnedObject):
                    yield value
            return

    def _track_owned_object(self, obj: Optional[_OwnedObject], *, role: str) -> None:
        if obj is None:
            return

        state = obj._donut_owner_state
        if state is None:
            obj._donut_owner_state = _OwnershipState(
                runtime_generation=self._runtime_generation,
                scene_token=self._scene_token,
                owner_kind=role,
                owner_name=self._object_name(obj),
            )
        elif state.destroyed:
            raise InvalidStateError(
                f"{self._describe_object(obj)} belongs to a destroyed Scene and cannot be reused. "
                f"Create a new {type(obj).__name__} instance."
            )
        elif state.runtime_generation != self._runtime_generation:
            raise InvalidStateError(
                f"{self._describe_object(obj)} belongs to a previous DonutRenderPy runtime and cannot be reused "
                "after destroy(). Create a fresh object after re-initializing the module."
            )
        elif state.scene_token != self._scene_token:
            raise InvalidStateError(
                f"{self._describe_object(obj)} is already attached to another Scene and cannot be reused here."
            )

        self._owned_objects.add(obj)
        for dependency in self._iter_owned_dependencies(obj):
            self._track_owned_object(dependency, role=f"{role}-dependency")

    def _mark_owned_objects_destroyed(self) -> None:
        for obj in list(self._owned_objects):
            state = obj._donut_owner_state
            if state is not None:
                state.destroyed = True
        self._owned_objects.clear()

    def init(self, render: Render) -> None:
        self._check_alive()
        if self._initialized:
            raise InvalidStateError("Scene.init(render) can only be called once per Scene.")
        self._track_owned_object(render, role="render")
        self._render = render
        self._initialized = True

    def _stage_environment(self, environment: Environment) -> None:
        state = _environment_fingerprint(environment)
        previous = self._environment_snapshots.get(environment.name)
        self._environments[environment.name] = environment
        self._environment_snapshots[environment.name] = state
        if previous != state:
            self._mark_dirty("environment", f"environment:{environment.name}")

    def update_environment(self, environment: Environment) -> None:
        self._check_ready()
        self._track_owned_object(environment, role="environment")
        self._stage_environment(environment)

    def _stage_emission(self, light: Light) -> None:
        state = _light_fingerprint(light)
        previous = self._emission_snapshots.get(light.name)
        self._emissions[light.name] = light
        self._emission_snapshots[light.name] = state
        if previous != state:
            self._mark_dirty("surface", f"emission:{light.name}")

    def update_emission(self, light: Light) -> None:
        self._check_ready()
        self._track_owned_object(light, role="light")
        self._stage_emission(light)

    def _stage_subsurface(self, subsurface: UniformSubsurface) -> None:
        state = _subsurface_fingerprint(subsurface)
        previous = self._subsurface_snapshots.get(subsurface.name)
        self._subsurfaces[subsurface.name] = subsurface
        self._subsurface_snapshots[subsurface.name] = state
        if previous != state:
            self._mark_dirty("surface", f"subsurface:{subsurface.name}")

    def update_subsurface(self, subsurface: UniformSubsurface) -> None:
        self._check_ready()
        self._track_owned_object(subsurface, role="subsurface")
        self._stage_subsurface(subsurface)

    def _stage_surface(self, surface: Surface) -> None:
        state = _surface_fingerprint(surface)
        previous = self._surface_snapshots.get(surface.name)
        self._surfaces[surface.name] = surface
        self._surface_snapshots[surface.name] = state
        if previous != state:
            self._mark_dirty("surface", f"surface:{surface.name}")

    def update_surface(self, surface: Surface) -> None:
        self._check_ready()
        self._track_owned_object(surface, role="surface")
        self._stage_surface(surface)

    def _stage_shape(self, shape: Shape) -> None:
        state = _shape_state(shape)
        previous = self._shape_snapshots.get(shape.name)

        if isinstance(shape.surface, Surface):
            self._surfaces[shape.surface.name] = shape.surface
            self._surface_snapshots[shape.surface.name] = _surface_fingerprint(shape.surface)
        if isinstance(shape.emission, Light):
            self._emissions[shape.emission.name] = shape.emission
            self._emission_snapshots[shape.emission.name] = _light_fingerprint(shape.emission)
        if isinstance(shape.subsurface, UniformSubsurface):
            self._subsurfaces[shape.subsurface.name] = shape.subsurface
            self._subsurface_snapshots[shape.subsurface.name] = _subsurface_fingerprint(shape.subsurface)

        self._shapes[shape.name] = shape
        self._shape_snapshots[shape.name] = state

        if previous is None:
            self._mark_dirty("geometry", f"shape:{shape.name}.create")
            return

        if previous["geometry"] != state["geometry"]:
            self._mark_dirty("geometry", f"shape:{shape.name}.geometry")
        if previous["transform"] != state["transform"]:
            self._mark_dirty("transform", f"shape:{shape.name}.transform")
        if previous["binding"] != state["binding"]:
            self._mark_dirty("surface", f"shape:{shape.name}.binding")

    def update_shape(self, shape: Shape) -> None:
        self._check_ready()
        self._track_owned_object(shape, role="shape")
        self._stage_shape(shape)

    def _stage_camera(self, camera: Camera, denoise: bool) -> None:
        state = _camera_fingerprint(camera)
        previous = self._camera_snapshots.get(camera.name)
        self._cameras[camera.name] = (camera, bool(denoise))
        self._camera_snapshots[camera.name] = state
        if previous != state:
            self._mark_dirty("camera", f"camera:{camera.name}")

    def update_camera(self, camera: Camera, denoise: bool) -> None:
        self._check_ready()
        self._track_owned_object(camera, role="camera")
        self._stage_camera(camera, denoise)

    def update_scene(self, time: float) -> None:
        self._check_ready()
        self._time = float(time)
        plan = self._build_update_plan(time_value=self._time)
        started = _time.perf_counter()
        self._execute_update_plan(plan)
        duration_ms = (_time.perf_counter() - started) * 1000.0
        self._record_update_report(
            plan=plan,
            duration_ms=duration_ms,
        )
        self._clear_dirty_state()

    def render_frame(self, camera: Camera | str) -> bytes:
        self._check_ready()
        camera_object = self._resolve_camera(camera)
        self.update_scene(time=self._time)
        if self._backend is None:
            raise InvalidStateError("Scene backend is unavailable after update_scene().")
        rgba = self._backend.render_camera_rgba(
            _camera_to_backend_desc(camera_object),
            force_render=False,
            time=self._time,
        )
        return np.ascontiguousarray(rgba, dtype=np.uint8).tobytes()

    def destroy(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        if self._backend is not None:
            self._backend.destroy()
            self._backend = None
        self._environments.clear()
        self._emissions.clear()
        self._subsurfaces.clear()
        self._surfaces.clear()
        self._shapes.clear()
        self._cameras.clear()
        self._environment_snapshots.clear()
        self._emission_snapshots.clear()
        self._subsurface_snapshots.clear()
        self._surface_snapshots.clear()
        self._shape_snapshots.clear()
        self._camera_snapshots.clear()
        self._clear_dirty_state()
        self._mark_owned_objects_destroyed()

    def _resolve_surface(self, surface_ref: Optional[Surface | str]) -> Optional[Surface]:
        if surface_ref is None:
            return None
        if isinstance(surface_ref, Surface):
            return surface_ref
        if surface_ref not in self._surfaces:
            raise InvalidStateError(f"Unknown surface reference '{surface_ref}'.")
        return self._surfaces[surface_ref]

    def _resolve_light(self, light_ref: Optional[Light | str]) -> Optional[Light]:
        if light_ref is None:
            return None
        if isinstance(light_ref, Light):
            return light_ref
        if light_ref not in self._emissions:
            raise InvalidStateError(f"Unknown emission reference '{light_ref}'.")
        return self._emissions[light_ref]

    def _resolve_camera(self, camera_ref: Camera | str) -> Camera:
        if isinstance(camera_ref, Camera):
            self._track_owned_object(camera_ref, role="camera")
            denoise = self._cameras.get(camera_ref.name, (camera_ref, False))[1]
            self._stage_camera(camera_ref, denoise)
            return camera_ref
        if camera_ref not in self._cameras:
            raise InvalidStateError(f"Unknown camera reference '{camera_ref}'.")
        return self._cameras[camera_ref][0]

    def _rebuild_backend(self) -> None:
        if self._backend is not None:
            self._backend.destroy()

        self._backend = GenesisStyleRenderer(
            module_dir=self._options.module_dir,
            runtime_dir=self._options.runtime_dir,
            backend=self._options.backend,
            device_index=self._options.device_index,
            enable_debug=self._options.enable_debug,
        )
        self._apply_environment()

        for camera, _denoise in self._cameras.values():
            self._backend.add_camera(_camera_to_backend_desc(camera))

        for shape in self._shapes.values():
            self._sync_shape(shape)

    def _apply_environment(self) -> None:
        if not self._environments:
            return

        latest = next(reversed(self._environments.values()))
        if latest.emission is None:
            return

        rgb = tuple(float(v) for v in _texture_to_values(latest.emission, 3, (0.03, 0.04, 0.06)))
        self._backend.set_ambient(rgb, rgb)
        # The native backend rejects non-positive irradiance, so disable the fallback
        # directional light by zeroing its color instead of passing irradiance=0.
        self._backend.set_default_light(direction=(-0.4, -1.0, -0.6), color=(0.0, 0.0, 0.0), irradiance=1.0)

    def _sync_shape(self, shape: Shape) -> None:
        surface = self._resolve_surface(shape.surface)
        light = self._resolve_light(shape.emission)
        backend_surface = _surface_to_backend_desc(surface, light)
        self._backend.add_surface(shape.name, backend_surface)

        if isinstance(shape, RigidShape):
            if shape.obj_path:
                raise UnsupportedFeatureError("RigidShape.obj_path is part of the draft API but not executed yet.")
            self._backend.add_rigid(shape.name, shape.vertices, shape.triangles, shape.normals, shape.uvs)
            if shape.transform is not None:
                self._backend.update_rigid(shape.name, _resolve_transform_matrix(shape.transform))
            return

        if isinstance(shape, DeformableShape):
            self._backend.add_deformable(shape.name)
            self._backend.update_deformable(shape.name, shape.vertices, shape.triangles, shape.normals, shape.uvs)
            return

        if isinstance(shape, ParticlesShape):
            radius = float(shape.radii[0]) if shape.radii.size > 0 else 0.05
            self._backend.add_particles(shape.name, radius=radius)
            self._backend.update_particles(shape.name, shape.centers, radius=radius, particles_radii=shape.radii)
            return

        raise UnsupportedFeatureError(f"Unsupported shape type: {type(shape).__name__}")


def init(
    context_path: str | os.PathLike[str] | None = None,
    context_id: str = "",
    backend: str = "vulkan",
    device_index: int = -1,
    log_level: object = None,
    *,
    enable_debug: bool = False,
    module_dir: str | os.PathLike[str] | None = None,
    runtime_dir: str | os.PathLike[str] | None = None,
) -> None:
    resolved_module_dir = Path(module_dir) if module_dir is not None else _default_module_dir()
    if runtime_dir is not None:
        resolved_runtime_dir = Path(runtime_dir)
    elif context_path is not None:
        resolved_runtime_dir = Path(context_path)
    else:
        resolved_runtime_dir = _repo_root()

    options = RuntimeOptions(
        context_path=Path(context_path) if context_path is not None else resolved_runtime_dir,
        context_id=str(context_id),
        backend=str(backend),
        device_index=int(device_index),
        log_level=log_level,
        enable_debug=bool(enable_debug),
        module_dir=resolved_module_dir,
        runtime_dir=resolved_runtime_dir,
    )
    if _RUNTIME.options is None:
        _RUNTIME.options = options
        return
    if _RUNTIME.options != options:
        raise InvalidStateError("DonutRenderPy is already initialized with different options.")


def create_scene() -> Scene:
    options = _RUNTIME.ensure_initialized()
    scene = Scene(options, runtime_generation=_RUNTIME.generation, scene_token=_RUNTIME.allocate_scene_token())
    _RUNTIME.register_scene(scene)
    return scene


def destroy() -> None:
    for scene in list(_RUNTIME.scenes):
        scene.destroy()
    _RUNTIME.reset()
