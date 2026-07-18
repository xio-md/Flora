from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence


Vec3 = tuple[float, float, float]
QuatWxyz = tuple[float, float, float, float]
Matrix4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]

_TRANSLATION_ORIGINS = {"ASSET_LOCAL", "COM", "UNKNOWN"}


def _float_tuple(values: Sequence[float], length: int, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != length:
        raise ValueError(f"{name} must contain {length} values, got {len(result)}.")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite values.")
    return result


def vector3(values: Sequence[float], name: str = "vector") -> Vec3:
    x, y, z = _float_tuple(values, 3, name)
    return x, y, z


def normalize_quaternion_wxyz(values: Sequence[float]) -> QuatWxyz:
    w, x, y, z = _float_tuple(values, 4, "rotation_wxyz")
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1.0e-12:
        raise ValueError("rotation_wxyz must be non-zero.")
    return w / norm, x / norm, y / norm, z / norm


def normalize_translation_origin(value: object) -> str:
    origin = str(value).strip().upper()
    if origin not in _TRANSLATION_ORIGINS:
        expected = ", ".join(sorted(_TRANSLATION_ORIGINS))
        raise ValueError(f"translation_origin must be one of {expected}, got {value!r}.")
    return origin


def compose_transform_matrix(
    translation: Sequence[float], rotation_wxyz: Sequence[float]
) -> Matrix4:
    tx, ty, tz = vector3(translation, "translation")
    w, x, y, z = normalize_quaternion_wxyz(rotation_wxyz)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy), tx),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx), ty),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy), tz),
        (0.0, 0.0, 0.0, 1.0),
    )


@dataclass(frozen=True)
class PoseDesc:
    translation: Vec3 = (0.0, 0.0, 0.0)
    rotation_wxyz: QuatWxyz = (1.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "translation", vector3(self.translation, "translation"))
        object.__setattr__(
            self, "rotation_wxyz", normalize_quaternion_wxyz(self.rotation_wxyz)
        )

    @property
    def matrix_row_major(self) -> Matrix4:
        return compose_transform_matrix(self.translation, self.rotation_wxyz)


@dataclass(frozen=True)
class ManifestWarning:
    code: str
    message: str
    source_path: Optional[Path] = None
    context: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", str(self.code))
        object.__setattr__(self, "message", str(self.message))
        if self.source_path is not None:
            object.__setattr__(self, "source_path", Path(self.source_path).resolve())
        normalized = tuple(sorted((str(key), str(value)) for key, value in self.context))
        object.__setattr__(self, "context", normalized)


@dataclass(frozen=True)
class VisualAssetDesc:
    source_path: Path
    scale: Vec3 = (1.0, 1.0, 1.0)
    template_path: Optional[Path] = None
    collision_source_path: Optional[Path] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", Path(self.source_path).resolve())
        object.__setattr__(self, "scale", vector3(self.scale, "asset scale"))
        if self.template_path is not None:
            object.__setattr__(self, "template_path", Path(self.template_path).resolve())
        if self.collision_source_path is not None:
            object.__setattr__(
                self, "collision_source_path", Path(self.collision_source_path).resolve()
            )


@dataclass(frozen=True)
class StageTemplateDesc:
    handle: str
    config_path: Path
    visual_asset: VisualAssetDesc
    up: Vec3
    front: Vec3
    requires_lighting: bool


@dataclass(frozen=True)
class ObjectTemplateDesc:
    handle: str
    config_path: Path
    visual_asset: VisualAssetDesc
    mass: Optional[float]
    com: Vec3
    semantic_id: int


@dataclass(frozen=True)
class UrdfVisualDesc:
    link_name: str
    mesh_path: Path
    origin_xyz: Vec3 = (0.0, 0.0, 0.0)
    origin_rpy: Vec3 = (0.0, 0.0, 0.0)
    scale: Vec3 = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "mesh_path", Path(self.mesh_path).resolve())
        object.__setattr__(self, "origin_xyz", vector3(self.origin_xyz, "visual xyz"))
        object.__setattr__(self, "origin_rpy", vector3(self.origin_rpy, "visual rpy"))
        object.__setattr__(self, "scale", vector3(self.scale, "visual scale"))


@dataclass(frozen=True)
class UrdfJointDesc:
    name: str
    joint_type: str
    parent_link: str
    child_link: str
    origin_xyz: Vec3 = (0.0, 0.0, 0.0)
    origin_rpy: Vec3 = (0.0, 0.0, 0.0)
    axis_xyz: Vec3 = (1.0, 0.0, 0.0)
    limit_lower: Optional[float] = None
    limit_upper: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin_xyz", vector3(self.origin_xyz, "joint xyz"))
        object.__setattr__(self, "origin_rpy", vector3(self.origin_rpy, "joint rpy"))
        object.__setattr__(self, "axis_xyz", vector3(self.axis_xyz, "joint axis"))


@dataclass(frozen=True)
class ArticulationTemplateDesc:
    handle: str
    urdf_path: Path
    links: tuple[str, ...]
    visuals: tuple[UrdfVisualDesc, ...]
    joints: tuple[UrdfJointDesc, ...]
    warnings: tuple[ManifestWarning, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "urdf_path", Path(self.urdf_path).resolve())


@dataclass(frozen=True)
class InstanceDesc:
    name: str
    template_name: str
    visual_asset: VisualAssetDesc
    pose: PoseDesc
    motion_type: str
    semantic_id: int
    instance_id: int
    com: Vec3 = (0.0, 0.0, 0.0)
    mass: Optional[float] = None
    translation_origin: str = "ASSET_LOCAL"

    def __post_init__(self) -> None:
        object.__setattr__(self, "motion_type", str(self.motion_type).upper())
        object.__setattr__(self, "com", vector3(self.com, "center of mass"))
        object.__setattr__(
            self,
            "translation_origin",
            normalize_translation_origin(self.translation_origin),
        )


@dataclass(frozen=True)
class ArticulationDesc:
    name: str
    template_name: str
    urdf_path: Path
    pose: PoseDesc
    motion_type: str
    instance_id: int
    fixed_base: bool
    uniform_scale: float
    translation_origin: str
    auto_clamp_joint_limits: bool
    visuals: tuple[UrdfVisualDesc, ...]
    joints: tuple[UrdfJointDesc, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "urdf_path", Path(self.urdf_path).resolve())
        object.__setattr__(self, "motion_type", str(self.motion_type).upper())
        object.__setattr__(
            self,
            "translation_origin",
            normalize_translation_origin(self.translation_origin),
        )
        scale = float(self.uniform_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("uniform_scale must be positive and finite.")
        object.__setattr__(self, "uniform_scale", scale)


@dataclass(frozen=True)
class SceneDesc:
    name: str
    source_path: Path
    stage: InstanceDesc
    objects: tuple[InstanceDesc, ...]
    articulated: tuple[ArticulationDesc, ...]
    lighting_path: Optional[Path] = None
    navmesh_path: Optional[Path] = None
    warnings: tuple[ManifestWarning, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", Path(self.source_path).resolve())
        if self.lighting_path is not None:
            object.__setattr__(self, "lighting_path", Path(self.lighting_path).resolve())
        if self.navmesh_path is not None:
            object.__setattr__(self, "navmesh_path", Path(self.navmesh_path).resolve())
