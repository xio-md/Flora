from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

import numpy as np


def _float_array(values: Sequence[float], *, length: Optional[int] = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if length is not None and array.size != length:
        raise ValueError(f"Expected {length} values, got {array.size}.")
    return array


def _matrix4(values: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (4, 4):
        raise ValueError("Expected a 4x4 transform matrix.")
    return np.ascontiguousarray(array)


def _vector_array(values: Sequence[Sequence[float]], *, width: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"Expected an array of shape (N, {width}).")
    return np.ascontiguousarray(array)


def _triangle_array(values: Sequence[Sequence[int]]) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("Expected triangles with shape (N, 3).")
    if array.size > 0 and int(array.min()) < 0:
        raise ValueError("Triangle indices must be non-negative.")
    return np.ascontiguousarray(array.astype(np.uint32))


def _radius_array(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size > 0 and float(array.min()) < 0.0:
        raise ValueError("Particle radii must be non-negative.")
    return np.ascontiguousarray(array)


class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"


@dataclass
class _OwnershipState:
    runtime_generation: int
    scene_token: str
    owner_kind: str
    owner_name: str
    destroyed: bool = False


class _OwnedObject:
    def __init__(self) -> None:
        self._donut_owner_state: Optional[_OwnershipState] = None


class Transform:
    pass


class MatrixTransform(Transform):
    def __init__(self, matrix: Sequence[Sequence[float]]) -> None:
        self._matrix = np.eye(4, dtype=np.float32)
        self.update(matrix)

    def update(self, matrix: Sequence[Sequence[float]]) -> None:
        self._matrix = _matrix4(matrix)

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix.copy()


class Texture(_OwnedObject):
    def __init__(self) -> None:
        super().__init__()


class ColorTexture(Texture):
    def __init__(self, color: Sequence[float]) -> None:
        super().__init__()
        values = _float_array(color)
        if values.size == 0:
            raise ValueError("ColorTexture requires at least one channel.")
        self.color = tuple(float(v) for v in values)


class ImageTexture(Texture):
    def __init__(
        self,
        file: str = "",
        image_data: bytes | str = b"",
        width: int = 0,
        height: int = 0,
        channel: int = 0,
        scale: Optional[Sequence[float]] = None,
        encoding: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.file = str(file)
        self.image_data = image_data if isinstance(image_data, bytes) else image_data.encode("utf-8")
        self.width = int(width)
        self.height = int(height)
        self.channel = int(channel)
        self.scale = None if scale is None else tuple(float(v) for v in _float_array(scale))
        self.encoding = encoding


class Light(_OwnedObject):
    def __init__(
        self,
        name: str,
        emission: Optional[Texture] = None,
        intensity: float = 1.0,
        two_sided: bool = False,
        beam_angle: float = 180.0,
    ) -> None:
        super().__init__()
        self.name = str(name)
        self.emission = emission
        self.intensity = float(intensity)
        self.two_sided = bool(two_sided)
        self.beam_angle = float(beam_angle)


class Subsurface(_OwnedObject):
    def __init__(self) -> None:
        super().__init__()


class UniformSubsurface(Subsurface):
    def __init__(self, name: str, thickness: Optional[Texture] = None) -> None:
        super().__init__()
        self.name = str(name)
        self.thickness = thickness


class Surface(_OwnedObject):
    def __init__(self, name: str, double_sided: bool = False) -> None:
        super().__init__()
        self.name = str(name)
        self.double_sided = bool(double_sided)


class PlasticSurface(Surface):
    def __init__(
        self,
        name: str,
        roughness: Optional[Texture] = None,
        opacity: Optional[Texture] = None,
        normal_map: Optional[Texture] = None,
        kd: Optional[Texture] = None,
        ks: Optional[Texture] = None,
        eta: Optional[Texture] = None,
        double_sided: bool = False,
    ) -> None:
        super().__init__(name, double_sided=double_sided)
        self.roughness = roughness
        self.opacity = opacity
        self.normal_map = normal_map
        self.kd = kd
        self.ks = ks
        self.eta = eta


class DisneySurface(Surface):
    def __init__(
        self,
        name: str,
        roughness: Optional[Texture] = None,
        opacity: Optional[Texture] = None,
        normal_map: Optional[Texture] = None,
        kd: Optional[Texture] = None,
        eta: Optional[Texture] = None,
        metallic: Optional[Texture] = None,
        specular_tint: Optional[Texture] = None,
        specular_trans: Optional[Texture] = None,
        diffuse_trans: Optional[Texture] = None,
        double_sided: bool = False,
    ) -> None:
        super().__init__(name, double_sided=double_sided)
        self.roughness = roughness
        self.opacity = opacity
        self.normal_map = normal_map
        self.kd = kd
        self.eta = eta
        self.metallic = metallic
        self.specular_tint = specular_tint
        self.specular_trans = specular_trans
        self.diffuse_trans = diffuse_trans


class MetalSurface(Surface):
    def __init__(
        self,
        name: str,
        roughness: Optional[Texture] = None,
        opacity: Optional[Texture] = None,
        normal_map: Optional[Texture] = None,
        kd: Optional[Texture] = None,
        eta: Optional[str | Texture] = None,
        double_sided: bool = False,
    ) -> None:
        super().__init__(name, double_sided=double_sided)
        self.roughness = roughness
        self.opacity = opacity
        self.normal_map = normal_map
        self.kd = kd
        self.eta = eta


class GlassSurface(Surface):
    def __init__(
        self,
        name: str,
        roughness: Optional[Texture] = None,
        opacity: Optional[Texture] = None,
        normal_map: Optional[Texture] = None,
        ks: Optional[Texture] = None,
        kt: Optional[Texture] = None,
        eta: Optional[Texture] = None,
        double_sided: bool = False,
    ) -> None:
        super().__init__(name, double_sided=double_sided)
        self.roughness = roughness
        self.opacity = opacity
        self.normal_map = normal_map
        self.ks = ks
        self.kt = kt
        self.eta = eta


class Shape(_OwnedObject):
    def __init__(
        self,
        name: str,
        surface: Optional[Surface | str] = None,
        emission: Optional[Light | str] = None,
        subsurface: Optional[Subsurface | str] = None,
        clamp_normal: float = 180.0,
    ) -> None:
        super().__init__()
        self.name = str(name)
        self.surface = surface
        self.emission = emission
        self.subsurface = subsurface
        self.clamp_normal = float(clamp_normal)


class RigidShape(Shape):
    def __init__(
        self,
        name: str,
        obj_path: str = "",
        vertices: Optional[Sequence[Sequence[float]]] = None,
        triangles: Optional[Sequence[Sequence[int]]] = None,
        normals: Optional[Sequence[Sequence[float]]] = None,
        uvs: Optional[Sequence[Sequence[float]]] = None,
        transform: Optional[Transform] = None,
        surface: Optional[Surface | str] = None,
        emission: Optional[Light | str] = None,
        subsurface: Optional[Subsurface | str] = None,
        clamp_normal: float = 180.0,
    ) -> None:
        super().__init__(name, surface=surface, emission=emission, subsurface=subsurface, clamp_normal=clamp_normal)
        self.obj_path = str(obj_path)
        self.vertices = np.empty((0, 3), dtype=np.float32)
        self.triangles = np.empty((0, 3), dtype=np.uint32)
        self.normals = np.empty((0, 3), dtype=np.float32)
        self.uvs = np.empty((0, 2), dtype=np.float32)
        self.transform = transform
        if vertices is not None:
            self.vertices = _vector_array(vertices, width=3)
        if triangles is not None:
            self.triangles = _triangle_array(triangles)
        if normals is not None:
            self.normals = _vector_array(normals, width=3)
        if uvs is not None:
            self.uvs = _vector_array(uvs, width=2)

    def update(self, transform: Optional[Transform] = None) -> None:
        self.transform = transform


class DeformableShape(Shape):
    def __init__(
        self,
        name: str,
        vertices: Optional[Sequence[Sequence[float]]] = None,
        triangles: Optional[Sequence[Sequence[int]]] = None,
        normals: Optional[Sequence[Sequence[float]]] = None,
        uvs: Optional[Sequence[Sequence[float]]] = None,
        surface: Optional[Surface | str] = None,
        emission: Optional[Light | str] = None,
        subsurface: Optional[Subsurface | str] = None,
        clamp_normal: float = 180.0,
    ) -> None:
        super().__init__(name, surface=surface, emission=emission, subsurface=subsurface, clamp_normal=clamp_normal)
        self.vertices = np.empty((0, 3), dtype=np.float32)
        self.triangles = np.empty((0, 3), dtype=np.uint32)
        self.normals = np.empty((0, 3), dtype=np.float32)
        self.uvs = np.empty((0, 2), dtype=np.float32)
        if vertices is not None and triangles is not None:
            self.update(vertices, triangles, normals=normals, uvs=uvs)

    def update(
        self,
        vertices: Sequence[Sequence[float]],
        triangles: Sequence[Sequence[int]],
        normals: Optional[Sequence[Sequence[float]]] = None,
        uvs: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        self.vertices = _vector_array(vertices, width=3)
        self.triangles = _triangle_array(triangles)
        self.normals = np.empty((0, 3), dtype=np.float32) if normals is None else _vector_array(normals, width=3)
        self.uvs = np.empty((0, 2), dtype=np.float32) if uvs is None else _vector_array(uvs, width=2)


class ParticlesShape(Shape):
    def __init__(
        self,
        name: str,
        centers: Optional[Sequence[Sequence[float]]] = None,
        radii: Optional[Sequence[float]] = None,
        subdivision: int = 0,
        surface: Optional[Surface | str] = None,
        emission: Optional[Light | str] = None,
        subsurface: Optional[Subsurface | str] = None,
        clamp_normal: float = 180.0,
    ) -> None:
        super().__init__(name, surface=surface, emission=emission, subsurface=subsurface, clamp_normal=clamp_normal)
        self.subdivision = int(subdivision)
        self.centers = np.empty((0, 3), dtype=np.float32)
        self.radii = np.empty((0,), dtype=np.float32)
        if centers is not None:
            radii_arg: Sequence[float] = [] if radii is None else radii
            self.update(centers, radii_arg)

    def update(
        self,
        centers: Sequence[Sequence[float]] = (),
        radii: Sequence[float] = (),
    ) -> None:
        self.centers = _vector_array(centers, width=3) if len(centers) > 0 else np.empty((0, 3), dtype=np.float32)
        self.radii = _radius_array(radii)
        if self.radii.size not in (0, self.centers.shape[0]):
            raise ValueError("ParticlesShape.radii must be empty or match the number of centers.")


class Film:
    def __init__(self, resolution: Sequence[int]) -> None:
        array = np.asarray(resolution, dtype=np.int32).reshape(-1)
        if array.size != 2:
            raise ValueError("Film resolution must contain exactly two integers.")
        self.resolution = (int(array[0]), int(array[1]))


class Filter:
    def __init__(self, radius: float = 1.0) -> None:
        self.radius = 1.0
        self.update(radius)

    def update(self, radius: float = 1.0) -> None:
        self.radius = float(radius)


class Camera(_OwnedObject):
    def __init__(self, name: str, pose: Optional[Transform], film: Film, filter: Optional[Filter], spp: int) -> None:
        super().__init__()
        self.name = str(name)
        self.pose = pose
        self.film = film
        self.filter = Filter() if filter is None else filter
        self.spp = int(spp)


class PinholeCamera(Camera):
    def __init__(
        self,
        name: str,
        pose: Optional[Transform],
        film: Film,
        filter: Optional[Filter],
        spp: int,
        fov: float,
    ) -> None:
        super().__init__(name, pose=pose, film=film, filter=filter, spp=spp)
        self.fov = float(fov)

    def update(self, pose: Optional[Transform] = None, fov: Optional[float] = None) -> None:
        if pose is not None:
            self.pose = pose
        if fov is not None:
            self.fov = float(fov)


class ThinLensCamera(Camera):
    def __init__(
        self,
        name: str,
        pose: Optional[Transform],
        film: Film,
        filter: Optional[Filter],
        spp: int,
        aperture: float,
        focal_len: float,
        focus_dis: float,
    ) -> None:
        super().__init__(name, pose=pose, film=film, filter=filter, spp=spp)
        self.aperture = float(aperture)
        self.focal_len = float(focal_len)
        self.focus_dis = float(focus_dis)

    def update(
        self,
        pose: Optional[Transform] = None,
        aperture: Optional[float] = None,
        focal_len: Optional[float] = None,
        focus_dis: Optional[float] = None,
    ) -> None:
        if pose is not None:
            self.pose = pose
        if aperture is not None:
            self.aperture = float(aperture)
        if focal_len is not None:
            self.focal_len = float(focal_len)
        if focus_dis is not None:
            self.focus_dis = float(focus_dis)


class Environment(_OwnedObject):
    def __init__(self, name: str, emission: Optional[Texture] = None, transform: Optional[Transform] = None) -> None:
        super().__init__()
        self.name = str(name)
        self.emission = emission
        self.transform = transform


class Integrator:
    pass


class WavePathIntegrator(Integrator):
    def __init__(
        self,
        log_level: LogLevel = LogLevel.WARNING,
        enable_cache: bool = True,
        max_depth: int = 32,
        rr_depth: int = 0,
        rr_threshold: float = 0.95,
    ) -> None:
        self.log_level = log_level
        self.enable_cache = bool(enable_cache)
        self.max_depth = int(max_depth)
        self.rr_depth = int(rr_depth)
        self.rr_threshold = float(rr_threshold)


class Spectrum:
    pass


class SRGBSpectrum(Spectrum):
    pass


class Render(_OwnedObject):
    def __init__(self, name: str, spectrum: Spectrum, integrator: Integrator, clamp_normal: float = 180.0) -> None:
        super().__init__()
        self.name = str(name)
        self.spectrum = spectrum
        self.integrator = integrator
        self.clamp_normal = float(clamp_normal)
