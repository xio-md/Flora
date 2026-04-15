from __future__ import annotations

import importlib
import shutil
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .glb_builder import GlbSceneBuilder


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_module_dir() -> Path:
    return _repo_root() / "bin" / "windows-x64"


def _default_runtime_dir() -> Path:
    return _repo_root()


def _import_native_renderer_module():
    errors: list[Exception] = []
    for module_name in ("DonutRenderPyNative", "RtxRenderPy"):
        try:
            return importlib.import_module(module_name)
        except ImportError as exc:
            errors.append(exc)
    details = "; ".join(str(exc) for exc in errors)
    raise ImportError(f"Failed to import Donut native renderer module. {details}")


def _shape_name(name: str, batch_index: Optional[int]) -> str:
    return name if batch_index is None else f"{name}_{batch_index}"


def _as_float_array(values: Any, length: Optional[int] = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if length is not None and array.size != length:
        raise ValueError(f"Expected {length} values, got {array.size}.")
    return array


def _coerce_vertices(vertices: Any) -> np.ndarray:
    array = np.asarray(vertices, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("Vertices must have shape (N, 3).")
    return np.ascontiguousarray(array)


def _coerce_triangles(triangles: Any) -> np.ndarray:
    array = np.asarray(triangles, dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("Triangles must have shape (N, 3).")
    if array.size > 0 and np.min(array) < 0:
        raise ValueError("Triangle indices must be non-negative.")
    return np.ascontiguousarray(array.astype(np.uint32))


def _coerce_normals(normals: Any, vertex_count: int) -> np.ndarray:
    if normals is None:
        return np.empty((0, 3), dtype=np.float32)
    array = np.asarray(normals, dtype=np.float32)
    if array.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] != vertex_count:
        raise ValueError("Normals must have shape (N, 3) and match the vertex count.")
    return np.ascontiguousarray(array)


def _coerce_uvs(uvs: Any, vertex_count: int) -> np.ndarray:
    if uvs is None:
        return np.empty((0, 2), dtype=np.float32)
    array = np.asarray(uvs, dtype=np.float32)
    if array.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] != vertex_count:
        raise ValueError("UVs must have shape (N, 2) and match the vertex count.")
    return np.ascontiguousarray(array)


def _coerce_transform(matrix: Any) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float32)
    if array.shape != (4, 4):
        raise ValueError("Transforms must have shape (4, 4).")
    return np.ascontiguousarray(array)


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    valid = lengths[:, 0] > 1.0e-12
    result = np.zeros_like(vectors, dtype=np.float32)
    if np.any(valid):
        result[valid] = vectors[valid] / lengths[valid]
    if np.any(~valid):
        result[~valid] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return result


def _compute_vertex_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float32)
    if triangles.size == 0 or vertices.size == 0:
        return normals

    tri_vertices = vertices[triangles]
    face_normals = np.cross(
        tri_vertices[:, 1] - tri_vertices[:, 0],
        tri_vertices[:, 2] - tri_vertices[:, 0],
    )
    for corner in range(3):
        np.add.at(normals, triangles[:, corner], face_normals)
    return _normalize_vectors(normals)


def _transform_vertices(vertices: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    linear = matrix[:3, :3]
    translation = matrix[:3, 3]
    return np.ascontiguousarray(vertices @ linear.T + translation, dtype=np.float32)


def _transform_normals(normals: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if normals.size == 0:
        return normals
    linear = matrix[:3, :3]
    normal_matrix = np.linalg.inv(linear).T
    return np.ascontiguousarray(_normalize_vectors(normals @ normal_matrix.T), dtype=np.float32)


def _extract_texture_value(texture: Any, channels: int, default: Sequence[float]) -> np.ndarray:
    if texture is None:
        return np.asarray(default, dtype=np.float32)

    if isinstance(texture, Mapping):
        if "textures" in texture:
            items = texture.get("textures") or []
            return _extract_texture_value(items[0] if items else None, channels, default)
        if "color" in texture:
            texture = texture["color"]
        elif "image_color" in texture:
            texture = texture["image_color"]

    if hasattr(texture, "textures"):
        items = getattr(texture, "textures")
        texture = items[0] if items else None
        return _extract_texture_value(texture, channels, default)

    if texture is None:
        return np.asarray(default, dtype=np.float32)

    if hasattr(texture, "color"):
        values = np.asarray(getattr(texture, "color"), dtype=np.float32).reshape(-1)
    elif hasattr(texture, "mean_color"):
        values = np.asarray(texture.mean_color(), dtype=np.float32).reshape(-1)
        image_color = getattr(texture, "image_color", None)
        if image_color is not None:
            scale = np.asarray(image_color, dtype=np.float32).reshape(-1)
            values = values * scale[: values.size]
    elif hasattr(texture, "image_color"):
        values = np.asarray(getattr(texture, "image_color"), dtype=np.float32).reshape(-1)
    elif np.isscalar(texture):
        values = np.asarray([texture], dtype=np.float32)
    else:
        values = np.asarray(texture, dtype=np.float32).reshape(-1)

    if values.size == 0:
        values = np.asarray(default, dtype=np.float32)
    if values.size == 1 and channels > 1:
        values = np.repeat(values, channels)
    elif values.size < channels:
        tail = np.asarray(default, dtype=np.float32)
        values = np.concatenate([values, tail[values.size : channels]])
    else:
        values = values[:channels]
    return np.clip(values.astype(np.float32), 0.0, 1.0)


def _coerce_surface_desc(surface: Any) -> "SurfaceDesc":
    if isinstance(surface, SurfaceDesc):
        return surface

    if surface is None:
        return SurfaceDesc()

    if isinstance(surface, Mapping):
        base = surface.get("base_color", surface.get("color", (0.8, 0.8, 0.8, 1.0)))
        if "opacity" in surface and len(base) < 4:
            base = tuple(base) + (surface["opacity"],)
        emissive = surface.get("emissive", (0.0, 0.0, 0.0))
        return SurfaceDesc(
            base_color=tuple(_extract_texture_value(base, 4, (0.8, 0.8, 0.8, 1.0))),
            roughness=float(np.clip(surface.get("roughness", 1.0), 0.0, 1.0)),
            metallic=float(np.clip(surface.get("metallic", 0.0), 0.0, 1.0)),
            emissive=tuple(_extract_texture_value(emissive, 3, (0.0, 0.0, 0.0))),
            double_sided=bool(surface.get("double_sided", False)),
        )

    rgba_source = surface.get_rgba() if hasattr(surface, "get_rgba") else getattr(surface, "color", None)
    emission_source = surface.get_emission() if hasattr(surface, "get_emission") else getattr(surface, "emissive", None)
    roughness_source = getattr(surface, "roughness_texture", getattr(surface, "roughness", None))
    metallic_source = getattr(surface, "metallic_texture", getattr(surface, "metallic", None))

    return SurfaceDesc(
        base_color=tuple(_extract_texture_value(rgba_source, 4, (0.8, 0.8, 0.8, 1.0))),
        roughness=float(_extract_texture_value(roughness_source, 1, (1.0,))[0]),
        metallic=float(_extract_texture_value(metallic_source, 1, (0.0,))[0]),
        emissive=tuple(_extract_texture_value(emission_source, 3, (0.0, 0.0, 0.0))),
        double_sided=bool(getattr(surface, "double_sided", False) or False),
    )


def _coerce_camera_desc(camera: Any) -> "CameraDesc":
    if isinstance(camera, CameraDesc):
        return camera

    if isinstance(camera, Mapping):
        source = camera
        uid = str(source.get("uid", source.get("name", "camera")))
        model = str(source.get("model", "pinhole"))
        pos = source.get("pos", source.get("position"))
        lookat = source.get("lookat", source.get("target"))
        up = source.get("up", (0.0, 1.0, 0.0))
        res = source.get("res", source.get("resolution", (512, 512)))
        fov = float(source.get("fov", 45.0))
        near = float(source.get("near", 0.1))
        far = float(source.get("far", 1000.0))
    else:
        uid = str(getattr(camera, "uid", getattr(camera, "name", "camera")))
        model = str(getattr(camera, "model", "pinhole"))
        pos = getattr(camera, "pos", getattr(camera, "position", None))
        lookat = getattr(camera, "lookat", getattr(camera, "target", None))
        up = getattr(camera, "up", (0.0, 1.0, 0.0))
        res = getattr(camera, "res", getattr(camera, "resolution", (512, 512)))
        fov = float(getattr(camera, "fov", 45.0))
        near = float(getattr(camera, "near", 0.1))
        far = float(getattr(camera, "far", 1000.0))

    if pos is None or lookat is None:
        raise ValueError("Camera must provide both position/pos and target/lookat.")
    if model != "pinhole":
        raise ValueError("GenesisStyleRenderer MVP-1 only supports pinhole cameras.")

    res_array = np.asarray(res, dtype=np.int32).reshape(-1)
    if res_array.size != 2:
        raise ValueError("Camera resolution must contain two integers.")

    return CameraDesc(
        uid=uid,
        model=model,
        pos=tuple(_as_float_array(pos, 3)),
        lookat=tuple(_as_float_array(lookat, 3)),
        up=tuple(_as_float_array(up, 3)),
        res=(int(res_array[0]), int(res_array[1])),
        fov=fov,
        near=near,
        far=far,
    )


def _make_unit_sphere(latitudes: int, longitudes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    normals: list[list[float]] = []
    uvs: list[list[float]] = []
    triangles: list[list[int]] = []

    for lat in range(latitudes + 1):
        v = lat / latitudes
        phi = np.pi * v
        sin_phi = float(np.sin(phi))
        cos_phi = float(np.cos(phi))

        for lon in range(longitudes + 1):
            u = lon / longitudes
            theta = 2.0 * np.pi * u
            sin_theta = float(np.sin(theta))
            cos_theta = float(np.cos(theta))

            x = sin_phi * cos_theta
            y = cos_phi
            z = sin_phi * sin_theta
            normals.append([x, y, z])
            vertices.append([x, y, z])
            uvs.append([u, 1.0 - v])

    row = longitudes + 1
    for lat in range(latitudes):
        for lon in range(longitudes):
            a = lat * row + lon
            b = a + row
            c = a + 1
            d = b + 1
            if lat != 0:
                triangles.append([a, b, c])
            if lat != latitudes - 1:
                triangles.append([c, b, d])

    return (
        np.asarray(vertices, dtype=np.float32),
        np.asarray(triangles, dtype=np.uint32),
        np.asarray(uvs, dtype=np.float32),
    )


class _RuntimeHandle:
    _lock = threading.Lock()
    _module = None
    _options = None
    _refcount = 0

    @classmethod
    def acquire(
        cls,
        module_dir: Path,
        runtime_dir: Path,
        backend: str,
        device_index: int,
        enable_debug: bool,
    ):
        module_dir = module_dir.resolve()
        runtime_dir = runtime_dir.resolve()
        if str(module_dir) not in sys.path:
            sys.path.insert(0, str(module_dir))

        module = _import_native_renderer_module()
        options = (module.__name__, str(module_dir), str(runtime_dir), backend, int(device_index), bool(enable_debug))

        with cls._lock:
            if cls._refcount == 0:
                module.init(
                    runtime_dir=str(runtime_dir),
                    backend=backend,
                    device_index=int(device_index),
                    enable_debug=bool(enable_debug),
                )
                cls._module = module
                cls._options = options
            elif cls._options != options:
                raise RuntimeError(f"{cls._module.__name__} is already initialized with different options.")

            cls._refcount += 1
            return cls._module

    @classmethod
    def release(cls) -> None:
        with cls._lock:
            if cls._refcount == 0:
                return

            cls._refcount -= 1
            if cls._refcount == 0 and cls._module is not None:
                cls._module.destroy()
                cls._module = None
                cls._options = None


@dataclass(frozen=True)
class SurfaceDesc:
    base_color: tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)
    roughness: float = 1.0
    metallic: float = 0.0
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0)
    double_sided: bool = False


@dataclass(frozen=True)
class CameraDesc:
    uid: str
    pos: tuple[float, float, float]
    lookat: tuple[float, float, float]
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    res: tuple[int, int] = (512, 512)
    fov: float = 45.0
    near: float = 0.1
    far: float = 1000.0
    model: str = "pinhole"


@dataclass
class _ShapeRecord:
    kind: str
    surface_name: str
    vertices: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    triangles: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.uint32))
    normals: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    uvs: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float32))
    transform: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))
    particle_radius: float = 0.05
    particle_centers: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    particle_radii: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float32))


class GenesisStyleRenderer:
    def __init__(
        self,
        module_dir: Optional[Path | str] = None,
        runtime_dir: Optional[Path | str] = None,
        backend: str = "vulkan",
        device_index: int = -1,
        enable_debug: bool = False,
        rendered_envs_idx: Optional[Sequence[int]] = None,
        particle_sphere_segments: tuple[int, int] = (10, 20),
    ) -> None:
        self.module_dir = Path(module_dir) if module_dir is not None else _default_module_dir()
        self.runtime_dir = Path(runtime_dir) if runtime_dir is not None else _default_runtime_dir()
        self.rendered_envs_idx = list(rendered_envs_idx) if rendered_envs_idx is not None else [0]

        self._rr = _RuntimeHandle.acquire(
            module_dir=self.module_dir,
            runtime_dir=self.runtime_dir,
            backend=backend,
            device_index=device_index,
            enable_debug=enable_debug,
        )
        self._scene = self._rr.create_scene()

        self._surfaces: dict[str, SurfaceDesc] = {}
        self._shapes: dict[str, _ShapeRecord] = {}
        self._cameras: dict[str, CameraDesc] = {}
        self._scene_dirty = True
        self.camera_updated = False
        self._t = -1
        self._destroyed = False

        self._ambient_top = (0.03, 0.04, 0.06)
        self._ambient_bottom = (0.01, 0.01, 0.01)
        self._default_light_direction = (-0.4, -1.0, -0.6)
        self._default_light_color = (1.0, 1.0, 1.0)
        self._default_light_irradiance = 2.0
        self._scene.set_ambient(self._ambient_top, self._ambient_bottom)
        self._scene.set_default_light(
            self._default_light_direction,
            self._default_light_color,
            self._default_light_irradiance,
        )

        latitudes, longitudes = particle_sphere_segments
        self._sphere_vertices, self._sphere_triangles, self._sphere_uvs = _make_unit_sphere(
            max(3, int(latitudes)),
            max(3, int(longitudes)),
        )

        self._temp_dir = _repo_root() / ".temp" / "rtxns_genesis_style"
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._scene_path = self._temp_dir / "scene.glb"

    def __enter__(self) -> "GenesisStyleRenderer":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.destroy()

    def __del__(self) -> None:
        try:
            self.destroy()
        except Exception:
            pass

    def add_surface(self, shape_name: str, surface: Any) -> SurfaceDesc:
        desc = _coerce_surface_desc(surface)
        self._surfaces[str(shape_name)] = desc
        self._scene_dirty = True
        return desc

    def update_surface(self, shape_name: str, surface: Any) -> SurfaceDesc:
        return self.add_surface(shape_name, surface)

    def add_rigid(
        self,
        name: str,
        vertices: Any,
        triangles: Any,
        normals: Any = None,
        uvs: Any = None,
        batch_index: Optional[int] = None,
    ) -> None:
        shape_name = _shape_name(str(name), batch_index)
        vertex_array = _coerce_vertices(vertices)
        triangle_array = _coerce_triangles(triangles)
        normal_array = _coerce_normals(normals, vertex_array.shape[0])
        if normal_array.size == 0:
            normal_array = _compute_vertex_normals(vertex_array, triangle_array)
        uv_array = _coerce_uvs(uvs, vertex_array.shape[0])

        self._shapes[shape_name] = _ShapeRecord(
            kind="rigid",
            surface_name=str(name),
            vertices=vertex_array,
            triangles=triangle_array,
            normals=normal_array,
            uvs=uv_array,
        )
        self._scene_dirty = True

    def add_rigid_batch(self, name: str, vertices: Any, triangles: Any, normals: Any = None, uvs: Any = None) -> None:
        for batch_index in self.rendered_envs_idx:
            self.add_rigid(name, vertices, triangles, normals, uvs, batch_index=batch_index)

    def update_rigid(self, name: str, matrix: Any, batch_index: Optional[int] = None) -> None:
        shape_name = _shape_name(str(name), batch_index)
        if shape_name not in self._shapes:
            raise KeyError(f"Rigid shape '{shape_name}' has not been added.")
        record = self._shapes[shape_name]
        if record.kind != "rigid":
            raise ValueError(f"Shape '{shape_name}' is not rigid.")
        record.transform = _coerce_transform(matrix)
        self._scene_dirty = True

    def update_rigid_batch(self, name: str, matrices: Any) -> None:
        for batch_index in self.rendered_envs_idx:
            self.update_rigid(name, matrices[batch_index], batch_index=batch_index)

    def add_deformable(self, name: str, batch_index: Optional[int] = None) -> None:
        shape_name = _shape_name(str(name), batch_index)
        self._shapes[shape_name] = _ShapeRecord(
            kind="deformable",
            surface_name=str(name),
        )
        self._scene_dirty = True

    def update_deformable(
        self,
        name: str,
        vertices: Any,
        triangles: Any,
        normals: Any = None,
        uvs: Any = None,
        batch_index: Optional[int] = None,
    ) -> None:
        shape_name = _shape_name(str(name), batch_index)
        if shape_name not in self._shapes:
            self.add_deformable(name, batch_index=batch_index)
        record = self._shapes[shape_name]
        vertex_array = _coerce_vertices(vertices)
        triangle_array = _coerce_triangles(triangles)
        normal_array = _coerce_normals(normals, vertex_array.shape[0])
        if normal_array.size == 0:
            normal_array = _compute_vertex_normals(vertex_array, triangle_array)
        uv_array = _coerce_uvs(uvs, vertex_array.shape[0])

        record.kind = "deformable"
        record.vertices = vertex_array
        record.triangles = triangle_array
        record.normals = normal_array
        record.uvs = uv_array
        record.transform = np.eye(4, dtype=np.float32)
        self._scene_dirty = True

    def add_particles(self, name: str, radius: Optional[float] = None, density: Optional[float] = None) -> None:
        del density
        self._shapes[str(name)] = _ShapeRecord(
            kind="particles",
            surface_name=str(name),
            particle_radius=float(radius if radius is not None else 0.05),
        )
        self._scene_dirty = True

    def update_particles(
        self,
        name: str,
        particles: Any,
        radius: Optional[float] = None,
        particles_vel: Any = None,
        particles_radii: Any = None,
    ) -> None:
        del particles_vel
        shape_name = str(name)
        if shape_name not in self._shapes:
            self.add_particles(name, radius=radius)

        centers = _coerce_vertices(particles)
        record = self._shapes[shape_name]
        if particles_radii is None:
            particle_radius = float(radius if radius is not None else record.particle_radius)
            radii = np.full((centers.shape[0],), particle_radius, dtype=np.float32)
        else:
            radii = np.asarray(particles_radii, dtype=np.float32).reshape(-1)
            if radii.size != centers.shape[0]:
                raise ValueError("particles_radii must have the same length as particles.")
            particle_radius = float(radius if radius is not None else (float(radii[0]) if radii.size > 0 else 0.05))

        record.kind = "particles"
        record.particle_centers = np.ascontiguousarray(centers)
        record.particle_radii = np.ascontiguousarray(radii.astype(np.float32))
        record.particle_radius = particle_radius
        self._scene_dirty = True

    def add_camera(self, camera: Any) -> CameraDesc:
        desc = _coerce_camera_desc(camera)
        self._cameras[desc.uid] = desc
        self.camera_updated = True
        return desc

    def update_camera(self, camera: Any) -> CameraDesc:
        return self.add_camera(camera)

    def set_ambient(self, top_rgb: Sequence[float], bottom_rgb: Sequence[float]) -> None:
        self._ambient_top = tuple(_as_float_array(top_rgb, 3))
        self._ambient_bottom = tuple(_as_float_array(bottom_rgb, 3))
        self._scene.set_ambient(self._ambient_top, self._ambient_bottom)

    def set_default_light(
        self,
        direction: Sequence[float],
        color: Sequence[float] = (1.0, 1.0, 1.0),
        irradiance: float = 2.0,
    ) -> None:
        self._default_light_direction = tuple(_as_float_array(direction, 3))
        self._default_light_color = tuple(_as_float_array(color, 3))
        self._default_light_irradiance = float(irradiance)
        self._scene.set_default_light(
            self._default_light_direction,
            self._default_light_color,
            self._default_light_irradiance,
        )

    def reset(self) -> None:
        self._t = -1

    def update_scene(self, force_render: bool = False, time: Optional[float] = None) -> None:
        if not force_render and not self._scene_dirty:
            if time is not None:
                self._t = time
            self.camera_updated = False
            return

        builder = GlbSceneBuilder()
        material_indices: dict[str, int] = {}
        renderable_count = 0

        for shape_name, record in self._shapes.items():
            vertices, triangles, normals, uvs = self._shape_to_mesh(record)
            if vertices.size == 0 or triangles.size == 0:
                continue

            material_name = record.surface_name
            if material_name not in material_indices:
                surface = self._surfaces.get(material_name, SurfaceDesc())
                material_indices[material_name] = builder.add_material(
                    name=material_name,
                    base_color=np.asarray(surface.base_color, dtype=np.float32),
                    roughness=float(surface.roughness),
                    metallic=float(surface.metallic),
                    emissive=np.asarray(surface.emissive, dtype=np.float32),
                    double_sided=bool(surface.double_sided),
                )

            builder.add_mesh(
                name=shape_name,
                vertices=vertices,
                triangles=triangles,
                normals=normals,
                uvs=uvs,
                material_index=material_indices[material_name],
            )
            renderable_count += 1

        if renderable_count == 0:
            raise RuntimeError("update_scene() found no renderable shapes. Add geometry before rendering.")

        self._scene_path.write_bytes(builder.build())
        self._scene.load_scene(str(self._scene_path))
        self._scene.set_ambient(self._ambient_top, self._ambient_bottom)
        self._scene.set_default_light(
            self._default_light_direction,
            self._default_light_color,
            self._default_light_irradiance,
        )

        self._scene_dirty = False
        self.camera_updated = False
        self._t = self._t + 1 if time is None else time

    def render_camera(self, camera: Any, force_render: bool = False, time: Optional[float] = None) -> np.ndarray:
        desc = _coerce_camera_desc(camera)
        self._cameras[desc.uid] = desc
        self.update_scene(force_render=force_render, time=time)
        self._scene.set_camera(
            desc.pos,
            desc.lookat,
            desc.up,
            float(desc.fov),
            int(desc.res[0]),
            int(desc.res[1]),
            float(desc.near),
            float(desc.far),
        )
        rgba = self._scene.render_frame()
        image = np.frombuffer(rgba, dtype=np.uint8).reshape(desc.res[1], desc.res[0], 4)
        return np.ascontiguousarray(image[:, :, :3])

    def render_frame(self, camera: Any, force_render: bool = False, time: Optional[float] = None) -> np.ndarray:
        return self.render_camera(camera, force_render=force_render, time=time)

    def render_camera_rgba(self, camera: Any, force_render: bool = False, time: Optional[float] = None) -> np.ndarray:
        desc = _coerce_camera_desc(camera)
        self._cameras[desc.uid] = desc
        self.update_scene(force_render=force_render, time=time)
        self._scene.set_camera(
            desc.pos,
            desc.lookat,
            desc.up,
            float(desc.fov),
            int(desc.res[0]),
            int(desc.res[1]),
            float(desc.near),
            float(desc.far),
        )
        rgba = self._scene.render_frame()
        return np.frombuffer(rgba, dtype=np.uint8).reshape(desc.res[1], desc.res[0], 4).copy()

    def destroy(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True

        self._cameras.clear()
        self._shapes.clear()
        self._surfaces.clear()
        self._scene = None

        shutil.rmtree(self._temp_dir, ignore_errors=True)
        _RuntimeHandle.release()

    @property
    def cameras(self) -> dict[str, CameraDesc]:
        return dict(self._cameras)

    def _shape_to_mesh(self, record: _ShapeRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if record.kind == "particles":
            return self._particles_to_mesh(record)

        vertices = record.vertices
        triangles = record.triangles
        normals = record.normals
        uvs = record.uvs
        if vertices.size == 0 or triangles.size == 0:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint32),
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
            )

        if record.kind == "rigid":
            vertices = _transform_vertices(vertices, record.transform)
            normals = _transform_normals(normals, record.transform)

        return (
            np.ascontiguousarray(vertices, dtype=np.float32),
            np.ascontiguousarray(triangles, dtype=np.uint32),
            np.ascontiguousarray(normals, dtype=np.float32),
            np.ascontiguousarray(uvs, dtype=np.float32),
        )

    def _particles_to_mesh(self, record: _ShapeRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        centers = record.particle_centers
        radii = record.particle_radii
        if centers.size == 0 or radii.size == 0:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint32),
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
            )

        template_vertices = self._sphere_vertices
        template_triangles = self._sphere_triangles
        template_normals = template_vertices
        template_uvs = self._sphere_uvs

        vertex_count = template_vertices.shape[0]
        face_count = template_triangles.shape[0]
        all_vertices = np.empty((centers.shape[0] * vertex_count, 3), dtype=np.float32)
        all_normals = np.empty_like(all_vertices)
        all_uvs = np.tile(template_uvs, (centers.shape[0], 1)).astype(np.float32)
        all_triangles = np.empty((centers.shape[0] * face_count, 3), dtype=np.uint32)

        for index, (center, radius) in enumerate(zip(centers, radii, strict=False)):
            vertex_offset = index * vertex_count
            face_offset = index * face_count
            all_vertices[vertex_offset : vertex_offset + vertex_count] = template_vertices * radius + center
            all_normals[vertex_offset : vertex_offset + vertex_count] = template_normals
            all_triangles[face_offset : face_offset + face_count] = template_triangles + vertex_offset

        return all_vertices, all_triangles, all_normals, all_uvs
