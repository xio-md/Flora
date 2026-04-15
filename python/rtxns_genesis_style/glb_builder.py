from __future__ import annotations

import json
import struct
from typing import Optional

import numpy as np

_GLB_MAGIC = 0x46546C67
_GLB_VERSION = 2
_JSON_CHUNK_TYPE = 0x4E4F534A
_BIN_CHUNK_TYPE = 0x004E4942

_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963

_COMPONENT_TYPE_TO_SIZE = {
    5125: 4,
    5126: 4,
}

_ACCESSOR_TYPE_TO_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
}


def _pad_bytes(blob: bytes, pad_byte: bytes) -> bytes:
    padding = (-len(blob)) % 4
    if padding == 0:
        return blob
    return blob + pad_byte * padding


class GlbSceneBuilder:
    def __init__(self) -> None:
        self._binary = bytearray()
        self._buffer_views: list[dict] = []
        self._accessors: list[dict] = []
        self._materials: list[dict] = []
        self._meshes: list[dict] = []
        self._nodes: list[dict] = []

    def add_material(
        self,
        name: str,
        base_color: np.ndarray,
        roughness: float,
        metallic: float,
        emissive: np.ndarray,
        double_sided: bool,
    ) -> int:
        material: dict = {
            "name": name,
            "pbrMetallicRoughness": {
                "baseColorFactor": [float(x) for x in base_color],
                "roughnessFactor": float(roughness),
                "metallicFactor": float(metallic),
            },
        }

        if np.any(np.abs(emissive) > 1.0e-6):
            material["emissiveFactor"] = [float(x) for x in emissive]
        if double_sided:
            material["doubleSided"] = True
        if base_color[3] < 0.999:
            material["alphaMode"] = "BLEND"

        index = len(self._materials)
        self._materials.append(material)
        return index

    def add_mesh(
        self,
        name: str,
        vertices: np.ndarray,
        triangles: np.ndarray,
        normals: Optional[np.ndarray],
        uvs: Optional[np.ndarray],
        material_index: int,
        node_matrix: Optional[np.ndarray] = None,
    ) -> None:
        if vertices.size == 0 or triangles.size == 0:
            return

        vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        triangles = np.ascontiguousarray(triangles.reshape(-1), dtype=np.uint32)
        if normals is not None and normals.size > 0:
            normals = np.ascontiguousarray(normals, dtype=np.float32)
        else:
            normals = None
        if uvs is not None and uvs.size > 0:
            uvs = np.ascontiguousarray(uvs, dtype=np.float32)
        else:
            uvs = None

        attributes = {
            "POSITION": self._add_accessor(vertices, 5126, "VEC3", _ARRAY_BUFFER, include_bounds=True),
        }
        if normals is not None:
            attributes["NORMAL"] = self._add_accessor(normals, 5126, "VEC3", _ARRAY_BUFFER)
        if uvs is not None:
            attributes["TEXCOORD_0"] = self._add_accessor(uvs, 5126, "VEC2", _ARRAY_BUFFER)

        primitive = {
            "attributes": attributes,
            "indices": self._add_accessor(triangles, 5125, "SCALAR", _ELEMENT_ARRAY_BUFFER),
            "material": material_index,
        }

        mesh_index = len(self._meshes)
        self._meshes.append(
            {
                "name": name,
                "primitives": [primitive],
            }
        )
        node: dict = {
            "name": name,
            "mesh": mesh_index,
        }
        if node_matrix is not None:
            matrix = np.ascontiguousarray(node_matrix, dtype=np.float32)
            if matrix.shape != (4, 4):
                raise ValueError("node_matrix must have shape (4, 4).")
            # glTF stores matrices in column-major order.
            node["matrix"] = matrix.T.reshape(-1).astype(float).tolist()

        self._nodes.append(node)

    def build(self) -> bytes:
        if not self._nodes:
            raise RuntimeError("No renderable geometry was added to the GLB scene.")

        binary_blob = _pad_bytes(bytes(self._binary), b"\x00")
        document = {
            "asset": {
                "version": "2.0",
                "generator": "RTXNS GenesisStyleRenderer",
            },
            "scene": 0,
            "scenes": [{"nodes": list(range(len(self._nodes)))}],
            "nodes": self._nodes,
            "meshes": self._meshes,
            "materials": self._materials,
            "buffers": [{"byteLength": len(binary_blob)}],
            "bufferViews": self._buffer_views,
            "accessors": self._accessors,
        }

        json_blob = _pad_bytes(
            json.dumps(document, separators=(",", ":"), ensure_ascii=True).encode("utf-8"),
            b" ",
        )

        total_length = 12 + 8 + len(json_blob) + 8 + len(binary_blob)
        header = struct.pack("<III", _GLB_MAGIC, _GLB_VERSION, total_length)
        json_chunk = struct.pack("<II", len(json_blob), _JSON_CHUNK_TYPE) + json_blob
        bin_chunk = struct.pack("<II", len(binary_blob), _BIN_CHUNK_TYPE) + binary_blob
        return header + json_chunk + bin_chunk

    def _add_accessor(
        self,
        array: np.ndarray,
        component_type: int,
        accessor_type: str,
        target: int,
        include_bounds: bool = False,
    ) -> int:
        array = np.ascontiguousarray(array)
        buffer_view = self._add_blob(array.tobytes(), target)

        accessor = {
            "bufferView": buffer_view,
            "componentType": component_type,
            "count": int(self._count_for(array, accessor_type)),
            "type": accessor_type,
        }

        if include_bounds:
            reshaped = self._reshape_for_bounds(array, accessor_type)
            accessor["min"] = reshaped.min(axis=0).astype(float).tolist()
            accessor["max"] = reshaped.max(axis=0).astype(float).tolist()

        index = len(self._accessors)
        self._accessors.append(accessor)
        return index

    def _add_blob(self, payload: bytes, target: int) -> int:
        while len(self._binary) % 4 != 0:
            self._binary.append(0)

        offset = len(self._binary)
        self._binary.extend(payload)

        buffer_view = {
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(payload),
            "target": target,
        }
        index = len(self._buffer_views)
        self._buffer_views.append(buffer_view)
        return index

    @staticmethod
    def _count_for(array: np.ndarray, accessor_type: str) -> int:
        components = _ACCESSOR_TYPE_TO_COMPONENTS[accessor_type]
        if components == 1:
            return int(array.size)
        if array.ndim != 2 or array.shape[1] != components:
            raise ValueError(f"Expected an array of shape (N, {components}) for accessor type {accessor_type}.")
        return int(array.shape[0])

    @staticmethod
    def _reshape_for_bounds(array: np.ndarray, accessor_type: str) -> np.ndarray:
        components = _ACCESSOR_TYPE_TO_COMPONENTS[accessor_type]
        if components == 1:
            return array.reshape(-1, 1)
        return array.reshape(-1, components)
