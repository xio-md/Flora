from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from .scene_desc import (
    ArticulationTemplateDesc,
    ManifestWarning,
    UrdfJointDesc,
    UrdfVisualDesc,
    Vec3,
    vector3,
)


def _values(text: Optional[str], default: Vec3, name: str) -> Vec3:
    if text is None or not text.strip():
        return default
    return vector3(tuple(float(value) for value in text.split()), name)


def _origin(element: Optional[ET.Element], prefix: str) -> tuple[Vec3, Vec3]:
    if element is None:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    return (
        _values(element.get("xyz"), (0.0, 0.0, 0.0), f"{prefix} xyz"),
        _values(element.get("rpy"), (0.0, 0.0, 0.0), f"{prefix} rpy"),
    )


def _optional_float(element: Optional[ET.Element], attribute: str) -> Optional[float]:
    if element is None or element.get(attribute) is None:
        return None
    return float(element.get(attribute, ""))


def parse_urdf_manifest(path: Path, handle: Optional[str] = None) -> ArticulationTemplateDesc:
    urdf_path = Path(path).resolve()
    root = ET.parse(urdf_path).getroot()
    links: list[str] = []
    visuals: list[UrdfVisualDesc] = []
    joints: list[UrdfJointDesc] = []
    warnings: list[ManifestWarning] = []

    for link in root.findall("link"):
        link_name = link.get("name", "")
        links.append(link_name)
        for visual_index, visual in enumerate(link.findall("visual")):
            mesh = visual.find("geometry/mesh")
            if mesh is None or not mesh.get("filename"):
                warnings.append(
                    ManifestWarning(
                        code="unsupported_urdf_visual",
                        message="URDF visual does not contain a mesh filename.",
                        source_path=urdf_path,
                        context=(("link", link_name), ("visual_index", str(visual_index))),
                    )
                )
                continue
            mesh_path = (urdf_path.parent / mesh.get("filename", "")).resolve()
            xyz, rpy = _origin(visual.find("origin"), "visual origin")
            scale = _values(mesh.get("scale"), (1.0, 1.0, 1.0), "mesh scale")
            if not mesh_path.is_file():
                warnings.append(
                    ManifestWarning(
                        code="missing_urdf_visual_asset",
                        message="URDF visual mesh does not exist.",
                        source_path=mesh_path,
                        context=(("link", link_name),),
                    )
                )
            visuals.append(
                UrdfVisualDesc(
                    link_name=link_name,
                    mesh_path=mesh_path,
                    origin_xyz=xyz,
                    origin_rpy=rpy,
                    scale=scale,
                )
            )

    for joint in root.findall("joint"):
        origin_xyz, origin_rpy = _origin(joint.find("origin"), "joint origin")
        parent = joint.find("parent")
        child = joint.find("child")
        axis = joint.find("axis")
        limit = joint.find("limit")
        joints.append(
            UrdfJointDesc(
                name=joint.get("name", ""),
                joint_type=joint.get("type", "fixed"),
                parent_link="" if parent is None else parent.get("link", ""),
                child_link="" if child is None else child.get("link", ""),
                origin_xyz=origin_xyz,
                origin_rpy=origin_rpy,
                axis_xyz=_values(
                    None if axis is None else axis.get("xyz"),
                    (1.0, 0.0, 0.0),
                    "joint axis",
                ),
                limit_lower=_optional_float(limit, "lower"),
                limit_upper=_optional_float(limit, "upper"),
            )
        )

    return ArticulationTemplateDesc(
        handle=urdf_path.stem if handle is None else str(handle),
        urdf_path=urdf_path,
        links=tuple(links),
        visuals=tuple(visuals),
        joints=tuple(joints),
        warnings=tuple(warnings),
    )
