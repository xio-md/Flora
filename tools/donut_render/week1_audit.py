from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

_CLASS_START_RE = re.compile(r'py::class_<.*?>\(m,\s*"([^"]+)"\)')
_ENUM_START_RE = re.compile(r'py::enum_<.*?>\(m,\s*"([^"]+)"\)')
_DEF_RE = re.compile(r'\.def(?:_property_readonly)?\(\s*"([^"]+)"')
_ENUM_VALUE_RE = re.compile(r'\.value\(\s*"([^"]+)"')
_MODULE_DEF_RE = re.compile(r'm\.def\(\s*"([^"]+)"')
_LUISA_SYMBOL_RE = re.compile(r'LuisaRenderPy\.([A-Za-z_][A-Za-z0-9_]*)')
_SCENE_CALL_RE = re.compile(r'self\._scene\.([A-Za-z_][A-Za-z0-9_]*)')
_METHOD_DEF_RE = re.compile(r"^ {4}def ([A-Za-z_][A-Za-z0-9_]*)\(")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_genesis_root() -> Path:
    return _repo_root().parent / "Genesis"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _parse_cpp_bindings(source: str) -> dict[str, Any]:
    lines = source.splitlines()
    classes: dict[str, list[str]] = {}
    enums: dict[str, list[str]] = {}
    module_functions = _ordered_unique(_MODULE_DEF_RE.findall(source))

    index = 0
    while index < len(lines):
        line = lines[index]

        enum_match = _ENUM_START_RE.search(line)
        if enum_match:
            name = enum_match.group(1)
            values = _ENUM_VALUE_RE.findall(line)
            brace_depth = line.count("{") - line.count("}")
            while True:
                if brace_depth == 0 and lines[index].strip().endswith(";"):
                    break
                index += 1
                brace_depth += lines[index].count("{") - lines[index].count("}")
                values.extend(_ENUM_VALUE_RE.findall(lines[index]))
            enums[name] = _ordered_unique(values)
            index += 1
            continue

        class_match = _CLASS_START_RE.search(line)
        if class_match:
            name = class_match.group(1)
            methods = _DEF_RE.findall(line)
            brace_depth = line.count("{") - line.count("}")
            while True:
                if brace_depth == 0 and lines[index].strip().endswith(";"):
                    break
                index += 1
                brace_depth += lines[index].count("{") - lines[index].count("}")
                methods.extend(_DEF_RE.findall(lines[index]))
            classes[name] = _ordered_unique(methods)
            index += 1
            continue

        index += 1

    return {
        "enums": enums,
        "classes": classes,
        "module_functions": module_functions,
    }


def _collect_raytracer_usage(source: str) -> dict[str, list[str]]:
    module_symbols: list[str] = []
    scene_methods: list[str] = []

    for line in source.splitlines():
        if line.lstrip().startswith("#"):
            continue
        module_symbols.extend(symbol for symbol in _LUISA_SYMBOL_RE.findall(line) if not symbol.startswith("__"))
        scene_methods.extend(_SCENE_CALL_RE.findall(line))

    return {
        "module_symbols": _ordered_unique(module_symbols),
        "scene_methods": _ordered_unique(scene_methods),
    }


def _collect_class_methods(source: str, class_name: str) -> list[str]:
    lines = source.splitlines()
    inside_class = False
    methods: list[str] = []

    for line in lines:
        if not inside_class:
            if line.startswith(f"class {class_name}"):
                inside_class = True
            continue

        if line and not line.startswith(" "):
            break

        method_match = _METHOD_DEF_RE.match(line)
        if method_match:
            method_name = method_match.group(1)
            if not method_name.startswith("_"):
                methods.append(method_name)

    return _ordered_unique(methods)


def _build_summary(
    luisa_bindings: dict[str, Any],
    raytracer_usage: dict[str, list[str]],
    rtx_bindings: dict[str, Any],
    renderer_methods: list[str],
) -> dict[str, Any]:
    luisa_classes = set(luisa_bindings["classes"].keys())
    luisa_enums = set(luisa_bindings["enums"].keys())
    luisa_module_functions = set(luisa_bindings["module_functions"])

    raytracer_symbols = set(raytracer_usage["module_symbols"])
    raytracer_scene_methods = set(raytracer_usage["scene_methods"])

    rtx_classes = set(rtx_bindings["classes"].keys())
    rtx_enums = set(rtx_bindings["enums"].keys())
    rtx_module_functions = set(rtx_bindings["module_functions"])
    rtx_scene_methods = set(rtx_bindings["classes"].get("Scene", []))

    luisa_api_surface = luisa_classes | luisa_enums | luisa_module_functions
    rtx_api_surface = rtx_classes | rtx_enums | rtx_module_functions

    return {
        "luisa_class_count": len(luisa_classes),
        "luisa_enum_count": len(luisa_enums),
        "luisa_module_function_count": len(luisa_module_functions),
        "raytracer_required_symbols": sorted(raytracer_symbols),
        "raytracer_missing_binding_symbols": sorted(raytracer_symbols - rtx_api_surface),
        "raytracer_required_scene_methods": sorted(raytracer_scene_methods),
        "missing_scene_methods_in_rtx_binding": sorted(raytracer_scene_methods - rtx_scene_methods),
        "rtx_binding_class_count": len(rtx_classes),
        "rtx_binding_module_function_count": len(rtx_module_functions),
        "genesis_style_renderer_methods": renderer_methods,
        "luisa_only_surface_symbols": sorted(luisa_api_surface - rtx_api_surface),
    }


def _format_text(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("Week 1 audit: LuisaRenderPy -> RTXNS/Donut prototype")
    lines.append("")
    lines.append(
        "LuisaRenderPy surface: "
        f"{report['summary']['luisa_class_count']} classes, "
        f"{report['summary']['luisa_enum_count']} enums, "
        f"{report['summary']['luisa_module_function_count']} module functions"
    )
    lines.append(
        "RtxRenderPy surface: "
        f"{report['summary']['rtx_binding_class_count']} classes, "
        f"{report['summary']['rtx_binding_module_function_count']} module functions"
    )
    lines.append("")
    lines.append("raytracer.py required LuisaRenderPy symbols:")
    for symbol in report["summary"]["raytracer_required_symbols"]:
        lines.append(f"  - {symbol}")
    lines.append("")
    lines.append("raytracer.py required Scene methods:")
    for method in report["summary"]["raytracer_required_scene_methods"]:
        lines.append(f"  - {method}")
    lines.append("")
    lines.append("Scene methods missing in current RtxRenderPy binding:")
    for method in report["summary"]["missing_scene_methods_in_rtx_binding"]:
        lines.append(f"  - {method}")
    lines.append("")
    lines.append("Luisa symbols referenced by raytracer.py but absent from RtxRenderPy:")
    for symbol in report["summary"]["raytracer_missing_binding_symbols"]:
        lines.append(f"  - {symbol}")
    lines.append("")
    lines.append("Current GenesisStyleRenderer public methods:")
    for method in report["summary"]["genesis_style_renderer_methods"]:
        lines.append(f"  - {method}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit LuisaRenderPy, raytracer.py, and the current RTXNS Python prototype for Week 1 planning."
    )
    parser.add_argument("--rtxns-root", type=Path, default=_repo_root())
    parser.add_argument("--genesis-root", type=Path, default=_default_genesis_root())
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args()

    luisa_binding_path = args.genesis_root / "genesis" / "ext" / "LuisaRender" / "src" / "apps" / "py_interface.cpp"
    raytracer_path = args.genesis_root / "genesis" / "vis" / "raytracer.py"
    rtx_binding_path = args.rtxns_root / "src" / "PythonBindings" / "py_interface.cpp"
    prototype_path = args.rtxns_root / "python" / "rtxns_genesis_style" / "renderer.py"

    luisa_bindings = _parse_cpp_bindings(_read_text(luisa_binding_path))
    raytracer_usage = _collect_raytracer_usage(_read_text(raytracer_path))
    rtx_bindings = _parse_cpp_bindings(_read_text(rtx_binding_path))
    renderer_methods = _collect_class_methods(_read_text(prototype_path), "GenesisStyleRenderer")

    report = {
        "paths": {
            "luisa_binding": str(luisa_binding_path),
            "raytracer": str(raytracer_path),
            "rtx_binding": str(rtx_binding_path),
            "prototype": str(prototype_path),
        },
        "luisa_bindings": luisa_bindings,
        "raytracer_usage": raytracer_usage,
        "rtx_bindings": rtx_bindings,
        "summary": _build_summary(luisa_bindings, raytracer_usage, rtx_bindings, renderer_methods),
    }

    if args.format == "json":
        print(json.dumps(report, indent=2, ensure_ascii=True))
    else:
        print(_format_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
