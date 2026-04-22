from __future__ import annotations

import argparse
import os
from pathlib import Path


def _default_module_dir(repo_root: Path) -> Path:
    platform_dir = "windows-x64" if os.name == "nt" else "linux-x64"
    return repo_root / "bin" / platform_dir


def _make_render(dr, name: str):
    return dr.Render(
        name=name,
        spectrum=dr.SRGBSpectrum(),
        integrator=dr.WavePathIntegrator(log_level=dr.LogLevel.INFO),
        clamp_normal=45.0,
    )


def _make_camera(dr, name: str):
    return dr.PinholeCamera(
        name=name,
        pose=dr.MatrixTransform(
            (
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 3.0),
                (0.0, 0.0, 0.0, 1.0),
            )
        ),
        film=dr.Film((320, 240)),
        filter=dr.Filter(radius=1.0),
        spp=4,
        fov=45.0,
    )


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    python_root = repo_root / "python"

    parser = argparse.ArgumentParser(description="Validate ownership and lifecycle error semantics.")
    parser.add_argument("--module-dir", type=Path, default=_default_module_dir(repo_root))
    parser.add_argument("--runtime-dir", type=Path, default=repo_root)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(python_root))

    import DonutRenderPy as dr

    results: list[str] = []

    dr.init(
        context_path=repo_root,
        runtime_dir=args.runtime_dir,
        module_dir=args.module_dir,
        backend="vulkan",
        device_index=-1,
        log_level=dr.LogLevel.INFO,
    )

    scene_a = dr.create_scene()
    scene_b = dr.create_scene()

    shared_render = _make_render(dr, "shared-render")
    scene_a.init(shared_render)
    try:
        scene_b.init(shared_render)
        raise AssertionError("Render handles should not be reusable across scenes.")
    except dr.InvalidStateError:
        results.append("Render handles correctly reject cross-scene reuse.")

    scene_b.init(_make_render(dr, "scene-b-render"))
    results.append("A second scene can still initialize with a fresh render handle.")

    shared_texture = dr.ColorTexture((0.3, 0.4, 0.5, 1.0))
    surface_a = dr.PlasticSurface(name="mat_a", kd=shared_texture)
    surface_b = dr.PlasticSurface(name="mat_b", kd=shared_texture)
    scene_a.update_surface(surface_a)
    try:
        scene_b.update_surface(surface_b)
        raise AssertionError("Nested texture ownership should reject cross-scene reuse.")
    except dr.InvalidStateError:
        results.append("Nested texture ownership correctly rejects cross-scene reuse.")

    shared_camera = _make_camera(dr, "cam_shared")
    scene_a.update_camera(shared_camera, denoise=False)
    try:
        scene_b.update_camera(shared_camera, denoise=False)
        raise AssertionError("Camera handles should not be reusable across scenes.")
    except dr.InvalidStateError:
        results.append("Camera handles correctly reject cross-scene reuse.")

    scene_a.destroy()
    scene_a.destroy()
    results.append("Scene.destroy() is idempotent.")

    try:
        scene_b.update_surface(surface_a)
        raise AssertionError("Destroyed handles should not be reusable.")
    except dr.InvalidStateError:
        results.append("Destroyed scene-owned objects correctly reject reuse.")

    stale_scene = dr.create_scene()
    stale_scene.init(_make_render(dr, "stale-render"))
    stale_camera = _make_camera(dr, "cam_stale")
    stale_scene.update_camera(stale_camera, denoise=False)

    dr.destroy()
    dr.destroy()
    results.append("Module-level destroy() is idempotent.")

    try:
        stale_scene.update_camera(_make_camera(dr, "after-destroy"), denoise=False)
        raise AssertionError("Module destroy should invalidate outstanding Scene handles.")
    except dr.SceneDestroyedError:
        results.append("Module-level destroy() correctly invalidates outstanding Scene handles.")

    dr.init(
        context_path=repo_root,
        runtime_dir=args.runtime_dir,
        module_dir=args.module_dir,
        backend="vulkan",
        device_index=-1,
        log_level=dr.LogLevel.INFO,
    )
    fresh_scene = dr.create_scene()
    fresh_scene.init(_make_render(dr, "fresh-render"))
    try:
        fresh_scene.update_camera(stale_camera, denoise=False)
        raise AssertionError("Objects from a destroyed runtime should not be reusable after re-init.")
    except dr.InvalidStateError:
        results.append("Objects from a destroyed runtime correctly reject reuse after re-init.")

    fresh_scene.destroy()
    dr.destroy()

    if not args.quiet:
        for line in results:
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
