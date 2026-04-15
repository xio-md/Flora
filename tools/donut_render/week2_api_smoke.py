from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    python_root = repo_root / "python"

    import sys

    sys.path.insert(0, str(python_root))

    import DonutRenderPy as dr

    results: list[str] = []

    try:
        dr.create_scene()
        raise AssertionError("create_scene() should fail before init().")
    except dr.RuntimeNotInitializedError:
        results.append("create_scene() correctly rejects calls before init().")

    dr.init(
        context_path=repo_root,
        runtime_dir=repo_root,
        module_dir=repo_root / "bin" / "windows-x64",
        backend="vulkan",
        device_index=-1,
        log_level=dr.LogLevel.INFO,
    )
    dr.init(
        context_path=repo_root,
        runtime_dir=repo_root,
        module_dir=repo_root / "bin" / "windows-x64",
        backend="vulkan",
        device_index=-1,
        log_level=dr.LogLevel.INFO,
    )
    results.append("init() is idempotent for identical options.")

    scene = dr.create_scene()
    render = dr.Render(
        name="week2-smoke",
        spectrum=dr.SRGBSpectrum(),
        integrator=dr.WavePathIntegrator(log_level=dr.LogLevel.INFO),
        clamp_normal=45.0,
    )
    scene.init(render)
    results.append("Scene.init(render) succeeds.")

    try:
        scene.init(render)
        raise AssertionError("Scene.init(render) should fail on second call.")
    except dr.InvalidStateError:
        results.append("Scene.init(render) correctly rejects double initialization.")

    scene.update_environment(
        dr.Environment(
            name="env",
            emission=dr.ColorTexture((0.1, 0.15, 0.2)),
            transform=dr.MatrixTransform(
                (
                    (1.0, 0.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                )
            ),
        )
    )
    surface = dr.PlasticSurface(
        name="mat_box",
        kd=dr.ColorTexture((0.8, 0.5, 0.2, 1.0)),
        roughness=dr.ColorTexture((0.6,)),
    )
    emission = dr.Light(
        name="light_box",
        emission=dr.ColorTexture((0.2, 0.1, 0.05)),
        intensity=2.0,
    )
    shape = dr.RigidShape(
        name="box",
        vertices=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        triangles=((0, 1, 2),),
        transform=dr.MatrixTransform(
            (
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            )
        ),
        surface=surface,
        emission=emission,
    )
    camera = dr.PinholeCamera(
        name="cam_main",
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

    scene.update_surface(surface)
    scene.update_emission(emission)
    scene.update_shape(shape)
    scene.update_camera(camera, denoise=False)
    results.append("Scene registries accept environment/surface/light/shape/camera objects.")

    thinlens = dr.ThinLensCamera(
        name="cam_dof",
        pose=dr.MatrixTransform(
            (
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 2.0),
                (0.0, 0.0, 0.0, 1.0),
            )
        ),
        film=dr.Film((64, 64)),
        filter=dr.Filter(radius=1.0),
        spp=1,
        aperture=2.8,
        focal_len=50.0,
        focus_dis=3.0,
    )
    scene.update_camera(thinlens, denoise=False)
    results.append("ThinLensCamera is accepted at the API level.")

    scene.destroy()
    try:
        scene.update_surface(surface)
        raise AssertionError("Destroyed scene should reject further updates.")
    except dr.SceneDestroyedError:
        results.append("Destroyed scenes correctly reject reuse.")

    dr.destroy()
    results.append("Module-level destroy() succeeds.")

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--quiet", action="store_true")
    args, _ = parser.parse_known_args()
    if not args.quiet:
        for line in results:
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
