#!/usr/bin/env python3
"""Render frl_apartment_stage.glb with SAPIEN, matching Flora's camera exactly.

SAPIEN receives the geometry in the GLB's Y-up coordinate system; it does not
need a Y-up-to-Z-up conversion for this off-screen render.  The camera can
therefore use Flora's position, target, and up vector verbatim.

SAPIEN 3.0.3 applies the ReplicaCAD GLB's node hierarchy itself, including its
root rotation and centimetre-to-metre scale.  Keep the original GLB (including
its embedded PBR textures) and do not apply that transform a second time.  A
trimesh round-trip is deliberately avoided: its exported GLB loses the image
texture bindings that make the Flora reference colourful.
"""

from __future__ import annotations

import os
import sys
import numpy as np


def main() -> int:
    import sapien
    import transforms3d
    from PIL import Image

    gi_mode = "--gi" in sys.argv or "--quality" in sys.argv
    # `--reference-tonemap` renders the ordinary raster/direct-light path
    # with the same HDR display mapping as --gi.  It is an A/B control for
    # separating display exposure from indirect-light transport.
    reference_tonemap = gi_mode or "--reference-tonemap" in sys.argv
    if gi_mode:
        # Must be configured before creating the renderer/scene.  The RT
        # camera adds multi-bounce path tracing to the *same* direct lighting
        # used below for the Flora-aligned render.  OptiX denoising is
        # unstable with this SAPIEN build on the current RTX driver, so use
        # clean native accumulation instead.
        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(256)
        sapien.render.set_ray_tracing_path_depth(8)
        sapien.render.set_ray_tracing_denoiser("none")

    # Keep the original GLB so embedded material textures are preserved.
    glb_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "replica_cad",
        "frl_apartment_stage.glb"
    )
    glb_path = os.path.abspath(glb_path)

    # Flora camera params (Y-up, used EXACTLY as-is in SAPIEN)
    # derrickliao begin: 相机参数与 Flora 完全一致 (Y-up)
    pos = np.array([3.5, 2.0, 3.5])
    lookat = np.array([0.0, 1.0, 0.0])
    up = np.array([0.0, 1.0, 0.0])  # Y-up, NOT Z-up!
    fov_deg = 60.0
    width, height = 1280, 720
    # derrickliao end

    print("[1/5] Creating SAPIEN scene...")
    scene = sapien.Scene()

    # No skybox or constant ambient: this keeps the empty background black,
    # in both modes.  The only difference in --quality mode is GI.
    rs = scene.get_system("render")
    rs.set_cubemap(None)
    rs.set_ambient_light(np.array([0.0, 0.0, 0.0]))

    print("[2/5] Setting up lighting...")
    # With SAPIEN's native camera axes, this is the same light propagation
    # vector used by Flora.  Its colour parameter is scaled differently from
    # Donut irradiance, hence this exposure-calibrated equivalent intensity.
    scene.add_directional_light(
        direction=np.array([-0.4, -1.0, -0.6]),
        color=np.array([0.36, 0.342, 0.324]),
        shadow=True,
    )
    # derrickliao end

    print("[3/5] Loading GLB: {}".format(os.path.basename(glb_path)))
    # In SAPIEN 3.0.3 this path correctly applies the complete glTF transform
    # hierarchy and preserves the GLB's embedded PBR textures.
    builder = scene.create_actor_builder()
    builder.add_visual_from_file(glb_path)
    actor = builder.build_static(name="apartment_stage")

    print("[4/5] Setting up camera (Y-up, matching Flora exactly)...")
    # SAPIEN camera local axes are +X forward, +Y left, +Z up.  This differs
    # from the OpenGL-style (-Z forward, +Y up) convention used by many
    # look-at snippets.
    forward = lookat - pos
    forward = forward / np.linalg.norm(forward)
    left = np.cross(up, forward)
    left = left / np.linalg.norm(left)
    cam_up = np.cross(forward, left)
    R = np.column_stack([forward, left, cam_up])
    cam_q = transforms3d.quaternions.mat2quat(R)
    cam_pose = sapien.Pose(p=pos.tolist(), q=cam_q.tolist())

    camera = scene.add_camera(
        "flora_match",
        width,
        height,
        float(np.deg2rad(fov_deg)),
        0.1,
        10000.0,
    )
    camera.set_entity_pose(cam_pose)
    print("  pos={}, lookat={}, up={}, fov={}deg, {}x{}".format(
        pos, lookat, up, fov_deg, width, height))

    print("[5/5] Rendering{}...".format(" (RT GI mode)" if gi_mode else ""))
    scene.update_render()
    camera.take_picture()

    # Check geometry visibility
    pos_img = camera.get_picture("Position")
    depth = np.linalg.norm(pos_img[:, :, :3], axis=-1)
    valid = depth > 0.01
    print("  Valid pixels: {}/{} ({:.1f}%)".format(
        valid.sum(), valid.size, 100 * valid.sum() / valid.size))

    rgba = camera.get_picture("Color")
    print("  Color: mean={:.3f}, std={:.3f}".format(rgba.mean(), rgba.std()))

    if rgba.dtype != np.uint8:
        rgb_linear = np.clip(rgba[:, :, :3], 0, None)
        if reference_tonemap:
            # Preserve high-dynamic-range detail from the path tracer instead
            # of clipping bright indirect illumination to flat white.
            # Path-tracer radiance is HDR whereas the raster path writes its
            # display-referred result directly.  This is an output exposure
            # calibration, not a light-intensity change: it keeps directly
            # lit wall regions comparable to the no-GI control so the visible
            # difference is indirect transport in occluded regions.
            # This is a display transform, not a light-intensity change.
            # Apply it to both --gi and --reference-tonemap so the resulting
            # A/B isolates indirect transport from presentation brightness.
            rgb_linear *= 0.60                      # common reference exposure
            # Match Flora's shadow_composite_cs.hlsl: Hejl-Burgess-Dawson
            # filmic curve.  Keeping this operator and exposure identical in
            # all four reference images prevents display mapping from being
            # mistaken for a GI improvement.
            x = np.maximum(rgb_linear - 0.004, 0.0)
            rgb = (x * (6.2 * x + 0.5)) / (x * (6.2 * x + 1.7) + 0.06)
            rgb = (rgb * 255).astype(np.uint8)
        else:
            rgb = (np.clip(rgb_linear, 0, 1) * 255).astype(np.uint8)
    else:
        rgb = rgba[:, :, :3]

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "replica_cad")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if gi_mode:
        output_name = "maniskill_rt_gi_output.png"
    elif reference_tonemap:
        output_name = "maniskill_direct_reference_tonemap_output.png"
    else:
        output_name = "maniskill_demo_output.png"
    png_path = os.path.join(out_dir, output_name)
    Image.fromarray(rgb, "RGB").save(png_path)
    print("  Saved: {} ({:.1f} KB)".format(png_path, os.path.getsize(png_path) / 1024))

    # Compare only the raster render with Flora.  The RT image keeps the same
    # direct light but intentionally includes indirect light, so a pixel
    # difference quantifies GI rather than camera or asset mismatch.
    flora_path = os.path.join(out_dir, "demo_output.png")
    if not gi_mode and os.path.exists(flora_path):
        flora_img = np.array(Image.open(flora_path)).astype(float)
        diff = np.abs(flora_img - rgb.astype(float))
        match_pct = 100 * (diff.mean(axis=2) < 30).sum() / (flora_img.shape[0] * flora_img.shape[1])
        print("\n=== Comparison with Flora ===")
        print("  Mean diff: {:.1f}".format(diff.mean()))
        print("  Pixels matching (<30): {:.1f}%".format(match_pct))
        print("  Row means:")
        for r in range(0, 720, 100):
            print("    Row {:3d}: Flora={:.0f}, SAPIEN={:.0f}".format(
                r, flora_img[r].mean(), rgb[r].mean()))

    print("Done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
