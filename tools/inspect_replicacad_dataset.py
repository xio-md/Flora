"""Parse every ReplicaCAD scene and write a deterministic coverage report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "ReplicaCAD"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "replicacad_manifest_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-scenes", type=int, default=91)
    parser.add_argument("--expected-stage-templates", type=int, default=5)
    parser.add_argument("--expected-object-templates", type=int, default=92)
    parser.add_argument("--expected-urdf-templates", type=int, default=12)
    parser.add_argument("--expected-object-instances", type=int, default=2293)
    parser.add_argument("--expected-articulated-instances", type=int, default=540)
    parser.add_argument("--fail-on-warning", action="store_true")
    return parser.parse_args()


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(REPO_ROOT / "python"))
    from donut_render_py import load_replicacad_manifest

    manifest = load_replicacad_manifest(args.dataset)
    scenes = []
    total = len(manifest.scene_handles)
    for index, handle in enumerate(manifest.scene_handles, start=1):
        scenes.append(manifest.parse_scene(handle))
        if index == 1 or index % 10 == 0 or index == total:
            print(f"[{index:02d}/{total:02d}] parsed {handle}", flush=True)

    report = manifest.build_report(scenes)
    summary = report["summary"]
    checks = {
        "scene_coverage": summary["parsed_scenes"] == args.expected_scenes,
        "stage_template_coverage": (
            summary["stage_templates"] == args.expected_stage_templates
        ),
        "object_template_coverage": (
            summary["object_templates"] == args.expected_object_templates
        ),
        "urdf_template_coverage": (
            summary["urdf_templates"] == args.expected_urdf_templates
        ),
        "object_instance_coverage": (
            summary["object_instances"] == args.expected_object_instances
        ),
        "articulated_instance_coverage": (
            summary["articulated_instances"] == args.expected_articulated_instances
        ),
        "required_visual_assets_present": (
            summary["missing_required_visual_assets"] == 0
        ),
    }
    report["acceptance"] = checks
    write_json_atomic(args.output, report)

    print(
        "Summary: "
        f"scenes={summary['parsed_scenes']}/{summary['registered_scenes']}, "
        f"objects={summary['object_instances']}, "
        f"articulated={summary['articulated_instances']}, "
        f"warnings={summary['warnings']}",
        flush=True,
    )
    print(f"Report: {args.output}", flush=True)

    if not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        print(f"FAIL: {failed}", flush=True)
        return 1
    if args.fail_on_warning and summary["warnings"]:
        print("FAIL: warnings were reported", flush=True)
        return 1
    print("PASS: ReplicaCAD manifest acceptance checks passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
