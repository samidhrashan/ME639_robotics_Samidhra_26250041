
#!/usr/bin/env python3
"""Run one visible HEAL pick-and-place episode using Mink."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    scripts_dir = Path(__file__).resolve().parent
    project_dir = scripts_dir.parent

    controller = scripts_dir / "heal_pick_place_mink.py"
    default_xml = project_dir / "robot_descriptions" / "heal_pnp_scene.xml"
    default_json = (
        project_dir / "results" / "recorded_episodes" / "mink_episode.json"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", default=str(default_xml))
    parser.add_argument("--result-json", default=str(default_json))
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()

    if not controller.exists():
        print(f"[ERROR] Controller not found: {controller}")
        return 1

    output_json = Path(args.result_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(controller),
        "--xml", str(Path(args.xml).resolve()),
        "--result-json", str(output_json),
        "--viewer" if not args.no_viewer else "--no-viewer",
    ]

    print("[EPISODE] Method: MINK")
    print("[EPISODE] Viewer:", "OFF" if args.no_viewer else "ON")
    print("[EPISODE] Starting pick-and-place...")

    return subprocess.call(command, cwd=str(project_dir))


if __name__ == "__main__":
    raise SystemExit(main())
