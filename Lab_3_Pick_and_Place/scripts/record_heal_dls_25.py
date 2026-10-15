
#!/usr/bin/env python3
"""Run 25 consecutive visible HEAL DLS pick-and-place episodes."""

import subprocess
import sys
from pathlib import Path


def main():
    scripts_dir = Path(__file__).resolve().parent
    project_dir = scripts_dir.parent

    controller = scripts_dir / "heal_pick_place_dls.py"
    scene = project_dir / "robot_descriptions" / "heal_pnp_scene.xml"

    output_dir = (
        project_dir / "results" / "recorded_episodes" / "dls"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    total_episodes = 25

    if not controller.exists():
        print(f"Controller not found: {controller}")
        return 1

    print(f"Starting {total_episodes} DLS episodes.")
    print("Start your screen recording now.")

    for episode in range(1, total_episodes + 1):
        result_json = output_dir / f"episode_{episode:02d}.json"

        command = [
            sys.executable,
            str(controller),
            "--xml", str(scene),
            "--result-json", str(result_json),
            "--viewer",
        ]

        print(f"\n===== DLS EPISODE {episode}/{total_episodes} =====")

        result = subprocess.call(command, cwd=str(project_dir))

        if result != 0:
            print(f"Episode {episode} returned error code {result}.")
            print("Stopping the sequence.")
            return result

        print(f"Episode {episode} finished.")

    print("\nAll 25 DLS episodes completed.")
    print("Stop your screen recording.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
