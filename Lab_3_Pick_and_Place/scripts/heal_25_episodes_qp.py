#!/usr/bin/env python3
"""Run 25 HEAL benchmark episodes using QP-IK."""

import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPTS_DIR.parent
RUNNER = SCRIPTS_DIR / "heal_seeded_episodes.py"


def main():
    if not RUNNER.exists():
        print(f"[ERROR] Missing benchmark runner: {RUNNER}")
        return 1

    command = [
        sys.executable, str(RUNNER),
        "--method", "qp",
        "--episodes", "25",
        "--seed", "2026",
    ]

    return subprocess.call(command, cwd=str(PROJECT_DIR))


if __name__ == "__main__":
    raise SystemExit(main())