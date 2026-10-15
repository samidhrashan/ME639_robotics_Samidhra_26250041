#!/usr/bin/env python3
"""Run paired randomized HEAL episodes for Mink, DLS, and/or QP IK.

The benchmark fixes each randomized cube pose and reuses it for every selected
method, then saves episode JSON/logs, a CSV, method summary, and comparison plots.
Use --episodes 2 first; scale to 25 after confirming the smoke test succeeds.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys

METHODS = ("mink", "dls", "qp")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xml", default="../robot_descriptions/heal_pnp_scene.xml",
                    help="Scene path relative to the scripts folder by default")
    ap.add_argument("--method", choices=[*METHODS, "all"], default="all",
                    help="Method to evaluate; use 'all' for side-by-side comparison")
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--output-dir", default=None,
                    help="Default: ../results/ik_benchmark for all methods, or a per-method folder")
    ap.add_argument("--x-range", nargs=2, type=float, default=[0.35, 0.37])
    ap.add_argument("--y-range", nargs=2, type=float, default=[0.14, 0.16])
    ap.add_argument("--yaw-range-deg", nargs=2, type=float, default=[-45.0, 45.0])
    args = ap.parse_args()
    if args.episodes < 1:
        ap.error("--episodes must be >= 1")

    scripts_dir = Path(__file__).resolve().parent
    benchmark = scripts_dir / "run_ik_benchmark.py"
    if not benchmark.is_file():
        print(f"Missing benchmark engine: {benchmark}")
        return 2

    selected = list(METHODS) if args.method == "all" else [args.method]
    if args.output_dir:
        output_dir = args.output_dir
    elif args.method == "all":
        output_dir = "../results/ik_benchmark"
    else:
        output_dir = f"../results/ik_benchmark_{args.method}"

    cmd = [
        sys.executable, str(benchmark),
        "--xml", args.xml,
        "--episodes", str(args.episodes),
        "--seed", str(args.seed),
        "--methods", *selected,
        "--x-range", *(str(v) for v in args.x_range),
        "--y-range", *(str(v) for v in args.y_range),
        "--yaw-range-deg", *(str(v) for v in args.yaw_range_deg),
        "--output-dir", output_dir,
    ]
    print("[HEAL episodes] methods:", ", ".join(selected))
    print(f"[HEAL episodes] episodes per method: {args.episodes}; seed: {args.seed}")
    print("[HEAL episodes] benchmark outputs will be written by run_ik_benchmark.py")
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
