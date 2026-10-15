#!/usr/bin/env python3
"""Run reproducible HEAL pick-and-place benchmarks for Mink, DLS, and QP.

Each episode pose is sampled once and then reused unchanged across all IK methods.
The default randomization bounds are deliberately local to the known-good nominal
pose; expand them only after running workspace_analysis.py and checking reachability.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

METHODS_DEFAULT = ["mink", "dls", "qp"]


def classify_failure(reason: str | None) -> str:
    s = (reason or "").lower()
    if not s:
        return "none"
    if "qp-ik" in s or "no feasible" in s or "ik stalled" in s or "reachable" in s or "infeasible" in s:
        return "ik_infeasible_or_solver"
    if "contact" in s or "gripper" in s or "pad force" in s:
        return "grasp_failure"
    if "lift" in s or "lost pad" in s or "drop" in s or "slip" in s:
        return "grasp_slip_or_drop"
    if "joint" in s or "limit" in s:
        return "joint_limit"
    if "collision" in s or "table" in s:
        return "collision_or_clearance"
    return "other"


def safe_mean(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key) is not None and np.isfinite(float(r[key]))]
    return float(np.mean(vals)) if vals else None


def write_plots(rows, summary_rows, plot_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib missing; CSV/JSON will still be written")
        return
    plot_dir.mkdir(parents=True, exist_ok=True)
    methods = [r["method"] for r in summary_rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(methods, [r["success_rate"] * 100 for r in summary_rows])
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Pick-and-place success rate by IK method")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "success_rate.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    values = [r["mean_solver_time_ms"] for r in summary_rows]
    ax.bar(methods, [0 if v is None else v for v in values])
    ax.set_ylabel("Mean time per IK solver call (ms)")
    ax.set_title("IK solve latency")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "ik_latency.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    values = [r["mean_min_pad_table_clearance_mm"] for r in summary_rows]
    ax.bar(methods, [0 if v is None else v for v in values])
    ax.set_ylabel("Mean minimum pad-table clearance (mm)")
    ax.set_title("Clearance during episodes (negative means penetration)")
    ax.axhline(0, linewidth=1)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "clearance.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    values = [r.get("joint_limit_violations", 0) for r in summary_rows]
    ax.bar(methods, values)
    ax.set_ylabel("Recorded raw joint-limit violations")
    ax.set_title("Joint-limit violations before clipping (DLS) / constrained output (QP)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "joint_limit_violations.png", dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xml", default="../robot_descriptions/heal_pnp_scene.xml")
    ap.add_argument("--controller", default=str(Path(__file__).with_name("heal_pick_place_ik_compare.py")))
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--methods", nargs="+", choices=METHODS_DEFAULT, default=METHODS_DEFAULT)
    ap.add_argument("--x-range", nargs=2, type=float, default=[0.35, 0.37], metavar=("XMIN", "XMAX"))
    ap.add_argument("--y-range", nargs=2, type=float, default=[0.14, 0.16], metavar=("YMIN", "YMAX"))
    ap.add_argument("--yaw-range-deg", nargs=2, type=float, default=[-45.0, 45.0], metavar=("YAWMIN", "YAWMAX"))
    ap.add_argument("--output-dir", default="../results/ik_benchmark")
    ap.add_argument("--stop-on-first-success", action="store_true", help="debug option: stop each method at its first success")
    args = ap.parse_args()

    if args.episodes < 1:
        ap.error("--episodes must be >= 1")
    if not args.x_range[0] < args.x_range[1] or not args.y_range[0] < args.y_range[1]:
        ap.error("x/y range minimums must be less than maximums")

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    xml_path = Path(args.xml)
    if not xml_path.is_absolute():
        xml_path = (script_dir / xml_path).resolve()
    controller = Path(args.controller)
    if not controller.is_absolute():
        controller = (script_dir / controller).resolve()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (script_dir / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    metric_dir = out_dir / "episode_json"
    log_dir.mkdir(exist_ok=True)
    metric_dir.mkdir(exist_ok=True)

    if not xml_path.is_file():
        raise SystemExit(f"Scene XML not found: {xml_path}")
    if not controller.is_file():
        raise SystemExit(f"Controller script not found: {controller}")
    if "qp" in args.methods and importlib.util.find_spec("osqp") is None:
        print("[required] QP method needs OSQP. Run: python -m pip install osqp")
        raise SystemExit(2)

    rng = np.random.default_rng(args.seed)
    yaw_lo, yaw_hi = np.deg2rad(args.yaw_range_deg)
    episode_poses = []
    for ep in range(args.episodes):
        episode_poses.append({
            "episode": ep + 1,
            "seed": args.seed + ep,
            "cube_x": float(rng.uniform(*args.x_range)),
            "cube_y": float(rng.uniform(*args.y_range)),
            "cube_yaw": float(rng.uniform(yaw_lo, yaw_hi)),
        })
    with (out_dir / "episode_poses.json").open("w", encoding="utf-8") as f:
        json.dump(episode_poses, f, indent=2)

    all_rows = []
    start_all = time.perf_counter()
    for method in args.methods:
        print(f"\n================ METHOD: {method.upper()} ================")
        for pose in episode_poses:
            ep = pose["episode"]
            stem = f"{method}_ep{ep:03d}"
            result_json = metric_dir / f"{stem}.json"
            log_path = log_dir / f"{stem}.log"
            cmd = [
                sys.executable, str(controller),
                "--xml", str(xml_path),
                "--ik-method", method,
                "--no-viewer",
                "--cube-x", f"{pose['cube_x']:.8f}",
                "--cube-y", f"{pose['cube_y']:.8f}",
                "--cube-yaw", f"{pose['cube_yaw']:.8f}",
                "--result-json", str(result_json),
            ]
            print(f"[{method} {ep}/{args.episodes}] cube=({pose['cube_x']:.4f}, {pose['cube_y']:.4f}), yaw={np.rad2deg(pose['cube_yaw']):.1f} deg")
            t0 = time.perf_counter()
            try:
                proc = subprocess.run(
                    cmd, cwd=str(project_root), text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    timeout=900,
                )
                log_text = proc.stdout or ""
                exit_code = proc.returncode
            except subprocess.TimeoutExpired as exc:
                log_text = (exc.stdout or "") + "\n[TIMEOUT] episode exceeded 900 seconds\n"
                exit_code = 124
            log_path.write_text(log_text, encoding="utf-8", errors="replace")
            row = {
                **pose,
                "method": method,
                "exit_code": exit_code,
                "log_file": str(log_path),
                "result_json": str(result_json),
                "wall_clock_subprocess_s": time.perf_counter() - t0,
            }
            if result_json.is_file():
                try:
                    with result_json.open(encoding="utf-8") as f:
                        payload = json.load(f)
                    row.update(payload)
                except (OSError, json.JSONDecodeError) as exc:
                    row["failure_reason"] = f"could not parse result JSON: {exc}"
            else:
                row["success"] = False
                row["failure_reason"] = f"controller produced no result JSON (exit {exit_code})"
            row["failure_class"] = classify_failure(row.get("failure_reason"))
            all_rows.append(row)
            print(f"    success={row.get('success', False)}, failure={row.get('failure_class')}, elapsed={row['wall_clock_subprocess_s']:.1f}s")
            if args.stop_on_first_success and row.get("success"):
                break

    # One row per episode/method.
    csv_path = out_dir / "episode_metrics.csv"
    fields = sorted({key for row in all_rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    with (out_dir / "episode_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2)

    summaries = []
    for method in args.methods:
        rows = [r for r in all_rows if r["method"] == method]
        n = len(rows)
        successes = sum(bool(r.get("success", False)) for r in rows)
        reasons = {}
        for row in rows:
            if not row.get("success", False):
                cls = row.get("failure_class", "other")
                reasons[cls] = reasons.get(cls, 0) + 1
        clear_mm = safe_mean(rows, "min_pad_table_clearance_m")
        summaries.append({
            "method": method,
            "episodes_run": n,
            "successes": successes,
            "success_rate": successes / max(n, 1),
            "mean_episode_wall_time_s": safe_mean(rows, "episode_wall_time_s"),
            "mean_subprocess_time_s": safe_mean(rows, "wall_clock_subprocess_s"),
            "mean_solver_time_ms": safe_mean(rows, "mean_solver_time_ms"),
            "mean_ik_solver_calls": safe_mean(rows, "ik_solver_calls"),
            "mean_min_pad_table_clearance_mm": None if clear_mm is None else clear_mm * 1000,
            "velocity_limit_saturations_total": int(sum(int(r.get("velocity_limit_saturations", 0) or 0) for r in rows)),
            "joint_limit_saturations_total": int(sum(int(r.get("joint_limit_saturations", 0) or 0) for r in rows)),
            "joint_limit_violations_total": int(sum(int(r.get("joint_limit_violations", 0) or 0) for r in rows)),
            "failure_breakdown": reasons,
        })
    with (out_dir / "method_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    write_plots(all_rows, summaries, out_dir / "plots")
    print("\n================ SUMMARY ================")
    for row in summaries:
        print(f"{row['method']:>5}: {row['successes']}/{row['episodes_run']} success ({100*row['success_rate']:.1f}%), mean IK call={row['mean_solver_time_ms']} ms, mean min clearance={row['mean_min_pad_table_clearance_mm']} mm")
    print(f"\nBenchmark duration: {time.perf_counter()-start_all:.1f} s")
    print(f"Results CSV: {csv_path}")
    print(f"Summary JSON: {out_dir / 'method_summary.json'}")
    print(f"Plots: {out_dir / 'plots'}")
    print("NOTE: default cube randomization is intentionally limited to a small neighbourhood around the verified nominal position; do workspace analysis before broadening it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
