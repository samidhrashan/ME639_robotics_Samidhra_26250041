#!/usr/bin/env python3
"""Sample HEAL's joint-limited TCP workspace and approximate downward-tool workspace.

This is a configuration-sampling map, not a proof that every TCP target is IK
reachable. It records each sampled q and a downward-approach test, and creates CSV,
JSON and 3D/2D plots. The nominal cube/tray markers match heal_pnp_scene.xml.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

import mujoco
import numpy as np

ARM_JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
TCP_SITE = "pinch"
NOMINAL_CUBE = (0.36, 0.15)
TRAY_CENTER = (0.36, -0.22)


def locate_xml(arg: str) -> Path:
    p = Path(arg)
    if p.is_file():
        return p.resolve()
    script_dir = Path(__file__).resolve().parent
    for candidate in (script_dir / arg, script_dir.parent / arg, script_dir.parent / "robot_descriptions" / Path(arg).name):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not locate scene XML: {arg}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xml", default="../robot_descriptions/heal_pnp_scene.xml")
    ap.add_argument("--samples", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--downward-tolerance-deg", type=float, default=15.0)
    ap.add_argument("--voxel-size", type=float, default=0.025)
    ap.add_argument("--output-dir", default="../results/workspace")
    args = ap.parse_args()
    if args.samples < 100:
        ap.error("--samples should be at least 100")

    xml_path = locate_xml(args.xml)
    script_dir = Path(__file__).resolve().parent
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (script_dir / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    jid = []
    qadr = []
    ranges = []
    for name in ARM_JOINTS:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j < 0:
            raise RuntimeError(f"Missing required joint: {name}")
        if not model.jnt_limited[j]:
            raise RuntimeError(f"Joint {name} is not position-limited; workspace sampler expects explicit ranges.")
        jid.append(j)
        qadr.append(int(model.jnt_qposadr[j]))
        lo, hi = map(float, model.jnt_range[j])
        # Stay 0.02 rad off each hard stop to avoid sampling invalid endpoints.
        margin = min(0.02, 0.05 * (hi - lo))
        ranges.append((lo + margin, hi - margin))
    qadr = np.asarray(qadr, dtype=int)
    ranges = np.asarray(ranges, dtype=float)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
    if site_id < 0:
        raise RuntimeError(f"Missing TCP site: {TCP_SITE}")

    table_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_geom")
    table = None
    if table_geom >= 0:
        table_center = np.asarray(model.geom_pos[table_geom], dtype=float)
        body_id = int(model.geom_bodyid[table_geom])
        table_world = np.asarray(model.body_pos[body_id], dtype=float) + table_center
        half = np.asarray(model.geom_size[table_geom], dtype=float)
        table = {
            "body_position_local_m": np.asarray(model.body_pos[body_id]).tolist(),
            "geom_half_sizes_m": half.tolist(),
            "top_z_m": float(table_world[2] + half[2]),
            "x_bounds_m": [float(table_world[0] - half[0]), float(table_world[0] + half[0])],
            "y_bounds_m": [float(table_world[1] - half[1]), float(table_world[1] + half[1])],
            "dimensions_m": (2 * half).tolist(),
        }

    rng = np.random.default_rng(args.seed)
    nominal_qpos = data.qpos.copy()
    downward_cos = math.cos(math.radians(args.downward_tolerance_deg))
    rows = []
    xyz_all, xyz_down = [], []
    voxel_yaw_bins: dict[tuple[int, int, int], set[int]] = {}

    left_driver = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_driver")
    right_driver = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_driver")

    for idx in range(args.samples):
        data.qpos[:] = nominal_qpos
        q = rng.uniform(ranges[:, 0], ranges[:, 1])
        data.qpos[qadr] = q
        mujoco.mj_forward(model, data)

        p = np.asarray(data.site_xpos[site_id], dtype=float).copy()
        R = data.site_xmat[site_id].reshape(3, 3)
        approach_world = R @ np.array([0.0, 0.0, 1.0])
        alignment = float(np.clip(np.dot(approach_world, np.array([0.0, 0.0, -1.0])), -1.0, 1.0))
        tilt_deg = float(math.degrees(math.acos(alignment)))
        downward_ok = bool(alignment >= downward_cos)

        if left_driver >= 0 and right_driver >= 0:
            close = data.xpos[right_driver] - data.xpos[left_driver]
            close = close - np.dot(close, approach_world) * approach_world
            if np.linalg.norm(close[:2]) > 1e-8:
                yaw_deg = float(math.degrees(math.atan2(close[1], close[0])))
            else:
                yaw_deg = 0.0
        else:
            yaw_deg = 0.0

        row = {f"q{i+1}_rad": float(q[i]) for i in range(6)}
        row.update({
            "x_m": float(p[0]), "y_m": float(p[1]), "z_m": float(p[2]),
            "downward_tilt_deg": tilt_deg,
            "closing_yaw_deg": yaw_deg,
            "downward_approach_feasible_sample": downward_ok,
        })
        rows.append(row)
        xyz_all.append(p)
        if downward_ok:
            xyz_down.append(p)
            yaw_bin = int(((yaw_deg + 180.0) % 360.0) / 30.0)  # 12 yaw bins
            voxel = tuple(np.round(p / args.voxel_size).astype(int).tolist())
            voxel_yaw_bins.setdefault(voxel, set()).add(yaw_bin)

    csv_path = out_dir / "workspace_points.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    xyz_all = np.asarray(xyz_all)
    xyz_down = np.asarray(xyz_down) if xyz_down else np.empty((0, 3))
    dex_voxels = sum(len(bins) >= 3 for bins in voxel_yaw_bins.values())
    summary = {
        "scene_xml": str(xml_path),
        "seed": args.seed,
        "samples": args.samples,
        "joint_ranges_rad": {ARM_JOINTS[i]: ranges[i].tolist() for i in range(6)},
        "downward_tolerance_deg": args.downward_tolerance_deg,
        "downward_approach_samples": int(len(xyz_down)),
        "downward_approach_fraction": float(len(xyz_down) / args.samples),
        "all_workspace_xyz_min_m": xyz_all.min(axis=0).tolist(),
        "all_workspace_xyz_max_m": xyz_all.max(axis=0).tolist(),
        "downward_workspace_xyz_min_m": None if len(xyz_down) == 0 else xyz_down.min(axis=0).tolist(),
        "downward_workspace_xyz_max_m": None if len(xyz_down) == 0 else xyz_down.max(axis=0).tolist(),
        "approx_dexterous_voxels_3_or_more_yaw_bins": int(dex_voxels),
        "dexterous_voxel_size_m": args.voxel_size,
        "yaw_bins": 12,
        "table_geometry": table,
        "interpretation_note": "Random configuration sample cloud, not a formal IK reachability proof. Dexterous voxel count is an observed yaw-diversity proxy, not a guarantee of all orientations.",
    }
    with (out_dir / "workspace_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(xyz_all[:, 0], xyz_all[:, 1], xyz_all[:, 2], c=xyz_all[:, 2], s=2, alpha=0.35)
        ax.scatter([NOMINAL_CUBE[0]], [NOMINAL_CUBE[1]], [0.132], marker="s", s=65, label="Nominal cube TCP target")
        ax.scatter([TRAY_CENTER[0]], [TRAY_CENTER[1]], [0.195], marker="^", s=65, label="Tray/place TCP target")
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
        ax.set_title("Joint-limit sampled TCP workspace")
        fig.colorbar(sc, ax=ax, shrink=0.6, label="TCP Z (m)")
        ax.legend(loc="best")
        fig.tight_layout(); fig.savefig(plot_dir / "workspace_3d.png", dpi=180); plt.close(fig)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        if len(xyz_down):
            sc = ax.scatter(xyz_down[:, 0], xyz_down[:, 1], xyz_down[:, 2], c=xyz_down[:, 2], s=3, alpha=0.5)
            fig.colorbar(sc, ax=ax, shrink=0.6, label="TCP Z (m)")
        ax.scatter([NOMINAL_CUBE[0]], [NOMINAL_CUBE[1]], [0.132], marker="s", s=65, label="Nominal cube TCP target")
        ax.scatter([TRAY_CENTER[0]], [TRAY_CENTER[1]], [0.195], marker="^", s=65, label="Tray/place TCP target")
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
        ax.set_title(f"Downward-tool sample workspace (tilt ≤ {args.downward_tolerance_deg:g}°)")
        ax.legend(loc="best")
        fig.tight_layout(); fig.savefig(plot_dir / "workspace_downward_3d.png", dpi=180); plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 6))
        if len(xyz_down):
            sc = ax.scatter(xyz_down[:, 0], xyz_down[:, 1], c=xyz_down[:, 2], s=4, alpha=0.55)
            fig.colorbar(sc, ax=ax, label="TCP Z (m)")
        if table:
            ax.add_patch(plt.Rectangle((table["x_bounds_m"][0], table["y_bounds_m"][0]),
                                       table["dimensions_m"][0], table["dimensions_m"][1],
                                       fill=False, linewidth=1.4, label="Table top footprint"))
        ax.scatter([NOMINAL_CUBE[0]], [NOMINAL_CUBE[1]], marker="s", s=60, label="Cube target")
        ax.scatter([TRAY_CENTER[0]], [TRAY_CENTER[1]], marker="^", s=60, label="Tray target")
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_aspect("equal", adjustable="datalim")
        ax.set_title("XY downward-approach workspace samples")
        ax.legend(loc="best")
        fig.tight_layout(); fig.savefig(plot_dir / "workspace_xy.png", dpi=180); plt.close(fig)
    except ImportError:
        print("[warn] matplotlib unavailable; wrote CSV and JSON only")

    table_md = [
        "# Table design and workspace summary", "",
        f"Scene: `{xml_path}`", "",
        "## Table dimensions extracted from MuJoCo XML", "",
    ]
    if table:
        table_md += [
            f"- X dimension: {table['dimensions_m'][0]:.3f} m",
            f"- Y dimension: {table['dimensions_m'][1]:.3f} m",
            f"- Height: {table['dimensions_m'][2]:.3f} m",
            f"- Top surface: z = {table['top_z_m']:.3f} m",
            f"- X bounds: {table['x_bounds_m'][0]:.3f} to {table['x_bounds_m'][1]:.3f} m",
            f"- Y bounds: {table['y_bounds_m'][0]:.3f} to {table['y_bounds_m'][1]:.3f} m",
        ]
    else:
        table_md.append("Table geometry `table_geom` was not found; dimensions could not be extracted.")
    table_md += [
        "", "## Sampled workspace (empirical, not a formal reachability proof)", "",
        f"- Random configurations sampled: {args.samples}",
        f"- Downward-approach samples (tilt ≤ {args.downward_tolerance_deg:g}°): {len(xyz_down)} ({100*len(xyz_down)/args.samples:.1f}%)",
        f"- Approximate dexterous voxels with at least 3 observed yaw bins: {dex_voxels}",
        "", "## Table sizing justification", "",
        "The table footprint should be kept within the empirically sampled TCP XY workspace, with both the cube pick area and tray approach confirmed as feasible by inverse kinematics. The table top height must permit the approach pose and gripper fingertips to remain above the surface during descent. The sampled workspace plots provide the empirical boundary; final bounds should be selected from dense samples and confirmed by running IK at each proposed cube/tray corner.",
        "", "Do not interpret random joint-sampled points as guaranteed reachable Cartesian targets; use IK feasibility tests on the final task region before expanding episode randomization.",
    ]
    (out_dir / "table_design_justification.md").write_text("\n".join(table_md) + "\n", encoding="utf-8")
    print(f"[done] points CSV: {csv_path}")
    print(f"[done] summary JSON: {out_dir / 'workspace_summary.json'}")
    print(f"[done] plots: {plot_dir}")
    print(f"[done] table memo: {out_dir / 'table_design_justification.md'}")


if __name__ == "__main__":
    main()
