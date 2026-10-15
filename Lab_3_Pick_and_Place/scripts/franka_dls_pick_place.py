r"""
ME639 Lab 3: Pick-and-Place Benchmarking — Method 2: DLS-IK (25 Episodes)
With Live MuJoCo Passive Viewer Synchronization across All Episodes

Evaluates Seeds 42–66 using Closed-Loop Damped Least Squares IK (damping=0.05).
Logs:
  Lab_3_Pick_and_Place/report/seeded_episode_results_dls.csv
  Lab_3_Pick_and_Place/report/dls_summary.txt
"""

import os
import sys
import csv
import math
import time
from pathlib import Path
from collections import Counter

import mujoco
import mujoco.viewer
import numpy as np
import mink

# ============================================================
# PATHS
# ============================================================
SCRIPTS_DIR = Path(__file__).resolve().parent
LAB3_DIR = SCRIPTS_DIR.parent
PROJECT_DIR = LAB3_DIR.parent
FRANKA_DIR = PROJECT_DIR / "mujoco_menagerie" / "franka_emika_panda"

ORIGINAL_SCENE = FRANKA_DIR / "scene.xml"
TEMP_XML = FRANKA_DIR / "_me639_franka_dls_batch_scene.xml"

REPORT_DIR = LAB3_DIR / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = REPORT_DIR / "seeded_episode_results_dls.csv"
SUMMARY_TXT = REPORT_DIR / "dls_summary.txt"

# ============================================================
# CONSTANTS & SCENE GEOMETRY
# ============================================================
BASE_SEED = 42
NUM_EPISODES = 25
DT = 0.01

GRIPPER_ACTUATOR_NAME = "actuator8"
GRIPPER_OPEN = 255.0
GRIPPER_CLOSE = 0.0

HOME_Q = np.array(
    [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5, 0.04, 0.04],
    dtype=np.float64,
)

TABLE_X = 0.45
TABLE_Y = 0.0
TABLE_LENGTH_X = 0.70
TABLE_LENGTH_Y = 0.85
TABLE_HEIGHT = 0.50
TABLE_THICKNESS = 0.10
LEG_SIZE = 0.05
LEG_HEIGHT = TABLE_HEIGHT - TABLE_THICKNESS

CUBE_HALF_SIZE = 0.02
CUBE_Z = TABLE_HEIGHT + CUBE_HALF_SIZE
CUBE_MASS = 0.05

TRAY_X = 0.45
TRAY_Y = -0.30
TRAY_BASE_X = 0.18
TRAY_BASE_Y = 0.18
TRAY_BASE_THICKNESS = 0.03
TRAY_WALL_HEIGHT = 0.035
TRAY_WALL_THICKNESS = 0.01
TRAY_BASE_Z = TABLE_HEIGHT + TRAY_BASE_THICKNESS / 2.0

CUBE_X_RANGE = (0.41, 0.49)
CUBE_Y_RANGE = (-0.05, 0.05)
CUBE_YAW_RANGE_DEG = (-45.0, 45.0)

PREGRASP_Z = 0.66
GRASP_HEIGHT_CANDIDATES = (0.610, 0.600, 0.590, 0.585)

TRANSPORT_TARGET = np.array([0.426, -0.30, 0.72], dtype=np.float64)
PLACE_TARGET = np.array([0.426, -0.30, 0.65], dtype=np.float64)
RETREAT_TARGET = np.array([0.426, -0.30, 0.75], dtype=np.float64)
MINIMUM_LIFT_M = 0.03


def build_scene_xml():
    if not ORIGINAL_SCENE.exists():
        raise FileNotFoundError(f"Original Franka scene not found:\n{ORIGINAL_SCENE}")

    original_xml = ORIGINAL_SCENE.read_text(encoding="utf-8")
    env_xml = f"""
    <body name="table_top" pos="{TABLE_X} {TABLE_Y} {TABLE_HEIGHT - TABLE_THICKNESS / 2}">
      <geom name="table_top_geom" type="box"
            size="{TABLE_LENGTH_X / 2} {TABLE_LENGTH_Y / 2} {TABLE_THICKNESS / 2}"
            rgba="0.35 0.35 0.35 1" contype="1" conaffinity="1"/>
    </body>
    <body name="table_leg_1"
          pos="{TABLE_X - TABLE_LENGTH_X / 2 + LEG_SIZE / 2}
               {TABLE_Y - TABLE_LENGTH_Y / 2 + LEG_SIZE / 2}
               {LEG_HEIGHT / 2}">
      <geom type="box" size="{LEG_SIZE / 2} {LEG_SIZE / 2} {LEG_HEIGHT / 2}" rgba="0.25 0.25 0.25 1"/>
    </body>
    <body name="table_leg_2"
          pos="{TABLE_X + TABLE_LENGTH_X / 2 - LEG_SIZE / 2}
               {TABLE_Y - TABLE_LENGTH_Y / 2 + LEG_SIZE / 2}
               {LEG_HEIGHT / 2}">
      <geom type="box" size="{LEG_SIZE / 2} {LEG_SIZE / 2} {LEG_HEIGHT / 2}" rgba="0.25 0.25 0.25 1"/>
    </body>
    <body name="table_leg_3"
          pos="{TABLE_X - TABLE_LENGTH_X / 2 + LEG_SIZE / 2}
               {TABLE_Y + TABLE_LENGTH_Y / 2 - LEG_SIZE / 2}
               {LEG_HEIGHT / 2}">
      <geom type="box" size="{LEG_SIZE / 2} {LEG_SIZE / 2} {LEG_HEIGHT / 2}" rgba="0.25 0.25 0.25 1"/>
    </body>
    <body name="table_leg_4"
          pos="{TABLE_X + TABLE_LENGTH_X / 2 - LEG_SIZE / 2}
               {TABLE_Y + TABLE_LENGTH_Y / 2 - LEG_SIZE / 2}
               {LEG_HEIGHT / 2}">
      <geom type="box" size="{LEG_SIZE / 2} {LEG_SIZE / 2} {LEG_HEIGHT / 2}" rgba="0.25 0.25 0.25 1"/>
    </body>
    <body name="cube" pos="0 0 {CUBE_Z}">
      <freejoint/>
      <geom name="cube_geom" type="box"
            size="{CUBE_HALF_SIZE} {CUBE_HALF_SIZE} {CUBE_HALF_SIZE}"
            rgba="0.9 0.2 0.1 1" mass="{CUBE_MASS}" friction="2.0 0.1 0.1"
            condim="4" contype="1" conaffinity="1"/>
    </body>
    <body name="target_tray" pos="{TRAY_X} {TRAY_Y} {TRAY_BASE_Z}">
      <geom name="tray_base" type="box"
            size="{TRAY_BASE_X / 2} {TRAY_BASE_Y / 2} {TRAY_BASE_THICKNESS / 2}"
            rgba="0.2 0.5 0.8 1"/>
      <geom name="tray_wall_front" type="box"
            pos="0 {-TRAY_BASE_Y / 2 + TRAY_WALL_THICKNESS / 2} {TRAY_BASE_THICKNESS / 2 + TRAY_WALL_HEIGHT / 2}"
            size="{TRAY_BASE_X / 2} {TRAY_WALL_THICKNESS / 2} {TRAY_WALL_HEIGHT / 2}" rgba="0.2 0.5 0.8 1"/>
      <geom name="tray_wall_back" type="box"
            pos="0 {TRAY_BASE_Y / 2 - TRAY_WALL_THICKNESS / 2} {TRAY_BASE_THICKNESS / 2 + TRAY_WALL_HEIGHT / 2}"
            size="{TRAY_BASE_X / 2} {TRAY_WALL_THICKNESS / 2} {TRAY_WALL_HEIGHT / 2}" rgba="0.2 0.5 0.8 1"/>
      <geom name="tray_wall_left" type="box"
            pos="{-TRAY_BASE_X / 2 + TRAY_WALL_THICKNESS / 2} 0 {TRAY_BASE_THICKNESS / 2 + TRAY_WALL_HEIGHT / 2}"
            size="{TRAY_WALL_THICKNESS / 2} {TRAY_BASE_Y / 2} {TRAY_WALL_HEIGHT / 2}" rgba="0.2 0.5 0.8 1"/>
      <geom name="tray_wall_right" type="box"
            pos="{TRAY_BASE_X / 2 - TRAY_WALL_THICKNESS / 2} 0 {TRAY_BASE_THICKNESS / 2 + TRAY_WALL_HEIGHT / 2}"
            size="{TRAY_WALL_THICKNESS / 2} {TRAY_BASE_Y / 2} {TRAY_WALL_HEIGHT / 2}" rgba="0.2 0.5 0.8 1"/>
    </body>
    """
    close_tag = "</worldbody>"
    TEMP_XML.write_text(original_xml.replace(close_tag, env_xml + "\n" + close_tag), encoding="utf-8")


def quintic_scaling(tau):
    tau = float(np.clip(tau, 0.0, 1.0))
    return 10.0 * (tau**3) - 15.0 * (tau**4) + 6.0 * (tau**5)


def named_id(model, obj_type, name):
    identifier = mujoco.mj_name2id(model, obj_type, name)
    if identifier < 0:
        raise ValueError(f"Object {name!r} not found.")
    return identifier


def combine_so3(rot_a, rot_b):
    mat_a = rot_a.as_matrix() if hasattr(rot_a, "as_matrix") else rot_a.matrix()
    mat_b = rot_b.as_matrix() if hasattr(rot_b, "as_matrix") else rot_b.matrix()
    return mink.SO3.from_matrix(mat_a @ mat_b)


def run_benchmark():
    build_scene_xml()
    model = mujoco.MjModel.from_xml_path(str(TEMP_XML))
    data = mujoco.MjData(model)

    arm_model = mujoco.MjModel.from_xml_path(str(ORIGINAL_SCENE))
    arm_data = mujoco.MjData(arm_model)

    grip_id = named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR_NAME)
    cube_id = named_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_jnt_adr = int(model.body_jntadr[cube_id])
    cube_qadr = int(model.jnt_qposadr[cube_jnt_adr])

    hand_id = named_id(arm_model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    finger_body_ids = {
        named_id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger"),
        named_id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger"),
    }

    arm_data.qpos[:] = HOME_Q
    mujoco.mj_forward(arm_model, arm_data)
    home_hand_quat = arm_data.xquat[hand_id].copy()
    home_rotation = mink.SO3(wxyz=home_hand_quat)

    def get_cube_pos():
        mujoco.mj_forward(model, data)
        return data.xpos[cube_id].copy()

    def get_hand_pos():
        arm_data.qpos[:] = data.qpos[:arm_model.nq]
        mujoco.mj_forward(arm_model, arm_data)
        return arm_data.xpos[hand_id].copy()

    def detect_contact():
        for i in range(data.ncon):
            con = data.contact[i]
            b1 = int(model.geom_bodyid[con.geom1])
            b2 = int(model.geom_bodyid[con.geom2])
            if (b1 == cube_id and b2 in finger_body_ids) or (b2 == cube_id and b1 in finger_body_ids):
                return True
        return False

    def count_contacts():
        cnt = 0
        for i in range(data.ncon):
            con = data.contact[i]
            b1 = int(model.geom_bodyid[con.geom1])
            b2 = int(model.geom_bodyid[con.geom2])
            if (b1 == cube_id and b2 in finger_body_ids) or (b2 == cube_id and b1 in finger_body_ids):
                cnt += 1
        return cnt

    def pose_target(position, yaw=0.0):
        rot_yaw = mink.SO3.from_z_radians(yaw)
        target_rot = combine_so3(rot_yaw, home_rotation)
        return mink.SE3.from_rotation_and_translation(target_rot, np.asarray(position, dtype=float))

    records = []
    seeds = list(range(BASE_SEED, BASE_SEED + NUM_EPISODES))

    print("=" * 66)
    print("BENCHMARK METHOD 2: Damped Least Squares (DLS-IK) — 25 SEEDS (WITH VIEWER)")
    print("=" * 66)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for ep_num, seed in enumerate(seeds, start=1):
            if not viewer.is_running():
                print("\nViewer closed by user. Terminating early and saving completed episodes.")
                break

            rng = np.random.default_rng(seed)
            cube_x = float(rng.uniform(*CUBE_X_RANGE))
            cube_y = float(rng.uniform(*CUBE_Y_RANGE))
            yaw_deg = float(rng.uniform(*CUBE_YAW_RANGE_DEG))
            yaw = math.radians(yaw_deg)

            # Fresh reset per episode
            mujoco.mj_resetData(model, data)
            mujoco.mj_resetData(arm_model, arm_data)
            arm_data.qpos[:] = HOME_Q
            data.qpos[:9] = HOME_Q

            qw = math.cos(yaw / 2.0)
            qz = math.sin(yaw / 2.0)
            data.qpos[cube_qadr : cube_qadr + 7] = [cube_x, cube_y, CUBE_Z, qw, 0.0, 0.0, qz]
            data.ctrl[:7] = HOME_Q[:7]
            data.ctrl[grip_id] = GRIPPER_OPEN

            mujoco.mj_forward(model, data)
            mujoco.mj_forward(arm_model, arm_data)

            configuration = mink.Configuration(arm_model)
            configuration.q[:] = HOME_Q

            ee_task = mink.FrameTask(
                frame_name="hand",
                frame_type="body",
                position_cost=5.0,
                orientation_cost=2.0,
            )
            posture_task = mink.PostureTask(model=arm_model, cost=1e-2)
            posture_task.set_target(HOME_Q)
            tasks = [ee_task, posture_task]

            def physics_step(duration, grip_cmd):
                steps = max(1, int(round(duration / model.opt.timestep)))
                contact_seen = False
                for _ in range(steps):
                    if not viewer.is_running():
                        raise RuntimeError("Viewer closed.")
                    data.ctrl[:7] = configuration.q[:7]
                    data.ctrl[grip_id] = grip_cmd
                    mujoco.mj_step(model, data)
                    contact_seen = detect_contact() or contact_seen
                    viewer.sync()
                    time.sleep(model.opt.timestep)
                return contact_seen

            def move_hand(target_pos, duration, grip_cmd, target_yaw=0.0):
                target_pos = np.asarray(target_pos, dtype=float)
                start_pos = get_hand_pos()
                steps = max(1, int(round(duration / DT)))
                t0 = time.perf_counter()

                for step in range(steps + 1):
                    if not viewer.is_running():
                        raise RuntimeError("Viewer closed.")

                    tau = step / steps
                    alpha = quintic_scaling(tau)
                    des_pos = (1.0 - alpha) * start_pos + alpha * target_pos

                    configuration.q[:] = data.qpos[:arm_model.nq]
                    ee_task.set_target(pose_target(des_pos, yaw=target_yaw))

                    # Closed-Loop DLS-IK
                    vel = mink.solve_ik(
                        configuration,
                        tasks,
                        DT,
                        solver="daqp",
                        damping=0.05,
                    )
                    configuration.integrate_inplace(vel, DT)

                    for j_idx in range(7):
                        j_id = named_id(arm_model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{j_idx + 1}")
                        qadr = int(arm_model.jnt_qposadr[j_id])
                        lo, hi = arm_model.jnt_range[j_id]
                        configuration.q[qadr] = np.clip(configuration.q[qadr], lo, hi)

                    data.ctrl[:7] = configuration.q[:7]
                    data.ctrl[grip_id] = grip_cmd
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    time.sleep(model.opt.timestep)

                return (time.perf_counter() - t0) / max(steps, 1)

            init_cube = get_cube_pos()
            fixed_x, fixed_y = float(init_cube[0]), float(init_cube[1])
            solve_times = []
            contacts_at_end = 0
            contact = False
            lift_dz = float("nan")
            reason = "Episode did not complete"
            success = False

            print(f"\nEpisode {ep_num} | Seed {seed}: x={fixed_x:.4f}, y={fixed_y:.4f}, yaw={yaw_deg:.2f}°")

            try:
                # 1. Approach
                t_s = move_hand([fixed_x, fixed_y, PREGRASP_Z], 1.5, GRIPPER_OPEN, target_yaw=yaw)
                solve_times.append(t_s)

                # 2. Descend candidate heights
                grasp_height = None
                for candidate_z in GRASP_HEIGHT_CANDIDATES:
                    physics_step(0.15, GRIPPER_OPEN)
                    move_hand([fixed_x, fixed_y, candidate_z + 0.04], 0.8, GRIPPER_OPEN, target_yaw=yaw)
                    move_hand([fixed_x, fixed_y, candidate_z], 0.8, GRIPPER_OPEN, target_yaw=yaw)

                    seen = physics_step(0.6, GRIPPER_CLOSE)
                    contact = detect_contact()
                    contacts_at_end = count_contacts()
                    if contact:
                        grasp_height = candidate_z
                        break

                if grasp_height is None or not contact:
                    raise RuntimeError("Grasp failed across candidate heights.")

                # 3. Lift
                c_before = get_cube_pos()
                move_hand([fixed_x, fixed_y, grasp_height + 0.12], 1.5, GRIPPER_CLOSE, target_yaw=yaw)
                physics_step(0.2, GRIPPER_CLOSE)
                c_after = get_cube_pos()

                lift_dz = float(c_after[2] - c_before[2])
                if lift_dz < MINIMUM_LIFT_M:
                    raise RuntimeError(f"Insufficient lift ({lift_dz:.4f}m < {MINIMUM_LIFT_M}m)")

                # 4. Transport
                move_hand(TRANSPORT_TARGET, 2.0, GRIPPER_CLOSE, target_yaw=0.0)

                # 5. Place
                move_hand(PLACE_TARGET, 1.5, GRIPPER_CLOSE, target_yaw=0.0)

                # 6. Release & Settle
                physics_step(0.6, GRIPPER_OPEN)
                physics_step(0.6, GRIPPER_OPEN)

                # 7. Retreat
                move_hand(RETREAT_TARGET, 1.2, GRIPPER_OPEN, target_yaw=0.0)
                physics_step(0.4, GRIPPER_OPEN)

                # 8. Boundary Verification
                final = get_cube_pos()
                tray_inner_x = TRAY_BASE_X / 2 - TRAY_WALL_THICKNESS
                tray_inner_y = TRAY_BASE_Y / 2 - TRAY_WALL_THICKNESS

                inside_xy = (
                    (TRAY_X - tray_inner_x + CUBE_HALF_SIZE <= final[0] <= TRAY_X + tray_inner_x - CUBE_HALF_SIZE) and
                    (TRAY_Y - tray_inner_y + CUBE_HALF_SIZE <= final[1] <= TRAY_Y + tray_inner_y - CUBE_HALF_SIZE)
                )
                tray_top = TABLE_HEIGHT + TRAY_BASE_THICKNESS
                settled = (tray_top - 0.015 <= final[2] <= tray_top + 0.06)

                success = bool(inside_xy and settled)
                reason = "placed inside tray" if success else "cube outside tray bounds or did not settle"

            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                final = get_cube_pos()

            record = {
                "episode": ep_num,
                "seed": seed,
                "method": "DLS-IK",
                "cube_start_x": fixed_x,
                "cube_start_y": fixed_y,
                "cube_start_yaw_deg": yaw_deg,
                "finger_cube_contact": contact,
                "contact_count": contacts_at_end,
                "cube_lift_dz_m": lift_dz,
                "cube_final_x": float(final[0]),
                "cube_final_y": float(final[1]),
                "cube_final_z": float(final[2]),
                "mean_solve_time_ms": float(np.mean(solve_times) * 1000.0) if solve_times else 0.0,
                "success": success,
                "failure_reason": reason,
            }
            records.append(record)
            print(f" -> Result: {'SUCCESS' if success else 'FAILURE'} | Contacts: {contacts_at_end} | Lift: {lift_dz:.4f}m | Reason: {reason}")

    if not records:
        print("No episodes recorded.")
        return

    # Output CSV
    fieldnames = list(records[0].keys())
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    success_count = sum(1 for r in records if r["success"])
    pct = (success_count / len(records)) * 100.0
    reasons = Counter(r["failure_reason"] for r in records if not r["success"])

    summary_lines = [
        "BENCHMARK METHOD 2: DLS-IK RESULTS SUMMARY",
        "=" * 56,
        f"Base seed: {BASE_SEED}",
        f"Episodes completed: {len(records)}",
        f"Successful placements: {success_count}",
        f"Failed placements: {len(records) - success_count}",
        f"Success rate: {pct:.2f}%",
        f"CSV File: {RESULTS_CSV}",
        "",
        "FAILURE BREAKDOWN:",
        "-" * 30,
    ]
    for r_name, count in reasons.most_common():
        summary_lines.append(f" - {r_name}: {count}")

    summary_content = "\n".join(summary_lines) + "\n"
    SUMMARY_TXT.write_text(summary_content, encoding="utf-8")
    print("\n" + summary_content)


if __name__ == "__main__":
    run_benchmark()