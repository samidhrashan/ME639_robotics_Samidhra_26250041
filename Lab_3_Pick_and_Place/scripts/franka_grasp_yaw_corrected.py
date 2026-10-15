r"""
ME639 Lab 3: Franka + Mink Pick-and-Place with Yaw-Corrected Grasping

Guaranteed per-episode fresh IK configuration to prevent solver drift/freezing.
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
TEMP_XML = FRANKA_DIR / "_me639_franka_random_yaw_scene.xml"

REPORT_DIR = LAB3_DIR / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

STAGE1_REPORT_TXT = REPORT_DIR / "stage1_seed42_report.txt"
RESULTS_CSV = REPORT_DIR / "seeded_episode_results_random_yaw.csv"
SUMMARY_TXT = REPORT_DIR / "randomized_yaw_summary.txt"

# ============================================================
# EXPERIMENT & VIEWER SETTINGS
# ============================================================
BASE_SEED = 42
NUM_EPISODES = 25

# Headless for high-speed deterministic evaluation
RUN_WITH_VIEWER = False

DT = 0.01
GRIPPER_ACTUATOR_NAME = "actuator8"
GRIPPER_OPEN = 255.0
GRIPPER_CLOSE = 0.0

HOME_Q = np.array(
    [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5, 0.04, 0.04],
    dtype=float,
)

# Table and cube geometry
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

# Tray geometry
TRAY_X = 0.45
TRAY_Y = -0.30
TRAY_BASE_X = 0.18
TRAY_BASE_Y = 0.18
TRAY_BASE_THICKNESS = 0.03
TRAY_WALL_HEIGHT = 0.035
TRAY_WALL_THICKNESS = 0.01
TRAY_BASE_Z = TABLE_HEIGHT + TRAY_BASE_THICKNESS / 2.0

# Sampling bounds
CUBE_X_RANGE = (0.41, 0.49)
CUBE_Y_RANGE = (-0.05, 0.05)
CUBE_YAW_RANGE_DEG = (-45.0, 45.0)

# Grasp targets
PREGRASP_Z = 0.66
GRASP_HEIGHT_CANDIDATES = (0.615, 0.605, 0.595, 0.585)

TRANSPORT_TARGET = np.array([0.426, -0.30, 0.72])
PLACE_TARGET = np.array([0.426, -0.30, 0.64])
RETREAT_TARGET = np.array([0.426, -0.30, 0.75])
MINIMUM_LIFT_M = 0.03


def build_scene_xml():
    if not ORIGINAL_SCENE.exists():
        raise FileNotFoundError(f"Original Franka scene not found:\n{ORIGINAL_SCENE}")

    original_xml = ORIGINAL_SCENE.read_text(encoding="utf-8")

    environment_xml = f"""
    <body name="table_top"
          pos="{TABLE_X} {TABLE_Y} {TABLE_HEIGHT - TABLE_THICKNESS / 2}">
      <geom name="table_top_geom" type="box"
            size="{TABLE_LENGTH_X / 2} {TABLE_LENGTH_Y / 2} {TABLE_THICKNESS / 2}"
            rgba="0.35 0.35 0.35 1"
            contype="1" conaffinity="1"/>
    </body>

    <body name="table_leg_1"
          pos="{TABLE_X - TABLE_LENGTH_X / 2 + LEG_SIZE / 2}
               {TABLE_Y - TABLE_LENGTH_Y / 2 + LEG_SIZE / 2}
               {LEG_HEIGHT / 2}">
      <geom type="box"
            size="{LEG_SIZE / 2} {LEG_SIZE / 2} {LEG_HEIGHT / 2}"
            rgba="0.25 0.25 0.25 1"/>
    </body>

    <body name="table_leg_2"
          pos="{TABLE_X + TABLE_LENGTH_X / 2 - LEG_SIZE / 2}
               {TABLE_Y - TABLE_LENGTH_Y / 2 + LEG_SIZE / 2}
               {LEG_HEIGHT / 2}">
      <geom type="box"
            size="{LEG_SIZE / 2} {LEG_SIZE / 2} {LEG_HEIGHT / 2}"
            rgba="0.25 0.25 0.25 1"/>
    </body>

    <body name="table_leg_3"
          pos="{TABLE_X - TABLE_LENGTH_X / 2 + LEG_SIZE / 2}
               {TABLE_Y + TABLE_LENGTH_Y / 2 - LEG_SIZE / 2}
               {LEG_HEIGHT / 2}">
      <geom type="box"
            size="{LEG_SIZE / 2} {LEG_SIZE / 2} {LEG_HEIGHT / 2}"
            rgba="0.25 0.25 0.25 1"/>
    </body>

    <body name="table_leg_4"
          pos="{TABLE_X + TABLE_LENGTH_X / 2 - LEG_SIZE / 2}
               {TABLE_Y + TABLE_LENGTH_Y / 2 - LEG_SIZE / 2}
               {LEG_HEIGHT / 2}">
      <geom type="box"
            size="{LEG_SIZE / 2} {LEG_SIZE / 2} {LEG_HEIGHT / 2}"
            rgba="0.25 0.25 0.25 1"/>
    </body>

    <body name="cube" pos="0 0 {CUBE_Z}">
      <freejoint/>
      <geom name="cube_geom" type="box"
            size="{CUBE_HALF_SIZE} {CUBE_HALF_SIZE} {CUBE_HALF_SIZE}"
            rgba="0.9 0.2 0.1 1"
            mass="{CUBE_MASS}"
            friction="2.0 0.1 0.1"
            condim="4"
            contype="1" conaffinity="1"/>
    </body>

    <body name="target_tray" pos="{TRAY_X} {TRAY_Y} {TRAY_BASE_Z}">
      <geom name="tray_base" type="box"
            size="{TRAY_BASE_X / 2} {TRAY_BASE_Y / 2} {TRAY_BASE_THICKNESS / 2}"
            rgba="0.2 0.5 0.8 1"/>

      <geom name="tray_wall_front" type="box"
            pos="0 {-TRAY_BASE_Y / 2 + TRAY_WALL_THICKNESS / 2}
                 {TRAY_BASE_THICKNESS / 2 + TRAY_WALL_HEIGHT / 2}"
            size="{TRAY_BASE_X / 2} {TRAY_WALL_THICKNESS / 2} {TRAY_WALL_HEIGHT / 2}"
            rgba="0.2 0.5 0.8 1"/>

      <geom name="tray_wall_back" type="box"
            pos="0 {TRAY_BASE_Y / 2 - TRAY_WALL_THICKNESS / 2}
                 {TRAY_BASE_THICKNESS / 2 + TRAY_WALL_HEIGHT / 2}"
            size="{TRAY_BASE_X / 2} {TRAY_WALL_THICKNESS / 2} {TRAY_WALL_HEIGHT / 2}"
            rgba="0.2 0.5 0.8 1"/>

      <geom name="tray_wall_left" type="box"
            pos="{-TRAY_BASE_X / 2 + TRAY_WALL_THICKNESS / 2} 0
                 {TRAY_BASE_THICKNESS / 2 + TRAY_WALL_HEIGHT / 2}"
            size="{TRAY_WALL_THICKNESS / 2} {TRAY_BASE_Y / 2} {TRAY_WALL_HEIGHT / 2}"
            rgba="0.2 0.5 0.8 1"/>

      <geom name="tray_wall_right" type="box"
            pos="{TRAY_BASE_X / 2 - TRAY_WALL_THICKNESS / 2} 0
                 {TRAY_BASE_THICKNESS / 2 + TRAY_WALL_HEIGHT / 2}"
            size="{TRAY_WALL_THICKNESS / 2} {TRAY_BASE_Y / 2} {TRAY_WALL_HEIGHT / 2}"
            rgba="0.2 0.5 0.8 1"/>
    </body>
    """

    close_tag = "</worldbody>"
    if close_tag not in original_xml:
        raise RuntimeError("Could not find </worldbody> in Franka scene.xml")

    injected_xml = original_xml.replace(close_tag, environment_xml + "\n" + close_tag)
    TEMP_XML.write_text(injected_xml, encoding="utf-8")


def quintic_scaling(tau):
    tau = float(np.clip(tau, 0.0, 1.0))
    return 10.0 * (tau ** 3) - 15.0 * (tau ** 4) + 6.0 * (tau ** 5)


def named_id(model, obj_type, name):
    identifier = mujoco.mj_name2id(model, obj_type, name)
    if identifier < 0:
        raise ValueError(f"Object {name!r} not found in model.")
    return identifier


def combine_so3(rot_a, rot_b):
    mat_a = rot_a.as_matrix() if hasattr(rot_a, "as_matrix") else rot_a.matrix()
    mat_b = rot_b.as_matrix() if hasattr(rot_b, "as_matrix") else rot_b.matrix()
    return mink.SO3.from_matrix(mat_a @ mat_b)


def execute_pipeline(seeds_to_run, is_stage_1=False):
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

    # Extract nominal home rotation
    arm_data.qpos[:] = HOME_Q
    mujoco.mj_forward(arm_model, arm_data)
    home_hand_quat = arm_data.xquat[hand_id].copy()
    home_rotation = mink.SO3(wxyz=home_hand_quat)

    def sync_arm_fk():
        arm_data.qpos[:] = data.qpos[:arm_model.nq]
        mujoco.mj_forward(arm_model, arm_data)

    def get_cube_position():
        mujoco.mj_forward(model, data)
        return data.xpos[cube_id].copy()

    def get_hand_position():
        sync_arm_fk()
        return arm_data.xpos[hand_id].copy()

    def detect_finger_cube_contact():
        for index in range(data.ncon):
            contact = data.contact[index]
            body_a = int(model.geom_bodyid[contact.geom1])
            body_b = int(model.geom_bodyid[contact.geom2])
            if (body_a == cube_id and body_b in finger_body_ids) or (body_b == cube_id and body_a in finger_body_ids):
                return True
        return False

    def count_finger_cube_contacts():
        count = 0
        for index in range(data.ncon):
            contact = data.contact[index]
            body_a = int(model.geom_bodyid[contact.geom1])
            body_b = int(model.geom_bodyid[contact.geom2])
            if (body_a == cube_id and body_b in finger_body_ids) or (body_b == cube_id and body_a in finger_body_ids):
                count += 1
        return count

    def pose_target(position, yaw=0.0):
        rot_yaw = mink.SO3.from_z_radians(yaw)
        target_rot = combine_so3(rot_yaw, home_rotation)
        return mink.SE3.from_rotation_and_translation(
            target_rot,
            np.asarray(position, dtype=float),
        )

    def run_single(episode_num, seed_val, viewer=None):
        rng = np.random.default_rng(seed_val)
        cube_x = float(rng.uniform(*CUBE_X_RANGE))
        cube_y = float(rng.uniform(*CUBE_Y_RANGE))
        yaw_deg = float(rng.uniform(*CUBE_YAW_RANGE_DEG))
        yaw = math.radians(yaw_deg)

        # 1. COMPLETE RESET: Physics AND Fresh Mink solver instance
        mujoco.mj_resetData(model, data)
        mujoco.mj_resetData(arm_model, arm_data)

        arm_data.qpos[:] = HOME_Q
        data.qpos[:9] = HOME_Q

        qw = math.cos(yaw / 2.0)
        qz = math.sin(yaw / 2.0)
        data.qpos[cube_qadr:cube_qadr + 7] = [cube_x, cube_y, CUBE_Z, qw, 0.0, 0.0, qz]
        data.ctrl[:7] = HOME_Q[:7]
        data.ctrl[grip_id] = GRIPPER_OPEN

        mujoco.mj_forward(model, data)
        mujoco.mj_forward(arm_model, arm_data)

        # Fresh Mink IK solver per episode to eliminate all accumulated nullspace drift
        configuration = mink.Configuration(arm_model)
        configuration.q[:] = HOME_Q

        ee_task = mink.FrameTask(
            frame_name="hand",
            frame_type="body",
            position_cost=5.0,
            orientation_cost=2.0,
        )
        posture_task = mink.PostureTask(
            model=arm_model,
            cost=1e-2,
        )
        posture_task.set_target(HOME_Q)
        tasks = [ee_task, posture_task]

        def physics_step(duration, grip_command):
            steps = max(1, int(round(duration / model.opt.timestep)))
            contact_seen = False
            for _ in range(steps):
                data.ctrl[:7] = configuration.q[:7]
                data.ctrl[grip_id] = grip_command
                mujoco.mj_step(model, data)
                contact_seen = detect_finger_cube_contact() or contact_seen
                if viewer is not None and viewer.is_running():
                    viewer.sync()
                    time.sleep(model.opt.timestep)
            return contact_seen

        def move_hand(target_position, duration, grip_command, target_yaw=0.0):
            target_position = np.asarray(target_position, dtype=float)
            sync_arm_fk()
            start_position = arm_data.xpos[hand_id].copy()
            steps = max(1, int(round(duration / DT)))

            for step in range(steps + 1):
                tau = step / steps
                alpha = quintic_scaling(tau)
                desired_position = (1.0 - alpha) * start_position + alpha * target_position

                configuration.q[:] = data.qpos[:arm_model.nq]
                ee_task.set_target(pose_target(desired_position, yaw=target_yaw))

                velocity = mink.solve_ik(
                    configuration,
                    tasks,
                    DT,
                    solver="daqp",
                    damping=1e-3,
                )
                configuration.integrate_inplace(velocity, DT)

                for joint_index in range(7):
                    j_id = named_id(arm_model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{joint_index + 1}")
                    qadr = int(arm_model.jnt_qposadr[j_id])
                    lower, upper = arm_model.jnt_range[j_id]
                    configuration.q[qadr] = np.clip(configuration.q[qadr], lower, upper)

                data.ctrl[:7] = configuration.q[:7]
                data.ctrl[grip_id] = grip_command
                mujoco.mj_step(model, data)

                if viewer is not None and viewer.is_running():
                    viewer.sync()
                    time.sleep(model.opt.timestep)

        initial = get_cube_position()
        fixed_cube_x, fixed_cube_y = float(initial[0]), float(initial[1])

        success = False
        contact = False
        contacts_at_end = 0
        lift_dz = float("nan")
        reason = "Episode did not complete."

        print(f"\nEpisode {episode_num} | Seed {seed_val}: x={fixed_cube_x:.4f}, y={fixed_cube_y:.4f}, yaw={yaw_deg:.2f}°")

        try:
            # 1. Approach above cube
            move_hand([fixed_cube_x, fixed_cube_y, PREGRASP_Z], 1.5, GRIPPER_OPEN, target_yaw=yaw)

            # 2. Descend to grasp height
            grasp_height = None
            for candidate_z in GRASP_HEIGHT_CANDIDATES:
                physics_step(0.15, GRIPPER_OPEN)
                move_hand([fixed_cube_x, fixed_cube_y, candidate_z + 0.05], 0.8, GRIPPER_OPEN, target_yaw=yaw)
                move_hand([fixed_cube_x, fixed_cube_y, candidate_z], 0.8, GRIPPER_OPEN, target_yaw=yaw)

                physics_step(0.6, GRIPPER_CLOSE)
                contact = detect_finger_cube_contact()
                contacts_at_end = count_finger_cube_contacts()

                if contact:
                    grasp_height = candidate_z
                    break

            if grasp_height is None or not contact:
                raise RuntimeError(f"Grasp failed across candidates {GRASP_HEIGHT_CANDIDATES}")

            # 3. Lift
            cube_before_lift = get_cube_position()
            move_hand([fixed_cube_x, fixed_cube_y, grasp_height + 0.12], 1.5, GRIPPER_CLOSE, target_yaw=yaw)
            physics_step(0.2, GRIPPER_CLOSE)
            cube_after_lift = get_cube_position()

            lift_dz = float(cube_after_lift[2] - cube_before_lift[2])
            if lift_dz < MINIMUM_LIFT_M:
                raise RuntimeError(f"Insufficient lift displacement ({lift_dz:.4f}m < {MINIMUM_LIFT_M}m)")

            # 4. Transport above tray
            move_hand(TRANSPORT_TARGET, 2.0, GRIPPER_CLOSE, target_yaw=0.0)

            # 5. Lower into tray
            move_hand(PLACE_TARGET, 1.5, GRIPPER_CLOSE, target_yaw=0.0)

            # 6. Release and settle
            physics_step(0.6, GRIPPER_OPEN)
            physics_step(0.6, GRIPPER_OPEN)

            # 7. Retreat
            move_hand(RETREAT_TARGET, 1.2, GRIPPER_OPEN, target_yaw=0.0)
            physics_step(0.4, GRIPPER_OPEN)

            # 8. Placement verification
            final = get_cube_position()
            tray_inner_x = TRAY_BASE_X / 2 - TRAY_WALL_THICKNESS
            tray_inner_y = TRAY_BASE_Y / 2 - TRAY_WALL_THICKNESS

            inside_xy = (
                (TRAY_X - tray_inner_x + CUBE_HALF_SIZE <= final[0] <= TRAY_X + tray_inner_x - CUBE_HALF_SIZE) and
                (TRAY_Y - tray_inner_y + CUBE_HALF_SIZE <= final[1] <= TRAY_Y + tray_inner_y - CUBE_HALF_SIZE)
            )

            tray_top = TABLE_HEIGHT + TRAY_BASE_THICKNESS
            settled_on_tray = (tray_top - 0.015 <= final[2] <= tray_top + 0.06)

            success = bool(inside_xy and settled_on_tray)
            reason = "placed inside tray" if success else "cube outside tray bounds or did not settle"

        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            final = get_cube_position()

        record = {
            "episode": episode_num,
            "seed": seed_val,
            "cube_start_x": fixed_cube_x,
            "cube_start_y": fixed_cube_y,
            "cube_start_yaw_deg": yaw_deg,
            "finger_cube_contact": contact,
            "contact_count": contacts_at_end,
            "cube_lift_dz_m": lift_dz,
            "cube_final_x": float(final[0]),
            "cube_final_y": float(final[1]),
            "cube_final_z": float(final[2]),
            "success": success,
            "failure_reason": reason,
        }

        print(f" -> Result: {'SUCCESS' if success else 'FAILURE'} | Contacts: {contacts_at_end} | Lift: {lift_dz:.4f}m | Reason: {reason}")
        return record

    results = []
    if RUN_WITH_VIEWER:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            for idx, s in enumerate(seeds_to_run, start=1):
                rec = run_single(idx, s, viewer=viewer)
                results.append(rec)
                if is_stage_1 and not rec["success"]:
                    break
    else:
        for idx, s in enumerate(seeds_to_run, start=1):
            rec = run_single(idx, s, viewer=None)
            results.append(rec)
            if is_stage_1 and not rec["success"]:
                break

    return results


def run_stage_1():
    print("=" * 66)
    print("STAGE 1: Single-Episode Verification (Seed 42)")
    print("=" * 66)

    records = execute_pipeline([42], is_stage_1=True)
    if not records:
        return False
    rec = records[0]

    with open(STAGE1_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("STAGE 1 VERIFICATION REPORT (Seed 42)\n")
        f.write("=" * 45 + "\n")
        for k, v in rec.items():
            f.write(f"{k}: {v}\n")

    print(f"\nStage 1 report written to: {STAGE1_REPORT_TXT}")
    return rec["success"]


def run_stage_3():
    print("\n" + "=" * 66)
    print("STAGE 3: 25 Randomized-Yaw Episodes (Seeds 42–66)")
    print("=" * 66)

    seeds = list(range(42, 42 + NUM_EPISODES))
    records = execute_pipeline(seeds, is_stage_1=False)

    fieldnames = list(records[0].keys())
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    success_count = sum(1 for r in records if r["success"])
    pct = (success_count / len(records)) * 100.0

    reasons = Counter(r["failure_reason"] for r in records if not r["success"])

    summary_lines = [
        "FRANKA + MINK RANDOMIZED-YAW PICK-AND-PLACE EVALUATION",
        "=" * 56,
        f"Base seed: {BASE_SEED}",
        f"Episodes requested: {len(seeds)}",
        f"Episodes completed: {len(records)}",
        f"Successful placements: {success_count}",
        f"Failed placements: {len(records) - success_count}",
        f"Success rate: {pct:.2f}%",
        f"Cube X range: {CUBE_X_RANGE}",
        f"Cube Y range: {CUBE_Y_RANGE}",
        f"Cube yaw range (deg): {CUBE_YAW_RANGE_DEG}",
        f"Results CSV: {RESULTS_CSV}",
        "",
        "FAILURE MODES BREAKDOWN:",
        "-" * 30,
    ]
    if not reasons:
        summary_lines.append("No failures recorded.")
    else:
        for r_name, count in reasons.most_common():
            summary_lines.append(f" - {r_name}: {count}")

    summary_content = "\n".join(summary_lines) + "\n"
    SUMMARY_TXT.write_text(summary_content, encoding="utf-8")
    print("\n" + summary_content)


if __name__ == "__main__":
    stage1_ok = run_stage_1()
    if stage1_ok:
        print("\nStage 1 PASSED! Proceeding directly to Stage 3 evaluation.")
        run_stage_3()
    else:
        print("\nStage 1 FAILED. Halting per workflow guardrails.")
        sys.exit(1)