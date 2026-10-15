r"""
ME639 Lab 3 — Franka + Mink randomized-yaw pick-and-place evaluation.

Runs 25 deterministic episodes (seeds 42–66).
Records contact, cube lift, placement success, and failure reason.

Outputs:
    Lab_3_Pick_and_Place/report/seeded_episode_results_random_yaw.csv
    Lab_3_Pick_and_Place/report/randomized_yaw_summary.txt

Run from the project root:
    .\.venv\Scripts\python.exe .\Lab_3_Pick_and_Place\scripts\franka_seeded_episodes.py
"""

from pathlib import Path
import csv
import math
import time
from collections import Counter

import mujoco
import mujoco.viewer
import numpy as np
import mink


# ============================================================
# PATHS
# ============================================================

LAB3_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = LAB3_DIR.parent
FRANKA_DIR = PROJECT_DIR / "mujoco_menagerie" / "franka_emika_panda"

ORIGINAL_SCENE = FRANKA_DIR / "scene.xml"
TEMP_XML = FRANKA_DIR / "_me639_franka_random_yaw_scene.xml"

REPORT_DIR = LAB3_DIR / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = REPORT_DIR / "seeded_episode_results_random_yaw.csv"
SUMMARY_TXT = REPORT_DIR / "randomized_yaw_summary.txt"


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

BASE_SEED = 42
NUM_EPISODES = 25

# Keep False for the first run. Set True to inspect the robot.
RUN_WITH_VIEWER = True

DT = 0.01
GRIPPER_ACTUATOR_NAME = "actuator8"

# Franka Menagerie gripper actuator:
# open = 255, closed = 0
GRIPPER_OPEN = 255.0
GRIPPER_CLOSE = 0.0

HOME_Q = np.array(
    [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5, 0.04, 0.04],
    dtype=float,
)


# ============================================================
# TABLE, CUBE, AND TRAY
# ============================================================

TABLE_X = 0.45
TABLE_Y = 0.0
TABLE_LENGTH_X = 0.70
TABLE_LENGTH_Y = 0.90
TABLE_HEIGHT = 0.50
TABLE_THICKNESS = 0.10

LEG_SIZE = 0.06
LEG_HEIGHT = 0.50

CUBE_HALF_SIZE = 0.02
CUBE_Z = TABLE_HEIGHT + CUBE_HALF_SIZE
CUBE_MASS = 0.05

TRAY_X = 0.45
TRAY_Y = -0.30
TRAY_BASE_X = 0.18
TRAY_BASE_Y = 0.18
TRAY_BASE_THICKNESS = 0.03
TRAY_WALL_HEIGHT = 0.035
TRAY_WALL_THICKNESS = 0.015
TRAY_BASE_Z = TABLE_HEIGHT + TRAY_BASE_THICKNESS / 2.0

CUBE_X_RANGE = (0.41, 0.49)
CUBE_Y_RANGE = (-0.05, 0.05)
CUBE_YAW_RANGE_DEG = (-45.0, 45.0)


# ============================================================
# CARTESIAN MOTION TARGETS
# ============================================================

PREGRASP_Z = 0.66
GRASP_Z = 0.62

# Test a sequence of hand-frame heights if the nominal target misses.
GRASP_HEIGHT_CANDIDATES = (0.62, 0.61, 0.60, 0.59, 0.58)

TRANSPORT_TARGET = np.array([0.426, -0.30, 0.72])
PLACE_TARGET = np.array([0.426, -0.30, 0.65])
RETREAT_TARGET = np.array([0.426, -0.30, 0.75])

MINIMUM_LIFT_M = 0.03


# ============================================================
# SCENE GENERATION
# ============================================================

def build_scene_xml():
    """Create a temporary Franka scene with a table, free cube, and tray."""

    if not ORIGINAL_SCENE.exists():
        raise FileNotFoundError(
            f"Original Franka scene not found:\n{ORIGINAL_SCENE}"
        )

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
            size="{TRAY_BASE_X / 2} {TRAY_BASE_Y / 2} {TRAY_WALL_HEIGHT / 2}"
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

    if "</worldbody>" not in original_xml:
        raise RuntimeError("Could not find </worldbody> in scene.xml.")

    TEMP_XML.write_text(
        original_xml.replace(
            "</worldbody>",
            environment_xml + "\n</worldbody>",
            1,
        ),
        encoding="utf-8",
    )


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def quintic_scaling(tau):
    """Smooth interpolation with zero endpoint velocity and acceleration."""
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def named_id(model, object_type, name):
    result = mujoco.mj_name2id(model, object_type, name)
    if result < 0:
        raise RuntimeError(f"Could not find model object named '{name}'.")
    return result


def save_results(records):
    """Overwrite the CSV with the current run; never mix runs."""
    if not records:
        return

    fieldnames = list(records[0].keys())

    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_summary(records):
    completed = len(records)
    successes = sum(bool(row["success"]) for row in records)
    failures = completed - successes
    rate = 100.0 * successes / completed if completed else 0.0

    reasons = Counter(
        row["failure_reason"]
        for row in records
        if not row["success"]
    )

    lines = [
        "FRANKA + MINK RANDOMIZED-YAW PICK-AND-PLACE",
        "=" * 52,
        f"Base seed: {BASE_SEED}",
        f"Episodes requested: {NUM_EPISODES}",
        f"Episodes completed: {completed}",
        f"Successful placements: {successes}/{completed}",
        f"Failed episodes: {failures}",
        f"Success rate: {rate:.1f}%",
        f"Cube X range (m): {CUBE_X_RANGE}",
        f"Cube Y range (m): {CUBE_Y_RANGE}",
        f"Cube yaw range (degrees): {CUBE_YAW_RANGE_DEG}",
        f"CSV file: {RESULTS_CSV}",
        "",
        "FAILURE BREAKDOWN",
        "-" * 24,
    ]

    if reasons:
        for reason, count in reasons.most_common():
            lines.append(f"{reason}: {count}")
    else:
        lines.append("No failures.")

    summary = "\n".join(lines)
    SUMMARY_TXT.write_text(summary + "\n", encoding="utf-8")

    print("\n" + summary)
    print(f"\nSummary saved to: {SUMMARY_TXT}")


# ============================================================
# EPISODE EVALUATION
# ============================================================

def main():
    build_scene_xml()

    # Mink uses the original arm model; physics uses the full scene.
    arm_model = mujoco.MjModel.from_xml_path(str(ORIGINAL_SCENE))
    arm_data = mujoco.MjData(arm_model)

    model = mujoco.MjModel.from_xml_path(str(TEMP_XML))
    data = mujoco.MjData(model)

    if arm_model.nq != 9:
        raise RuntimeError(
            f"Expected Franka arm model nq=9; received nq={arm_model.nq}."
        )

    if model.nq != 16:
        raise RuntimeError(
            f"Expected full scene nq=16; received nq={model.nq}."
        )

    hand_id = named_id(
        arm_model, mujoco.mjtObj.mjOBJ_BODY, "hand"
    )
    cube_id = named_id(
        model, mujoco.mjtObj.mjOBJ_BODY, "cube"
    )
    left_finger_id = named_id(
        model, mujoco.mjtObj.mjOBJ_BODY, "left_finger"
    )
    right_finger_id = named_id(
        model, mujoco.mjtObj.mjOBJ_BODY, "right_finger"
    )
    grip_id = named_id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR_NAME
    )

    cube_joint_id = int(model.body_jntadr[cube_id])
    cube_qadr = int(model.jnt_qposadr[cube_joint_id])

    # Establish the hand orientation at HOME_Q.
    arm_data.qpos[:] = HOME_Q
    mujoco.mj_forward(arm_model, arm_data)
    home_hand_quat = arm_data.xquat[hand_id].copy()

    configuration = mink.Configuration(
        arm_model,
        q=HOME_Q.copy(),
    )

    task = mink.FrameTask(
        frame_name="hand",
        frame_type="body",
        position_cost=1.0,
        orientation_cost=1.0,
        gain=1.0,
    )

    def pose_target(position):
        """Create an SE3 target using the fixed home hand orientation."""
        rotation = mink.SO3(wxyz=home_hand_quat)
        translation = np.asarray(position, dtype=float)

        return mink.SE3.from_rotation_and_translation(
            rotation,
            translation,
        )

    def sync_arm_fk():
        """Update the arm-only FK model from the physics state."""
        arm_data.qpos[:] = data.qpos[:arm_model.nq]
        mujoco.mj_forward(arm_model, arm_data)

    def get_cube_position():
        mujoco.mj_forward(model, data)
        return data.xpos[cube_id].copy()

    def get_hand_position():
        sync_arm_fk()
        return arm_data.xpos[hand_id].copy()

    def detect_finger_cube_contact():
        """Return True only when a finger body contacts the cube body."""
        for index in range(data.ncon):
            contact = data.contact[index]

            body_a = int(model.geom_bodyid[contact.geom1])
            body_b = int(model.geom_bodyid[contact.geom2])

            cube_and_finger = (
                body_a == cube_id
                and body_b in (left_finger_id, right_finger_id)
            ) or (
                body_b == cube_id
                and body_a in (left_finger_id, right_finger_id)
            )

            if cube_and_finger:
                return True

        return False

    def viewer_is_running(viewer):
        return viewer is None or viewer.is_running()

    def sync_configuration_from_physics():
        """
        Refresh Mink's configuration from the simulated robot state.

        The full scene and arm-only model share the first nine qpos entries:
        seven arm joints and two gripper joints.
        """
        arm_data.qpos[:] = data.qpos[:arm_model.nq]
        mujoco.mj_forward(arm_model, arm_data)

        configuration.q[:] = arm_data.qpos
        configuration.update()

    def physics_step(viewer, duration, grip_command):
        """Advance physics and track whether finger-cube contact occurred."""
        steps = max(1, int(round(duration / model.opt.timestep)))
        contact_seen = False

        for _ in range(steps):
            if not viewer_is_running(viewer):
                raise RuntimeError("Viewer closed during episode.")

            data.ctrl[:7] = configuration.q[:7]
            data.ctrl[grip_id] = grip_command

            mujoco.mj_step(model, data)

            contact_seen = (
                detect_finger_cube_contact() or contact_seen
            )

            if viewer is not None:
                viewer.sync()
                time.sleep(model.opt.timestep)

        return contact_seen

    def move_hand(target_position, duration, viewer, grip_command):
        """
        Move with Mink IK and position actuators.

        The physical robot joint positions are not overwritten during motion.
        The cube remains a free physical body and must be lifted through contact.
        """
        target_position = np.asarray(target_position, dtype=float)
        start_position = get_hand_position()
        steps = max(1, int(round(duration / DT)))

        for step in range(steps + 1):
            if not viewer_is_running(viewer):
                raise RuntimeError("Viewer closed during hand movement.")

            tau = step / steps
            alpha = quintic_scaling(tau)

            desired_position = (
                (1.0 - alpha) * start_position
                + alpha * target_position
            )

            # Closed-loop IK: use actual simulated robot state each cycle.
            sync_configuration_from_physics()
            task.set_target(pose_target(desired_position))

            velocity = mink.solve_ik(
                configuration,
                [task],
                DT,
                solver="daqp",
                damping=1e-4,
            )

            configuration.integrate_inplace(velocity, DT)

            # Clamp the seven arm joints to their joint limits.
            for joint_index in range(7):
                joint_name = f"joint{joint_index + 1}"
                joint_id = named_id(
                    arm_model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    joint_name,
                )

                qadr = int(arm_model.jnt_qposadr[joint_id])
                lower, upper = arm_model.jnt_range[joint_id]

                configuration.q[qadr] = np.clip(
                    configuration.q[qadr],
                    lower,
                    upper,
                )

            data.ctrl[:7] = configuration.q[:7]
            data.ctrl[grip_id] = grip_command

            mujoco.mj_step(model, data)

            if viewer is not None:
                viewer.sync()
                time.sleep(model.opt.timestep)

        actual_position = get_hand_position()
        return float(np.linalg.norm(actual_position - target_position))

    def reset_episode(cube_x, cube_y, yaw):
        """Reset robot and cube; the cube remains a free physical body."""
        mujoco.mj_resetData(model, data)
        mujoco.mj_resetData(arm_model, arm_data)

        configuration.q[:] = HOME_Q
        arm_data.qpos[:] = HOME_Q
        data.qpos[:9] = HOME_Q

        qw = math.cos(yaw / 2.0)
        qz = math.sin(yaw / 2.0)

        # Freejoint pose is x, y, z, qw, qx, qy, qz.
        data.qpos[cube_qadr:cube_qadr + 7] = [
            cube_x,
            cube_y,
            CUBE_Z,
            qw,
            0.0,
            0.0,
            qz,
        ]

        data.ctrl[:7] = HOME_Q[:7]
        data.ctrl[grip_id] = GRIPPER_OPEN

        mujoco.mj_forward(model, data)
        mujoco.mj_forward(arm_model, arm_data)

    def run_episode(episode_number, episode_seed, viewer):
        rng = np.random.default_rng(episode_seed)

        cube_x = float(rng.uniform(*CUBE_X_RANGE))
        cube_y = float(rng.uniform(*CUBE_Y_RANGE))
        yaw_deg = float(rng.uniform(*CUBE_YAW_RANGE_DEG))
        yaw = math.radians(yaw_deg)

        reset_episode(cube_x, cube_y, yaw)

        initial = get_cube_position()
        final = initial.copy()

        success = False
        contact = False
        lift_dz = float("nan")
        reason = "Episode did not complete."

        print("\n" + "#" * 66)
        print(
            f"EPISODE {episode_number}/{NUM_EPISODES} | seed={episode_seed}"
        )
        print(
            f"Cube start: x={cube_x:.4f}, y={cube_y:.4f}, "
            f"yaw={yaw_deg:.2f} degrees"
        )
        print("#" * 66)

        try:
            # 1. Move above the cube.
            move_hand(
                [cube_x, cube_y, PREGRASP_Z],
                2.0,
                viewer,
                GRIPPER_OPEN,
            )

            # 2. Search several hand-frame heights for finger contact.
            contact_during_close = False
            grasp_height = None

            for attempt_index, candidate_z in enumerate(
                GRASP_HEIGHT_CANDIDATES,
                start=1,
            ):
                # Correct viewer check: do not call an undefined function.
                if not viewer_is_running(viewer):
                    raise RuntimeError("Viewer closed during grasp search.")

                current_cube = get_cube_position()
                target_x = float(current_cube[0])
                target_y = float(current_cube[1])

                print(
                    f"Grasp height attempt {attempt_index}/"
                    f"{len(GRASP_HEIGHT_CANDIDATES)}: "
                    f"cube=({target_x:.4f}, {target_y:.4f}), "
                    f"hand target z={candidate_z:.3f}"
                )

                # Open the fingers and approach from above.
                physics_step(viewer, 0.25, GRIPPER_OPEN)

                move_hand(
                    [
                        target_x,
                        target_y,
                        max(PREGRASP_Z, candidate_z + 0.04),
                    ],
                    1.0,
                    viewer,
                    GRIPPER_OPEN,
                )

                move_hand(
                    [target_x, target_y, candidate_z],
                    1.0,
                    viewer,
                    GRIPPER_OPEN,
                )

                seen_this_attempt = physics_step(
                    viewer,
                    0.8,
                    GRIPPER_CLOSE,
                )

                contact_during_close = (
                    contact_during_close or seen_this_attempt
                )
                contact = detect_finger_cube_contact()

                print(
                    f"  contact during closure={seen_this_attempt}; "
                    f"contact at end={contact}"
                )

                if contact:
                    grasp_height = candidate_z
                    cube_x, cube_y = target_x, target_y
                    break

            if grasp_height is None or not contact:
                raise RuntimeError(
                    "Grasp failed at all tested heights "
                    f"{GRASP_HEIGHT_CANDIDATES}; "
                    f"contact_seen_during_close={contact_during_close}; "
                    "lift cancelled."
                )

            # 3. Lift and verify that the cube physically rises.
            cube_before_lift = get_cube_position()

            move_hand(
                [cube_x, cube_y, grasp_height + 0.10],
                2.0,
                viewer,
                GRIPPER_CLOSE,
            )

            physics_step(viewer, 0.25, GRIPPER_CLOSE)
            cube_after_lift = get_cube_position()

            lift_dz = float(
                cube_after_lift[2] - cube_before_lift[2]
            )

            print(f"Cube lift displacement: {lift_dz:.4f} m")

            if lift_dz < MINIMUM_LIFT_M:
                raise RuntimeError(
                    f"Insufficient cube lift ({lift_dz:.4f} m)."
                )

            # 4. Transport above the tray.
            move_hand(
                TRANSPORT_TARGET,
                2.5,
                viewer,
                GRIPPER_CLOSE,
            )

            # 5. Lower into the tray.
            move_hand(
                PLACE_TARGET,
                2.0,
                viewer,
                GRIPPER_CLOSE,
            )

            # 6. Release, allow the cube to settle, and retreat.
            physics_step(viewer, 1.0, GRIPPER_OPEN)
            physics_step(viewer, 1.0, GRIPPER_OPEN)

            move_hand(
                RETREAT_TARGET,
                2.0,
                viewer,
                GRIPPER_OPEN,
            )

            physics_step(viewer, 0.5, GRIPPER_OPEN)
            final = get_cube_position()

            tray_inner_x = (
                TRAY_BASE_X / 2 - TRAY_WALL_THICKNESS
            )
            tray_inner_y = (
                TRAY_BASE_Y / 2 - TRAY_WALL_THICKNESS
            )

            inside_xy = (
                TRAY_X - tray_inner_x + CUBE_HALF_SIZE
                <= final[0]
                <= TRAY_X + tray_inner_x - CUBE_HALF_SIZE
                and
                TRAY_Y - tray_inner_y + CUBE_HALF_SIZE
                <= final[1]
                <= TRAY_Y + tray_inner_y - CUBE_HALF_SIZE
            )

            tray_top = TABLE_HEIGHT + TRAY_BASE_THICKNESS

            settled_on_tray = (
                tray_top - 0.015
                <= final[2]
                <= tray_top + 0.06
            )

            success = bool(inside_xy and settled_on_tray)

            reason = (
                "placed inside tray"
                if success
                else "cube outside tray bounds or did not settle on tray"
            )

            print("Final cube position:", np.round(final, 5))
            print("Inside tray XY:", inside_xy)
            print("Settled on tray:", settled_on_tray)
            print(
                "EPISODE RESULT:",
                "SUCCESS" if success else "FAILURE",
            )

        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            final = get_cube_position()

            print("EPISODE RESULT: FAILURE")
            print("Failure reason:", reason)

        record = {
            "episode": episode_number,
            "seed": episode_seed,
            "cube_start_x": cube_x,
            "cube_start_y": cube_y,
            "cube_start_yaw_deg": yaw_deg,
            "finger_cube_contact": contact,
            "cube_lift_dz_m": lift_dz,
            "cube_final_x": float(final[0]),
            "cube_final_y": float(final[1]),
            "cube_final_z": float(final[2]),
            "success": success,
            "failure_reason": reason,
        }

        return record

    print("=" * 66)
    print("FRANKA + MINK RANDOMIZED-YAW PICK-AND-PLACE")
    print("=" * 66)
    print(f"Base seed: {BASE_SEED}")
    print(f"Episodes requested: {NUM_EPISODES}")
    print(f"Cube X range: {CUBE_X_RANGE}")
    print(f"Cube Y range: {CUBE_Y_RANGE}")
    print(f"Cube yaw range: {CUBE_YAW_RANGE_DEG} degrees")
    print(f"Viewer enabled: {RUN_WITH_VIEWER}")
    print(f"CSV output: {RESULTS_CSV}")
    print(f"Summary output: {SUMMARY_TXT}")

    # Clear old output so results from previous runs are not mixed in.
    RESULTS_CSV.write_text("", encoding="utf-8")

    records = []

    def run_all(viewer):
        for episode_index in range(NUM_EPISODES):
            seed = BASE_SEED + episode_index

            record = run_episode(
                episode_index + 1,
                seed,
                viewer,
            )
            records.append(record)

            # Save after each episode so partial progress is retained.
            save_results(records)
            write_summary(records)

            print(
                f"Progress: {len(records)}/{NUM_EPISODES} | "
                f"successes: "
                f"{sum(bool(r['success']) for r in records)} | "
                f"CSV: {RESULTS_CSV}"
            )

            if viewer is not None and not viewer.is_running():
                print("Viewer closed; stopping remaining episodes.")
                break

            # Keep the final state visible briefly before resetting.
            if viewer is not None:
                for _ in range(20):
                    if not viewer.is_running():
                        break
                    viewer.sync()
                    time.sleep(0.02)

    if RUN_WITH_VIEWER:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance = 1.5
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
            run_all(viewer)
    else:
        run_all(None)

    save_results(records)
    write_summary(records)

    print("\nEvaluation finished.")
    print(f"Results CSV: {RESULTS_CSV}")
    print(f"Summary: {SUMMARY_TXT}")


if __name__ == "__main__":
    main()