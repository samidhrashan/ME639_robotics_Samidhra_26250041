"""Final deterministic Franka Mink pick-and-place run for ME639 Lab 3.

Run from the project root or this script's folder. The original successful
single-episode script is not modified. Results are written to a CSV file.
"""
from pathlib import Path
import csv
import math
import time

import mujoco
import mujoco.viewer
import numpy as np
import mink

# ----------------------------------------------------------------------
# Paths and experiment settings
# ----------------------------------------------------------------------
LAB3_DIR = Path(__file__).resolve().parents[1]
FRANKA_DIR = LAB3_DIR.parent / "mujoco_menagerie" / "franka_emika_panda"
ORIGINAL_SCENE = FRANKA_DIR / "scene.xml"
TEMP_XML = FRANKA_DIR / "_me639_franka_seeded_scene.xml"
RESULTS_CSV = LAB3_DIR / "franka_pick_place_result.csv"

BASE_SEED = 42
NUM_EPISODES = 1
DT = 0.01
PHYSICS_DT = 0.01  # the model's own timestep is used when stepping

GRIPPER_ACTUATOR_NAME = "actuator8"
GRIPPER_OPEN = 255.0
GRIPPER_CLOSE = 0.0
HOME_Q = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5, 0.04, 0.04])

TABLE_X, TABLE_Y = 0.45, 0.0
TABLE_LENGTH_X, TABLE_LENGTH_Y = 0.70, 0.90
TABLE_HEIGHT, TABLE_THICKNESS = 0.50, 0.10
LEG_SIZE, LEG_HEIGHT = 0.06, 0.50
CUBE_HALF_SIZE = 0.02
CUBE_Z = TABLE_HEIGHT + CUBE_HALF_SIZE

TRAY_X, TRAY_Y = 0.45, -0.30
TRAY_BASE_X, TRAY_BASE_Y = 0.18, 0.18
TRAY_BASE_THICKNESS = 0.03
TRAY_WALL_HEIGHT = 0.035
TRAY_WALL_THICKNESS = 0.015
TRAY_BASE_Z = TABLE_HEIGHT + TRAY_BASE_THICKNESS / 2

# Randomized start ranges. Keep these modest for the first robustness test.
CUBE_X_RANGE = (0.45, 0.45)
CUBE_Y_RANGE = (0.0, 0.0)
CUBE_YAW_RANGE = (0.0, 0.0)

PREGRASP_Z = 0.66
GRASP_Z = 0.62
TRANSPORT_TARGET = np.array([0.426, -0.30, 0.72])
PLACE_TARGET = np.array([0.426, -0.30, 0.65])
RETREAT_TARGET = np.array([0.426, -0.30, 0.75])


def build_scene_xml():
    """Create the same table/cube/tray environment as the working script."""
    original_xml = ORIGINAL_SCENE.read_text(encoding="utf-8")
    environment_xml = f"""
    <body name="table_top" pos="{TABLE_X} {TABLE_Y} {TABLE_HEIGHT - TABLE_THICKNESS / 2}">
      <geom name="table_top_geom" type="box"
            size="{TABLE_LENGTH_X / 2} {TABLE_LENGTH_Y / 2} {TABLE_THICKNESS / 2}"
            rgba="0.35 0.35 0.35 1" contype="1" conaffinity="1"/>
    </body>
    <body name="table_leg_1" pos="{TABLE_X - TABLE_LENGTH_X / 2 + LEG_SIZE / 2} {TABLE_Y - TABLE_LENGTH_Y / 2 + LEG_SIZE / 2} {LEG_HEIGHT / 2}">
      <geom type="box" size="{LEG_SIZE / 2} {LEG_SIZE / 2} {LEG_HEIGHT / 2}" rgba="0.25 0.25 0.25 1"/>
    </body>
    <body name="table_leg_2" pos="{TABLE_X + TABLE_LENGTH_X / 2 - LEG_SIZE / 2} {TABLE_Y - TABLE_LENGTH_Y / 2 + LEG_SIZE / 2} {LEG_HEIGHT / 2}">
      <geom type="box" size="{LEG_SIZE / 2} {LEG_SIZE / 2} {LEG_HEIGHT / 2}" rgba="0.25 0.25 0.25 1"/>
    </body>
    <body name="table_leg_3" pos="{TABLE_X - TABLE_LENGTH_X / 2 + LEG_SIZE / 2} {TABLE_Y + TABLE_LENGTH_Y / 2 - LEG_SIZE / 2} {LEG_HEIGHT / 2}">
      <geom type="box" size="{LEG_SIZE / 2} {LEG_SIZE / 2} {LEG_HEIGHT / 2}" rgba="0.25 0.25 0.25 1"/>
    </body>
    <body name="table_leg_4" pos="{TABLE_X + TABLE_LENGTH_X / 2 - LEG_SIZE / 2} {TABLE_Y + TABLE_LENGTH_Y / 2 - LEG_SIZE / 2} {LEG_HEIGHT / 2}">
      <geom type="box" size="{LEG_SIZE / 2} {LEG_SIZE / 2} {LEG_HEIGHT / 2}" rgba="0.25 0.25 0.25 1"/>
    </body>
    <body name="cube" pos="0 0 {CUBE_Z}">
      <freejoint/>
      <geom name="cube_geom" type="box" size="{CUBE_HALF_SIZE} {CUBE_HALF_SIZE} {CUBE_HALF_SIZE}"
            rgba="0.9 0.2 0.1 1" mass="0.05" contype="1" conaffinity="1"/>
    </body>
    <body name="target_tray" pos="{TRAY_X} {TRAY_Y} {TRAY_BASE_Z}">
      <geom name="tray_base" type="box" size="{TRAY_BASE_X / 2} {TRAY_BASE_Y / 2} {TRAY_BASE_THICKNESS / 2}" rgba="0.2 0.5 0.8 1"/>
      <geom name="tray_wall_front" type="box" pos="0 {-TRAY_BASE_Y / 2 + TRAY_WALL_THICKNESS / 2} {TRAY_BASE_THICKNESS / 2 + TRAY_WALL_HEIGHT / 2}" size="{TRAY_BASE_X / 2} {TRAY_WALL_THICKNESS / 2} {TRAY_WALL_HEIGHT / 2}" rgba="0.2 0.5 0.8 1"/>
      <geom name="tray_wall_back" type="box" pos="0 {TRAY_BASE_Y / 2 - TRAY_WALL_THICKNESS / 2} {TRAY_BASE_THICKNESS / 2 + TRAY_WALL_HEIGHT / 2}" size="{TRAY_BASE_X / 2} {TRAY_WALL_THICKNESS / 2} {TRAY_WALL_HEIGHT / 2}" rgba="0.2 0.5 0.8 1"/>
      <geom name="tray_wall_left" type="box" pos="{-TRAY_BASE_X / 2 + TRAY_WALL_THICKNESS / 2} 0 {TRAY_BASE_THICKNESS / 2 + TRAY_WALL_HEIGHT / 2}" size="{TRAY_WALL_THICKNESS / 2} {TRAY_BASE_Y / 2} {TRAY_WALL_HEIGHT / 2}" rgba="0.2 0.5 0.8 1"/>
      <geom name="tray_wall_right" type="box" pos="{TRAY_BASE_X / 2 - TRAY_WALL_THICKNESS / 2} 0 {TRAY_BASE_THICKNESS / 2 + TRAY_WALL_HEIGHT / 2}" size="{TRAY_WALL_THICKNESS / 2} {TRAY_BASE_Y / 2} {TRAY_WALL_HEIGHT / 2}" rgba="0.2 0.5 0.8 1"/>
    </body>
    """
    if "</worldbody>" not in original_xml:
        raise RuntimeError("Could not find </worldbody> in the original Franka scene.")
    TEMP_XML.write_text(
        original_xml.replace("</worldbody>", environment_xml + "\n</worldbody>", 1),
        encoding="utf-8",
    )


def quintic_scaling(tau):
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def main():
    if not ORIGINAL_SCENE.exists():
        raise FileNotFoundError(f"Franka scene not found: {ORIGINAL_SCENE}")
    build_scene_xml()

    arm_model = mujoco.MjModel.from_xml_path(str(ORIGINAL_SCENE))
    arm_data = mujoco.MjData(arm_model)
    model = mujoco.MjModel.from_xml_path(str(TEMP_XML))
    data = mujoco.MjData(model)
    if arm_model.nq != 9 or model.nq != 16:
        raise RuntimeError(f"Unexpected model dimensions: arm nq={arm_model.nq}, full nq={model.nq}")

    hand_id = mujoco.mj_name2id(arm_model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    left_finger_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_finger_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
    grip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR_NAME)
    if min(hand_id, cube_id, left_finger_id, right_finger_id, grip_id) < 0:
        raise RuntimeError("Could not find hand/cube/finger bodies or actuator; check model names.")

    print("Gripper actuator control range:", model.actuator_ctrlrange[grip_id])
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        if body_id in (left_finger_id, right_finger_id):
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"
            print(
                f"Finger collision geom: {geom_name} | "
                f"contype={int(model.geom_contype[geom_id])} | "
                f"conaffinity={int(model.geom_conaffinity[geom_id])}"
            )

    cube_joint_id = int(model.body_jntadr[cube_id])
    cube_qadr = int(model.jnt_qposadr[cube_joint_id])
    configuration = mink.Configuration(arm_model, q=HOME_Q.copy())
    task = mink.FrameTask(frame_name="hand", frame_type="body", position_cost=1.0,
                          orientation_cost=0.0, gain=1.0)
    rng = np.random.default_rng(BASE_SEED)
    records = []

    def cube_position():
        mujoco.mj_forward(model, data)
        return data.xpos[cube_id].copy()

    def hand_position():
        mujoco.mj_forward(arm_model, arm_data)
        return arm_data.xpos[hand_id].copy()

    def contact_detected():
        for i in range(data.ncon):
            contact = data.contact[i]
            b1 = int(model.geom_bodyid[contact.geom1])
            b2 = int(model.geom_bodyid[contact.geom2])
            if (b1 == cube_id and b2 in (left_finger_id, right_finger_id)) or \
               (b2 == cube_id and b1 in (left_finger_id, right_finger_id)):
                return True
        return False

    def step_physics(viewer, duration, grip_command):
        count = max(1, int(duration / model.opt.timestep))
        for _ in range(count):
            data.ctrl[:7] = configuration.q[:7]
            data.ctrl[grip_id] = grip_command
            mujoco.mj_step(model, data)
            if viewer.is_running():
                viewer.sync()
                time.sleep(model.opt.timestep)
            else:
                break

    def move_hand(target, duration, viewer, grip_command):
        target = np.asarray(target, dtype=float)
        start = hand_position()
        steps = max(1, int(duration / DT))
        for i in range(steps + 1):
            desired = (1.0 - quintic_scaling(i / steps)) * start + quintic_scaling(i / steps) * target
            task.set_target(mink.SE3.from_translation(desired))
            velocity = mink.solve_ik(configuration, [task], DT, solver="daqp", damping=1e-4)
            configuration.integrate_inplace(velocity, DT)
            # Clamp the seven arm joints using their qpos addresses.
            for j in range(7):
                qadr = int(arm_model.jnt_qposadr[j])
                lower, upper = arm_model.jnt_range[j]
                configuration.q[qadr] = np.clip(configuration.q[qadr], lower, upper)
            data.ctrl[:7] = configuration.q[:7]
            data.ctrl[grip_id] = grip_command
            mujoco.mj_step(model, data)
            arm_data.qpos[:] = configuration.q[:]
            mujoco.mj_forward(arm_model, arm_data)
            if viewer.is_running():
                viewer.sync()
                time.sleep(model.opt.timestep)
            else:
                break
        error = float(np.linalg.norm(hand_position() - target))
        return error

    def reset_episode(cube_x, cube_y, yaw):
        mujoco.mj_resetData(model, data)
        mujoco.mj_resetData(arm_model, arm_data)
        configuration.q[:] = HOME_Q
        arm_data.qpos[:] = HOME_Q
        data.qpos[:9] = HOME_Q
        qw, qz = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
        data.qpos[cube_qadr:cube_qadr + 7] = [cube_x, cube_y, CUBE_Z, qw, 0.0, 0.0, qz]
        data.qpos[7] = 0.04
        data.qpos[8] = 0.04
        data.ctrl[:7] = HOME_Q[:7]
        data.ctrl[grip_id] = GRIPPER_OPEN
        mujoco.mj_forward(model, data)
        mujoco.mj_forward(arm_model, arm_data)

    print("=" * 72)
    print("FRANKA MINK PICK-AND-PLACE — SINGLE BASELINE RUN")
    print(f"Base seed: {BASE_SEED} | Episodes: {NUM_EPISODES}")
    print(f"Cube x range: {CUBE_X_RANGE}; y range: {CUBE_Y_RANGE}; yaw range: {CUBE_YAW_RANGE}")
    print("=" * 72)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 1.5
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20

        for episode in range(NUM_EPISODES):
            episode_seed = BASE_SEED + episode
            # One independent deterministic random stream per episode seed.
            episode_rng = np.random.default_rng(episode_seed)
            x = float(episode_rng.uniform(*CUBE_X_RANGE))
            y = float(episode_rng.uniform(*CUBE_Y_RANGE))
            yaw = float(episode_rng.uniform(*CUBE_YAW_RANGE))
            reset_episode(x, y, yaw)
            initial = cube_position()
            print("\n" + "#" * 72)
            print(f"EPISODE {episode + 1}/{NUM_EPISODES} | seed={episode_seed}")
            print(f"Cube start: x={x:.4f}, y={y:.4f}, yaw={yaw:.4f} rad")
            print("#" * 72)

            success = False
            reason = ""
            contact = False
            lift_dz = float("nan")
            final = initial.copy()
            try:
                move_hand([x, y, PREGRASP_Z], 2.0, viewer, GRIPPER_OPEN)
                move_hand([x, y, GRASP_Z], 2.0, viewer, GRIPPER_OPEN)
                step_physics(viewer, 2.0, GRIPPER_CLOSE)
                contact = contact_detected()
                print("Finger-cube contact:", contact)
                if not contact:
                    raise RuntimeError("No finger-cube contact; lift cancelled")

                cube_before_lift = cube_position()
                move_hand([x, y, GRASP_Z + 0.10], 2.0, viewer, GRIPPER_CLOSE)
                cube_after_lift = cube_position()
                lift_dz = float(cube_after_lift[2] - cube_before_lift[2])
                print(f"Cube lift displacement: {lift_dz:.4f} m")
                if lift_dz < 0.03:
                    raise RuntimeError("Cube did not lift by at least 3 cm")

                move_hand(TRANSPORT_TARGET, 2.5, viewer, GRIPPER_CLOSE)
                move_hand(PLACE_TARGET, 2.0, viewer, GRIPPER_CLOSE)
                step_physics(viewer, 1.0, GRIPPER_OPEN)
                step_physics(viewer, 1.5, GRIPPER_OPEN)
                move_hand(RETREAT_TARGET, 2.0, viewer, GRIPPER_OPEN)
                final = cube_position()
                inside_xy = (
                    TRAY_X - (TRAY_BASE_X / 2 - TRAY_WALL_THICKNESS) <= final[0] <= TRAY_X + (TRAY_BASE_X / 2 - TRAY_WALL_THICKNESS)
                    and TRAY_Y - (TRAY_BASE_Y / 2 - TRAY_WALL_THICKNESS) <= final[1] <= TRAY_Y + (TRAY_BASE_Y / 2 - TRAY_WALL_THICKNESS)
                )
                settled_low = final[2] < TABLE_HEIGHT + 0.12
                success = bool(inside_xy and settled_low)
                reason = "placed inside tray" if success else "cube ended outside tray bounds or did not settle"
                print("Final cube position:", np.round(final, 5))
                print("Inside tray XY:", inside_xy)
                print("EPISODE RESULT:", "SUCCESS" if success else "FAILURE")
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                final = cube_position()
                print("EPISODE RESULT: FAILURE")
                print("Failure reason:", reason)

            records.append({
                "episode": episode + 1,
                "seed": episode_seed,
                "cube_start_x": x,
                "cube_start_y": y,
                "cube_start_yaw_rad": yaw,
                "finger_cube_contact": contact,
                "cube_lift_dz_m": lift_dz,
                "cube_final_x": float(final[0]),
                "cube_final_y": float(final[1]),
                "cube_final_z": float(final[2]),
                "success": success,
                "failure_reason": reason,
            })
            # Brief pause so the viewer can show the final state before reset.
            for _ in range(30):
                if not viewer.is_running():
                    break
                viewer.sync()
                time.sleep(0.02)
            if not viewer.is_running():
                break

    fieldnames = list(records[0].keys()) if records else []
    if records:
        with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        successes = sum(bool(row["success"]) for row in records)
        print("\n" + "=" * 72)
        print("SEEDED EPISODE SUMMARY")
        print(f"Episodes completed: {len(records)}")
        print(f"Successful placements: {successes}/{len(records)}")
        print(f"Success rate: {100.0 * successes / len(records):.1f}%")
        print(f"CSV saved to: {RESULTS_CSV}")
        print("=" * 72)


if __name__ == "__main__":
    main()
