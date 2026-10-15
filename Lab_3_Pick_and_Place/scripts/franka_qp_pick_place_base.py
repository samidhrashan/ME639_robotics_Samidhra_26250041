r"""
ME639 Lab 3: Franka Pick-and-Place — Method 3: QP-IK Base Interactive Script
Single Episode (Seed 42) with Quintic Trajectory and Live Passive Viewer

Fixes:
  - Table shifted forward and resized so it does not collide with the robot base flange.
  - Symmetrical tray box geometry (corrected tray_wall_right dimensions).
  - DAQP QP-IK optimization with joint limits and velocity limits.
"""

import math
import os
import sys
import time
from pathlib import Path

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
TEMP_XML = FRANKA_DIR / "_me639_franka_qp_base_scene.xml"

# ============================================================
# CONSTANTS & SCENE GEOMETRY
# ============================================================
SEED = 42
DT = 0.01

GRIPPER_ACTUATOR_NAME = "actuator8"
GRIPPER_OPEN = 255.0
GRIPPER_CLOSE = 0.0

HOME_Q = np.array(
    [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5, 0.04, 0.04],
    dtype=np.float64,
)

# Table adjusted so x_edge = 0.50 - 0.30 = 0.20m (leaves 5.5cm clearance to base)
TABLE_X = 0.50
TABLE_Y = 0.0
TABLE_LENGTH_X = 0.60
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
        raise ValueError(f"Object {name!r} not found in model.")
    return identifier


def combine_so3(rot_a, rot_b):
    mat_a = rot_a.as_matrix() if hasattr(rot_a, "as_matrix") else rot_a.matrix()
    mat_b = rot_b.as_matrix() if hasattr(rot_b, "as_matrix") else rot_b.matrix()
    return mink.SO3.from_matrix(mat_a @ mat_b)


def main():
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

    # Initialize Seed 42
    rng = np.random.default_rng(SEED)
    cube_x = float(rng.uniform(*CUBE_X_RANGE))
    cube_y = float(rng.uniform(*CUBE_Y_RANGE))
    yaw_deg = float(rng.uniform(*CUBE_YAW_RANGE_DEG))
    yaw = math.radians(yaw_deg)

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

    # Hard box bounds on configuration and velocity
    limits = [
        mink.ConfigurationLimit(model=arm_model),
        mink.VelocityLimit(model=arm_model),
    ]

    def physics_step(duration, grip_cmd, viewer=None):
        steps = max(1, int(round(duration / model.opt.timestep)))
        contact_seen = False
        for _ in range(steps):
            data.ctrl[:7] = configuration.q[:7]
            data.ctrl[grip_id] = grip_cmd
            mujoco.mj_step(model, data)
            contact_seen = detect_contact() or contact_seen
            if viewer is not None and viewer.is_running():
                viewer.sync()
                time.sleep(model.opt.timestep)
        return contact_seen

    def move_hand(target_pos, duration, grip_cmd, target_yaw=0.0, viewer=None):
        target_pos = np.asarray(target_pos, dtype=float)
        start_pos = get_hand_pos()
        steps = max(1, int(round(duration / DT)))

        for step in range(steps + 1):
            tau = step / steps
            alpha = quintic_scaling(tau)
            des_pos = (1.0 - alpha) * start_pos + alpha * target_pos

            configuration.q[:] = data.qpos[:arm_model.nq]
            ee_task.set_target(pose_target(des_pos, yaw=target_yaw))

            # DAQP active-set quadratic program solve
            vel = mink.solve_ik(
                configuration,
                tasks,
                DT,
                solver="daqp",
                limits=limits,
                damping=1e-4,
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

            if viewer is not None and viewer.is_running():
                viewer.sync()
                time.sleep(model.opt.timestep)

    print("=" * 66)
    print("FRANKA QP-IK BASE PICK-AND-PLACE (SEED 42)")
    print(f"Cube Start: x={cube_x:.4f}, y={cube_y:.4f}, yaw={yaw_deg:.2f}°")
    print("=" * 66)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        time.sleep(0.5)

        # 1. Pre-Grasp Approach
        print("1. Pre-grasp approach...")
        move_hand([cube_x, cube_y, PREGRASP_Z], 1.5, GRIPPER_OPEN, target_yaw=yaw, viewer=viewer)

        # 2. Candidate Grasp Descent
        print("2. Searching grasp height...")
        grasp_height = None
        for candidate_z in GRASP_HEIGHT_CANDIDATES:
            physics_step(0.15, GRIPPER_OPEN, viewer=viewer)
            move_hand([cube_x, cube_y, candidate_z + 0.04], 0.8, GRIPPER_OPEN, target_yaw=yaw, viewer=viewer)
            move_hand([cube_x, cube_y, candidate_z], 0.8, GRIPPER_OPEN, target_yaw=yaw, viewer=viewer)

            seen = physics_step(0.6, GRIPPER_CLOSE, viewer=viewer)
            contact = detect_contact()
            contacts = count_contacts()
            print(f"   Testing candidate z={candidate_z:.3f}m -> Seen: {seen}, Contacts: {contacts}")

            if contact:
                grasp_height = candidate_z
                print(f"   --> Solid grasp acquired at candidate z={grasp_height:.3f}m")
                break

        if grasp_height is None or not contact:
            raise RuntimeError("Grasp failed across candidates.")

        # 3. Lift
        print("3. Lifting cube...")
        cube_before = get_cube_pos()
        move_hand([cube_x, cube_y, grasp_height + 0.12], 1.5, GRIPPER_CLOSE, target_yaw=yaw, viewer=viewer)
        physics_step(0.2, GRIPPER_CLOSE, viewer=viewer)
        cube_after = get_cube_pos()
        lift_dz = float(cube_after[2] - cube_before[2])
        print(f"   Lift displacement: {lift_dz:.4f}m (Threshold: {MINIMUM_LIFT_M}m)")

        if lift_dz < MINIMUM_LIFT_M:
            raise RuntimeError(f"Insufficient lift ({lift_dz:.4f}m < {MINIMUM_LIFT_M}m)")

        # 4. Transport above tray
        print("4. Transporting above tray...")
        move_hand(TRANSPORT_TARGET, 2.0, GRIPPER_CLOSE, target_yaw=0.0, viewer=viewer)

        # 5. Place into tray
        print("5. Lowering into tray...")
        move_hand(PLACE_TARGET, 1.5, GRIPPER_CLOSE, target_yaw=0.0, viewer=viewer)

        # 6. Release & Settle
        print("6. Releasing cube...")
        physics_step(0.8, GRIPPER_OPEN, viewer=viewer)
        physics_step(0.8, GRIPPER_OPEN, viewer=viewer)

        # 7. Retreat
        print("7. Retreating hand...")
        move_hand(RETREAT_TARGET, 1.2, GRIPPER_OPEN, target_yaw=0.0, viewer=viewer)
        physics_step(0.4, GRIPPER_OPEN, viewer=viewer)

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
        print("\n" + "=" * 45)
        print("EPISODE RESULT:", "SUCCESS" if success else "FAILURE")
        print(f"Final Pos: [{final[0]:.4f}, {final[1]:.4f}, {final[2]:.4f}]")
        print(f"Inside XY: {inside_xy} | Settled: {settled}")
        print("=" * 45)

        time.sleep(1.0)


if __name__ == "__main__":
    main()