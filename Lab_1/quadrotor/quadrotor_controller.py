import time
import math
import sys
import termios
import tty
import select
import numpy as np

import mujoco
import mujoco.viewer

import rclpy
from rclpy.node import Node


# ============================================================
# MUJOCO MODEL
# ============================================================

MODEL_PATH = "/home/vboxuser/mujoco_menagerie/bitcraze_crazyflie_2/scene.xml"


# ============================================================
# QUATERNION -> YAW
# ============================================================

def quaternion_to_yaw(q):

    w, x, y, z = q

    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z)
    )


# ============================================================
# ANGLE ERROR
# ============================================================

def angle_error(target, current):

    error = target - current

    while error > math.pi:
        error -= 2.0 * math.pi

    while error < -math.pi:
        error += 2.0 * math.pi

    return error


# ============================================================
# ROS 2 NODE
# ============================================================

class QuadrotorROS(Node):

    def __init__(self):

        super().__init__("quadrotor_mujoco")

        self.get_logger().info(
            "Quadrotor controller started"
        )


# ============================================================
# NON-BLOCKING KEYBOARD
# ============================================================

def read_key():

    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)

    return None


# ============================================================
# BODY AXIS
# ============================================================

def set_axis(
    geom,
    start,
    direction,
    length,
    radius,
    rgba
):

    direction = np.asarray(
        direction,
        dtype=float
    )

    direction /= np.linalg.norm(direction)

    # Cylinder points along local Z axis.
    z_axis = direction

    if abs(z_axis[2]) < 0.9:

        reference = np.array([
            0.0,
            0.0,
            1.0
        ])

    else:

        reference = np.array([
            1.0,
            0.0,
            0.0
        ])

    x_axis = np.cross(
        reference,
        z_axis
    )

    x_axis /= np.linalg.norm(
        x_axis
    )

    y_axis = np.cross(
        z_axis,
        x_axis
    )

    rotation = np.column_stack([
        x_axis,
        y_axis,
        z_axis
    ])

    center = (
        np.asarray(start)
        + direction * length / 2.0
    )

    mujoco.mjv_initGeom(

        geom,

        mujoco.mjtGeom.mjGEOM_CYLINDER,

        np.array([
            radius,
            radius,
            length / 2.0
        ]),

        center,

        rotation.reshape(-1),

        np.asarray(
            rgba,
            dtype=np.float32
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    rclpy.init()

    ros_node = QuadrotorROS()

    # ========================================================
    # LOAD MUJOCO
    # ========================================================

    model = mujoco.MjModel.from_xml_path(
        MODEL_PATH
    )

    data = mujoco.MjData(model)

    # ========================================================
    # INITIAL POSITION
    # ========================================================

    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    data.qpos[2] = 1.0

    # ========================================================
    # INITIAL ORIENTATION
    # ========================================================

    data.qpos[3] = 1.0
    data.qpos[4] = 0.0
    data.qpos[5] = 0.0
    data.qpos[6] = 0.0

    mujoco.mj_forward(
        model,
        data
    )

    # ========================================================
    # CONTROLLER PARAMETERS
    # ========================================================

    hover_thrust = 0.26487

    Kx = 0.20
    Ky = 0.20
    Kz = 0.15
    Kyaw = 0.40

    max_roll = 0.15
    max_pitch = 0.15
    max_yaw = 0.60

    min_thrust = 0.0
    max_thrust = 0.35

    # ========================================================
    # TARGET
    # ========================================================

    target_x = 0.0
    target_y = 0.0
    target_z = 1.0
    target_yaw = 0.0

    # Keyboard movement amount
    move_step = 0.30

    # Yaw movement
    yaw_step = math.radians(10.0)

    # ========================================================
    # TERMINAL KEYBOARD MODE
    # ========================================================

    old_settings = termios.tcgetattr(
        sys.stdin
    )

    try:

        tty.setcbreak(
            sys.stdin.fileno()
        )

        # ====================================================
        # START VIEWER
        # ====================================================

        with mujoco.viewer.launch_passive(
            model,
            data
        ) as viewer:

            # =================================================
            # MAKE DRONE LOOK BIGGER
            # =================================================

            viewer.cam.distance = 1.5

            # =================================================
            # BODY AXES
            # =================================================

            viewer.user_scn.ngeom = 3

            # Small axes relative to drone
            axis_length = 0.08
            axis_radius = 0.005

            # =================================================
            # UPDATE TIMERS
            # =================================================

            last_axis_update = 0.0
            last_display = 0.0

            # =================================================
            # MAIN LOOP
            # =================================================

            while viewer.is_running():

                # ------------------------------------------------
                # ROS
                # ------------------------------------------------

                rclpy.spin_once(
                    ros_node,
                    timeout_sec=0.0
                )

                # ------------------------------------------------
                # KEYBOARD
                # ------------------------------------------------

                key = read_key()

                if key:

                    # ESC
                    if ord(key) == 27:
                        break

                    # Current yaw
                    current_yaw = quaternion_to_yaw(
                        data.qpos[3:7]
                    )

                    c = math.cos(
                        current_yaw
                    )

                    s = math.sin(
                        current_yaw
                    )

                    # =================================================
                    # SPACE = FORWARD
                    # Body +X
                    # =================================================

                    if key == " ":

                        target_x += (
                            move_step * c
                        )

                        target_y += (
                            move_step * s
                        )

                    # =================================================
                    # S = BACKWARD
                    # =================================================

                    elif key.lower() == "s":

                        target_x -= (
                            move_step * c
                        )

                        target_y -= (
                            move_step * s
                        )

                    # =================================================
                    # A = LEFT
                    # =================================================

                    elif key.lower() == "a":

                        target_x -= (
                            move_step * s
                        )

                        target_y += (
                            move_step * c
                        )

                    # =================================================
                    # D = RIGHT
                    # =================================================

                    elif key.lower() == "d":

                        target_x += (
                            move_step * s
                        )

                        target_y -= (
                            move_step * c
                        )

                    # =================================================
                    # Q = UP
                    # =================================================

                    elif key.lower() == "q":

                        target_z += move_step

                    # =================================================
                    # E = DOWN
                    # =================================================

                    elif key.lower() == "e":

                        target_z -= move_step

                    # =================================================
                    # J = YAW LEFT
                    # =================================================

                    elif key.lower() == "j":

                        target_yaw += yaw_step

                    # =================================================
                    # L = YAW RIGHT
                    # =================================================

                    elif key.lower() == "l":

                        target_yaw -= yaw_step

                    # =================================================
                    # X = HOLD CURRENT POSITION
                    # =================================================

                    elif key.lower() == "x":

                        target_x = data.qpos[0]
                        target_y = data.qpos[1]
                        target_z = data.qpos[2]

                # =================================================
                # CURRENT POSITION
                # =================================================

                x = data.qpos[0]
                y = data.qpos[1]
                z = data.qpos[2]

                # =================================================
                # CURRENT YAW
                # =================================================

                current_yaw = quaternion_to_yaw(
                    data.qpos[3:7]
                )

                c = math.cos(
                    current_yaw
                )

                s = math.sin(
                    current_yaw
                )

                # =================================================
                # WORLD FRAME ERROR
                # =================================================

                error_x = target_x - x
                error_y = target_y - y
                error_z = target_z - z

                # =================================================
                # ROTATION MATRIX
                #
                # WORLD -> BODY
                # =================================================

                R = np.array([

                    [
                        c,
                        s,
                        0.0
                    ],

                    [
                        -s,
                        c,
                        0.0
                    ],

                    [
                        0.0,
                        0.0,
                        1.0
                    ]

                ])

                # =================================================
                # BODY FRAME ERROR
                # =================================================

                world_error = np.array([

                    error_x,
                    error_y,
                    error_z

                ])

                body_error = R @ world_error

                error_x_body = body_error[0]
                error_y_body = body_error[1]
                error_z_body = body_error[2]

                # =================================================
                # YAW ERROR
                # =================================================

                error_yaw = angle_error(
                    target_yaw,
                    current_yaw
                )

                # =================================================
                # CONTROLLER
                # =================================================

                thrust = (
                    hover_thrust
                    + Kz * error_z_body
                )

                # Direction corrected
                pitch = (
                    Kx * error_x_body
                )

                roll = (
                    -Ky * error_y_body
                )

                yaw = (
                    -Kyaw * error_yaw
                )

                # =================================================
                # LIMIT CONTROLS
                # =================================================

                thrust = np.clip(
                    thrust,
                    min_thrust,
                    max_thrust
                )

                roll = np.clip(
                    roll,
                    -max_roll,
                    max_roll
                )

                pitch = np.clip(
                    pitch,
                    -max_pitch,
                    max_pitch
                )

                yaw = np.clip(
                    yaw,
                    -max_yaw,
                    max_yaw
                )

                # =================================================
                # APPLY CONTROLS
                # =================================================

                data.ctrl[0] = thrust
                data.ctrl[1] = roll
                data.ctrl[2] = pitch
                data.ctrl[3] = yaw

                # =================================================
                # MUJOCO STEP
                # =================================================

                mujoco.mj_step(
                    model,
                    data
                )

                now = time.time()

                # =================================================
                # BODY AXES
                # UPDATE 20 TIMES / SECOND
                # =================================================

                if now - last_axis_update >= 0.05:

                    px = data.qpos[0]
                    py = data.qpos[1]
                    pz = data.qpos[2]

                    axis_start = np.array([
                        px,
                        py,
                        pz
                    ])

                    # ---------------------------------------------
                    # BODY X
                    # ---------------------------------------------

                    body_x = np.array([
                        c,
                        s,
                        0.0
                    ])

                    # ---------------------------------------------
                    # BODY Y
                    # ---------------------------------------------

                    body_y = np.array([
                        -s,
                        c,
                        0.0
                    ])

                    # ---------------------------------------------
                    # BODY Z
                    # ---------------------------------------------

                    body_z = np.array([
                        0.0,
                        0.0,
                        1.0
                    ])

                    # Red = X
                    set_axis(
                        viewer.user_scn.geoms[0],
                        axis_start,
                        body_x,
                        axis_length,
                        axis_radius,
                        [
                            1.0,
                            0.0,
                            0.0,
                            1.0
                        ]
                    )

                    # Green = Y
                    set_axis(
                        viewer.user_scn.geoms[1],
                        axis_start,
                        body_y,
                        axis_length,
                        axis_radius,
                        [
                            0.0,
                            1.0,
                            0.0,
                            1.0
                        ]
                    )

                    # Blue = Z
                    set_axis(
                        viewer.user_scn.geoms[2],
                        axis_start,
                        body_z,
                        axis_length,
                        axis_radius,
                        [
                            0.0,
                            0.0,
                            1.0,
                            1.0
                        ]
                    )

                    last_axis_update = now

                # =================================================
                # TERMINAL DISPLAY
                # UPDATE 2 TIMES / SECOND
                # =================================================

                if now - last_display >= 0.5:

                    # Clear terminal
                    print(
                        "\033[2J\033[H",
                        end=""
                    )

                    print(
                        "=" * 64
                    )

                    print(
                        "          QUADROTOR BODY-FRAME DEMO"
                    )

                    print(
                        "=" * 64
                    )

                    # ------------------------------------------------
                    # POSITION
                    # ------------------------------------------------

                    print()
                    print("POSITION")

                    print(
                        f"Current : "
                        f"X={x:7.3f} "
                        f"Y={y:7.3f} "
                        f"Z={z:7.3f}"
                    )

                    print(
                        f"Target  : "
                        f"X={target_x:7.3f} "
                        f"Y={target_y:7.3f} "
                        f"Z={target_z:7.3f}"
                    )

                    # ------------------------------------------------
                    # YAW
                    # ------------------------------------------------

                    print()
                    print("YAW")

                    print(
                        f"Current : "
                        f"{math.degrees(current_yaw):7.2f}°"
                    )

                    print(
                        f"Target  : "
                        f"{math.degrees(target_yaw):7.2f}°"
                    )

                    # ------------------------------------------------
                    # BODY FRAME
                    # ------------------------------------------------

                    print()
                    print(
                        "BODY FRAME ERROR"
                    )

                    print(
                        f"Xb = {error_x_body:8.3f}"
                    )

                    print(
                        f"Yb = {error_y_body:8.3f}"
                    )

                    print(
                        f"Zb = {error_z_body:8.3f}"
                    )

                    # ------------------------------------------------
                    # ROTATION MATRIX
                    # ------------------------------------------------

                    print()
                    print(
                        "ROTATION MATRIX"
                    )

                    print(
                        "WORLD  ->  BODY"
                    )

                    print()

                    print(
                        f"[ "
                        f"{R[0,0]:7.3f} "
                        f"{R[0,1]:7.3f} "
                        f"{R[0,2]:7.3f} "
                        f"]"
                    )

                    print(
                        f"[ "
                        f"{R[1,0]:7.3f} "
                        f"{R[1,1]:7.3f} "
                        f"{R[1,2]:7.3f} "
                        f"]"
                    )

                    print(
                        f"[ "
                        f"{R[2,0]:7.3f} "
                        f"{R[2,1]:7.3f} "
                        f"{R[2,2]:7.3f} "
                        f"]"
                    )

                    # ------------------------------------------------
                    # CONTROLS
                    # ------------------------------------------------

                    print()
                    print(
                        "KEYBOARD"
                    )

                    print(
                        "SPACE : Forward"
                    )

                    print(
                        "S     : Backward"
                    )

                    print(
                        "A / D : Left / Right"
                    )

                    print(
                        "Q / E : Up / Down"
                    )

                    print(
                        "J / L : Yaw"
                    )

                    print(
                        "X     : Hold position"
                    )

                    print(
                        "ESC   : Exit"
                    )

                    print()
                    print(
                        "RED=X   GREEN=Y   BLUE=Z"
                    )

                    print(
                        "=" * 64
                    )

                    last_display = now

                # =================================================
                # VIEWER
                # =================================================

                viewer.sync()

    finally:

        termios.tcsetattr(
            sys.stdin,
            termios.TCSADRAIN,
            old_settings
        )

        ros_node.destroy_node()

        rclpy.shutdown()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()
