import time
import numpy as np
import mujoco
import mujoco.viewer


MODEL_PATH = "scene_turtlebot3_waffle_pi.xml"


# =========================================================
# LOAD MODEL
# =========================================================

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)


# =========================================================
# FIND ROBOT BODY
# =========================================================

base_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "base"
)


# =========================================================
# FIND ACTUATORS
# =========================================================

left_actuator = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "wheel_left"
)

right_actuator = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "wheel_right"
)


# =========================================================
# FIND JOINTS
# =========================================================

left_joint = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "wheel_left"
)

right_joint = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "wheel_right"
)


# =========================================================
# IMPORTANT:
# INCREASE VELOCITY ACTUATOR STRENGTH
#
# Original XML:
#     kv = 0.1
#
# That is extremely weak.
#
# We change it to 2.0 here so we don't have to
# =========================================================

model.actuator_gainprm[left_actuator, 0] = 2.0
model.actuator_gainprm[right_actuator, 0] = 2.0


# =========================================================
# KEYBOARD STATE
# =========================================================

keys = {
    "w": False,
    "s": False,
    "a": False,
    "d": False,
}


def key_callback(keycode):

    key = chr(keycode).lower()

    # W
    if key == "w":
        keys["w"] = True
        keys["s"] = False
        keys["a"] = False
        keys["d"] = False

    # S
    elif key == "s":
        keys["s"] = True
        keys["w"] = False
        keys["a"] = False
        keys["d"] = False

    # A
    elif key == "a":
        keys["a"] = True
        keys["w"] = False
        keys["s"] = False
        keys["d"] = False

    # D
    elif key == "d":
        keys["d"] = True
        keys["w"] = False
        keys["s"] = False
        keys["a"] = False

    # SPACE = stop
    elif key == " ":
        keys["w"] = False
        keys["s"] = False
        keys["a"] = False
        keys["d"] = False


# =========================================================
# PRINT STARTUP INFORMATION
# =========================================================

print()
print("============================================")
print("      TURTLEBOT3 WAFFLE PI")
print("============================================")
print()
print("W      = Forward")
print("S      = Backward")
print("A      = Rotate Left")
print("D      = Rotate Right")
print("SPACE  = Stop")
print("ESC    = Quit")
print()
print("Actuator gain: 10.0")
print()


# =========================================================
# START VIEWER
# =========================================================

with mujoco.viewer.launch_passive(
    model,
    data,
    key_callback=key_callback
) as viewer:

    last_print = 0.0

    while viewer.is_running():

        # =================================================
        # WHEEL COMMANDS
        # =================================================

        left_cmd = 0.0
        right_cmd = 0.0

        # Forward
        if keys["w"]:
            left_cmd = 3.0
            right_cmd = 3.0

        # Backward
        elif keys["s"]:
            left_cmd = -3.0
            right_cmd = -3.0

        # Rotate left
        elif keys["a"]:
            left_cmd = -2.0
            right_cmd = 2.0

        # Rotate right
        elif keys["d"]:
            left_cmd = 2.0
            right_cmd = -2.0


        # =================================================
        # APPLY COMMANDS
        # =================================================

        data.ctrl[left_actuator] = left_cmd
        data.ctrl[right_actuator] = right_cmd


        # =================================================
        # STEP PHYSICS
        # =================================================

        mujoco.mj_step(model, data)


        # =================================================
        # ROBOT POSITION
        # =================================================

        position = data.xpos[base_id].copy()

        x = position[0]
        y = position[1]
        z = position[2]


        # =================================================
        # ROTATION MATRIX
        # =================================================

        R = data.xmat[base_id].reshape(3, 3)


        # =================================================
        # YAW
        # =================================================

        yaw = np.arctan2(
            R[1, 0],
            R[0, 0]
        )

        yaw_deg = np.degrees(yaw)


        # =================================================
        # BODY FRAME AXES
        # =================================================

        x_axis = R[:, 0]
        y_axis = R[:, 1]
        z_axis = R[:, 2]


        # =================================================
        # DRAW BODY FRAME
        #
        # RED   = X axis = forward
        # GREEN = Y axis
        # BLUE  = Z axis
        # =================================================

        viewer.user_scn.ngeom = 0

        axis_length = 0.30
        radius = 0.012

        axes = [
            (
                x_axis,
                np.array(
                    [1, 0, 0, 1],
                    dtype=np.float32
                )
            ),

            (
                y_axis,
                np.array(
                    [0, 1, 0, 1],
                    dtype=np.float32
                )
            ),

            (
                z_axis,
                np.array(
                    [0, 0, 1, 1],
                    dtype=np.float32
                )
            ),
        ]


        for axis, rgba in axes:

            start = position.copy()

            end = (
                position
                + axis_length * axis
            )

            direction = end - start

            length = np.linalg.norm(direction)

            if length < 1e-8:
                continue

            direction = direction / length


            # ---------------------------------------------
            # Construct rotation matrix for capsule
            # ---------------------------------------------

            temp = np.array(
                [0.0, 0.0, 1.0]
            )

            if abs(
                np.dot(temp, direction)
            ) > 0.9:

                temp = np.array(
                    [0.0, 1.0, 0.0]
                )


            side = np.cross(
                temp,
                direction
            )

            side = side / np.linalg.norm(side)


            up = np.cross(
                direction,
                side
            )

            mat = np.column_stack(
                (
                    side,
                    up,
                    direction
                )
            )


            # ---------------------------------------------
            # Add capsule
            # ---------------------------------------------

            geom = viewer.user_scn.geoms[
                viewer.user_scn.ngeom
            ]

            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,

                np.array(
                    [
                        radius,
                        length / 2.0,
                        0.0
                    ],
                    dtype=np.float64
                ),

                (start + end) / 2.0,

                mat.reshape(9).astype(
                    np.float64
                ),

                rgba
            )

            viewer.user_scn.ngeom += 1


        # =================================================
        # PRINT DATA
        # =================================================

        current_time = time.time()

        if current_time - last_print > 0.5:

            left_dof = model.jnt_dofadr[
                left_joint
            ]

            right_dof = model.jnt_dofadr[
                right_joint
            ]

            left_velocity = data.qvel[
                left_dof
            ]

            right_velocity = data.qvel[
                right_dof
            ]

            print()
            print(
                f"Position: "
                f"x={x:.3f}, "
                f"y={y:.3f}, "
                f"z={z:.3f}"
            )

            print(
                f"Yaw: {yaw_deg:.2f} degrees"
            )

            print(
                f"Wheel velocity: "
                f"L={left_velocity:.3f}, "
                f"R={right_velocity:.3f}"
            )

            print("Rotation matrix:")

            print(
                np.round(R, 3)
            )

            last_print = current_time


        # =================================================
        # UPDATE VIEWER
        # =================================================

        viewer.sync()


        # =================================================
        # TIMING
        # =================================================

        time.sleep(0.002)
