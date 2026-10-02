import numpy as np
import mujoco
import mujoco.viewer
import time

MODEL_PATH = "robot_descriptions/single_arm_heal_effort_actuation_rs_mj.xml"

JOINTS = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
]

# ============================================================
# CHANGE JOINT ANGLES HERE
# ============================================================

joint_angles_deg = [
    0.0,
    -30.0,
    0.0,
    -90.0,
    0.0,
    60.0
]

q = np.radians(joint_angles_deg)


# ============================================================
# LOAD MUJOCO MODEL
# ============================================================

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)


# ============================================================
# SET JOINT ANGLES
# ============================================================

for i, joint_name in enumerate(JOINTS):

    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name
    )

    qpos_address = model.jnt_qposadr[joint_id]

    data.qpos[qpos_address] = q[i]


# ============================================================
# FORWARD KINEMATICS
# ============================================================

mujoco.mj_forward(model, data)


# ============================================================
# GET END-EFFECTOR
# ============================================================

site_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "right_center"
)

position = data.site_xpos[site_id].copy()
rotation = data.site_xmat[site_id].reshape(3, 3).copy()


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("HEAL MUJOCO FORWARD KINEMATICS")
print("=" * 50)

print("\nJoint angles:")
for i, angle in enumerate(joint_angles_deg):
    print(f"Joint {i+1}: {angle:.2f} degrees")

print("\nEnd-effector position:")
print(f"x = {position[0]:.6f} m")
print(f"y = {position[1]:.6f} m")
print(f"z = {position[2]:.6f} m")

print("\nEnd-effector rotation matrix:")
print(np.round(rotation, 6))

print("\nOpening MuJoCo viewer...")


# ============================================================
# OPEN VIEWER
# ============================================================

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        viewer.sync()

        time.sleep(0.01)
