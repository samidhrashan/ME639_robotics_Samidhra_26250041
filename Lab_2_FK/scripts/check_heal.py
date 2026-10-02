import mujoco
import numpy as np


MODEL = "robot_descriptions/single_arm_heal_effort_actuation_rs_mj.xml"

# Same angles used in heal_fk.py
joint_angles_deg = [
    0.0,
    -30.0,
    0.0,
    -90.0,
    0.0,
    60.0
]

joint_angles = np.radians(joint_angles_deg)


# Load MuJoCo model
model = mujoco.MjModel.from_xml_path(MODEL)
data = mujoco.MjData(model)


# Set the six joint positions
for i in range(6):
    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        f"joint_{i + 1}"
    )

    qpos_index = model.jnt_qposadr[joint_id]
    data.qpos[qpos_index] = joint_angles[i]


# Calculate forward kinematics
mujoco.mj_forward(model, data)


# Find HEAL end-effector site
site_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "right_center"
)


position = data.site_xpos[site_id]
rotation = data.site_xmat[site_id].reshape(3, 3)


print("\nMuJoCo HEAL Forward Kinematics")
print("==============================")

print("\nEnd-effector position:")
print(f"x = {position[0]:.6f} m")
print(f"y = {position[1]:.6f} m")
print(f"z = {position[2]:.6f} m")

print("\nEnd-effector rotation matrix:")
print(np.round(rotation, 6))
