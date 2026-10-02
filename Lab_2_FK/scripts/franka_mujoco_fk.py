import time
import numpy as np
import mujoco
import mujoco.viewer

MODEL_PATH = "robot_descriptions/franka/panda.xml"

JOINTS = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
]

# Change these angles whenever you want
joint_angles_deg = [20, -20, 30, -40, 20, 30, -20]


# Load MuJoCo model
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# Set joint angles
q = np.radians(joint_angles_deg)

for i, joint_name in enumerate(JOINTS):
    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name
    )

    qpos_id = model.jnt_qposadr[joint_id]
    data.qpos[qpos_id] = q[i]

# Calculate forward kinematics
mujoco.mj_forward(model, data)

# Find hand body
hand_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "hand"
)

position = data.xpos[hand_id].copy()
rotation = data.xmat[hand_id].reshape(3, 3).copy()

print("FRANKA PANDA - MUJOCO")
print("=====================")
print("Joint angles:", joint_angles_deg)

print("\nEnd-effector position:")
print("x =", round(position[0], 6))
print("y =", round(position[1], 6))
print("z =", round(position[2], 6))

print("\nEnd-effector rotation:")
print(np.round(rotation, 6))

# Open MuJoCo viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.01)
