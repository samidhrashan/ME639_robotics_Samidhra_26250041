import numpy as np
import mujoco

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

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# All joints at zero
for name in JOINTS:
    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name
    )

    data.qpos[model.jnt_qposadr[jid]] = 0.0

mujoco.mj_forward(model, data)

print()
print("FRANKA PANDA JOINT GEOMETRY AT q = 0")
print("=" * 45)

for name in JOINTS:

    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name
    )

    body_id = model.jnt_bodyid[jid]

    # Joint position in world coordinates
    joint_pos = (
        data.xpos[body_id]
        +
        data.xmat[body_id].reshape(3, 3)
        @ model.jnt_pos[jid]
    )

    # Joint axis in world coordinates
    R = data.xmat[body_id].reshape(3, 3)

    joint_axis = R @ model.jnt_axis[jid]

    print()
    print(name)
    print("position =", np.round(joint_pos, 6))
    print("axis     =", np.round(joint_axis, 6))
