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

# Test configuration
q = np.radians([
    20.0,
    -20.0,
    30.0,
    -40.0,
    20.0,
    30.0,
    -20.0
])

for i, name in enumerate(JOINTS):
    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name
    )
    data.qpos[model.jnt_qposadr[jid]] = q[i]

mujoco.mj_forward(model, data)

print()
print("FRANKA PANDA JOINT GEOMETRY")
print("=" * 60)
print("Configuration:")
print([20, -20, 30, -40, 20, 30, -20])

for name in JOINTS:

    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name
    )

    body_id = model.jnt_bodyid[jid]

    R = data.xmat[body_id].reshape(3, 3)

    position = (
        data.xpos[body_id]
        + R @ model.jnt_pos[jid]
    )

    axis = R @ model.jnt_axis[jid]

    print()
    print(name)
    print("position =", np.round(position, 6))
    print("axis     =", np.round(axis, 6))


# Hand pose
hand_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "hand"
)

print()
print("HAND")
print("-" * 60)

print("position =", np.round(data.xpos[hand_id], 6))

print("rotation =")
print(
    np.round(
        data.xmat[hand_id].reshape(3, 3),
        6
    )
)
