import mujoco
import numpy as np

MODEL = "robot_descriptions/single_arm_heal_effort_actuation_rs_mj.xml"

model = mujoco.MjModel.from_xml_path(MODEL)
data = mujoco.MjData(model)

# All joints at zero
data.qpos[:] = 0

mujoco.mj_forward(model, data)

print("\nHEAL joint geometry at q = 0")
print("============================")

for i in range(1, 7):

    joint_name = f"joint_{i}"

    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name
    )

    body_id = model.jnt_bodyid[joint_id]

    # Joint position in world frame
    joint_pos = data.xpos[body_id] + data.xmat[body_id].reshape(3, 3) @ model.jnt_pos[joint_id]

    # Joint axis in world frame
    R = data.xmat[body_id].reshape(3, 3)
    joint_axis = R @ model.jnt_axis[joint_id]

    print(f"\n{joint_name}")
    print("position =", np.round(joint_pos, 6))
    print("axis     =", np.round(joint_axis, 6))
