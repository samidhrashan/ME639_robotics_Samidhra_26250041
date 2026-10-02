import numpy as np
import mujoco

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
# STANDARD DH MATRIX
# ============================================================

def dh_matrix(theta, d, a, alpha):

    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,       sa,       ca,      d],
        [0,        0,        0,      1]
    ])


# ============================================================
# VERIFIED HEAL DH PARAMETERS
#
# Columns:
# theta_offset, d, a, alpha
# ============================================================

DH = np.array([

    [ 3.141593,   0.320800,   0.000000,   1.570792],

    [ 1.570796,   0.087500,   0.300000,   3.141593],

    [-3.140797,   0.087373,   0.000000,  -1.570000],

    [ 3.141592,   0.377354,   0.000001,  -0.509511],

    [-1.571592,   0.000103,   0.000000,  -1.570792],

    [ 0.000000,  -0.122700,   0.000000,   0.000000]
])


# ============================================================
# D-H FORWARD KINEMATICS
# ============================================================

def forward_kinematics(q):

    T = np.eye(4)

    for i in range(6):

        theta_offset = DH[i, 0]
        d = DH[i, 1]
        a = DH[i, 2]
        alpha = DH[i, 3]

        theta = theta_offset + q[i]

        T = T @ dh_matrix(
            theta,
            d,
            a,
            alpha
        )

    return T


# ============================================================
# MUJOCO FORWARD KINEMATICS
# ============================================================

def mujoco_fk(q):

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    for i, name in enumerate(JOINTS):

        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name
        )

        data.qpos[model.jnt_qposadr[jid]] = q[i]

    mujoco.mj_forward(model, data)

    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "right_center"
    )

    position = data.site_xpos[site_id].copy()

    rotation = data.site_xmat[site_id].reshape(3, 3).copy()

    return position, rotation


# ============================================================
# CALCULATE D-H FK
# ============================================================

T_dh = forward_kinematics(q)

p_dh = T_dh[:3, 3]


# ============================================================
# CALCULATE MUJOCO FK
# ============================================================

p_mj, R_mj = mujoco_fk(q)


# ============================================================
# DISPLAY
# ============================================================

print()
print("HEAL ADDVERB - D-H FORWARD KINEMATICS")
print("=" * 55)

print("\nJoint angles:")
print(joint_angles_deg)

print("\nD-H parameter table")
print("-" * 55)

print(
    "Joint   theta_offset(deg)     d(m)       a(m)     alpha(deg)"
)

for i in range(6):

    print(
        f"J{i+1:<5}"
        f"{np.degrees(DH[i,0]):>15.3f}"
        f"{DH[i,1]:>12.6f}"
        f"{DH[i,2]:>12.6f}"
        f"{np.degrees(DH[i,3]):>14.3f}"
    )


print("\nD-H Transformation T0_6")
print("-" * 55)
print(np.round(T_dh, 6))


print("\nD-H end-effector position")
print(
    f"x = {p_dh[0]:.6f} m\n"
    f"y = {p_dh[1]:.6f} m\n"
    f"z = {p_dh[2]:.6f} m"
)


print("\nMuJoCo end-effector position")
print(
    f"x = {p_mj[0]:.6f} m\n"
    f"y = {p_mj[1]:.6f} m\n"
    f"z = {p_mj[2]:.6f} m"
)


print("\nMuJoCo rotation matrix")
print(np.round(R_mj, 6))


# ============================================================
# POSITION ERROR
# ============================================================

error = np.linalg.norm(p_dh - p_mj)

print("\nComparison")
print("-" * 55)

print(f"Position error = {error:.9f} m")

if error < 1e-3:
    print("PASS: D-H and MuJoCo positions agree.")
else:
    print("CHECK: D-H and MuJoCo positions differ.")


# ============================================================
# POSITION DIFFERENCE
# ============================================================

print("\nPosition difference [D-H - MuJoCo]")
print(
    np.round(p_dh - p_mj, 9)
)

