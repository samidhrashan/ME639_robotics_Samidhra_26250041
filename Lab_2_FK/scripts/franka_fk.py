import numpy as np


# ============================================================
# Standard Denavit-Hartenberg transformation
#
# A_i = Rot_z(theta) Trans_z(d) Trans_x(a) Rot_x(alpha)
# ============================================================

def dh_matrix(theta, d, a, alpha):
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,     sa,       ca,       d],
        [0.0,    0.0,      0.0,     1.0]
    ])


# ============================================================
# Franka Panda standard D-H parameters
#
#        theta       d        a       alpha
# ============================================================

DH = [
    [0.0,  0.333,   0.000,    np.pi / 2],
    [0.0,  0.000,   0.316,    0.0],
    [0.0,  0.000,   0.0825,   np.pi / 2],
    [0.0,  0.000,  -0.0825,   np.pi / 2],
    [0.0,  0.384,   0.000,    np.pi / 2],
    [0.0,  0.000,   0.088,   -np.pi / 2],
    [0.0,  0.107,   0.000,    0.0]
]


# ============================================================
# CHANGE YOUR JOINT ANGLES HERE
# ============================================================

joint_angles_deg = [
    0.0,     # Joint 1
    -30.0,   # Joint 2
    0.0,     # Joint 3
    -90.0,   # Joint 4
    0.0,     # Joint 5
    60.0,    # Joint 6
    0.0      # Joint 7
]


# Convert degrees -> radians
joint_angles = np.radians(joint_angles_deg)


# ============================================================
# Forward Kinematics
# ============================================================

T = np.eye(4)

print("\nFranka Panda Forward Kinematics")
print("===============================")

for i in range(7):

    theta = joint_angles[i]
    d = DH[i][1]
    a = DH[i][2]
    alpha = DH[i][3]

    A = dh_matrix(theta, d, a, alpha)

    T = T @ A

    print(f"\nA{i + 1}:")
    print(np.round(A, 6))


# ============================================================
# Final transformation
# ============================================================

print("\n======================")
print("Final Transformation T0_7")
print("======================")

print(np.round(T, 6))


# ============================================================
# End-effector position
# ============================================================

position = T[:3, 3]

print("\nEnd-effector position:")
print(f"x = {position[0]:.6f} m")
print(f"y = {position[1]:.6f} m")
print(f"z = {position[2]:.6f} m")


# ============================================================
# End-effector rotation matrix
# ============================================================

rotation = T[:3, :3]

print("\nEnd-effector rotation matrix:")
print(np.round(rotation, 6))

