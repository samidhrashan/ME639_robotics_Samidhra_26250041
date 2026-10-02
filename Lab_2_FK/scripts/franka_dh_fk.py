import numpy as np

# Franka Panda D-H parameters
# columns: d, a, alpha
DH = np.array([
    [0.333,  0.000,   -np.pi/2],
    [0.000,  0.000,    np.pi/2],
    [0.316,  0.0825,   np.pi/2],
    [0.000, -0.0825,  -np.pi/2],
    [0.384,  0.000,    np.pi/2],
    [0.000,  0.088,    np.pi/2],
    [0.107,  0.000,    0.000],
])

# MuJoCo Panda base position
BASE = np.array([-0.3, 0.0, 0.8])


def dh_matrix(theta, d, a, alpha):
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,        sa,       ca,      d],
        [0,         0,        0,      1]
    ])


def forward_kinematics(joint_angles_deg):
    q = np.radians(joint_angles_deg)

    T = np.eye(4)

    for i in range(7):
        d, a, alpha = DH[i]
        T = T @ dh_matrix(q[i], d, a, alpha)

    # Add MuJoCo base position
    T[:3, 3] += BASE

    return T


# Change these angles to test different configurations
joint_angles_deg = [20, -20, 30, -40, 20, 30, -20]

T = forward_kinematics(joint_angles_deg)

print("FRANKA PANDA - D-H FORWARD KINEMATICS")
print("======================================")
print("Joint angles:", joint_angles_deg)

print("\nT0_7 =")
print(np.round(T, 6))

print("\nEnd-effector position:")
print("x =", round(T[0, 3], 6))
print("y =", round(T[1, 3], 6))
print("z =", round(T[2, 3], 6))
