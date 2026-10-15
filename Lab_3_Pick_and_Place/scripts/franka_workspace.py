"""
ME639 Robotics - Lab 3
Franka Panda Workspace Analysis

Purpose:
    Compute the reachable workspace of the 7-DOF Franka Panda arm
    using forward kinematics and random joint-space sampling.

The actual Franka Menagerie model used here does not contain a
MuJoCo site for the end effector (nsite = 0). Therefore the
position of the 'hand' body is used as the end-effector reference.

The gripper joints are NOT included in workspace sampling.

Analysis:
    1. Load the existing Franka Menagerie scene.
    2. Identify the seven Panda arm joints.
    3. Use the actual joint limits from the model.
    4. Sample 20,000 arm configurations.
    5. Compute the hand-body position using forward kinematics.
    6. Report X/Y/Z workspace limits.
    7. Save workspace data.
    8. Generate workspace plots.
"""

from pathlib import Path

import numpy as np
import mujoco
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

NUM_SAMPLES = 20_000

# Reproducible random sampling
RNG_SEED = 42

# Project directories
LAB3_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = LAB3_DIR.parent

# Franka Menagerie scene
MODEL_PATH = (
    REPO_DIR
    / "mujoco_menagerie"
    / "franka_emika_panda"
    / "scene.xml"
)

# Actual end-effector reference in this model
EE_BODY_NAME = "hand"

# Seven Panda arm joints
ARM_JOINT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
]

# Output directories
OUTPUT_DIR = LAB3_DIR / "data" / "franka_workspace"
PLOT_DIR = LAB3_DIR / "plots" / "franka_workspace"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("FRANKA PANDA WORKSPACE ANALYSIS")
print("=" * 70)

print()
print("Scene:")
print(MODEL_PATH)


# ============================================================
# CHECK MODEL PATH
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Franka scene was not found:\n{MODEL_PATH}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

print()
print("Model loaded successfully.")
print(f"nq = {model.nq}")
print(f"nv = {model.nv}")
print(f"njnt = {model.njnt}")
print(f"nbody = {model.nbody}")
print(f"nsite = {model.nsite}")


# ============================================================
# FIND END-EFFECTOR BODY
# ============================================================

hand_body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    EE_BODY_NAME,
)

if hand_body_id < 0:
    raise RuntimeError(
        f"End-effector body '{EE_BODY_NAME}' was not found."
    )

print()
print("End-effector reference:")
print(f"Body name = {EE_BODY_NAME}")
print(f"Body ID   = {hand_body_id}")


# ============================================================
# FIND ARM JOINTS
# ============================================================

joint_ids = []
qpos_addresses = []
joint_ranges = []

print()
print("=" * 70)
print("FRANKA ARM JOINTS")
print("=" * 70)

for joint_name in ARM_JOINT_NAMES:

    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
    )

    if joint_id < 0:
        raise RuntimeError(
            f"Joint '{joint_name}' was not found."
        )

    qpos_address = int(model.jnt_qposadr[joint_id])

    if not model.jnt_limited[joint_id]:
        raise RuntimeError(
            f"Joint '{joint_name}' does not have a defined range."
        )

    joint_range = model.jnt_range[joint_id].copy()

    joint_ids.append(joint_id)
    qpos_addresses.append(qpos_address)
    joint_ranges.append(joint_range)

    print(
        f"{joint_name}: "
        f"joint_id={joint_id}, "
        f"qpos={qpos_address}, "
        f"range=[{joint_range[0]:.4f}, "
        f"{joint_range[1]:.4f}] rad "
        f"= [{np.degrees(joint_range[0]):.2f}, "
        f"{np.degrees(joint_range[1]):.2f}] deg"
    )

joint_ranges = np.asarray(
    joint_ranges,
    dtype=float,
)


# ============================================================
# RANDOM SAMPLING
# ============================================================

print()
print("=" * 70)
print("SAMPLING WORKSPACE")
print("=" * 70)

print(f"Number of configurations = {NUM_SAMPLES}")
print(f"Random seed = {RNG_SEED}")

rng = np.random.default_rng(RNG_SEED)

# Uniform random samples in [0, 1]
random_values = rng.random(
    (NUM_SAMPLES, len(ARM_JOINT_NAMES))
)

# Scale into actual joint limits
q_samples = (
    joint_ranges[:, 0]
    + random_values
    * (
        joint_ranges[:, 1]
        - joint_ranges[:, 0]
    )
)

print("Joint configurations generated successfully.")


# ============================================================
# FORWARD KINEMATICS
# ============================================================

workspace_positions = np.zeros(
    (NUM_SAMPLES, 3),
    dtype=float,
)

print()
print("Computing forward kinematics...")

for i in range(NUM_SAMPLES):

    # Set seven arm joint positions
    for j in range(len(ARM_JOINT_NAMES)):
        data.qpos[qpos_addresses[j]] = q_samples[i, j]

    # Forward kinematics
    mujoco.mj_forward(model, data)

    # Hand body world position
    workspace_positions[i] = data.xpos[hand_body_id]

    if (i + 1) % 2000 == 0:
        print(
            f"Processed "
            f"{i + 1:>6} / {NUM_SAMPLES}"
        )


# ============================================================
# EXTRACT XYZ
# ============================================================

x = workspace_positions[:, 0]
y = workspace_positions[:, 1]
z = workspace_positions[:, 2]


# ============================================================
# WORKSPACE LIMITS
# ============================================================

x_min = np.min(x)
x_max = np.max(x)

y_min = np.min(y)
y_max = np.max(y)

z_min = np.min(z)
z_max = np.max(z)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("FRANKA WORKSPACE RESULTS")
print("=" * 70)

print()
print(
    f"Number of sampled configurations: "
    f"{NUM_SAMPLES}"
)

print()
print("X range:")
print(f"  Minimum = {x_min:.6f} m")
print(f"  Maximum = {x_max:.6f} m")

print()
print("Y range:")
print(f"  Minimum = {y_min:.6f} m")
print(f"  Maximum = {y_max:.6f} m")

print()
print("Z range:")
print(f"  Minimum = {z_min:.6f} m")
print(f"  Maximum = {z_max:.6f} m")


# ============================================================
# SAVE WORKSPACE DATA
# ============================================================

workspace_file = (
    OUTPUT_DIR
    / "franka_workspace_positions.csv"
)

np.savetxt(
    workspace_file,
    workspace_positions,
    delimiter=",",
    header="x,y,z",
    comments="",
)

print()
print("Workspace data saved to:")
print(workspace_file)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_file = (
    OUTPUT_DIR
    / "franka_workspace_summary.txt"
)

with open(
    summary_file,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "ME639 Robotics Lab 3 - "
        "Franka Panda Workspace\n"
    )
    f.write("=" * 60 + "\n\n")

    f.write(
        f"Number of samples: "
        f"{NUM_SAMPLES}\n"
    )

    f.write(
        f"Random seed: "
        f"{RNG_SEED}\n"
    )

    f.write(
        f"End-effector reference body: "
        f"{EE_BODY_NAME}\n\n"
    )

    f.write("Workspace limits:\n")
    f.write(
        f"X: {x_min:.6f} to "
        f"{x_max:.6f} m\n"
    )
    f.write(
        f"Y: {y_min:.6f} to "
        f"{y_max:.6f} m\n"
    )
    f.write(
        f"Z: {z_min:.6f} to "
        f"{z_max:.6f} m\n\n"
    )

    f.write("Arm joint limits:\n")

    for i, joint_name in enumerate(
        ARM_JOINT_NAMES
    ):

        f.write(
            f"{joint_name}: "
            f"{joint_ranges[i, 0]:.6f} to "
            f"{joint_ranges[i, 1]:.6f} rad "
            f"("
            f"{np.degrees(joint_ranges[i, 0]):.2f} to "
            f"{np.degrees(joint_ranges[i, 1]):.2f}"
            f" deg)\n"
        )

print("Summary saved to:")
print(summary_file)


# ============================================================
# 3D WORKSPACE PLOT
# ============================================================

print()
print("Generating 3D workspace plot...")

fig = plt.figure(
    figsize=(10, 8)
)

ax = fig.add_subplot(
    111,
    projection="3d",
)

ax.scatter(
    x,
    y,
    z,
    s=1,
    alpha=0.25,
)

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")

ax.set_title(
    "Franka Panda Reachable Workspace\n"
    f"{NUM_SAMPLES:,} Random Joint Configurations"
)

plt.tight_layout()

plot_3d = (
    PLOT_DIR
    / "franka_workspace_3d.png"
)

plt.savefig(
    plot_3d,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print("3D workspace plot saved to:")
print(plot_3d)


# ============================================================
# XY PROJECTION
# ============================================================

print()
print("Generating XY workspace projection...")

plt.figure(figsize=(8, 7))

plt.scatter(
    x,
    y,
    s=1,
    alpha=0.25,
)

plt.xlabel("X (m)")
plt.ylabel("Y (m)")

plt.title(
    "Franka Panda Workspace - XY Projection"
)

plt.axis("equal")
plt.grid(True, alpha=0.3)

plt.tight_layout()

plot_xy = (
    PLOT_DIR
    / "franka_workspace_xy.png"
)

plt.savefig(
    plot_xy,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print("XY workspace plot saved to:")
print(plot_xy)


# ============================================================
# XZ PROJECTION
# ============================================================

print()
print("Generating XZ workspace projection...")

plt.figure(figsize=(8, 7))

plt.scatter(
    x,
    z,
    s=1,
    alpha=0.25,
)

plt.xlabel("X (m)")
plt.ylabel("Z (m)")

plt.title(
    "Franka Panda Workspace - XZ Projection"
)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plot_xz = (
    PLOT_DIR
    / "franka_workspace_xz.png"
)

plt.savefig(
    plot_xz,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print("XZ workspace plot saved to:")
print(plot_xz)


# ============================================================
# YZ PROJECTION
# ============================================================

print()
print("Generating YZ workspace projection...")

plt.figure(figsize=(8, 7))

plt.scatter(
    y,
    z,
    s=1,
    alpha=0.25,
)

plt.xlabel("Y (m)")
plt.ylabel("Z (m)")

plt.title(
    "Franka Panda Workspace - YZ Projection"
)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plot_yz = (
    PLOT_DIR
    / "franka_workspace_yz.png"
)

plt.savefig(
    plot_yz,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print("YZ workspace plot saved to:")
print(plot_yz)


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print("=" * 70)
print("FRANKA WORKSPACE ANALYSIS COMPLETE")
print("=" * 70)

print()
print("Final workspace:")
print(
    f"X = {x_min:.6f} to "
    f"{x_max:.6f} m"
)
print(
    f"Y = {y_min:.6f} to "
    f"{y_max:.6f} m"
)
print(
    f"Z = {z_min:.6f} to "
    f"{z_max:.6f} m"
)

print()
print("Workspace data:")
print(workspace_file)

print()
print("Workspace summary:")
print(summary_file)

print()
print("Workspace plots:")
print(PLOT_DIR)

print()
print("NEXT STEP:")
print("Franka dexterous workspace analysis.")