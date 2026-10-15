
"""
ME639 Robotics - Lab 3
Franka Panda Dexterous Workspace Analysis

Robot:
    Franka Emika Panda

Purpose:
    Determine the dexterous workspace of the Franka Panda arm
    using random joint-space sampling.

The hand body is used as the end-effector reference because the
Menagerie Franka scene contains no MuJoCo site.

Output:
    data/franka_dexterous_workspace/
        franka_dexterous_workspace.csv
        franka_dexterous_workspace_summary.txt

    plots/franka_dexterous_workspace/
        franka_dexterous_workspace_3d.png
        franka_dexterous_workspace_xy.png
        franka_dexterous_workspace_xz.png
        franka_dexterous_workspace_yz.png
"""

from pathlib import Path
import numpy as np
import mujoco
import matplotlib.pyplot as plt


# ==============================================================
# PATHS
# ==============================================================

LAB3_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    LAB3_DIR.parent
    / "mujoco_menagerie"
    / "franka_emika_panda"
    / "scene.xml"
)

DATA_DIR = LAB3_DIR / "data" / "franka_dexterous_workspace"
PLOT_DIR = LAB3_DIR / "plots" / "franka_dexterous_workspace"

DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================
# SETTINGS
# ==============================================================

NUM_SAMPLES = 20000
RNG_SEED = 42

# Spatial resolution of workspace cells.
# Smaller value = finer analysis but more cells.
CELL_SIZE = 0.05

# Number of different orientation bins required for a point
# to be considered dexterous.
MIN_ORIENTATION_BINS = 3


# ==============================================================
# FRANKA ARM
# ==============================================================

ARM_JOINT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
]

EE_BODY_NAME = "hand"


# ==============================================================
# HEADER
# ==============================================================

print("=" * 70)
print("FRANKA PANDA DEXTEROUS WORKSPACE ANALYSIS")
print("=" * 70)

print()
print("Scene:")
print(MODEL_PATH)


# ==============================================================
# LOAD MODEL
# ==============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Franka scene not found:\n{MODEL_PATH}"
    )

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

print()
print("Model loaded successfully.")
print(f"nq = {model.nq}")
print(f"nv = {model.nv}")
print(f"njnt = {model.njnt}")
print(f"nbody = {model.nbody}")
print(f"nsite = {model.nsite}")


# ==============================================================
# END EFFECTOR
# ==============================================================

hand_body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    EE_BODY_NAME
)

if hand_body_id < 0:
    raise RuntimeError(
        f"Body '{EE_BODY_NAME}' was not found in the Franka model."
    )

print()
print("End-effector reference:")
print(f"Body name = {EE_BODY_NAME}")
print(f"Body ID   = {hand_body_id}")


# ==============================================================
# JOINT INFORMATION
# ==============================================================

print()
print("=" * 70)
print("FRANKA ARM JOINTS")
print("=" * 70)

joint_info = []

for joint_name in ARM_JOINT_NAMES:

    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name
    )

    if joint_id < 0:
        raise RuntimeError(
            f"Joint '{joint_name}' was not found."
        )

    qpos_adr = model.jnt_qposadr[joint_id]

    lower = model.jnt_range[joint_id, 0]
    upper = model.jnt_range[joint_id, 1]

    joint_info.append(
        {
            "name": joint_name,
            "id": joint_id,
            "qpos": qpos_adr,
            "lower": lower,
            "upper": upper,
        }
    )

    print(
        f"{joint_name}: "
        f"joint_id={joint_id}, "
        f"qpos={qpos_adr}, "
        f"range=[{lower:.4f}, {upper:.4f}] rad"
    )


# ==============================================================
# RANDOM SAMPLING
# ==============================================================

print()
print("=" * 70)
print("SAMPLING FRANKA CONFIGURATIONS")
print("=" * 70)

print(f"Number of configurations = {NUM_SAMPLES}")
print(f"Random seed = {RNG_SEED}")

rng = np.random.default_rng(RNG_SEED)

joint_samples = np.zeros(
    (NUM_SAMPLES, len(joint_info))
)

for j, info in enumerate(joint_info):

    joint_samples[:, j] = rng.uniform(
        info["lower"],
        info["upper"],
        NUM_SAMPLES
    )

print("Joint configurations generated successfully.")


# ==============================================================
# STORAGE
# ==============================================================

positions = np.zeros(
    (NUM_SAMPLES, 3)
)

orientations = np.zeros(
    (NUM_SAMPLES, 3, 3)
)


# ==============================================================
# FORWARD KINEMATICS
# ==============================================================

print()
print("Computing forward kinematics...")

for i in range(NUM_SAMPLES):

    # Set only the seven Franka arm joints.
    for j, info in enumerate(joint_info):
        data.qpos[info["qpos"]] = joint_samples[i, j]

    # Forward kinematics.
    mujoco.mj_forward(model, data)

    # Hand position.
    positions[i] = data.xpos[hand_body_id]

    # Hand orientation matrix.
    orientations[i] = data.xmat[hand_body_id].reshape(3, 3)

    if (i + 1) % 2000 == 0:
        print(f"Processed {i + 1:6d} / {NUM_SAMPLES}")


# ==============================================================
# POSITION RANGE
# ==============================================================

x = positions[:, 0]
y = positions[:, 1]
z = positions[:, 2]

print()
print("=" * 70)
print("FRANKA POSITION RANGE")
print("=" * 70)

print()
print("X range:")
print(f"  Minimum = {x.min():.6f} m")
print(f"  Maximum = {x.max():.6f} m")

print()
print("Y range:")
print(f"  Minimum = {y.min():.6f} m")
print(f"  Maximum = {y.max():.6f} m")

print()
print("Z range:")
print(f"  Minimum = {z.min():.6f} m")
print(f"  Maximum = {z.max():.6f} m")


# ==============================================================
# CREATE POSITION CELLS
# ==============================================================

print()
print("=" * 70)
print("BUILDING WORKSPACE CELLS")
print("=" * 70)

# Convert each position into a spatial cell.
cell_indices = np.floor(
    positions / CELL_SIZE
).astype(int)


# ==============================================================
# ORIENTATION REPRESENTATION
# ==============================================================
#
# To identify different orientations without requiring exact
# floating-point matrix equality, each orientation is converted
# into a coarse orientation signature.
#
# The columns of the rotation matrix correspond to the hand
# coordinate axes expressed in the world frame.
#
# We use the direction of the local z-axis as the primary
# orientation descriptor. This provides a robust measure of
# orientation diversity while avoiding exact floating-point
# comparisons.
# ==============================================================

local_z = orientations[:, :, 2]


# Convert direction to spherical angular bins.
theta = np.arctan2(
    local_z[:, 1],
    local_z[:, 0]
)

phi = np.arccos(
    np.clip(local_z[:, 2], -1.0, 1.0)
)

# Orientation bin size.
ANGLE_BIN = np.deg2rad(30.0)

theta_bins = np.floor(
    theta / ANGLE_BIN
).astype(int)

phi_bins = np.floor(
    phi / ANGLE_BIN
).astype(int)


# ==============================================================
# COLLECT ORIENTATION DIVERSITY
# ==============================================================

print()
print("Calculating orientation diversity...")

cell_orientations = {}

for i in range(NUM_SAMPLES):

    cell = tuple(cell_indices[i])

    orientation_bin = (
        int(theta_bins[i]),
        int(phi_bins[i])
    )

    if cell not in cell_orientations:
        cell_orientations[cell] = set()

    cell_orientations[cell].add(
        orientation_bin
    )


# ==============================================================
# DEXTEROUS CELLS
# ==============================================================

dexterous_cells = []

for cell, orientation_set in cell_orientations.items():

    orientation_count = len(orientation_set)

    if orientation_count >= MIN_ORIENTATION_BINS:
        dexterous_cells.append(
            (
                cell,
                orientation_count
            )
        )


# ==============================================================
# CONVERT CELLS TO POSITIONS
# ==============================================================

dexterous_positions = []

dexterous_orientation_counts = []

for cell, orientation_count in dexterous_cells:

    center = (
        np.array(cell, dtype=float) + 0.5
    ) * CELL_SIZE

    dexterous_positions.append(center)

    dexterous_orientation_counts.append(
        orientation_count
    )

if len(dexterous_positions) > 0:

    dexterous_positions = np.array(
        dexterous_positions
    )

    dexterous_orientation_counts = np.array(
        dexterous_orientation_counts
    )

else:

    dexterous_positions = np.empty(
        (0, 3)
    )

    dexterous_orientation_counts = np.empty(
        (0,)
    )


# ==============================================================
# RESULTS
# ==============================================================

print()
print("=" * 70)
print("FRANKA DEXTEROUS WORKSPACE RESULTS")
print("=" * 70)

print()
print(
    f"Total occupied workspace cells = "
    f"{len(cell_orientations)}"
)

print(
    f"Dexterous workspace cells = "
    f"{len(dexterous_positions)}"
)

if len(cell_orientations) > 0:

    dexterous_percentage = (
        len(dexterous_positions)
        / len(cell_orientations)
        * 100.0
    )

else:

    dexterous_percentage = 0.0

print(
    f"Dexterous workspace percentage = "
    f"{dexterous_percentage:.4f}%"
)

print()
print(
    f"Minimum orientation bins required = "
    f"{MIN_ORIENTATION_BINS}"
)

if len(dexterous_orientation_counts) > 0:

    print()
    print("Orientation diversity:")
    print(
        f"  Minimum = "
        f"{dexterous_orientation_counts.min()}"
    )
    print(
        f"  Maximum = "
        f"{dexterous_orientation_counts.max()}"
    )
    print(
        f"  Mean = "
        f"{dexterous_orientation_counts.mean():.2f}"
    )


# ==============================================================
# SAVE CSV
# ==============================================================

csv_path = (
    DATA_DIR
    / "franka_dexterous_workspace.csv"
)

with open(csv_path, "w") as f:

    f.write(
        "x,y,z,orientation_bins\n"
    )

    for position, count in zip(
        dexterous_positions,
        dexterous_orientation_counts
    ):

        f.write(
            f"{position[0]:.6f},"
            f"{position[1]:.6f},"
            f"{position[2]:.6f},"
            f"{count}\n"
        )

print()
print("Dexterous workspace data saved to:")
print(csv_path)


# ==============================================================
# SAVE SUMMARY
# ==============================================================

summary_path = (
    DATA_DIR
    / "franka_dexterous_workspace_summary.txt"
)

with open(summary_path, "w") as f:

    f.write(
        "FRANKA PANDA DEXTEROUS WORKSPACE ANALYSIS\n"
    )
    f.write("=" * 60 + "\n\n")

    f.write(
        f"Number of sampled configurations: "
        f"{NUM_SAMPLES}\n"
    )

    f.write(
        f"Random seed: {RNG_SEED}\n"
    )

    f.write(
        f"Spatial cell size: "
        f"{CELL_SIZE:.4f} m\n"
    )

    f.write(
        f"Minimum orientation bins: "
        f"{MIN_ORIENTATION_BINS}\n\n"
    )

    f.write(
        "Workspace range:\n"
    )

    f.write(
        f"X = {x.min():.6f} to "
        f"{x.max():.6f} m\n"
    )

    f.write(
        f"Y = {y.min():.6f} to "
        f"{y.max():.6f} m\n"
    )

    f.write(
        f"Z = {z.min():.6f} to "
        f"{z.max():.6f} m\n\n"
    )

    f.write(
        f"Occupied workspace cells: "
        f"{len(cell_orientations)}\n"
    )

    f.write(
        f"Dexterous workspace cells: "
        f"{len(dexterous_positions)}\n"
    )

    f.write(
        f"Dexterous workspace percentage: "
        f"{dexterous_percentage:.4f}%\n"
    )

print("Summary saved to:")
print(summary_path)


# ==============================================================
# 3D PLOT
# ==============================================================

print()
print("Generating 3D dexterous workspace plot...")

fig = plt.figure(figsize=(9, 7))

ax = fig.add_subplot(
    111,
    projection="3d"
)

if len(dexterous_positions) > 0:

    scatter = ax.scatter(
        dexterous_positions[:, 0],
        dexterous_positions[:, 1],
        dexterous_positions[:, 2],
        c=dexterous_orientation_counts,
        cmap="viridis",
        s=8
    )

    fig.colorbar(
        scatter,
        ax=ax,
        label="Number of orientation bins"
    )

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")

ax.set_title(
    "Franka Panda Dexterous Workspace"
)

plt.tight_layout()

plot_path = (
    PLOT_DIR
    / "franka_dexterous_workspace_3d.png"
)

plt.savefig(
    plot_path,
    dpi=300
)

plt.close()

print("3D plot saved to:")
print(plot_path)


# ==============================================================
# XY PROJECTION
# ==============================================================

print("Generating XY dexterous workspace projection...")

plt.figure(figsize=(8, 7))

if len(dexterous_positions) > 0:

    plt.scatter(
        dexterous_positions[:, 0],
        dexterous_positions[:, 1],
        c=dexterous_orientation_counts,
        cmap="viridis",
        s=10
    )

    plt.colorbar(
        label="Number of orientation bins"
    )

plt.xlabel("X (m)")
plt.ylabel("Y (m)")

plt.title(
    "Franka Dexterous Workspace - XY Projection"
)

plt.axis("equal")
plt.grid(True)

plt.tight_layout()

plot_path = (
    PLOT_DIR
    / "franka_dexterous_workspace_xy.png"
)

plt.savefig(
    plot_path,
    dpi=300
)

plt.close()

print("XY plot saved to:")
print(plot_path)


# ==============================================================
# XZ PROJECTION
# ==============================================================

print("Generating XZ dexterous workspace projection...")

plt.figure(figsize=(8, 7))

if len(dexterous_positions) > 0:

    plt.scatter(
        dexterous_positions[:, 0],
        dexterous_positions[:, 2],
        c=dexterous_orientation_counts,
        cmap="viridis",
        s=10
    )

    plt.colorbar(
        label="Number of orientation bins"
    )

plt.xlabel("X (m)")
plt.ylabel("Z (m)")

plt.title(
    "Franka Dexterous Workspace - XZ Projection"
)

plt.grid(True)

plt.tight_layout()

plot_path = (
    PLOT_DIR
    / "franka_dexterous_workspace_xz.png"
)

plt.savefig(
    plot_path,
    dpi=300
)

plt.close()

print("XZ plot saved to:")
print(plot_path)


# ==============================================================
# YZ PROJECTION
# ==============================================================

print("Generating YZ dexterous workspace projection...")

plt.figure(figsize=(8, 7))

if len(dexterous_positions) > 0:

    plt.scatter(
        dexterous_positions[:, 1],
        dexterous_positions[:, 2],
        c=dexterous_orientation_counts,
        cmap="viridis",
        s=10
    )

    plt.colorbar(
        label="Number of orientation bins"
    )

plt.xlabel("Y (m)")
plt.ylabel("Z (m)")

plt.title(
    "Franka Dexterous Workspace - YZ Projection"
)

plt.grid(True)

plt.tight_layout()

plot_path = (
    PLOT_DIR
    / "franka_dexterous_workspace_yz.png"
)

plt.savefig(
    plot_path,
    dpi=300
)

plt.close()

print("YZ plot saved to:")
print(plot_path)


# ==============================================================
# COMPLETE
# ==============================================================

print()
print("=" * 70)
print("FRANKA DEXTEROUS WORKSPACE ANALYSIS COMPLETE")
print("=" * 70)

print()
print(
    f"Occupied cells = {len(cell_orientations)}"
)

print(
    f"Dexterous cells = {len(dexterous_positions)}"
)

print(
    f"Dexterous percentage = "
    f"{dexterous_percentage:.4f}%"
)

print()
print("Dexterous workspace data:")
print(csv_path)

print()
print("Dexterous workspace summary:")
print(summary_path)

print()
print("Dexterous workspace plots:")
print(PLOT_DIR)

