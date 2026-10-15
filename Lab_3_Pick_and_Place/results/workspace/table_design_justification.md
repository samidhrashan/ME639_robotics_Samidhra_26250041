# Table design and workspace summary

Scene: `C:\ME639_robotics_Samidhra_26250041\Lab_3_Pick_and_Place\robot_descriptions\heal_pnp_scene.xml`

## Table dimensions extracted from MuJoCo XML

- X dimension: 0.340 m
- Y dimension: 0.720 m
- Height: 0.120 m
- Top surface: z = 0.120 m
- X bounds: 0.190 to 0.530 m
- Y bounds: -0.360 to 0.360 m

## Sampled workspace (empirical, not a formal reachability proof)

- Random configurations sampled: 12000
- Downward-approach samples (tilt ≤ 15°): 144 (1.2%)
- Approximate dexterous voxels with at least 3 observed yaw bins: 0

## Table sizing justification

The table footprint should be kept within the empirically sampled TCP XY workspace, with both the cube pick area and tray approach confirmed as feasible by inverse kinematics. The table top height must permit the approach pose and gripper fingertips to remain above the surface during descent. The sampled workspace plots provide the empirical boundary; final bounds should be selected from dense samples and confirmed by running IK at each proposed cube/tray corner.

Do not interpret random joint-sampled points as guaranteed reachable Cartesian targets; use IK feasibility tests on the final task region before expanding episode randomization.
