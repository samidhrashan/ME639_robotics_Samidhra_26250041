"""HEAL pick-and-place controller — QP-IK entry point.


HEAL 6-DOF + Robotiq 2F-85 pick-and-place in MuJoCo (Mink diff-IK + septic).

Design
------
* Mink differential IK (daqp) drives an INDEPENDENT kinematic reference
  (a mink.Configuration that is never fed from data.qpos).  data.ctrl tracks
  that reference, so gravity sag cannot feed back into the integrator.
* Gravity is cancelled with qfrc_bias on the arm DOFs; a small clamped
  integral term on ctrl removes the remaining servo droop.
* All motions use a 7th-order (septic) polynomial time-scaling.
* Orientation: tool approach axis (joint_6 axis) and jaw-closing axis
  (left pad -> right pad) are measured from the model.  The desired rotation
  has approach = world -Z and closing axis = a cube-face normal.  Every
  candidate cube-face direction is checked offline with full-pose IK over ALL
  waypoints (yaw weight always strict; tilt weight relaxed only if needed).
  The first move (home -> pre-grasp) is a joint-space septic interpolation to
  the pre-solved configuration, which avoids singularities en route.
* Gripper: ctrl ramp on fingers_actuator (+ optional inward driver torque via
  qfrc_applied).  Closing is confirmed from real contacts between
  left_pad_geom / right_pad_geom and cube_geom (table contacts are ignored),
  and the grasp is monitored during lift/transport.

Scene requirements (all included in the patched XML files delivered with it)
----------------------------------------------------------------------------
* pads open to a 40 mm gap and are centred on the `pinch` site (z offset 0);
* pad slide joints are driven by the driver joints through
  `pad = -0.025 * driver` (the original had joint1/joint2 swapped, which
  locked the drivers near zero), with stiff equality solref;
* `fingers_actuator` has biastype="affine" (without it the spring links close
  the gripper by themselves);
* pad contact: solref 0.004 1 with priority 1 (so the pad's friction wins);
* scene option: cone=elliptic impratio=10 noslip_iterations=5 (stops a 20 g
  cube creeping out of the pads);
* the arm base is lowered 8 cm: with the tool vertical at pinch Z = 0.1376 the
  HEAL arm cannot reach closer than ~0.48 m from its base, and the cube (0.39 m)
  and tray (0.42 m) are inside that dead zone.

Usage
-----
    python heal_mink_pick_place_base.py --xml heal_pnp_scene.xml --viewer
"""

import argparse
import json
import os
import sys
import time

import mujoco
import numpy as np

import mink

# ----------------------------------------------------------------------------
# Scene names / constants
# ----------------------------------------------------------------------------
SCENE_XML = "heal_pnp_scene.xml"

ARM_JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
ARM_ACTUATORS = ["act_joint_1", "act_joint_2", "act_joint_3",
                 "act_joint_4", "act_joint_5", "act_joint_6"]
TCP_SITE = "pinch"

GRIPPER_ACTUATOR = "fingers_actuator"
DRIVER_JOINTS = ["left_driver_joint", "right_driver_joint"]
LEFT_PAD_GEOM = "left_pad_geom"
RIGHT_PAD_GEOM = "right_pad_geom"
CUBE_BODY = "cube"
CUBE_GEOM = "cube_geom"

# Optional start pose (6 rad) if the XML has no keyframe.  None -> model default
# (all zeros: tool already points straight down in this model).
HOME_Q = None

# Optional manual override of the tool axes in the pinch-site frame.
APPROACH_AXIS_LOCAL = None
CLOSING_AXIS_LOCAL = None

# Heights (m)
CUBE_START = np.array([0.36, 0.15, 0.132])
Z_PRE = 0.25
Z_GRASP = 0.1325
Z_CLEAR = 0.28
Z_PLACE = 0.195

# Tray (placement evaluation).  Tray floor top = 0.130 m, inner half-width 0.07 m.
TRAY_NAMES = ["target_tray", "tray", "place_tray", "tray_body",
              "tray_center", "tray_site", "tray_target", "tray_geom"]
TRAY_HALF_SIZE_XY = 0.055
TRAY_CUBE_Z_RANGE = (0.130, 0.170)   # cube centre resting on the tray floor ~0.142
SETTLE_LIN_VEL = 0.02
SETTLE_ANG_VEL = 0.30

# Timing (s)
T_WARMUP = 0.8
T_JOINT_TO_PRE = 3.0
T_DESCEND = 2.5
T_LIFT = 2.0
T_TRANSPORT = 3.0
T_LOWER = 2.0
T_RETREAT = 2.0
T_SETTLE_SEG = 0.3
T_FINAL_SETTLE = 1.5

# IK
IK_SUBSTEPS = 2
IK_ITERS = 2
POS_TOL = 2e-3
YAW_TOL = 0.05
MAX_TILT = 0.35
TILT_WEIGHTS = [1.0, 0.3, 0.1, 0.03]
IK_SOLVER = "daqp"
POSTURE_GAIN = 0.05      # low: the posture task must only gently regularise the null space
MAX_ARM_VELOCITY = 2.5   # rad/s, applied by DLS clipping and QP constraints
DLS_DAMPING = 0.08       # damping factor lambda for the custom DLS solver

# Servo helping terms
KI = 2.0
I_MAX = 0.15

# Gripper.  fingers_actuator (biastype affine): ctrl ~ 100 brings the pads to the
# 24 mm cube width; above that the squeeze force rises ~6 N per ctrl unit, so the
# close is force-limited instead of using a fixed ctrl = 255.
GRIP_CTRL_FAST = 80.0     # reached quickly; pads still ~27 mm apart (> cube)
GRIP_CTRL_MAX = 255.0     # hard ceiling of the slow ramp
GRIP_FAST_TIME = 0.6      # s to reach GRIP_CTRL_FAST
GRIP_SLOW_RATE = 50.0     # ctrl units / s while searching for the target force
GRASP_FORCE = 6.0         # N per pad at which the ramp stops and holds
GRIP_TORQUE = 0.0         # optional extra inward N*m per driver joint (qfrc_applied)
GRIP_OPEN_TIME = 0.8
GRIP_TIMEOUT = 5.0
MIN_PAD_FORCE = 1.0       # N per pad to count as genuine contact
CONFIRM_TIME = 0.10
PRELOAD_HOLD = 0.4
LOST_TIME = 0.15
LIFT_MIN = 0.03


class PickPlaceError(RuntimeError):
    pass


def septic(tau):
    """s(0)=0, s(1)=1, with zero velocity/acceleration/jerk at both ends."""
    tau = float(np.clip(tau, 0.0, 1.0))
    return tau ** 4 * (35.0 - 84.0 * tau + 70.0 * tau ** 2 - 20.0 * tau ** 3)


def septic_interp(a, b, tau):
    return a + septic(tau) * (b - a)


class HealPickPlace:
    def __init__(self, xml_path, use_viewer=False, tray_override=None, ik_method="mink", cube_pose=None):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        m, d = self.model, self.data
        self.dt = m.opt.timestep
        self.ik_dt = IK_SUBSTEPS * self.dt
        self.ik_method = str(ik_method).lower()
        if self.ik_method not in {"mink", "dls", "qp"}:
            raise ValueError("ik_method must be one of: mink, dls, qp")
        self.solve_calls = 0
        self.solve_time_seconds = 0.0
        self.velocity_limit_saturations = 0
        self.joint_limit_saturations = 0
        self.joint_limit_violations = 0
        self._qp_solver = None
        self._qp_pattern = None
        self.min_pad_table_clearance_m = float("inf")
        self.table_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_geom")

        J, A, G = mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_ACTUATOR, mujoco.mjtObj.mjOBJ_GEOM
        self.arm_jids = [self._id(J, n) for n in ARM_JOINTS]
        self.arm_qadr = np.array([m.jnt_qposadr[j] for j in self.arm_jids])
        self.arm_dadr = np.array([m.jnt_dofadr[j] for j in self.arm_jids])
        self.arm_act = np.array([self._id(A, n) for n in ARM_ACTUATORS])
        self.grip_act = self._id(A, GRIPPER_ACTUATOR)
        self.drv_jids = [self._id(J, n) for n in DRIVER_JOINTS]
        self.drv_qadr = np.array([m.jnt_qposadr[j] for j in self.drv_jids])
        self.drv_dadr = np.array([m.jnt_dofadr[j] for j in self.drv_jids])
        self.site = self._id(mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
        self.left_pad = self._id(G, LEFT_PAD_GEOM)
        self.right_pad = self._id(G, RIGHT_PAD_GEOM)
        self.cube_body = self._id(mujoco.mjtObj.mjOBJ_BODY, CUBE_BODY)
        self.cube_geom = self._id(G, CUBE_GEOM)
        cube_jadr = m.body_jntadr[self.cube_body]
        if cube_jadr < 0:
            raise PickPlaceError("Cube body has no joint (expected a freejoint).")
        self.cube_dadr = m.jnt_dofadr[cube_jadr]
        self.cube_qadr = m.jnt_qposadr[cube_jadr]

        lo = np.full(6, -np.inf)
        hi = np.full(6, np.inf)
        for k, a in enumerate(self.arm_act):
            if m.actuator_ctrllimited[a]:
                lo[k], hi[k] = m.actuator_ctrlrange[a]
        self.ctrl_lo, self.ctrl_hi = lo, hi

        # velocity feedforward time constants T = (kv + joint damping) / kp, so the
        # position servo has no steady lag while the reference is moving
        self.ff_T = np.zeros(6)
        for k, a in enumerate(self.arm_act):
            kp = m.actuator_gainprm[a, 0]
            kv = -m.actuator_biasprm[a, 2]
            damp = m.dof_damping[self.arm_dadr[k]]
            self.ff_T[k] = (kv + damp) / kp if kp > 0 else 0.0
        self.qd_ref = np.zeros(6)
        self._q_prev = None

        if m.nkey > 0:
            mujoco.mj_resetDataKeyframe(m, d, 0)
        else:
            mujoco.mj_resetData(m, d)
        if HOME_Q is not None:
            d.qpos[self.arm_qadr] = np.asarray(HOME_Q, dtype=float)
        if cube_pose is not None:
            cx, cy, cyaw = map(float, cube_pose)
            d.qpos[self.cube_qadr:self.cube_qadr + 3] = [cx, cy, float(CUBE_START[2])]
            d.qpos[self.cube_qadr + 3:self.cube_qadr + 7] = [
                np.cos(0.5 * cyaw), 0.0, 0.0, np.sin(0.5 * cyaw)
            ]
        mujoco.mj_forward(m, d)

        self.q_ref = d.qpos[self.arm_qadr].copy()
        self.integ = np.zeros(6)
        self.grip_ctrl = 0.0
        self.grip_torque = 0.0
        d.ctrl[self.arm_act] = self.q_ref
        d.ctrl[self.grip_act] = 0.0

        self.yaw_idx = 2
        self.tilt_w = 1.0
        self.cfg = mink.Configuration(m, q=d.qpos.copy())
        self.limits = self._build_limits()
        self.frame, self.posture = self._make_tasks(1.0)
        self.posture.set_target_from_configuration(self.cfg)
        self.q_home_full = d.qpos.copy()

        self.viewer = None
        self.step_count = 0
        if use_viewer:
            from mujoco import viewer as mj_viewer
            self.viewer = mj_viewer.launch_passive(m, d)

        self.monitor_grasp = False
        self.lost_steps = 0

        self.tray_xy = (np.asarray(tray_override, dtype=float)
                        if tray_override is not None else self._find_tray_xy())
        print(f"[init] tray xy = {self.tray_xy}")
        self.cube0 = d.xpos[self.cube_body].copy()
        self.check_scene()

    # ------------------------------------------------------------------ utils
    def check_scene(self):
        """Warn loudly if the XML still has the problems the patched files fix."""
        m, d = self.model, self.data
        mujoco.mj_forward(m, d)
        R = d.site_xmat[self.site].reshape(3, 3)
        l, r = d.geom_xpos[self.left_pad], d.geom_xpos[self.right_pad]
        c = (r - l) / max(np.linalg.norm(r - l), 1e-9)
        half = 0.0
        for g in (self.left_pad, self.right_pad):
            Rg = d.geom_xmat[g].reshape(3, 3)
            half += float(np.abs(Rg.T @ c) @ m.geom_size[g])
        gap = float(abs(np.dot(r - l, c)) - half)
        off = float(np.dot((l + r) / 2.0 - d.site_xpos[self.site], R[:, 2]))
        print(f"[check] open jaw gap = {gap * 1e3:.1f} mm, pad centre is {off * 1e3:+.1f} mm "
              f"from the pinch site along the tool axis")
        cube_w = 2.0 * float(m.geom_size[self.cube_geom][0])
        if gap < cube_w + 0.006:
            print(f"[WARN] open jaw gap {gap * 1e3:.1f} mm is not wider than the "
                  f"{cube_w * 1e3:.0f} mm cube: the pads cannot straddle it. Use the patched XML.")
        if abs(off) > 0.02:
            print("[WARN] pads are far from the pinch site along the tool axis, so the pinch "
                  "Z used for grasping will not put the pads on the cube. Use the patched XML.")
        if m.actuator_biastype[self.grip_act] != mujoco.mjtBias.mjBIAS_AFFINE:
            print("[WARN] fingers_actuator has no biastype=\"affine\": its biasprm is ignored and "
                  "the gripper springs close on their own. Use the patched XML.")
        if m.opt.noslip_iterations < 1:
            print("[WARN] option noslip_iterations is 0: a light cube will creep out of the "
                  "pads. Use the patched scene (cone=elliptic impratio=10 noslip_iterations=5).")
    def _id(self, objtype, name):
        i = mujoco.mj_name2id(self.model, objtype, name)
        if i < 0:
            raise PickPlaceError(f"Name '{name}' ({objtype.name}) not found in the model.")
        return i

    def _find_tray_xy(self):
        m, d = self.model, self.data
        mujoco.mj_forward(m, d)
        for name in TRAY_NAMES:
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
            if i >= 0:
                return d.xpos[i][:2].copy()
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)
            if i >= 0:
                return d.site_xpos[i][:2].copy()
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
            if i >= 0:
                return d.geom_xpos[i][:2].copy()
        raise PickPlaceError("Tray not found by name. Pass --tray-x/--tray-y or edit TRAY_NAMES.")

    def _build_limits(self):
        """Joint limits + zero velocity for every non-arm joint so IK only moves
        the six arm joints (the cube free joint is held by the posture task)."""
        m = self.model
        limits = [mink.ConfigurationLimit(m)]
        arm_set = set(self.arm_jids)
        frozen = {}
        for j in range(m.njnt):
            if j in arm_set or m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            if not name:
                continue
            nv = 3 if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL else 1
            frozen[name] = np.zeros(nv)
        if frozen:
            limits.append(mink.VelocityLimit(m, frozen))
        return limits

    def _make_tasks(self, tilt_w):
        """FrameTask: yaw (about the approach axis) weight 1, tilt weight tilt_w."""
        w = np.full(3, float(tilt_w))
        w[self.yaw_idx] = 1.0
        try:
            frame = mink.FrameTask(frame_name=TCP_SITE, frame_type="site",
                                   position_cost=1.0, orientation_cost=w,
                                   lm_damping=1e-3)
        except Exception:
            frame = mink.FrameTask(frame_name=TCP_SITE, frame_type="site",
                                   position_cost=1.0,
                                   orientation_cost=float(max(tilt_w, 0.05)),
                                   lm_damping=1e-3)
        posture = mink.PostureTask(self.model, cost=1e-2, gain=POSTURE_GAIN)
        return frame, posture

    def _task_residual(self, cfg, tasks):
        """Stack Mink's cost-weighted task residuals for custom solvers.

        Mink expresses the local task step as J*dq = target, where target is
        -gain*error. Using compute_qp_residual preserves each task's weights.
        """
        Js, targets = [], []
        mu_total = 0.0
        for task in tasks:
            residual = task.compute_qp_residual(cfg)
            if residual is None:
                # Fallback for task classes without the least-squares helper.
                J = np.asarray(task.compute_jacobian(cfg), dtype=float)
                err = np.asarray(task.compute_error(cfg), dtype=float)
                cost = np.asarray(getattr(task, "cost", 1.0), dtype=float)
                gain = float(getattr(task, "gain", 1.0))
                if cost.ndim == 0:
                    cost = np.full(err.size, float(cost))
                Jw = cost[:, None] * J
                target = cost * (-gain * err)
                mu = 0.0
            else:
                Jw, target, mu = residual
                Jw = np.asarray(Jw, dtype=float)
                target = np.asarray(target, dtype=float).reshape(-1)
                mu = float(mu)
            Js.append(Jw[:, self.arm_dadr])
            targets.append(target)
            mu_total += mu
        if not Js:
            raise PickPlaceError("No IK task residuals were supplied.")
        J_stack = np.vstack(Js)
        target_stack = np.concatenate(targets)
        if not np.all(np.isfinite(J_stack)) or not np.all(np.isfinite(target_stack)):
            raise PickPlaceError("Non-finite task residual encountered by IK solver.")
        return J_stack, target_stack, mu_total

    def _arm_delta_bounds(self, cfg, dt):
        q_now = np.asarray(cfg.q[self.arm_qadr], dtype=float)
        qlo = np.full(6, -np.inf)
        qhi = np.full(6, np.inf)
        for k, jid in enumerate(self.arm_jids):
            if self.model.jnt_limited[jid]:
                qlo[k], qhi[k] = self.model.jnt_range[jid]
        self._q_current_for_bounds = q_now
        self._joint_step_lo = qlo - q_now
        self._joint_step_hi = qhi - q_now
        self._velocity_step = MAX_ARM_VELOCITY * float(dt)
        lower = np.maximum(-self._velocity_step, self._joint_step_lo)
        upper = np.minimum(self._velocity_step, self._joint_step_hi)
        if np.any(lower > upper + 1e-12):
            raise PickPlaceError("Joint-position and velocity bounds are inconsistent.")
        return lower, upper

    def _solve_custom(self, cfg, tasks, dt, method):
        J, target, mu = self._task_residual(cfg, tasks)
        lam = DLS_DAMPING
        H = J.T @ J + (lam * lam + mu + 1e-8) * np.eye(6)
        rhs = J.T @ target
        lower, upper = self._arm_delta_bounds(cfg, dt)

        if method == "dls":
            # Damped least squares followed by explicit joint/velocity clipping.
            try:
                delta_raw = np.linalg.solve(H, rhs)
            except np.linalg.LinAlgError:
                delta_raw = np.linalg.lstsq(H, rhs, rcond=None)[0]
            delta = np.clip(delta_raw, lower, upper)
        elif method == "qp":
            try:
                import osqp
                from scipy import sparse
            except ImportError as exc:
                raise RuntimeError(
                    "QP IK requires OSQP. Install it in the project venv with: python -m pip install osqp"
                ) from exc

            # Convex box-constrained QP: min 0.5*dq' H dq - rhs' dq.
            # Create the fixed sparsity pattern once so repeated IK calls are fast.
            if self._qp_solver is None:
                self._qp_pattern = sparse.triu(
                    sparse.csc_matrix(np.ones((6, 6))), format="csc"
                )
                A = sparse.eye(6, format="csc")
                solver = osqp.OSQP()
                try:
                    solver.setup(
                        P=self._qp_pattern,
                        q=np.zeros(6), A=A,
                        l=lower, u=upper,
                        verbose=False, eps_abs=1e-5, eps_rel=1e-5,
                        max_iter=4000,
                    )
                except TypeError:
                    solver.setup(
                        P=self._qp_pattern,
                        q=np.zeros(6), A=A,
                        l=lower, u=upper,
                        verbose=False, eps_abs=1e-5, eps_rel=1e-5,
                    )
                self._qp_solver = solver
            # OSQP expects upper-triangle P entries in CSC column order.
            Px = np.array([H[i, j] for j in range(6) for i in range(j + 1)], dtype=float)
            self._qp_solver.update(Px=Px, q=-rhs, l=lower, u=upper)
            res = self._qp_solver.solve()
            status = str(getattr(res.info, "status", "")).lower()
            if res.x is None or "solved" not in status:
                raise PickPlaceError(f"QP-IK failed: {getattr(res.info, 'status', status)}")
            delta = np.clip(np.asarray(res.x, dtype=float), lower, upper)
        else:
            raise ValueError(f"Unsupported custom IK method: {method}")

        # Log both true violations and active constraints. DLS's raw result can
        # violate limits before clipping; QP returns a constrained solution.
        tol = 1e-6
        step_vel = max(self._velocity_step, 1e-12)
        if method == "dls":
            self.joint_limit_violations += int(np.count_nonzero(
                (delta_raw < self._joint_step_lo - tol) | (delta_raw > self._joint_step_hi + tol)
            ))
        self.velocity_limit_saturations += int(np.count_nonzero(np.abs(delta) >= step_vel - 1e-5))
        self.joint_limit_saturations += int(np.count_nonzero(
            (delta <= self._joint_step_lo + 1e-5) | (delta >= self._joint_step_hi - 1e-5)
        ))

        velocity = np.zeros(self.model.nv, dtype=float)
        velocity[self.arm_dadr] = delta / max(float(dt), 1e-9)
        return velocity

    def _solve(self, cfg, tasks, dt):
        t0 = time.perf_counter()
        try:
            if self.ik_method == "mink":
                try:
                    result = mink.solve_ik(cfg, tasks, dt, IK_SOLVER, damping=1e-3,
                                           limits=self.limits, safety_break=False)
                except TypeError:
                    result = mink.solve_ik(cfg, tasks, dt, IK_SOLVER, damping=1e-3,
                                           limits=self.limits)
            else:
                result = self._solve_custom(cfg, tasks, dt, self.ik_method)
            return result
        finally:
            self.solve_calls += 1
            self.solve_time_seconds += time.perf_counter() - t0

    # --------------------------------------------------------------- tool axes
    def measure_tool_axes(self):
        """Return E (3x3, columns = approach, closing, third) in site frame."""
        m, d = self.model, self.data
        mujoco.mj_forward(m, d)
        R_site = d.site_xmat[self.site].reshape(3, 3)

        if APPROACH_AXIS_LOCAL is not None and CLOSING_AXIS_LOCAL is not None:
            a = np.asarray(APPROACH_AXIS_LOCAL, float)
            c = np.asarray(CLOSING_AXIS_LOCAL, float)
        else:
            j6 = self.arm_jids[-1]
            axis_w = d.xaxis[j6].copy()
            anchor = d.xanchor[j6].copy()
            tcp = d.site_xpos[self.site].copy()
            if np.dot(tcp - anchor, axis_w) < 0.0:
                axis_w = -axis_w
            axis_w /= np.linalg.norm(axis_w)
            close_w = d.geom_xpos[self.right_pad] - d.geom_xpos[self.left_pad]
            close_w = close_w - np.dot(close_w, axis_w) * axis_w
            if np.linalg.norm(close_w) < 1e-6:
                raise PickPlaceError("Could not measure the jaw closing axis.")
            close_w /= np.linalg.norm(close_w)
            a = R_site.T @ axis_w
            c = R_site.T @ close_w
        a = a / np.linalg.norm(a)
        c = c - np.dot(c, a) * a
        c = c / np.linalg.norm(c)
        b = np.cross(a, c)
        self.yaw_idx = int(np.argmax(np.abs(a)))
        print(f"[axes] approach(local)={np.round(a, 3)} closing(local)={np.round(c, 3)} "
              f"yaw index={self.yaw_idx}")
        return np.stack([a, c, b], axis=1)

    @staticmethod
    def rotation_for(E, closing_dir_w):
        """Rotation (site->world): approach = world -Z, closing = closing_dir_w."""
        wa = np.array([0.0, 0.0, -1.0])
        wc = closing_dir_w / np.linalg.norm(closing_dir_w)
        wb = np.cross(wa, wc)
        return np.stack([wa, wc, wb], axis=1) @ E.T

    # -------------------------------------------------------------- offline IK
    def solve_pose(self, q_start, pos, quat, tilt_w, max_iter=500):
        cfg = mink.Configuration(self.model, q=q_start.copy())
        frame, posture = self._make_tasks(tilt_w)
        posture.set_target_from_configuration(cfg)
        frame.set_target(mink.SE3.from_rotation_and_translation(
            mink.SO3(wxyz=quat), np.asarray(pos, float)))
        for it in range(max_iter):
            v = self._solve(cfg, [frame, posture], 0.02)
            cfg.integrate_inplace(v, 0.02)
            if it % 5 == 4:
                err = frame.compute_error(cfg)
                if np.linalg.norm(err[:3]) < 1e-4 and np.linalg.norm(err[3:]) < 1e-3:
                    break
        err = frame.compute_error(cfg)
        ori = err[3:]
        return (cfg.q.copy(), float(np.linalg.norm(err[:3])),
                float(abs(ori[self.yaw_idx])),
                float(np.linalg.norm(np.delete(ori, self.yaw_idx))))

    def plan(self):
        """Pick grasp orientation (+ tilt weight) by IK over every waypoint."""
        E = self.measure_tool_axes()
        cx, cy = self.cube0[:2]
        tx, ty = self.tray_xy
        waypoints = [
            ("pre", (cx, cy, Z_PRE)),
            ("grasp", (cx, cy, Z_GRASP)),
            ("lift", (cx, cy, Z_CLEAR)),
            ("carry", (tx, ty, Z_CLEAR)),
            ("place", (tx, ty, Z_PLACE)),
        ]
        Rc = self.data.xmat[self.cube_body].reshape(3, 3)
        candidates = []
        for col in (0, 1):
            h = Rc[:, col].copy()
            h[2] = 0.0
            n = np.linalg.norm(h)
            if n < 1e-6:
                continue
            h /= n
            candidates.extend([h, -h])

        best = None
        print(f"[plan] waypoints: {[w[0] for w in waypoints]}")
        for tw in TILT_WEIGHTS:
            print(f"[plan] tilt weight {tw}:")
            for k, h in enumerate(candidates):
                quat = mink.SO3.from_matrix(self.rotation_for(E, h)).wxyz
                q = self.q_home_full.copy()
                path, sols = 0.0, {}
                pos_errs, worst_yaw, worst_tilt = [], 0.0, 0.0
                for name, pos in waypoints:
                    trials = [self.solve_pose(q, pos, quat, tw)]
                    if trials[0][1] >= POS_TOL:
                        trials.append(self.solve_pose(self.q_home_full, pos, quat, tw))
                    q_new, pe, ye, te = min(trials, key=lambda r: r[1] + 0.1 * (r[2] + r[3]))
                    path += float(np.linalg.norm(q_new[self.arm_qadr] - q[self.arm_qadr]))
                    pos_errs.append(pe)
                    worst_yaw, worst_tilt = max(worst_yaw, ye), max(worst_tilt, te)
                    sols[name] = q_new
                    q = q_new
                feasible = (max(pos_errs) < POS_TOL and worst_yaw < YAW_TOL
                            and worst_tilt < MAX_TILT)
                print(f"  cand {k}: dir={np.round(h, 2)} path={path:.2f} rad | "
                      f"pos err mm={np.round(np.array(pos_errs) * 1e3, 1).tolist()} | "
                      f"yaw={worst_yaw:.3f} tilt={np.degrees(worst_tilt):.1f} deg | "
                      f"feasible={feasible}")
                if feasible and (best is None or path < best[0]):
                    best = (path, quat, sols, h, tw, worst_tilt)
            if best is not None:
                break

        if best is None:
            raise PickPlaceError(
                "No grasp pose is reachable at all waypoints. See the per-waypoint position "
                "errors above: the waypoint with the large error is out of the arm's reach "
                "(lower the arm base / raise the table, or move the cube/tray closer to "
                "the radius range the arm can reach).")
        _, self.quat, sols, h, self.tilt_w, tilt = best
        self.q_pre = sols["pre"]
        self.frame, _ = self._make_tasks(self.tilt_w)
        print(f"[plan] chosen closing dir {np.round(h, 3)}, tilt weight {self.tilt_w}, "
              f"max tilt {np.degrees(tilt):.1f} deg")

    # ------------------------------------------------------------ contact info
    def pad_forces(self):
        m, d = self.model, self.data
        f = {self.left_pad: 0.0, self.right_pad: 0.0}
        buf = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if self.cube_geom not in (g1, g2):
                continue
            other = g2 if g1 == self.cube_geom else g1
            if other in f:
                mujoco.mj_contactForce(m, d, i, buf)
                f[other] += abs(buf[0])
        return f[self.left_pad], f[self.right_pad]

    # -------------------------------------------------------------- simulation
    def _step(self):
        m, d = self.model, self.data
        d.qfrc_applied[:] = 0.0
        d.qfrc_applied[self.arm_dadr] = d.qfrc_bias[self.arm_dadr]
        d.qfrc_applied[self.drv_dadr] += self.grip_torque

        if self._q_prev is None:
            self._q_prev = self.q_ref.copy()
        raw = (self.q_ref - self._q_prev) / self.dt
        self._q_prev = self.q_ref.copy()
        self.qd_ref = 0.97 * self.qd_ref + 0.03 * raw      # smoothed reference velocity

        err = self.q_ref - d.qpos[self.arm_qadr]
        self.integ = np.clip(self.integ + KI * err * self.dt, -I_MAX, I_MAX)
        ctrl = self.q_ref + self.ff_T * self.qd_ref + self.integ
        d.ctrl[self.arm_act] = np.clip(ctrl, self.ctrl_lo, self.ctrl_hi)
        d.ctrl[self.grip_act] = self.grip_ctrl

        mujoco.mj_step(m, d)
        self.step_count += 1
        self._update_table_clearance()

        if self.monitor_grasp and self.step_count % 5 == 0:
            fl, fr = self.pad_forces()
            if min(fl, fr) < 0.05:
                self.lost_steps += 5
            else:
                self.lost_steps = 0
            if self.lost_steps * self.dt > LOST_TIME:
                raise PickPlaceError(
                    f"Cube lost pad contact during transport (fl={fl:.2f}, fr={fr:.2f} N).")

        if self.viewer is not None and self.step_count % 8 == 0:
            if not self.viewer.is_running():
                raise PickPlaceError("Viewer closed.")
            self.viewer.sync()
            time.sleep(self.dt * 8)

    def _update_table_clearance(self):
        """Track the minimum signed Z-clearance of the physical pads over the run."""
        if self.table_geom < 0:
            return
        m, d = self.model, self.data
        table_top = float(d.geom_xpos[self.table_geom, 2] + m.geom_size[self.table_geom, 2])
        lowest_pad = float("inf")
        for gid in (self.left_pad, self.right_pad):
            R = d.geom_xmat[gid].reshape(3, 3)
            half_z = float(np.abs(R[2, :]) @ m.geom_size[gid])
            lowest_pad = min(lowest_pad, float(d.geom_xpos[gid, 2] - half_z))
        self.min_pad_table_clearance_m = min(
            self.min_pad_table_clearance_m, lowest_pad - table_top
        )

    def hold(self, duration):
        for _ in range(int(round(duration / self.dt))):
            self._step()

    # -------------------------------------------------------------- arm motion
    def _tcp_ref(self):
        return self.cfg.get_transform_frame_to_world(TCP_SITE, "site").translation().copy()

    def move_joint(self, q_goal_full, duration):
        q0 = self.cfg.q[self.arm_qadr].copy()
        q1 = q_goal_full[self.arm_qadr].copy()
        n = int(round(duration / self.dt))
        for i in range(1, n + 1):
            q_arm = septic_interp(q0, q1, i / n)
            qf = self.cfg.q.copy()
            qf[self.arm_qadr] = q_arm
            self.cfg.update(qf)
            self.q_ref = q_arm
            self._step()
        self.posture.set_target_from_configuration(self.cfg)
        self.hold(T_SETTLE_SEG)

    def move_cartesian(self, pos_goal, duration, label=""):
        p0 = self._tcp_ref()
        p1 = np.asarray(pos_goal, float)
        self.posture.set_target_from_configuration(self.cfg)   # no pull back to an old posture
        n_ik = max(1, int(round(duration / self.ik_dt)))
        for i in range(1, n_ik + 1):
            pos = septic_interp(p0, p1, i / n_ik)
            self.frame.set_target(mink.SE3.from_rotation_and_translation(
                mink.SO3(wxyz=self.quat), pos))
            for _ in range(IK_ITERS):
                v = self._solve(self.cfg, [self.frame, self.posture], self.ik_dt)
                self.cfg.integrate_inplace(v, self.ik_dt)
            self.q_ref = self.cfg.q[self.arm_qadr].copy()
            for _ in range(IK_SUBSTEPS):
                self._step()
        self.hold(T_SETTLE_SEG)

        ref_err = np.linalg.norm(self._tcp_ref() - p1)
        tcp = self.data.site_xpos[self.site]
        print(f"[{label}] ref err {ref_err * 1e3:.2f} mm | real TCP {np.round(tcp, 4)} "
              f"err {np.linalg.norm(tcp - p1) * 1e3:.2f} mm")
        if ref_err > 5e-3:
            raise PickPlaceError(f"IK stalled during '{label}' ({ref_err * 1e3:.1f} mm short).")

    # ----------------------------------------------------------------- gripper
    def print_pad_clearance(self):
        """Report pad/cube positions and physical clearance over the table."""
        m, d = self.model, self.data
        table = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_geom")
        if table < 0:
            print("[clearance] table_geom not found; cannot measure table clearance")
            return

        def vertical_bounds(gid):
            R = d.geom_xmat[gid].reshape(3, 3)
            hz = float(np.abs(R[2, :]) @ m.geom_size[gid])
            zc = float(d.geom_xpos[gid, 2])
            return zc - hz, zc + hz

        _, table_top = vertical_bounds(table)
        left_low, left_high = vertical_bounds(self.left_pad)
        right_low, right_high = vertical_bounds(self.right_pad)
        pad_low = min(left_low, right_low)
        cube = d.xpos[self.cube_body].copy()
        left_pos = d.geom_xpos[self.left_pad].copy()
        right_pos = d.geom_xpos[self.right_pad].copy()
        print(f"[geometry] cube centre={np.round(cube, 4)}")
        print(f"[geometry] left pad centre={np.round(left_pos, 4)}")
        print(f"[geometry] right pad centre={np.round(right_pos, 4)}")
        print(f"[clearance] table top z={table_top:.4f} m; lowest pad z={pad_low:.4f} m; "
              f"pad-to-table clearance={(pad_low-table_top)*1000:.1f} mm")

    def close_gripper(self):
        n_fast = int(round(GRIP_FAST_TIME / self.dt))
        n_max = int(round(GRIP_TIMEOUT / self.dt))
        need = int(round(CONFIRM_TIME / self.dt))
        consec = 0
        confirmed = False
        ctrl = 0.0
        for i in range(n_max):
            if i < n_fast:
                ctrl = GRIP_CTRL_FAST * septic((i + 1) / n_fast)
            else:
                fl, fr = self.pad_forces()
                if min(fl, fr) < GRASP_FORCE:      # keep squeezing slowly
                    ctrl = min(GRIP_CTRL_MAX, ctrl + GRIP_SLOW_RATE * self.dt)
            self.grip_ctrl = ctrl
            self.grip_torque = GRIP_TORQUE * min(1.0, ctrl / GRIP_CTRL_FAST)
            self._step()
            fl, fr = self.pad_forces()
            consec = consec + 1 if (fl >= MIN_PAD_FORCE and fr >= MIN_PAD_FORCE) else 0
            if consec >= need and min(fl, fr) >= GRASP_FORCE:
                confirmed = True
                break
        fl, fr = self.pad_forces()
        print(f"[close] ctrl={ctrl:.1f} pad forces L={fl:.2f} N R={fr:.2f} N, "
              f"drivers={np.round(self.data.qpos[self.drv_qadr], 3)}")
        if not confirmed:
            raise PickPlaceError("Gripper closed without genuine two-pad contact on the cube.")
        self.hold(PRELOAD_HOLD)
        fl, fr = self.pad_forces()
        print(f"[close] preload forces L={fl:.2f} N R={fr:.2f} N")

    def open_gripper(self):
        c0, t0 = self.grip_ctrl, self.grip_torque
        n = int(round(GRIP_OPEN_TIME / self.dt))
        for i in range(1, n + 1):
            a = septic(i / n)
            self.grip_ctrl = c0 * (1.0 - a)
            self.grip_torque = t0 * (1.0 - a)
            self._step()
        self.grip_ctrl = 0.0
        self.grip_torque = 0.0
        self.hold(0.3)
        fl, fr = self.pad_forces()
        print(f"[open] residual pad forces L={fl:.2f} N R={fr:.2f} N")

    # -------------------------------------------------------------- evaluation
    def evaluate(self):
        d = self.data
        cube = d.xpos[self.cube_body].copy()
        v = d.qvel[self.cube_dadr:self.cube_dadr + 6]
        lin, ang = np.linalg.norm(v[:3]), np.linalg.norm(v[3:])
        inside_xy = bool(np.all(np.abs(cube[:2] - self.tray_xy) <= TRAY_HALF_SIZE_XY))
        settled = bool(lin < SETTLE_LIN_VEL and ang < SETTLE_ANG_VEL)
        height_ok = bool(TRAY_CUBE_Z_RANGE[0] <= cube[2] <= TRAY_CUBE_Z_RANGE[1])
        return {
            "cube_pos": np.round(cube, 4),
            "inside_xy": inside_xy,
            "settled": settled,
            "tray_height_plausible": height_ok,
            "success": inside_xy and settled and height_ok,
        }

    # ------------------------------------------------------------ state machine
    def run(self):
        tx, ty = self.tray_xy

        print("[0] warm-up (cube settles on the table) / planning")
        self.hold(T_WARMUP)
        self.cube0 = self.data.xpos[self.cube_body].copy()
        print(f"[init] cube after settling = {np.round(self.cube0, 4)} (task: {CUBE_START})")
        cx, cy = self.cube0[:2]
        self.plan()

        print("[1] move to pre-grasp")
        self.move_joint(self.q_pre, T_JOINT_TO_PRE)
        self.move_cartesian([cx, cy, Z_PRE], 0.5, "pre-grasp refine")

        print("[2] descend to grasp")
        self.move_cartesian([cx, cy, Z_GRASP], T_DESCEND, "descend")
        self.print_pad_clearance()

        print("[3] close gripper")
        self.close_gripper()
        self.monitor_grasp = True
        self.lost_steps = 0

        print("[4] lift")
        z_before = self.data.xpos[self.cube_body][2]
        self.move_cartesian([cx, cy, Z_CLEAR], T_LIFT, "lift")
        lift_displacement = self.data.xpos[self.cube_body][2] - z_before
        print(f"[lift] lift_displacement = {lift_displacement * 1e3:.1f} mm")
        if lift_displacement <= LIFT_MIN:
            raise PickPlaceError(
                f"Lift displacement {lift_displacement:.4f} m <= {LIFT_MIN} m: cube not lifted.")

        print("[5] transport")
        self.move_cartesian([tx, ty, Z_CLEAR], T_TRANSPORT, "transport")

        print("[6] lower into tray")
        self.move_cartesian([tx, ty, Z_PLACE], T_LOWER, "lower")

        print("[7] release")
        self.monitor_grasp = False
        self.open_gripper()

        print("[8] retreat")
        self.move_cartesian([tx, ty, Z_CLEAR], T_RETREAT, "retreat")

        print("[9] settle + evaluate")
        self.hold(T_FINAL_SETTLE)
        result = self.evaluate()
        result["lift_displacement"] = round(float(lift_displacement), 4)
        return result


def find_xml(requested):
    if os.path.isfile(requested):
        return requested
    target = os.path.basename(requested)
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    for base in (here, root):
        for dirpath, _, files in os.walk(base):
            if target in files:
                return os.path.join(dirpath, target)
    return None


def main():
    ap = argparse.ArgumentParser(description="HEAL pick-and-place using constrained Quadratic Programming inverse kinematics (OSQP)")
    ap.add_argument("--xml", default=SCENE_XML)
    ap.add_argument("--ik-method", choices=["qp"], default="qp", help="fixed method for this script")
    vg = ap.add_mutually_exclusive_group()
    vg.add_argument("--viewer", dest="viewer", action="store_true")
    vg.add_argument("--no-viewer", dest="viewer", action="store_false")
    ap.set_defaults(viewer=True)
    ap.add_argument("--tray-x", type=float, default=None)
    ap.add_argument("--tray-y", type=float, default=None)
    ap.add_argument("--cube-x", type=float, default=None)
    ap.add_argument("--cube-y", type=float, default=None)
    ap.add_argument("--cube-yaw", type=float, default=0.0, help="initial cube yaw in radians")
    ap.add_argument("--result-json", default=None)
    args = ap.parse_args()

    xml_path = find_xml(args.xml)
    if xml_path is None:
        print(f"[FAILED] Could not find '{args.xml}'. Pass the full path with --xml.")
        return 1
    xml_path = os.path.realpath(xml_path)
    print(f"[init] using scene: {xml_path}")
    print(f"[init] IK method: {args.ik_method}")

    tray = None
    if args.tray_x is not None and args.tray_y is not None:
        tray = [args.tray_x, args.tray_y]
    cube_pose = None
    if args.cube_x is not None or args.cube_y is not None or abs(args.cube_yaw) > 0:
        cube_pose = (
            CUBE_START[0] if args.cube_x is None else args.cube_x,
            CUBE_START[1] if args.cube_y is None else args.cube_y,
            args.cube_yaw,
        )

    task = None
    result = None
    failure = None
    started = time.perf_counter()
    try:
        task = HealPickPlace(
            xml_path, use_viewer=args.viewer, tray_override=tray,
            ik_method=args.ik_method, cube_pose=cube_pose,
        )
        result = task.run()
    except PickPlaceError as e:
        failure = e
        print(f"\n[FAILED] {e}")
    except Exception as e:
        failure = e
        print(f"\n[FAILED] Unexpected error: {type(e).__name__}: {e}")

    elapsed = time.perf_counter() - started
    payload = {
        "ik_method": args.ik_method,
        "success": bool(result is not None and result.get("success", False)),
        "failure_reason": None if failure is None else str(failure),
        "episode_wall_time_s": float(elapsed),
        "ik_solver_calls": 0 if task is None else int(task.solve_calls),
        "ik_solver_time_s": 0.0 if task is None else float(task.solve_time_seconds),
        "mean_solver_time_ms": None if task is None or task.solve_calls == 0 else float(1000 * task.solve_time_seconds / task.solve_calls),
        "velocity_limit_saturations": 0 if task is None else int(task.velocity_limit_saturations),
        "joint_limit_saturations": 0 if task is None else int(task.joint_limit_saturations),
        "joint_limit_violations": 0 if task is None else int(task.joint_limit_violations),
        "min_pad_table_clearance_m": None if task is None or not np.isfinite(task.min_pad_table_clearance_m) else float(task.min_pad_table_clearance_m),
    }
    if result is not None:
        payload.update(result)
    elif task is not None:
        payload["cube_pos_at_failure"] = np.asarray(task.data.xpos[task.cube_body]).tolist()

    if args.result_json:
        out_path = os.path.abspath(args.result_json)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[metrics] result JSON written to: {out_path}")

    # Keep the viewer available after both success and failure.
    if task is not None and task.viewer is not None:
        print("\n[viewer] Simulation stopped; viewer will remain open. Close its window or press Ctrl+C here to exit.")
        try:
            while task.viewer.is_running():
                task.viewer.sync()
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass

    print("\n=== RESULT ===")
    for k, v in payload.items():
        print(f"{k:>28}: {v}")
    if failure is not None or result is None:
        return 1
    return 0 if result["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
