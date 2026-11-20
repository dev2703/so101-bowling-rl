"""4-active-DOF damped-least-squares IK and parameterized scripted push controller."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from so101_bowling.assets import SceneEntityIds
from so101_bowling.config import ACTIVE_JOINTS, ALL_JOINTS, WorkspaceConfig


@dataclass(frozen=True)
class PushParams:
    """Parameters controlling the 4-DOF scripted tabletop bowling push."""

    aim_yaw_offset_rad: float = 0.0
    push_speed_mps: float = 0.85
    follow_through_m: float = 0.14
    stage_offset_m: float = 0.085
    push_height_z: float = 0.052


def solve_4dof_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ids: SceneEntityIds,
    target_pos: np.ndarray,
    q_min: np.ndarray,
    q_max: np.ndarray,
    q_init: np.ndarray | None = None,
    prefer_fwd_vec: np.ndarray | None = None,
    pitch_weight: float = 0.025,
    damping: float = 1.5e-3,
    step_gain: float = 0.6,
    max_iters: int = 70,
    tol: float = 2e-4,
) -> np.ndarray:
    """Solve 3D gripperframe position IK over the 4 active joints using DLS.

    Held joints (wrist_roll, gripper) remain fixed at their values in ``data``.
    ``prefer_fwd_vec`` optionally specifies ``[ux, uy, target_dot]`` to softly
    regularize the gripper's forward axis projection onto horizontal direction ``(ux, uy)``.
    """
    d_ik = mujoco.MjData(model)
    d_ik.qpos[:] = data.qpos[:]
    q = (
        np.asarray(q_init, dtype=np.float64).copy()
        if q_init is not None
        else d_ik.qpos[ids.active_qpos_addrs].copy()
    )
    q = np.clip(q, q_min, q_max)
    target = np.asarray(target_pos, dtype=np.float64)

    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)

    for _ in range(max_iters):
        d_ik.qpos[ids.active_qpos_addrs] = q
        mujoco.mj_kinematics(model, d_ik)
        mujoco.mj_comPos(model, d_ik)

        err_p = target - d_ik.site_xpos[ids.gripper_site_id]
        if np.linalg.norm(err_p) < tol and prefer_fwd_vec is None:
            break

        mujoco.mj_jacSite(model, d_ik, jacp, jacr, ids.gripper_site_id)
        jp = jacp[:, ids.active_qvel_addrs]
        jr = jacr[:, ids.active_qvel_addrs]

        if prefer_fwd_vec is not None:
            fwd = d_ik.site_xmat[ids.gripper_site_id].reshape(3, 3)[:, 0]
            ux, uy, tgt_proj = float(prefer_fwd_vec[0]), float(prefer_fwd_vec[1]), float(prefer_fwd_vec[2])
            cur_proj = ux * float(fwd[0]) + uy * float(fwd[1])
            j0 = jr[1:2] * fwd[2] - jr[2:3] * fwd[1]
            j1 = jr[2:3] * fwd[0] - jr[0:1] * fwd[2]
            j_pitch = (ux * j0 + uy * j1) * pitch_weight
            e_pitch = np.array([(tgt_proj - cur_proj) * pitch_weight], dtype=np.float64)
            j_mat = np.vstack([jp, j_pitch])
            err = np.concatenate([err_p, e_pitch])
        else:
            j_mat = jp
            err = err_p

        reg = damping * np.eye(j_mat.shape[0], dtype=np.float64)
        dq = j_mat.T @ np.linalg.solve(j_mat @ j_mat.T + reg, err)
        q = np.clip(q + step_gain * dq, q_min, q_max)

    return q


class ScriptedPushController:
    """Parameterized 4-active-DOF closed-loop tabletop push controller at 20 Hz."""

    def __init__(
        self,
        model: mujoco.MjModel,
        ids: SceneEntityIds,
        cfg: WorkspaceConfig,
        q_min: np.ndarray,
        q_max: np.ndarray,
        params: PushParams | None = None,
    ) -> None:
        self.model = model
        self.ids = ids
        self.cfg = cfg
        self.q_min = np.asarray(q_min, dtype=np.float64)
        self.q_max = np.asarray(q_max, dtype=np.float64)
        self.params = params if params is not None else PushParams()

        self._ik_data = mujoco.MjData(model)
        home_all = np.asarray([cfg.robot.home_qpos[name] for name in ALL_JOINTS], dtype=np.float64)
        self._ik_data.qpos[model.jnt_qposadr[ids.all_joint_ids]] = home_all
        mujoco.mj_forward(model, self._ik_data)

        self.phase = "lift"
        self._push_steps = 0
        self._hold_steps = 0
        self.q_lift = np.zeros(len(ACTIVE_JOINTS), dtype=np.float64)
        self.q_stage = np.zeros(len(ACTIVE_JOINTS), dtype=np.float64)
        self.q_push_end = np.zeros(len(ACTIVE_JOINTS), dtype=np.float64)
        self.u_aim = np.array([1.0, 0.0], dtype=np.float64)

    def reset(self, ball_pos: np.ndarray, pin_positions: np.ndarray) -> None:
        """Plan lift, staging, and push-through joint targets for the episode layout."""
        ball_xy = np.asarray(ball_pos[:2], dtype=np.float64)
        rack_xy = np.mean(np.asarray(pin_positions, dtype=np.float64)[:, :2], axis=0)
        aim = rack_xy - ball_xy
        norm = float(np.linalg.norm(aim))
        u_nom = aim / norm if norm > 1e-8 else np.array([1.0, 0.0], dtype=np.float64)

        yaw = float(self.params.aim_yaw_offset_rad)
        c, s = float(np.cos(yaw)), float(np.sin(yaw))
        u_aim = np.array([c * u_nom[0] - s * u_nom[1], s * u_nom[0] + c * u_nom[1]], dtype=np.float64)
        self.u_aim = u_aim

        stage_off = max(float(self.params.stage_offset_m), float(self.cfg.ball.radius) + 0.030)
        stage_xy = ball_xy - stage_off * u_aim

        # Keep end waypoint strictly inside foul_line_x with safety margin for gripper geometry.
        max_end_x = float(self.cfg.delivery.foul_line_x) - 0.020
        end_xy = ball_xy + float(self.params.follow_through_m) * u_aim
        if end_xy[0] > max_end_x and end_xy[0] > stage_xy[0] + 1e-6:
            t_clamp = (max_end_x - stage_xy[0]) / (end_xy[0] - stage_xy[0])
            end_xy = stage_xy + np.clip(t_clamp, 0.0, 1.0) * (end_xy - stage_xy)

        min_clear_z = float(self.cfg.table.pad_thickness) + 0.044
        stage_z = max(float(self.params.push_height_z), min_clear_z)
        lift_z = max(stage_z + 0.078, 0.130)
        end_z = max(stage_z + 0.026, 0.078)

        p_lift = np.array([stage_xy[0], stage_xy[1], lift_z], dtype=np.float64)
        p_stage = np.array([stage_xy[0], stage_xy[1], stage_z], dtype=np.float64)
        p_end = np.array([end_xy[0], end_xy[1], end_z], dtype=np.float64)

        pan_seed = float(np.arctan2(stage_xy[1], stage_xy[0]))
        q_seed = np.array([pan_seed, -0.45, 0.75, 1.05], dtype=np.float64)

        self.q_lift = solve_4dof_ik(
            self.model,
            self._ik_data,
            self.ids,
            p_lift,
            self.q_min,
            self.q_max,
            q_init=q_seed,
            prefer_fwd_vec=np.array([u_aim[0], u_aim[1], 0.20]),
        )
        self.q_stage = solve_4dof_ik(
            self.model,
            self._ik_data,
            self.ids,
            p_stage,
            self.q_min,
            self.q_max,
            q_init=self.q_lift,
            prefer_fwd_vec=np.array([u_aim[0], u_aim[1], 0.20]),
        )
        self.q_push_end = solve_4dof_ik(
            self.model,
            self._ik_data,
            self.ids,
            p_end,
            self.q_min,
            self.q_max,
            q_init=self.q_stage,
            prefer_fwd_vec=np.array([u_aim[0], u_aim[1], 0.85]),
        )

        self.phase = "lift"
        self._push_steps = 0
        self._hold_steps = 0

    def step(self, q_measured: np.ndarray) -> np.ndarray:
        """Compute normalized 4-DOF position-delta action ``a in [-1, 1]^4`` at 20 Hz."""
        qm = np.asarray(q_measured, dtype=np.float64)
        speed_scale = float(np.clip(self.params.push_speed_mps / 0.85, 0.25, 1.0))

        if self.phase == "lift":
            q_ref = self.q_lift
            scale = 1.0
            if np.max(np.abs(qm - self.q_lift)) < 0.04:
                self.phase = "stage"
        elif self.phase == "stage":
            q_ref = self.q_stage
            scale = 1.0
            if np.max(np.abs(qm - self.q_stage)) < 0.04:
                self.phase = "push"
                self._push_steps = 0
        elif self.phase == "push":
            self._push_steps += 1
            q_ref = self.q_push_end
            scale = 0.47 * speed_scale if self._push_steps == 1 else speed_scale
            if self._push_steps >= 4:
                self.phase = "hold"
                self._hold_steps = 0
        elif self.phase == "hold":
            self._hold_steps += 1
            if self._hold_steps >= 6:
                self.phase = "retract"
            return np.zeros(len(ACTIVE_JOINTS), dtype=np.float32)
        else:
            q_ref = self.q_lift
            scale = 1.0

        raw_a = (q_ref - qm) / float(self.cfg.robot.max_action_rad)
        max_a = float(np.max(np.abs(raw_a)))
        if max_a > scale and max_a > 1e-8:
            raw_a = raw_a * (scale / max_a)
        return np.clip(raw_a, -1.0, 1.0).astype(np.float32)
