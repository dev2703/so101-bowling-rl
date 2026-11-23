"""Physics and visual domain randomization for sim-to-real transfer."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from so101_bowling.assets import SceneEntityIds, compute_hollow_pin_inertia
from so101_bowling.config import DRConfig, WorkspaceConfig


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two wxyz quaternions using MuJoCo's built-in utility."""
    out = np.empty(4, dtype=np.float64)
    mujoco.mju_mulQuat(out, np.asarray(q1, dtype=np.float64), np.asarray(q2, dtype=np.float64))
    return out


def _lookat_quat(pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Build MuJoCo camera wxyz quaternion looking from pos toward target (+Z up)."""
    fwd = np.asarray(target, dtype=np.float64) - np.asarray(pos, dtype=np.float64)
    fwd /= np.linalg.norm(fwd)
    up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(fwd, up_hint))) > 0.99:
        up_hint = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    right = np.cross(fwd, up_hint)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    rot = np.column_stack((right, up, -fwd)).reshape(9)
    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, rot)
    return quat


def _small_angle_quat(euler_rad: np.ndarray) -> np.ndarray:
    """Build wxyz quaternion from small xyz Euler perturbation."""
    rx, ry, rz = 0.5 * euler_rad
    q = np.array([1.0, rx, ry, rz], dtype=np.float64)
    return q / np.linalg.norm(q)


class DomainRandomizer:
    """Caches nominal MuJoCo model parameters and applies/restores episode DR."""

    def __init__(
        self, model: mujoco.MjModel, ids: SceneEntityIds, cfg: WorkspaceConfig
    ) -> None:
        self.model = model
        self.ids = ids
        self.cfg = cfg

        self._nom_body_mass = model.body_mass.copy()
        self._nom_body_inertia = model.body_inertia.copy()
        self._nom_body_pos = model.body_pos.copy()
        self._nom_pair_friction = model.pair_friction.copy()
        self._nom_dof_damping = model.dof_damping.copy()
        self._nom_actuator_gainprm = model.actuator_gainprm.copy()
        self._nom_actuator_biasprm = model.actuator_biasprm.copy()
        self._nom_cam_pos = model.cam_pos.copy()
        self._nom_cam_quat = model.cam_quat.copy()
        self._nom_cam_fovy = model.cam_fovy.copy()
        self._nom_light_pos = model.light_pos.copy()
        self._nom_light_diffuse = model.light_diffuse.copy()
        self._nom_mat_rgba = model.mat_rgba.copy()
        self._nom_geom_rgba = model.geom_rgba.copy()
        self._nom_geom_size = model.geom_size.copy()

        self._pad_pair_ids = [
            i
            for i in range(model.npair)
            if (
                int(model.pair_geom1[i]) == ids.ball_geom_id
                and int(model.pair_geom2[i]) in ids.pad_geom_ids
            )
            or (
                int(model.pair_geom2[i]) == ids.ball_geom_id
                and int(model.pair_geom1[i]) in ids.pad_geom_ids
            )
        ]
        self._desk_pair_ids = [
            i
            for i in range(model.npair)
            if (
                int(model.pair_geom1[i]) == ids.ball_geom_id
                and int(model.pair_geom2[i]) in ids.desk_geom_ids
            )
            or (
                int(model.pair_geom2[i]) == ids.ball_geom_id
                and int(model.pair_geom1[i]) in ids.desk_geom_ids
            )
        ]
        self._arm_dof_addrs = [int(model.jnt_dofadr[j]) for j in ids.all_joint_ids]
        self._arm_act_ids = ids.active_actuator_ids + ids.held_actuator_ids
        self._jitter_mat_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, mname)
            for mname in ("desk_mat", "pad_mat", "ball_mat")
        ]
        self._jitter_mat_ids = [m for m in self._jitter_mat_ids if m >= 0]
        self._distractor_mat_ids = [
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_MATERIAL, f"distractor_mat_{idx}"
            )
            for idx in range(len(ids.distractor_body_ids))
        ]

    def _restore_nominal(self) -> None:
        m = self.model
        m.body_mass[:] = self._nom_body_mass
        m.body_inertia[:] = self._nom_body_inertia
        m.body_pos[:] = self._nom_body_pos
        m.pair_friction[:] = self._nom_pair_friction
        m.dof_damping[:] = self._nom_dof_damping
        m.actuator_gainprm[:] = self._nom_actuator_gainprm
        m.actuator_biasprm[:] = self._nom_actuator_biasprm
        m.cam_pos[:] = self._nom_cam_pos
        m.cam_quat[:] = self._nom_cam_quat
        m.cam_fovy[:] = self._nom_cam_fovy
        m.light_pos[:] = self._nom_light_pos
        m.light_diffuse[:] = self._nom_light_diffuse
        m.mat_rgba[:] = self._nom_mat_rgba
        m.geom_rgba[:] = self._nom_geom_rgba
        m.geom_size[:] = self._nom_geom_size

    def apply_episode(
        self, data: mujoco.MjData, rng: np.random.Generator, dr_enabled: bool
    ) -> dict[str, Any]:
        """Apply episode randomization (or restore exact nominal parameters when disabled)."""
        self._restore_nominal()
        m = self.model
        dr = self.cfg.dr

        ball_x, ball_y = self.cfg.ball.spawn_pos
        ball_z = self.cfg.table.pad_thickness + self.cfg.ball.radius
        pin_positions = [list(p) for p in self.cfg.pin.positions]

        if not dr_enabled:
            mujoco.mj_setConst(m, data)
            self._set_free_bodies(data, (ball_x, ball_y, ball_z), pin_positions)
            mujoco.mj_forward(m, data)
            return {
                "dr_enabled": False,
                "ball_mass": float(self.cfg.ball.mass),
                "ball_spawn": (ball_x, ball_y, ball_z),
                "pin_masses": [float(self.cfg.pin.mass)] * len(self.ids.pin_body_ids),
                "pin_positions": [tuple(p) for p in pin_positions],
            }

        # 1. Ball mass + hollow sphere inertia + spawn jitter.
        ball_mass = float(rng.uniform(*dr.ball_mass_range))
        ball_i = (2.0 / 3.0) * ball_mass * (self.cfg.ball.radius**2)
        m.body_mass[self.ids.ball_body_id] = ball_mass
        m.body_inertia[self.ids.ball_body_id] = [ball_i, ball_i, ball_i]

        dx = float(rng.uniform(-dr.ball_spawn_dx, dr.ball_spawn_dx))
        dy = float(rng.uniform(-dr.ball_spawn_dy, dr.ball_spawn_dy))
        ball_spawn = (ball_x + dx, ball_y + dy, ball_z)

        # 2. Pin masses with coupled hollow thin-shell inertia + small position jitter.
        pin_masses: list[float] = []
        for idx, bid in enumerate(self.ids.pin_body_ids):
            pm = float(rng.uniform(*dr.pin_mass_range))
            pin_masses.append(pm)
            m.body_mass[bid] = pm
            m.body_inertia[bid] = list(compute_hollow_pin_inertia(pm, self.cfg.pin))
            pin_positions[idx][0] += float(
                rng.uniform(-dr.pin_pos_jitter, dr.pin_pos_jitter)
            )
            pin_positions[idx][1] += float(
                rng.uniform(-dr.pin_pos_jitter, dr.pin_pos_jitter)
            )

        # 3. Ball-pad & ball-desk condim=6 friction randomization.
        pad_slide = float(rng.uniform(*dr.slide_friction_scale))
        pad_roll = float(rng.uniform(*dr.pad_roll_friction_scale))
        for pid in self._pad_pair_ids:
            m.pair_friction[pid, 0:2] *= pad_slide
            m.pair_friction[pid, 3:5] *= pad_roll

        desk_slide = float(rng.uniform(*dr.slide_friction_scale))
        desk_roll = float(rng.uniform(*dr.desk_roll_friction_scale))
        for pid in self._desk_pair_ids:
            m.pair_friction[pid, 0:2] *= desk_slide
            m.pair_friction[pid, 3:5] *= desk_roll

        # 4. Arm joint damping and position actuator kp scaling.
        for dof in self._arm_dof_addrs:
            m.dof_damping[dof] *= float(rng.uniform(*dr.joint_damping_scale))
        for act in self._arm_act_ids:
            kp_scale = float(rng.uniform(*dr.actuator_kp_scale))
            m.actuator_gainprm[act, 0] *= kp_scale
            m.actuator_biasprm[act, 1] *= kp_scale

        # 5. Bounded camera pose, look-at target, & FOV jitter (keeps ball & pins in frame).
        tid = self.ids.tripod_cam_id
        tripod_pos = self._nom_cam_pos[tid] + rng.uniform(
            -dr.tripod_pos_jitter, dr.tripod_pos_jitter, size=3
        )
        tripod_target = np.asarray(self.cfg.camera.tripod_target, dtype=np.float64) + rng.uniform(
            -dr.tripod_target_jitter, dr.tripod_target_jitter, size=3
        )
        m.cam_pos[tid] = tripod_pos
        m.cam_quat[tid] = _quat_mul(
            _lookat_quat(tripod_pos, tripod_target),
            _small_angle_quat(np.array([0.0, 0.0, float(rng.uniform(-0.025, 0.025))])),
        )
        m.cam_fovy[tid] += float(rng.uniform(-dr.tripod_fovy_jitter, dr.tripod_fovy_jitter))

        wid = self.ids.wrist_cam_id
        m.cam_pos[wid] += rng.uniform(
            -dr.wrist_pos_jitter, dr.wrist_pos_jitter, size=3
        )
        m.cam_quat[wid] = _quat_mul(
            m.cam_quat[wid], _small_angle_quat(rng.uniform(-0.018, 0.018, size=3))
        )
        m.cam_fovy[wid] += float(rng.uniform(-dr.wrist_fovy_jitter, dr.wrist_fovy_jitter))

        # 6. Lighting & surface color jitter.
        for lid in range(m.nlight):
            m.light_pos[lid] += rng.uniform(
                -dr.light_pos_jitter, dr.light_pos_jitter, size=3
            )
            m.light_pos[lid, 2] = max(float(m.light_pos[lid, 2]), 0.5)
            iscale = float(rng.uniform(*dr.light_intensity_scale))
            m.light_diffuse[lid] = np.clip(m.light_diffuse[lid] * iscale, 0.05, 1.5)

        for mid in self._jitter_mat_ids:
            delta = rng.uniform(-dr.color_jitter, dr.color_jitter, size=3)
            m.mat_rgba[mid, :3] = np.clip(m.mat_rgba[mid, :3] + delta, 0.02, 0.98)

        # 7. Place distractors outside the lane / behind desk with random colors & sizes.
        for idx, (bid, gid) in enumerate(
            zip(self.ids.distractor_body_ids, self.ids.distractor_geom_ids)
        ):
            side = idx % 3
            if side == 0:
                dx_pos = float(rng.uniform(0.84, 1.15))
                dy_pos = float(rng.uniform(-0.45, 0.45))
                dz_pos = float(rng.uniform(-0.02, 0.12))
            elif side == 1:
                dx_pos = float(rng.uniform(0.10, 0.78))
                dy_pos = float(rng.uniform(0.39, 0.58))
                dz_pos = float(rng.uniform(0.01, 0.10))
            else:
                dx_pos = float(rng.uniform(0.10, 0.78))
                dy_pos = float(rng.uniform(-0.58, -0.39))
                dz_pos = float(rng.uniform(0.01, 0.10))

            m.body_pos[bid] = [dx_pos, dy_pos, dz_pos]
            scale = float(rng.uniform(0.6, 1.8))
            m.geom_size[gid] = self._nom_geom_size[gid] * scale
            if idx < len(self._distractor_mat_ids) and self._distractor_mat_ids[idx] >= 0:
                d_mid = self._distractor_mat_ids[idx]
                m.mat_rgba[d_mid, :3] = rng.uniform(0.08, 0.92, size=3)

        mujoco.mj_setConst(m, data)
        self._set_free_bodies(data, ball_spawn, pin_positions)
        mujoco.mj_forward(m, data)

        return {
            "dr_enabled": True,
            "ball_mass": ball_mass,
            "ball_spawn": ball_spawn,
            "pin_masses": pin_masses,
            "pin_positions": [tuple(p) for p in pin_positions],
        }

    def _set_free_bodies(
        self,
        data: mujoco.MjData,
        ball_spawn: tuple[float, float, float],
        pin_positions: list[list[float]],
    ) -> None:
        b_q = self.ids.ball_joint_qpos_addr
        b_v = self.ids.ball_joint_qvel_addr
        data.qpos[b_q : b_q + 3] = ball_spawn
        data.qpos[b_q + 3 : b_q + 7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[b_v : b_v + 6] = 0.0

        for idx, (p_q, p_v) in enumerate(
            zip(self.ids.pin_joint_qpos_addrs, self.ids.pin_joint_qvel_addrs)
        ):
            px, py = pin_positions[idx]
            data.qpos[p_q : p_q + 3] = [px, py, 0.0]
            data.qpos[p_q + 3 : p_q + 7] = [1.0, 0.0, 0.0, 0.0]
            data.qvel[p_v : p_v + 6] = 0.0


def augment_rgb(
    img: np.ndarray,
    rng: np.random.Generator,
    dr_enabled: bool,
    shift_xy: tuple[int, int] | None = None,
    dr_cfg: DRConfig | None = None,
) -> np.ndarray:
    """Apply photometric and small spatial shift augmentations preserving uint8 (H, W, 3)."""
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected uint8 (H, W, 3) image, got {img.dtype} {img.shape}")
    if not dr_enabled:
        return img.copy()

    dr = dr_cfg if dr_cfg is not None else DRConfig()
    h, w, _ = img.shape
    max_shift = int(dr.max_shift_px)
    if shift_xy is None:
        if max_shift > 0:
            sx = int(rng.integers(-max_shift, max_shift + 1))
            sy = int(rng.integers(-max_shift, max_shift + 1))
        else:
            sx = sy = 0
    else:
        sx, sy = int(shift_xy[0]), int(shift_xy[1])

    out = img
    if sx != 0 or sy != 0:
        pad_x = abs(sx)
        pad_y = abs(sy)
        padded = np.pad(
            out, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode="edge"
        )
        y0 = pad_y - sy
        x0 = pad_x - sx
        out = padded[y0 : y0 + h, x0 : x0 + w]

    arr = out.astype(np.float32) / 255.0

    # Contrast & brightness
    contrast = float(rng.uniform(*dr.contrast_scale))
    brightness = float(rng.uniform(-dr.brightness_delta, dr.brightness_delta))
    arr = (arr - 0.5) * contrast + 0.5 + brightness

    # Gamma
    gamma = float(rng.uniform(*dr.gamma_range))
    arr = np.power(np.clip(arr, 0.0, 1.0), gamma)

    # Saturation
    sat = float(rng.uniform(*dr.saturation_scale))
    gray = 0.299 * arr[:, :, 0:1] + 0.587 * arr[:, :, 1:2] + 0.114 * arr[:, :, 2:3]
    arr = gray + sat * (arr - gray)

    # Pixel noise
    if dr.noise_std > 0.0:
        noise = rng.normal(0.0, float(dr.noise_std) / 255.0, size=arr.shape).astype(np.float32)
        arr = arr + noise
    arr = np.clip(arr, 0.0, 1.0)

    return np.rint(arr * 255.0).astype(np.uint8)
