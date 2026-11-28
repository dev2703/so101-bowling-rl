"""State-based MuJoCo Gymnasium environment for SO-101 3-pin tabletop bowling."""

from __future__ import annotations

from dataclasses import asdict, replace

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from so101_bowling.assets import build_scene_xml, get_scene_entity_ids
from so101_bowling.config import (
    ACTIVE_JOINTS,
    ALL_JOINTS,
    HELD_JOINTS,
    CalibrationMap,
    WorkspaceConfig,
    intersect_joint_limits,
    load_calibration,
    load_config,
)
from so101_bowling.scorer import (
    DeliveryPhase,
    DeliveryScorer,
    ScorerConfig,
    SubstepSnapshot,
)

OBS_DIM = 53


class SO101BowlingEnv(gym.Env):
    """4-DOF continuous-control MuJoCo environment returning a 53D float32 state vector."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: WorkspaceConfig | None = None,
        cal: CalibrationMap | None = None,
        scorer_cfg: ScorerConfig | None = None,
        seed: int | None = None,
        dr: bool | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else load_config()
        self.cal = cal if cal is not None else load_calibration()
        self.dr_enabled = self.cfg.dr.enabled if dr is None else dr
        self._initial_seed = seed
        self._needs_reset = True

        self.model = mujoco.MjModel.from_xml_string(build_scene_xml(self.cfg))
        self.data = mujoco.MjData(self.model)
        self.ids = ids = get_scene_entity_ids(self.model)

        limits = intersect_joint_limits(self.cal)
        self.q_min, self.q_max = np.asarray([limits[name] for name in ACTIVE_JOINTS]).T
        self.max_action_rad = self.cfg.robot.max_action_rad
        self._home = np.asarray([self.cfg.robot.home_qpos[name] for name in ALL_JOINTS])
        self._held_home = np.asarray([self.cfg.robot.home_qpos[name] for name in HELD_JOINTS])
        self._all_qpos_addrs = self.model.jnt_qposadr[ids.all_joint_ids]

        self._arm_geoms = np.array(sorted(ids.arm_geom_ids), dtype=np.intp)
        self._arm_aabb = self.model.geom_aabb[self._arm_geoms].reshape(-1, 2, 3)
        self._is_arm_geom = np.zeros(self.model.ngeom, dtype=bool)
        self._is_arm_geom[list(ids.arm_geom_ids)] = True
        self._is_pusher_geom = np.zeros(self.model.ngeom, dtype=bool)
        self._is_pusher_geom[list(ids.pusher_geom_ids)] = True
        self._is_table_geom = np.zeros(self.model.ngeom, dtype=bool)
        self._is_table_geom[list(ids.table_geom_ids)] = True
        self._pin_geom_idx = np.full(self.model.ngeom, -1, dtype=np.intp)
        for pin, geoms in enumerate(ids.pin_geom_ids):
            self._pin_geom_idx[list(geoms)] = pin
        self._pin_vel = np.asarray(ids.pin_joint_qvel_addrs)[:, None] + np.arange(3)
        self._ft = np.zeros(6)

        geom_id = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
        self._pad_geoms = np.array([geom_id("mousepad"), geom_id("pad_ramp")], dtype=np.intp)
        self._desk_geoms = np.array(
            [geom_id(f"desk_{side}") for side in ("back", "front", "left", "right")], dtype=np.intp
        )
        self._nom_body_mass = self.model.body_mass.copy()
        self._nom_body_inertia = self.model.body_inertia.copy()
        self._nom_pair_friction = self.model.pair_friction.copy()
        self._nom_geom_friction = self.model.geom_friction.copy()

        self.substeps = round(1.0 / (self.cfg.robot.control_hz * self.cfg.robot.physics_dt))
        self._control_dt = self.substeps * self.cfg.robot.physics_dt
        self.delivery_timeout_s = self.cfg.delivery.max_steps * self._control_dt

        if scorer_cfg is None:
            d, r = self.cfg.delivery, self.cfg.reward
            scorer_cfg = ScorerConfig(
                gamma=r.gamma,
                sim_dt=self.cfg.robot.physics_dt,
                ball_radius=self.cfg.ball.radius,
                pin_belly_radius=self.cfg.pin.belly_radius,
                staging_clearance_m=d.staging_clearance_m,
                foul_line_x=d.foul_line_x,
                release_line_x=d.release_line_x,
                rack_x_nom=self.cfg.pin.positions[0][0],
                min_push_disp_m=d.min_push_disp_m,
                min_launch_speed_mps=d.min_launch_speed_mps,
                target_launch_speed_mps=d.target_launch_speed_mps,
                aim_sigma_rad=d.aim_sigma_rad,
                sep_dist_m=d.sep_dist_m,
                sep_duration_s=d.sep_duration_s,
                wobble_tilt_deg=d.wobble_tilt_deg,
                full_fall_tilt_deg=d.full_fall_tilt_deg,
                min_wobble_substeps=max(2, round(d.wobble_duration_s / self.cfg.robot.physics_dt)),
                off_table_z_m=d.off_table_z_m,
                quiet_vel_mps=d.quiet_vel_mps,
                quiet_dwell_s=d.quiet_dwell_s,
                max_coast_s=d.max_coast_s,
                w_stage_phi=r.w_stage_phi,
                w_push_phi=r.w_push_phi,
                r_first_contact=r.r_first_contact,
                w_contact_speed=r.w_contact_speed,
                w_launch_angle=r.w_launch_angle,
                w_ball_approach=r.w_ball_approach,
                ball_approach_sigma_m=r.ball_approach_sigma_m,
                w_partial_tilt=r.w_partial_tilt,
                w_step_penalty=r.w_step_penalty,
                w_action_jerk=r.w_action_jerk,
                r_pin_credit=r.r_pin_credit,
                r_strike_bonus=r.r_strike_bonus,
                r_illegal_penalty=r.r_illegal_penalty,
                r_soft_fault_penalty=r.r_soft_fault_penalty,
                r_timeout_penalty=r.r_timeout_penalty,
            )
        self.scorer = DeliveryScorer(replace(scorer_cfg, sim_dt=self.cfg.robot.physics_dt))

        self.action_space = spaces.Box(-1.0, 1.0, shape=(len(ACTIVE_JOINTS),), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=self._initial_seed if seed is None else seed)
        self._initial_seed = None
        self._restore_nominal_model()
        if self.dr_enabled:
            self._randomize_model()
        # mj_setConst overwrites data.qpos, so it must run before the state is initialized.
        mujoco.mj_setConst(self.model, self.data)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._all_qpos_addrs] = self._home
        self._hold(self._home[: len(ACTIVE_JOINTS)])
        if self.dr_enabled:
            self._randomize_positions()
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data, nstep=20)
        if self.dr_enabled:
            self._tilt_pins()
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._last_snapshot = snap = self._snapshot()
        self.scorer.reset(
            snap.ball_pos, snap.pin_positions, snap.pin_up_vectors, paddle_init_pos=snap.paddle_pos
        )
        self._step_count = 0
        self._prev_delta = np.zeros(len(ACTIVE_JOINTS))
        self._needs_reset = False
        return self._observation(), {}

    def _restore_nominal_model(self) -> None:
        self.model.body_mass[:] = self._nom_body_mass
        self.model.body_inertia[:] = self._nom_body_inertia
        self.model.pair_friction[:] = self._nom_pair_friction
        self.model.geom_friction[:] = self._nom_geom_friction

    def _randomize_model(self) -> None:
        """Sample ball/pin masses (inertia scaled to match) and contact friction."""
        model, dr, ids, rng = self.model, self.cfg.dr, self.ids, self.np_random
        mass_ranges = [dr.ball_mass_range] + [dr.pin_mass_range] * len(ids.pin_body_ids)
        for bid, mass_range in zip((ids.ball_body_id, *ids.pin_body_ids), mass_ranges):
            mass = float(rng.uniform(*mass_range))
            model.body_mass[bid] = mass
            model.body_inertia[bid] *= mass / self._nom_body_mass[bid]

        slide_s = float(rng.uniform(*dr.slide_friction_scale))
        pad_roll_s = float(rng.uniform(*dr.pad_roll_friction_scale))
        desk_roll_s = float(rng.uniform(*dr.desk_roll_friction_scale))
        for pair_ids, roll_s in ((ids.pad_pair_ids, pad_roll_s), (ids.desk_pair_ids, desk_roll_s)):
            model.pair_friction[pair_ids, 0:2] *= slide_s
            model.pair_friction[pair_ids, 3:5] *= roll_s
        model.geom_friction[self._pad_geoms, 0] *= slide_s
        model.geom_friction[self._pad_geoms, 2] *= pad_roll_s
        model.geom_friction[self._desk_geoms, 0] *= slide_s
        model.geom_friction[self._desk_geoms, 2] *= desk_roll_s

    def _randomize_positions(self) -> None:
        """Jitter the ball spawn, shift the whole pin rack, and jitter each pin."""
        qpos, dr, ids, rng = self.data.qpos, self.cfg.dr, self.ids, self.np_random
        ball = ids.ball_joint_qpos_addr
        qpos[ball] += rng.uniform(-dr.ball_spawn_dx, dr.ball_spawn_dx)
        qpos[ball + 1] += rng.uniform(-dr.ball_spawn_dy, dr.ball_spawn_dy)
        shift = (
            rng.uniform(-dr.rack_shift_x, dr.rack_shift_x),
            rng.uniform(-dr.rack_shift_y, dr.rack_shift_y),
        )
        addrs = ids.pin_joint_qpos_addrs
        base_xy = np.array([qpos[a : a + 2] for a in addrs])
        min_sep = (
            2.0 * self.cfg.pin.belly_radius
            + 2.0 * 0.068 * np.sin(np.radians(dr.pin_init_tilt_deg))
            + 0.003
        )
        jitters = np.zeros_like(base_xy)
        for _ in range(64):
            cand = rng.uniform(-dr.pin_pos_jitter, dr.pin_pos_jitter, size=base_xy.shape)
            pos = base_xy + cand
            diffs = pos[:, None, :] - pos[None, :, :]
            dists = np.hypot(diffs[:, :, 0], diffs[:, :, 1]) + np.eye(len(addrs))
            if np.all(dists >= min_sep):
                jitters = cand
                break
        for addr, jit in zip(addrs, jitters):
            qpos[addr : addr + 2] += shift + jit

    def _tilt_pins(self) -> None:
        """Give each settled pin a slight random lean, lifted so its foot clears the desk."""
        data, ids, rng = self.data, self.ids, self.np_random
        for addr, dof in zip(ids.pin_joint_qpos_addrs, ids.pin_joint_qvel_addrs):
            tilt = float(np.radians(rng.uniform(0.0, self.cfg.dr.pin_init_tilt_deg)))
            az = float(rng.uniform(0.0, 2.0 * np.pi))
            s = np.sin(0.5 * tilt)
            data.qpos[addr + 3 : addr + 7] = [np.cos(0.5 * tilt), -np.sin(az) * s, np.cos(az) * s, 0.0]
            data.qpos[addr + 2] += self.cfg.pin.foot_radius * np.sin(tilt)
            data.qvel[dof : dof + 6] = 0.0

    def _hold(self, target: np.ndarray) -> None:
        self.data.ctrl[self.ids.active_actuator_ids] = target
        self.data.ctrl[self.ids.held_actuator_ids] = self._held_home

    def step(self, action):
        if self._needs_reset:
            raise gym.error.ResetNeeded("Call reset() before stepping a new episode")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError("Expected 4 finite action values")

        q_meas = self.data.qpos[self.ids.active_qpos_addrs]
        scale = self.max_action_rad
        target = np.clip(q_meas + np.clip(action, -1.0, 1.0) * scale, self.q_min, self.q_max)
        self._hold(target)
        applied_delta = target - q_meas

        for _ in range(self.substeps):
            snap = self._advance()
            if self.scorer.phase == DeliveryPhase.INVALID:
                break

        self._step_count += 1
        done, reason = self.scorer.should_end_delivery(
            self._step_count * self._control_dt, self.delivery_timeout_s
        )
        reward = self.scorer.step_control_reward(
            snap.paddle_pos, applied_delta / scale, self._prev_delta / scale, terminated=done
        )
        self._prev_delta = applied_delta

        info: dict = {}
        if done:
            unresolved = False
            if self.scorer.phase != DeliveryPhase.INVALID:
                self._hold(self.data.qpos[self.ids.active_qpos_addrs])
                coast_steps = 0
                settled, unresolved = self.scorer.is_coast_settled(0.0)
                while not settled:
                    snap = self._advance()
                    coast_steps += 1
                    settled, unresolved = self.scorer.is_coast_settled(
                        coast_steps * self.scorer.config.sim_dt
                    )
            outcome = self.scorer.finalize_episode(snap, unresolved)
            reward += outcome.total_terminal_reward
            info.update(
                outcome=asdict(outcome),
                end_reason=reason,
                credited_pins=outcome.credited_pins,
                legal_strike=outcome.legal_strike,
                legal_delivery=outcome.legal_delivery,
                touched_ball=outcome.touched_ball,
            )

        mujoco.mj_forward(self.model, self.data)
        self._last_snapshot = self._snapshot()
        self._needs_reset = done
        return self._observation(), float(reward), done, False, info

    def _advance(self) -> SubstepSnapshot:
        mujoco.mj_step(self.model, self.data)
        snap = self._snapshot()
        self.scorer.update_substep(snap)
        return snap

    def _snapshot(self) -> SubstepSnapshot:
        data, ids = self.data, self.ids
        paddle = data.site_xpos[ids.gripper_site_id]
        x_axes = data.geom_xmat[self._arm_geoms, :3]
        max_x = data.geom_xpos[self._arm_geoms, 0] + np.sum(
            x_axes * self._arm_aabb[:, 0] + np.abs(x_axes) * self._arm_aabb[:, 1], axis=1
        )

        pusher_ball = other_ball = arm_pin = table_slam = False
        ball_pin = [False, False, False]
        pin_pin = np.zeros((3, 3), dtype=bool)
        ncon = int(data.ncon)
        if ncon > 0:
            geoms = data.contact.geom[:ncon]
            g1, g2 = geoms[:, 0], geoms[:, 1]
            is_g1_ball = g1 == ids.ball_geom_id
            is_ball = is_g1_ball | (g2 == ids.ball_geom_id)
            if is_ball.any():
                other = np.where(is_g1_ball[is_ball], g2[is_ball], g1[is_ball])
                is_push = self._is_pusher_geom[other]
                pusher_ball = bool(is_push.any())
                other_ball = bool((self._is_arm_geom[other] & ~is_push).any())
                for p in self._pin_geom_idx[other]:
                    if p >= 0:
                        ball_pin[int(p)] = True
            non_ball = ~is_ball
            if non_ball.any():
                nb_g1, nb_g2 = g1[non_ball], g2[non_ball]
                a1 = self._is_arm_geom[nb_g1]
                arm_mask = a1 | self._is_arm_geom[nb_g2]
                if arm_mask.any():
                    arm_other = np.where(a1[arm_mask], nb_g2[arm_mask], nb_g1[arm_mask])
                    arm_pin = bool((self._pin_geom_idx[arm_other] >= 0).any())
                    tbl_hits = self._is_table_geom[arm_other]
                    if tbl_hits.any():
                        for ci in np.flatnonzero(non_ball)[arm_mask][tbl_hits]:
                            mujoco.mj_contactForce(self.model, data, int(ci), self._ft)
                            if self._ft[0] > 8.0:
                                table_slam = True
                                break
                p1, p2 = self._pin_geom_idx[nb_g1], self._pin_geom_idx[nb_g2]
                pp = (p1 >= 0) & (p2 >= 0)
                if pp.any():
                    for i_p, j_p in zip(p1[pp], p2[pp]):
                        pin_pin[int(i_p), int(j_p)] = pin_pin[int(j_p), int(i_p)] = True

        pin_v = data.qvel[self._pin_vel]
        return SubstepSnapshot(
            t=data.time,
            paddle_pos=paddle.copy(),
            ball_pos=data.xpos[ids.ball_body_id].copy(),
            ball_vel=data.qvel[ids.ball_joint_qvel_addr : ids.ball_joint_qvel_addr + 3].copy(),
            pin_positions=data.xpos[ids.pin_body_ids].copy(),
            pin_up_vectors=data.xmat[ids.pin_body_ids, 2::3].copy(),
            pin_speeds=np.sqrt(np.sum(pin_v * pin_v, axis=1)),
            max_arm_x=max(float(max_x.max()), float(paddle[0])),
            pusher_ball_contact=pusher_ball,
            other_arm_ball_contact=other_ball,
            arm_pin_contact=arm_pin,
            table_slam=table_slam,
            ball_pin_contacts=tuple(ball_pin),
            pin_pin_contacts=pin_pin,
            paddle_ball_surface_dist=min(
                mujoco.mj_geomDistance(self.model, data, geom, ids.ball_geom_id, 0.5, None)
                for geom in ids.pusher_geom_ids
            ),
        )

    def _observation(self) -> np.ndarray:
        snap, ids = self._last_snapshot, self.ids
        q = self.data.qpos[ids.active_qpos_addrs]
        qd = self.data.qvel[ids.active_qvel_addrs]
        rack_xy = snap.pin_positions[:, :2].mean(axis=0)
        remaining = max(0.0, 1.0 - self._step_count * self._control_dt / self.delivery_timeout_s)
        return np.concatenate((
            q,
            qd,
            self._prev_delta / self.max_action_rad,
            snap.paddle_pos,
            snap.ball_pos,
            snap.ball_vel,
            snap.pin_positions.ravel(),
            snap.pin_up_vectors.ravel(),
            rack_xy - snap.ball_pos[:2],
            self.scorer.critic_state(remaining),
        )).astype(np.float32)
