"""Physics-substep delivery validation and three-pin scoring."""

from dataclasses import dataclass
from enum import IntEnum
import math

import numpy as np


class DeliveryPhase(IntEnum):
    APPROACH = 0
    PUSHING = 1
    RELEASED = 2
    INVALID = 3


SOFT_FAULT_REASONS: frozenset[str] = frozenset({
    "non_pusher_ball_contact",
    "insufficient_push_displacement_slap",
    "ball_moved_before_pusher_contact",
})


@dataclass
class ScorerConfig:
    gamma: float = 0.99
    sim_dt: float = 0.002
    ball_radius: float = 0.0485
    pin_belly_radius: float = 0.03435
    staging_clearance_m: float = 0.055
    foul_line_x: float = 0.44
    release_line_x: float = 0.44
    rack_x_nom: float = 0.62
    min_push_disp_m: float = 0.025
    min_launch_speed_mps: float = 0.25
    target_launch_speed_mps: float = 0.95
    aim_sigma_rad: float = 0.14
    sep_dist_m: float = 0.015
    sep_duration_s: float = 0.025
    wobble_tilt_deg: float = 12.0
    full_fall_tilt_deg: float = 46.0
    min_wobble_substeps: int = 10
    off_table_z_m: float = -0.05
    quiet_vel_mps: float = 0.02
    quiet_dwell_s: float = 0.25
    max_coast_s: float = 2.5
    w_stage_phi: float = 1.0
    w_push_phi: float = 1.5
    r_first_contact: float = 1.0
    w_contact_speed: float = 1.0
    w_launch_angle: float = 1.0
    w_ball_approach: float = 1.5
    ball_approach_sigma_m: float = 0.08
    w_partial_tilt: float = 1.0
    w_step_penalty: float = 0.01
    w_action_jerk: float = 0.005
    r_pin_credit: float = 5.0
    r_strike_bonus: float = 5.0
    r_illegal_penalty: float = -2.0
    r_soft_fault_penalty: float = -0.5
    r_timeout_penalty: float = -0.5


@dataclass
class SubstepSnapshot:
    t: float
    paddle_pos: np.ndarray
    ball_pos: np.ndarray
    ball_vel: np.ndarray
    pin_positions: np.ndarray
    pin_up_vectors: np.ndarray
    pin_speeds: np.ndarray
    max_arm_x: float
    pusher_ball_contact: bool
    other_arm_ball_contact: bool
    arm_pin_contact: bool
    table_slam: bool
    ball_pin_contacts: tuple[bool, bool, bool]
    pin_pin_contacts: np.ndarray
    paddle_ball_surface_dist: float


@dataclass
class EpisodeOutcome:
    legal_delivery: bool
    invalid_reason: str | None
    credited_pins: int
    full_knock_pins: int
    off_table_pins: int
    wobble_only_pins: int
    legal_strike: bool
    launch_speed: float
    launch_angle_err_rad: float
    contact_forward_disp_m: float
    unresolved_motion: bool
    launch_reward: float
    pin_reward: float
    penalty_reward: float
    total_terminal_reward: float
    approach_reward: float = 0.0
    partial_pin_reward: float = 0.0
    min_ball_head_pin_dist: float = float("inf")
    touched_ball: bool = False
    shaping_reward: float = 0.0
    contact_reward: float = 0.0
    cost_reward: float = 0.0


class DeliveryScorer:
    def __init__(self, config: ScorerConfig | None = None) -> None:
        self.config = config if config is not None else ScorerConfig()
        self.phase = DeliveryPhase.APPROACH
        self._ep_shaping_reward = self._ep_contact_reward = self._ep_cost_reward = 0.0

    def reset(
        self, ball_init_pos, pin_init_positions, pin_init_up_vectors, paddle_init_pos=None
    ) -> None:
        self.ball_init_pos = np.array(ball_init_pos, dtype=float, copy=True)
        self.head_pin_xy = np.array(pin_init_positions[0][:2], dtype=float, copy=True)
        aim = np.mean(pin_init_positions, axis=0)[:2] - self.ball_init_pos[:2]
        norm = np.linalg.norm(aim)
        self.u_aim = aim / norm if norm else np.array([1.0, 0.0])
        self.p_stage = self.ball_init_pos.copy()
        cfg = self.config
        self.p_stage[:2] -= (cfg.ball_radius + cfg.staging_clearance_m) * self.u_aim
        up = np.asarray(pin_init_up_vectors, dtype=float)
        self.pin_init_up_vectors = up / np.linalg.norm(up, axis=1, keepdims=True)
        self.phase = DeliveryPhase.APPROACH
        self.invalid_reason: str | None = None
        self.pin_impacted = np.zeros(3, dtype=bool)
        self.pin_wobbled = np.zeros(3, dtype=bool)
        self.pin_peak_tilt_deg = np.zeros(3)
        self._wobble_counts = np.zeros(3, dtype=int)
        self.contact_start_ball_xy = np.zeros(2)
        self.max_contact_fwd_disp = 0.0
        self.v_out = np.zeros(2)
        self._separation_x = 0.0
        self._sep_steps = 0
        self._quiet_steps = 0
        self._left_reach = False
        self._latest_t: float | None = None
        self._latest_ball_x = float(self.ball_init_pos[0])
        self._contact_bonus_awarded = False
        self._contact_bonus_pending = False
        self._min_ball_head_pin_dist = float("inf")
        self._phi_prev = 0.0
        self._ep_shaping_reward = self._ep_contact_reward = self._ep_cost_reward = 0.0
        self._terminated = False
        if paddle_init_pos is not None:
            self._phi_prev = self.compute_potential(paddle_init_pos)

    def compute_potential(self, paddle_pos) -> float:
        if self.phase == DeliveryPhase.INVALID:
            return 0.0
        cfg = self.config
        if self.phase == DeliveryPhase.PUSHING:
            push_progress = float(np.clip(self.max_contact_fwd_disp / 0.10, 0.0, 1.0))
            return float(cfg.w_stage_phi + cfg.w_push_phi * push_progress)
        if self.phase == DeliveryPhase.RELEASED:
            return float(cfg.w_stage_phi + cfg.w_push_phi)
        paddle = np.asarray(paddle_pos)
        distance_to_stage = np.linalg.norm(paddle - self.p_stage)
        clearance = np.linalg.norm(paddle - self.ball_init_pos)
        gate = np.clip((clearance - cfg.ball_radius) / cfg.staging_clearance_m, 0.0, 1.0)
        gate = gate**2 * (3.0 - 2.0 * gate)
        alignment = 1.0 - np.tanh(distance_to_stage / 0.12)
        return float(cfg.w_stage_phi * gate * alignment)

    def _invalidate(self, reason: str) -> None:
        if self.phase != DeliveryPhase.INVALID:
            self.phase = DeliveryPhase.INVALID
            self.invalid_reason = reason

    def _tilts_deg(self, up_vectors: np.ndarray) -> np.ndarray:
        up = np.asarray(up_vectors, dtype=np.float64)
        dots = np.einsum("ij,ij->i", up, self.pin_init_up_vectors)
        norms = np.sqrt(np.einsum("ij,ij->i", up, up))
        cosines = np.divide(dots, norms, out=np.ones(3), where=norms > 0.0)
        return np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))

    def _track_impacts(self, snap: SubstepSnapshot) -> None:
        if any(snap.ball_pin_contacts):
            self.pin_impacted |= np.asarray(snap.ball_pin_contacts, dtype=bool)
        if not self.pin_impacted.any():
            return
        c = np.asarray(snap.pin_pin_contacts, dtype=bool)
        if c.shape == (3, 3):
            p0, p1, p2 = bool(self.pin_impacted[0]), bool(self.pin_impacted[1]), bool(self.pin_impacted[2])
            c01 = bool(c[0, 1] or c[1, 0])
            c02 = bool(c[0, 2] or c[2, 0])
            c12 = bool(c[1, 2] or c[2, 1])
            for _ in range(2):
                if c01:
                    p0 = p1 = p0 or p1
                if c02:
                    p0 = p2 = p0 or p2
                if c12:
                    p1 = p2 = p1 or p2
            self.pin_impacted[:] = (p0, p1, p2)

        tilts = self._tilts_deg(snap.pin_up_vectors)
        self.pin_peak_tilt_deg = np.maximum(
            self.pin_peak_tilt_deg, np.where(self.pin_impacted, tilts, 0.0)
        )
        wobbling = self.pin_impacted & (tilts + 1e-9 >= self.config.wobble_tilt_deg)
        self._wobble_counts = np.where(wobbling, self._wobble_counts + 1, 0)
        self.pin_wobbled |= wobbling & (self._wobble_counts >= self.config.min_wobble_substeps)

    def update_substep(self, snap: SubstepSnapshot) -> None:
        cfg = self.config
        self._latest_t = snap.t
        ball_xy = np.asarray(snap.ball_pos[:2])
        ball_x, ball_y = float(ball_xy[0]), float(ball_xy[1])
        self._latest_ball_x = ball_x
        for illegal, reason in (
            (snap.arm_pin_contact, "arm_pin_contact"),
            (snap.table_slam, "table_slam"),
            (snap.max_arm_x > cfg.foul_line_x, "foul_line_crossing"),
            (snap.other_arm_ball_contact, "non_pusher_ball_contact"),
        ):
            if illegal:
                self._invalidate(reason)
                return
        if self.phase == DeliveryPhase.INVALID:
            return

        self._track_impacts(snap)
        if self.phase == DeliveryPhase.APPROACH:
            dx = ball_x - float(self.ball_init_pos[0])
            dy = ball_y - float(self.ball_init_pos[1])
            if dx * dx + dy * dy > 0.015 * 0.015:
                self._invalidate("ball_moved_before_pusher_contact")
            elif snap.pusher_ball_contact:
                self.phase = DeliveryPhase.PUSHING
                self.contact_start_ball_xy = ball_xy.copy()
                if not self._contact_bonus_awarded:
                    self._contact_bonus_awarded = True
                    self._contact_bonus_pending = True
            else:
                return

        if self.phase == DeliveryPhase.PUSHING:
            self._update_push(snap, ball_xy, ball_x)
        elif snap.pusher_ball_contact:
            self._invalidate("double_tap_restrike")

        if self.phase == DeliveryPhase.RELEASED:
            dx = ball_x - float(self.head_pin_xy[0])
            dy = ball_y - float(self.head_pin_xy[1])
            d_head = math.sqrt(dx * dx + dy * dy)
            self._min_ball_head_pin_dist = min(self._min_ball_head_pin_dist, d_head)
            self._left_reach |= ball_x - cfg.ball_radius > cfg.foul_line_x
            vx, vy = float(snap.ball_vel[0]), float(snap.ball_vel[1])
            quiet_vel_sq = cfg.quiet_vel_mps * cfg.quiet_vel_mps
            ball_done = float(snap.ball_pos[2]) < cfg.off_table_z_m or vx * vx + vy * vy < quiet_vel_sq
            pins_done = all(
                z < cfg.off_table_z_m or speed < cfg.quiet_vel_mps
                for z, speed in zip(snap.pin_positions[:, 2].tolist(), snap.pin_speeds.tolist())
            )
            self._quiet_steps = self._quiet_steps + 1 if (ball_done and pins_done) else 0

    def _update_push(self, snap: SubstepSnapshot, ball_xy: np.ndarray, ball_x: float) -> None:
        cfg = self.config
        if snap.pusher_ball_contact or snap.paddle_ball_surface_dist < cfg.sep_dist_m:
            self._sep_steps = 0
            if self.pin_impacted.any():
                self._invalidate("pre_release_pin_impact")
            elif ball_x > self.head_pin_xy[0] - 0.05:
                self._invalidate("late_release_past_rack")
            else:
                disp = float(np.dot(ball_xy - self.contact_start_ball_xy, self.u_aim))
                self.max_contact_fwd_disp = max(self.max_contact_fwd_disp, disp)
                self.v_out = np.array(snap.ball_vel[:2], dtype=float, copy=True)
            return

        if self._sep_steps == 0:
            self._separation_x = ball_x
            if not self.pin_impacted.any():
                self.v_out = np.array(snap.ball_vel[:2], dtype=float, copy=True)
        self._sep_steps += 1
        if self.pin_impacted.any():
            self._invalidate("pre_release_pin_impact")
        elif self._sep_steps * cfg.sim_dt >= cfg.sep_duration_s:
            if self._separation_x > self.head_pin_xy[0] - 0.05:
                self._invalidate("late_release_past_rack")
            elif self.max_contact_fwd_disp < cfg.min_push_disp_m:
                self._invalidate("insufficient_push_displacement_slap")
            else:
                self.phase = DeliveryPhase.RELEASED

    def step_control_reward(self, paddle_pos, action, prev_action, terminated: bool) -> float:
        phi_next = self.compute_potential(paddle_pos)
        delta = np.asarray(action) - np.asarray(prev_action)
        reward = self.config.gamma * phi_next - self._phi_prev
        self._ep_shaping_reward += float(reward)
        if self._contact_bonus_pending:
            if self.phase != DeliveryPhase.INVALID:
                reward += self.config.r_first_contact
                self._ep_contact_reward += float(self.config.r_first_contact)
            self._contact_bonus_pending = False
        self._phi_prev = phi_next
        self._terminated |= terminated
        jerk_cost = self.config.w_action_jerk * np.sum(delta**2)
        self._ep_cost_reward += float(-self.config.w_step_penalty - jerk_cost)
        return float(reward - self.config.w_step_penalty - jerk_cost)

    def should_end_delivery(self, elapsed_delivery_s: float, delivery_timeout_s: float) -> tuple[bool, str | None]:
        if self.phase == DeliveryPhase.INVALID:
            return True, self.invalid_reason
        if self.phase == DeliveryPhase.RELEASED and (self._latest_ball_x >= self.config.release_line_x or self._left_reach):
            return True, "released"
        if elapsed_delivery_s >= delivery_timeout_s:
            return True, "delivery_timeout"
        return False, None

    def is_coast_settled(self, coast_elapsed_s: float) -> tuple[bool, bool]:
        if self.phase == DeliveryPhase.INVALID:
            return True, False
        if self.phase == DeliveryPhase.APPROACH:
            return (coast_elapsed_s >= self.config.quiet_dwell_s), False
        if self.phase == DeliveryPhase.RELEASED and self._quiet_steps * self.config.sim_dt >= self.config.quiet_dwell_s:
            return True, False
        timed_out = coast_elapsed_s >= self.config.max_coast_s
        return timed_out, timed_out

    def finalize_episode(self, final_snap: SubstepSnapshot, unresolved_motion: bool) -> EpisodeOutcome:
        if self._latest_t is None or final_snap.t > self._latest_t:
            self.update_substep(final_snap)
        self._terminated = True
        cfg = self.config
        legal = self.phase == DeliveryPhase.RELEASED
        credited = full = off_table = wobble = 0
        speed = angle = launch_reward = approach_reward = pin_reward = partial_pin_reward = penalty = 0.0
        reason = self.invalid_reason
        if legal:
            d_head = float(np.linalg.norm(np.asarray(final_snap.ball_pos[:2]) - self.head_pin_xy))
            self._min_ball_head_pin_dist = min(self._min_ball_head_pin_dist, d_head)
            speed = float(np.linalg.norm(self.v_out))
            if speed:
                forward = np.dot(self.u_aim, self.v_out)
                lateral = self.u_aim[0] * self.v_out[1] - self.u_aim[1] * self.v_out[0]
                angle = float(np.arctan2(abs(lateral), forward))
            if speed >= cfg.min_launch_speed_mps:
                launch_reward = cfg.w_contact_speed * min(speed / cfg.target_launch_speed_mps, 1.0)
                launch_reward += cfg.w_launch_angle * float(np.exp(-0.5 * (angle / cfg.aim_sigma_rad) ** 2))
            if np.isfinite(self._min_ball_head_pin_dist):
                d_miss = max(0.0, self._min_ball_head_pin_dist - (cfg.ball_radius + cfg.pin_belly_radius))
                approach_reward = cfg.w_ball_approach * float(np.exp(-0.5 * (d_miss / cfg.ball_approach_sigma_m) ** 2))
            off_mask = self.pin_impacted & (final_snap.pin_positions[:, 2] < cfg.off_table_z_m)
            tilt = self._tilts_deg(final_snap.pin_up_vectors)
            full_mask = off_mask | (self.pin_impacted & (tilt > cfg.full_fall_tilt_deg))
            off_table = int(np.count_nonzero(off_mask))
            full = int(np.count_nonzero(full_mask))
            wobble = int(np.count_nonzero(self.pin_wobbled & ~full_mask))
            credited = full
            bins = np.floor(np.clip(self.pin_peak_tilt_deg, 0.0, 45.0) / 5.0)
            partial_pin_reward = float(
                np.sum(np.where(self.pin_impacted & ~full_mask, cfg.w_partial_tilt * (bins / 9.0), 0.0))
            )
            pin_reward = cfg.r_pin_credit * credited + (cfg.r_strike_bonus if credited == 3 else 0.0) + partial_pin_reward
        elif self.phase == DeliveryPhase.INVALID:
            penalty = cfg.r_soft_fault_penalty if reason in SOFT_FAULT_REASONS else cfg.r_illegal_penalty
        else:
            reason = "no_legal_release"
            penalty = cfg.r_timeout_penalty

        return EpisodeOutcome(
            legal_delivery=legal,
            invalid_reason=reason,
            credited_pins=credited,
            full_knock_pins=full,
            off_table_pins=off_table,
            wobble_only_pins=wobble,
            legal_strike=legal and credited == 3,
            launch_speed=speed,
            launch_angle_err_rad=angle,
            contact_forward_disp_m=self.max_contact_fwd_disp,
            unresolved_motion=unresolved_motion,
            launch_reward=launch_reward,
            pin_reward=pin_reward,
            penalty_reward=penalty,
            total_terminal_reward=launch_reward + approach_reward + pin_reward + penalty,
            approach_reward=approach_reward,
            partial_pin_reward=partial_pin_reward,
            min_ball_head_pin_dist=float(self._min_ball_head_pin_dist),
            touched_ball=self._contact_bonus_awarded,
            shaping_reward=float(self._ep_shaping_reward),
            contact_reward=float(self._ep_contact_reward),
            cost_reward=float(self._ep_cost_reward),
        )

    def critic_state(self, remaining_time_frac: float) -> np.ndarray:
        state = np.zeros(12, dtype=np.float32)
        state[self.phase] = 1.0
        state[4:6] = remaining_time_frac, self.max_contact_fwd_disp
        state[6:9] = np.radians(self.pin_peak_tilt_deg)
        state[9:12] = self.pin_impacted
        return state
