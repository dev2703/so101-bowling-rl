"""Dual-camera MuJoCo environment for SO-101 tabletop bowling."""

from collections import deque
from dataclasses import asdict, fields, replace

import cv2
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
from so101_bowling.perception_utils import (
    CameraDelayBuffer,
    CausalJointVelocityEstimator,
    TimestampedFrameStack,
    scale_proprio,
)
from so101_bowling.randomization import DomainRandomizer, augment_rgb
from so101_bowling.scorer import (
    DeliveryPhase,
    DeliveryScorer,
    ScorerConfig,
    SubstepSnapshot,
)


class SO101BowlingEnv(gym.Env):
    """Control joint deltas and observe two timestamped RGB histories."""

    def __init__(
        self,
        cfg: WorkspaceConfig | None = None,
        cal: CalibrationMap | None = None,
        dr_enabled: bool = False,
        render_rgb: bool = True,
        frame_stack: int = 2,
        gamma: float = 0.99,
        scorer_cfg: ScorerConfig | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.cfg = cfg if cfg is not None else load_config()
        self.cal = cal if cal is not None else load_calibration()
        self.dr_enabled = dr_enabled
        self.render_rgb = render_rgb
        self._initial_seed = seed
        self._renderer = None
        self._needs_reset = True
        self._stacks = [TimestampedFrameStack(k=frame_stack) for _ in range(2)]
        self.frame_stack = frame_stack
        cfg = self.cfg
        self.height, self.width = cfg.camera.render_height, cfg.camera.render_width
        self._blank = np.zeros((frame_stack, self.height, self.width, 3), dtype=np.uint8)
        self.model = mujoco.MjModel.from_xml_string(build_scene_xml(cfg))
        self.data = mujoco.MjData(self.model)
        self.ids = ids = get_scene_entity_ids(self.model)
        self.randomizer = DomainRandomizer(self.model, ids, cfg)
        self._scene_option = mujoco.MjvOption()
        self._scene_option.geomgroup[3] = 0
        self._camera_ids = (ids.wrist_cam_id, ids.tripod_cam_id)
        self._camera_names = ("wrist_cam", "tripod_cam")
        limits = intersect_joint_limits(self.cal)
        self.q_min, self.q_max = np.asarray([limits[name] for name in ACTIVE_JOINTS]).T
        self.max_action_rad = cfg.robot.max_action_rad
        self._home = np.asarray([cfg.robot.home_qpos[name] for name in ALL_JOINTS])
        self._held_home = np.asarray([cfg.robot.home_qpos[name] for name in HELD_JOINTS])
        self._all_qpos_addrs = self.model.jnt_qposadr[ids.all_joint_ids]
        self._arm_geoms = np.array(sorted(ids.arm_geom_ids), dtype=np.intp)
        self._arm_aabb = self.model.geom_aabb[self._arm_geoms].reshape(-1, 2, 3)
        self._table_geoms = ids.desk_geom_ids | ids.pad_geom_ids
        self._pin_by_geom = {
            geom: pin for pin, geoms in enumerate(ids.pin_geom_ids) for geom in geoms
        }
        self._pin_vel = np.asarray(ids.pin_joint_qvel_addrs)[:, None] + np.arange(6)
        self._ft = np.zeros(6)
        self._velocity = CausalJointVelocityEstimator()

        self.substeps = round(1.0 / (cfg.robot.control_hz * cfg.robot.physics_dt))
        self._control_dt = self.substeps * cfg.robot.physics_dt
        self.delivery_timeout_s = getattr(
            cfg.delivery, "delivery_timeout_s", cfg.delivery.max_steps * self._control_dt
        )
        if self.substeps < 1 or self.delivery_timeout_s <= 0 or self.max_action_rad <= 0:
            raise ValueError("Control interval, delivery timeout and action scale must be positive")
        if scorer_cfg is None:
            defaults = ScorerConfig()
            overrides = {
                field.name: getattr(
                    cfg.delivery, field.name,
                    getattr(cfg.reward, field.name, getattr(defaults, field.name)),
                )
                for field in fields(ScorerConfig)
            }
            overrides.update(
                gamma=gamma,
                ball_radius=cfg.ball.radius,
                rack_x_nom=cfg.pin.positions[0][0],
                min_wobble_substeps=getattr(
                    cfg.delivery, "min_wobble_substeps",
                    max(2, round(0.02 / cfg.robot.physics_dt)),
                ),
            )
            scorer_cfg = ScorerConfig(**overrides)
        self.scorer = DeliveryScorer(replace(scorer_cfg, sim_dt=cfg.robot.physics_dt))

        self.action_space = spaces.Box(-1.0, 1.0, shape=(len(ACTIVE_JOINTS),), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "wrist_rgb": spaces.Box(0, 255, shape=self._blank.shape, dtype=np.uint8),
            "tripod_rgb": spaces.Box(0, 255, shape=self._blank.shape, dtype=np.uint8),
            "camera_meta": spaces.Box(-10.0, 10.0, shape=(4,), dtype=np.float32),
            "proprio": spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32),
            "privileged_state": spaces.Box(-np.inf, np.inf, shape=(46,), dtype=np.float32),
            "privileged_aux_targets": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
            "privileged_aux_mask": spaces.Box(0.0, 1.0, shape=(3,), dtype=np.float32),
        })

    def reset(self, *, seed=None, options=None):
        super().reset(seed=self._initial_seed if seed is None else seed)
        self._initial_seed = None
        rng = self.np_random
        mujoco.mj_resetData(self.model, self.data)
        dr_info = self.randomizer.apply_episode(self.data, rng, self.dr_enabled)

        home = self._home.copy()
        n_active = len(ACTIVE_JOINTS)
        if self.dr_enabled:
            home[:n_active] = np.clip(
                home[:n_active] + rng.uniform(-0.02, 0.02, n_active), self.q_min, self.q_max
            )
        self.data.qpos[self._all_qpos_addrs] = home
        self._hold(home[:n_active])
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data, nstep=20)
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._last_snapshot = snap = self._snapshot()
        self.scorer.reset(
            snap.ball_pos, snap.pin_positions, snap.pin_up_vectors, paddle_init_pos=snap.paddle_pos
        )
        self._step_count = 0
        self._prev_delta = np.zeros(n_active)
        self._action_lag = int(rng.integers(3)) if self.dr_enabled else 0
        self._commands = deque(self._measured_q() for _ in range(self._action_lag))
        self._velocity.reset()
        self._delays = [
            CameraDelayBuffer(
                delay_frames=int(rng.integers(3)) if self.dr_enabled else 0,
                drop_probability=0.03 if self.dr_enabled else 0.0,
                rng=rng,
            )
            for _ in self._camera_names
        ]
        max_shift = self.cfg.dr.max_shift_px if self.dr_enabled else 0
        self._shift = tuple(rng.integers(-max_shift, max_shift + 1, size=2)) if max_shift else (0, 0)
        for stack in self._stacks:
            stack.reset()
        obs = self._observation(initial=True)
        self._needs_reset = False
        return obs, {"randomization": dr_info}

    def _measured_q(self):
        return self.data.qpos[self.ids.active_qpos_addrs]

    def _hold(self, target):
        self.data.ctrl[self.ids.active_actuator_ids] = target
        self.data.ctrl[self.ids.held_actuator_ids] = self._held_home

    def step(self, action):
        if self._needs_reset:
            raise gym.error.ResetNeeded("Call reset() before stepping a new episode")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError("Expected four finite action values")
        q_meas = self._measured_q()
        scale = self.max_action_rad
        q_cmd = np.clip(q_meas + np.clip(action, -1.0, 1.0) * scale, self.q_min, self.q_max)
        self._commands.append(q_cmd)
        target = self._commands.popleft()
        target = np.clip(np.clip(target, q_meas - scale, q_meas + scale), self.q_min, self.q_max)
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
        info = {}
        if done:
            unresolved = False
            if self.scorer.phase != DeliveryPhase.INVALID:
                self._hold(self._measured_q())
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
            info.update(outcome=asdict(outcome), end_reason=reason)
        self._last_snapshot = snap
        self._needs_reset = done
        return self._observation(), reward, done, False, info

    def _advance(self):
        mujoco.mj_step(self.model, self.data)
        # Refresh derived transforms and contact forces at the integrated state.
        mujoco.mj_forward(self.model, self.data)
        snap = self._snapshot()
        self.scorer.update_substep(snap)
        return snap

    def _snapshot(self):
        data, ids = self.data, self.ids
        paddle = data.site_xpos[ids.gripper_site_id]
        # Project each compiled local AABB onto world X. This bounds the mesh
        # without extending a bounding sphere equally in every direction.
        x_axes = data.geom_xmat[self._arm_geoms, :3]
        centers = self._arm_aabb[:, 0]
        extents = self._arm_aabb[:, 1]
        max_x = data.geom_xpos[self._arm_geoms, 0] + np.sum(
            x_axes * centers + np.abs(x_axes) * extents, axis=1
        )

        pusher_ball = other_ball = arm_pin = table_slam = False
        ball_pin = [False] * 3
        pin_pin = np.zeros((3, 3), dtype=bool)
        for index, contact in enumerate(data.contact):
            g1, g2 = contact.geom
            if ids.ball_geom_id in (g1, g2):
                other = g2 if g1 == ids.ball_geom_id else g1
                pusher_ball |= other in ids.pusher_geom_ids
                other_ball |= other in ids.arm_geom_ids and other not in ids.pusher_geom_ids
                pin = self._pin_by_geom.get(other)
                if pin is not None:
                    ball_pin[pin] = True
                continue
            if g2 in ids.arm_geom_ids:
                g1, g2 = g2, g1
            pin1, pin2 = self._pin_by_geom.get(g1), self._pin_by_geom.get(g2)
            if g1 in ids.arm_geom_ids:
                arm_pin |= pin2 is not None
                if g2 in self._table_geoms:
                    mujoco.mj_contactForce(self.model, data, index, self._ft)
                    table_slam |= bool(self._ft[0] > 8.0)
            elif pin1 is not None and pin2 is not None:
                pin_pin[pin1, pin2] = pin_pin[pin2, pin1] = True

        return SubstepSnapshot(
            t=data.time,
            paddle_pos=paddle.copy(),
            ball_pos=data.xpos[ids.ball_body_id].copy(),
            ball_vel=data.qvel[ids.ball_joint_qvel_addr:ids.ball_joint_qvel_addr + 3].copy(),
            pin_positions=data.xpos[ids.pin_body_ids],
            pin_up_vectors=data.xmat[ids.pin_body_ids, 2::3],
            pin_speeds=np.linalg.norm(data.qvel[self._pin_vel], axis=1),
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

    def _visible(self, point, body_id):
        for cam_id in self._camera_ids:
            origin = self.data.cam_xpos[cam_id]
            ray = point - origin
            x, y, z = self.data.cam_xmat[cam_id].reshape(3, 3).T @ ray
            if z >= 0:
                continue
            half_height = -z * np.tan(np.deg2rad(self.model.cam_fovy[cam_id]) / 2)
            half_width = half_height * self.width / self.height
            if (
                abs(x / half_width + 2 * self._shift[0] / self.width) > 1
                or abs(y / half_height - 2 * self._shift[1] / self.height) > 1
            ):
                continue
            hit = mujoco.mj_ray(
                self.model, self.data, origin, ray, self._scene_option.geomgroup,
                True, body_id, None,
            )
            if hit < 0 or hit >= 1.0:
                return True
        return False

    def _observation(self, initial=False):
        snap, ids = self._last_snapshot, self.ids
        t = self.data.time
        obs = {}
        meta = []
        for name, cam, stack, delay in zip(
            ("wrist_rgb", "tripod_rgb"), self._camera_names, self._stacks, self._delays
        ):
            if self.render_rgb:
                rgb = self.render_camera(cam, self.height, self.width)
                if self.dr_enabled:
                    rgb = augment_rgb(rgb, self.np_random, True, self._shift)
                delivered = delay.push(rgb, t)
                if initial:
                    stack.push(rgb, t)
                elif delivered is not None:
                    stack.push(*delivered)
                frames = stack.frames
                obs[name] = np.stack((frames[0],) * (self.frame_stack - len(frames)) + frames)
                meta.extend((min(10.0, stack.age_s(t)), float(
                    (initial or delivered is not None) and stack.is_fresh(t, self._control_dt * 2.01)
                )))
            else:
                obs[name] = self._blank.copy()
                meta.extend((10.0, 0.0))
        q_meas = self._measured_q()
        obs["camera_meta"] = np.asarray(meta, dtype=np.float32)
        obs["proprio"] = scale_proprio(
            q_meas, self._velocity.update(t, q_meas), self._prev_delta,
            self.q_min, self.q_max, self.max_action_rad,
        )
        ball_xy = snap.ball_pos[:2]
        rack_xy = snap.pin_positions[:, :2].mean(axis=0)
        remaining = max(0.0, 1.0 - self._step_count * self._control_dt / self.delivery_timeout_s)
        obs["privileged_state"] = np.concatenate((
            snap.ball_pos, snap.ball_vel, snap.pin_positions.ravel(),
            self.data.xquat[ids.pin_body_ids].ravel(), snap.paddle_pos, rack_xy - ball_xy,
            self.scorer.critic_state(remaining),
            (snap.pusher_ball_contact, snap.other_arm_ball_contact),
        )).astype(np.float32)
        obs["privileged_aux_targets"] = np.concatenate(
            (ball_xy, rack_xy, ball_xy - snap.paddle_pos[:2])
        ).astype(np.float32)
        ball_visible = rack_visible = relative_visible = False
        if self.render_rgb:
            ball_visible = self._visible(snap.ball_pos, ids.ball_body_id)
            rack_visible = any(
                self._visible(self.data.xipos[body_id], body_id) for body_id in ids.pin_body_ids
            )
            paddle_body = int(self.model.site_bodyid[ids.gripper_site_id])
            relative_visible = ball_visible and self._visible(snap.paddle_pos, paddle_body)
        obs["privileged_aux_mask"] = np.asarray(
            (ball_visible, rack_visible, relative_visible), dtype=np.float32
        )
        return obs

    def render_camera(self, cam_name: str, height: int, width: int) -> np.ndarray:
        """Render a named camera, resizing diagnostic output from the native frame."""
        if height < 1 or width < 1:
            raise ValueError("Camera output dimensions must be positive")
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        self._renderer.update_scene(self.data, camera=cam_name, scene_option=self._scene_option)
        rgb = self._renderer.render()
        if (height, width) != (self.height, self.width):
            rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        return rgb

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
