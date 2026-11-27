"""CI-safe mocked tests and hardware integration test for the SO-101 hardware bridge."""

from __future__ import annotations

from dataclasses import replace
import os
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import mujoco
import numpy as np
import pytest

from so101_bowling.config import (
    ACTIVE_JOINTS,
    ALL_JOINTS,
    HELD_JOINTS,
    hw_to_mjcf_rad,
    load_calibration,
    load_config,
    mjcf_rad_to_hw,
)
from so101_bowling.hardware import (
    HardwareConfig,
    OpenCVTimestampedCamera,
    SO101BowlingHardwareBridge,
    SafetyGuard,
    StepTrace,
    TimestampedCamera,
    TimestampedFrame,
    VideoFileTimestampedCamera,
    create_live_follower,
    load_hardware_config,
    rotate_frame,
)
from so101_bowling.perception_utils import preprocess_camera_frame, scale_proprio
from so101_bowling.ppo_models import ACTOR_OBS_KEYS, DualCameraActor


class FakeClock:
    """Deterministic controllable clock for timing and overrun tests."""

    def __init__(self, initial_t: float = 10.0, auto_step: float = 0.0) -> None:
        self.t = float(initial_t)
        self.auto_step = float(auto_step)

    def __call__(self) -> float:
        val = self.t
        self.t += self.auto_step
        return val

    def advance(self, dt: float) -> float:
        self.t += dt
        return self.t


class MockCamera(TimestampedCamera):
    """Controllable TimestampedCamera returning scripted frames or latest pushed frame."""

    def __init__(self, frame: TimestampedFrame | None = None) -> None:
        self.frame = frame
        self.opened = False
        self.closed = False

    def open(self) -> None:
        self.opened = True

    def read_latest(self) -> TimestampedFrame | None:
        return self.frame

    def close(self) -> None:
        self.closed = True


class SpyBus:
    def __init__(self, raise_on_disable: bool = False) -> None:
        self.enable_torque_calls = 0
        self.disable_torque_calls = 0
        self.raise_on_disable = raise_on_disable

    def enable_torque(self) -> None:
        self.enable_torque_calls += 1

    def disable_torque(self) -> None:
        self.disable_torque_calls += 1
        if self.raise_on_disable:
            raise RuntimeError("serial port disconnected")


class SpyRobot:
    """Spy stand-in for LeRobot SO101Follower."""

    def __init__(
        self,
        cal: Any,
        q_rad: dict[str, float],
        use_degrees: bool = True,
        max_relative_target: float | None = None,
        calibration_keys: tuple[str, ...] = ALL_JOINTS,
        calibration_overrides: dict[str, dict[str, Any]] | None = None,
        is_calibrated: bool = True,
        raise_on_disable: bool = False,
        obs_hook: Any = None,
        on_disconnect: Any = None,
        send_clip_offset_deg: float = 0.0,
    ) -> None:
        self.config = SimpleNamespace(
            use_degrees=use_degrees,
            max_relative_target=max_relative_target,
        )
        self.calibration: dict[str, Any] = {}
        overrides = calibration_overrides or {}
        for k in calibration_keys:
            jc = cal.joints[k]
            fields = {
                "id": jc.id,
                "drive_mode": jc.drive_mode,
                "homing_offset": jc.homing_offset,
                "range_min": jc.range_min,
                "range_max": jc.range_max,
            }
            fields.update(overrides.get(k, {}))
            self.calibration[k] = SimpleNamespace(**fields)
        self.is_calibrated = is_calibrated
        self.bus = SpyBus(raise_on_disable=raise_on_disable)
        self.cal = cal
        self.q_rad = dict(q_rad)
        self.is_connected = False
        self.connect_calls = 0
        self.connect_calibrate_args: list[bool] = []
        self.disconnect_calls = 0
        self.get_observation_calls = 0
        self.sent_commands: list[dict[str, float]] = []
        self._obs_hook = obs_hook
        self._on_disconnect = on_disconnect
        self._send_clip_offset_deg = send_clip_offset_deg

    def connect(self, calibrate: bool = True) -> None:
        self.connect_calls += 1
        self.connect_calibrate_args.append(calibrate)
        self.is_connected = True

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        if self._on_disconnect is not None:
            self._on_disconnect()
        self.is_connected = False

    def get_observation(self) -> dict[str, float]:
        self.get_observation_calls += 1
        if self._obs_hook is not None:
            self._obs_hook()
        return {
            f"{j}.pos": mjcf_rad_to_hw(j, self.q_rad[j], self.cal) for j in ALL_JOINTS
        }

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.sent_commands.append(dict(action))
        sent = {k: float(v) + self._send_clip_offset_deg for k, v in action.items()}
        for j in ALL_JOINTS:
            key = f"{j}.pos"
            if key in sent:
                self.q_rad[j] = hw_to_mjcf_rad(j, sent[key], self.cal)
        return sent


class ScriptedActor:
    """Actor stub recording received observations and returning a scripted action."""

    def __init__(
        self,
        action: np.ndarray | None = None,
        frame_stack: int = 2,
        on_act: Any = None,
    ) -> None:
        self.frame_stack = frame_stack
        self._action = (
            np.asarray(action, dtype=np.float32)
            if action is not None
            else np.array([0.25, 0.1, -0.1, 0.05], dtype=np.float32)
        )
        self._on_act = on_act
        self.seen_obs: list[dict[str, Any]] = []

    def act(self, obs: dict[str, Any], deterministic: bool = False) -> np.ndarray:
        assert deterministic is True
        self.seen_obs.append(dict(obs))
        if self._on_act is not None:
            self._on_act()
        return self._action.copy()


def _make_rgb(h: int = 720, w: int = 1280, value: int = 120) -> np.ndarray:
    img = np.full((h, w, 3), value, dtype=np.uint8)
    img[: h // 4, : w // 4, 0] = 240
    img[-h // 4 :, -w // 4 :, 2] = 200
    return img


def test_hardware_config_defaults_and_lerobot_lazy_import() -> None:
    assert "lerobot" not in sys.modules
    hw_cfg = load_hardware_config()
    assert hw_cfg.dry_run is True
    assert hw_cfg.port == "/dev/tty.usbmodem0000"
    assert hw_cfg.wrist_cam_index == 0
    assert hw_cfg.tripod_cam_index == 2
    assert hw_cfg.wrist_rotation_deg == 180
    assert hw_cfg.tripod_rotation_deg == 0
    assert hw_cfg.z_floor_margin_m == pytest.approx(0.015)
    assert hw_cfg.foul_line_x == pytest.approx(0.44)
    assert hw_cfg.stopping_margin_rad == pytest.approx(0.015)
    assert hw_cfg.max_joint_stale_s == pytest.approx(0.15)
    assert hw_cfg.max_camera_age_s == pytest.approx(0.20)
    assert hw_cfg.max_joint_read_latency_s == pytest.approx(0.05)

    with pytest.raises(ValueError):
        HardwareConfig(wrist_rotation_deg=45)

    for bad_field in (
        "control_hz",
        "z_floor_margin_m",
        "foul_line_x",
        "stopping_margin_rad",
        "max_joint_stale_s",
        "max_camera_age_s",
        "max_joint_read_latency_s",
    ):
        for non_finite in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(ValueError, match=bad_field):
                HardwareConfig(**{bad_field: non_finite})


def test_safety_guard_rejects_non_finite_bypassed_thresholds() -> None:
    wcfg = load_config()
    cal = load_calibration()
    home_6 = np.asarray([wcfg.robot.home_qpos[j] for j in ALL_JOINTS], dtype=np.float64)

    hw_cfg_z = HardwareConfig(dry_run=True)
    guard_z = SafetyGuard(wcfg, cal, hw_cfg_z)
    object.__setattr__(hw_cfg_z, "z_floor_margin_m", float("nan"))
    ok_z, reason_z, _, _ = guard_z.check_pose(home_6)
    assert ok_z is False
    assert reason_z == "z_floor_violation"

    hw_cfg_x = HardwareConfig(dry_run=True)
    guard_x = SafetyGuard(wcfg, cal, hw_cfg_x)
    object.__setattr__(hw_cfg_x, "foul_line_x", float("nan"))
    ok_x, reason_x, _, _ = guard_x.check_pose(home_6)
    assert ok_x is False
    assert reason_x == "foul_line_violation"


def test_dry_run_spy_robot_never_connects_writes_or_enables_torque() -> None:
    wcfg = load_config()
    cal = load_calibration()
    hw_cfg = HardwareConfig(dry_run=True)
    clock = FakeClock(initial_t=1.0, auto_step=0.005)
    spy = SpyRobot(cal=cal, q_rad=wcfg.robot.home_qpos)
    wrist_cam = MockCamera(TimestampedFrame(_make_rgb(), capture_t=1.0, frame_seq=1))
    tripod_cam = MockCamera(TimestampedFrame(_make_rgb(), capture_t=1.0, frame_seq=1))
    actor = ScriptedActor()

    bridge = SO101BowlingHardwareBridge(
        actor=actor,
        workspace_cfg=wcfg,
        cal=cal,
        hw_cfg=hw_cfg,
        wrist_cam=wrist_cam,
        tripod_cam=tripod_cam,
        robot=spy,
        clock=clock,
    )
    bridge.connect_robot()
    assert spy.connect_calls == 0

    for seq in range(2, 5):
        t = clock.t
        wrist_cam.frame = TimestampedFrame(_make_rgb(), capture_t=t, frame_seq=seq)
        tripod_cam.frame = TimestampedFrame(_make_rgb(), capture_t=t, frame_seq=seq)
        trace = bridge.step()
        assert isinstance(trace, StepTrace)
        assert not trace.aborted
        np.testing.assert_allclose(trace.applied_delta_rad, 0.0)

    stop_info = bridge.emergency_stop("manual_test")
    assert stop_info["estopped"] is True
    bridge.close()

    assert spy.connect_calls == 0
    assert spy.disconnect_calls == 0
    assert spy.sent_commands == []
    assert spy.bus.enable_torque_calls == 0
    assert spy.bus.disable_torque_calls == 0
    assert wrist_cam.closed and tripod_cam.closed

    # Dry-run also supports robot=None using synthetic home joints.
    bridge_none = SO101BowlingHardwareBridge(
        actor=actor,
        workspace_cfg=wcfg,
        cal=cal,
        hw_cfg=hw_cfg,
        wrist_cam=MockCamera(TimestampedFrame(_make_rgb(), capture_t=clock.t, frame_seq=1)),
        tripod_cam=MockCamera(TimestampedFrame(_make_rgb(), capture_t=clock.t, frame_seq=1)),
        robot=None,
        clock=clock,
    )
    bridge_none.connect_robot()
    trace_none = bridge_none.step()
    assert not trace_none.aborted
    home_vec = np.asarray([wcfg.robot.home_qpos[j] for j in ALL_JOINTS])
    np.testing.assert_allclose(trace_none.q_rad, home_vec)


def test_dry_run_closed_loop_evolves_state_and_rejects_non_bool(tmp_path: Path) -> None:
    assert HardwareConfig(dry_run=True).dry_run_closed_loop is False
    assert load_hardware_config().dry_run_closed_loop is True

    for bad_val in (1, "true", None):
        with pytest.raises(ValueError, match="dry_run_closed_loop"):
            HardwareConfig(dry_run_closed_loop=bad_val)  # type: ignore[arg-type]

    for bad_yaml in ('dry_run_closed_loop: "false"\n', "dry_run_closed_loop: 2\n"):
        cfg_file = tmp_path / "bad_hw.yaml"
        cfg_file.write_text(bad_yaml)
        with pytest.raises(ValueError, match="dry_run_closed_loop"):
            load_hardware_config(cfg_file)

    wcfg = load_config()
    cal = load_calibration()
    hw_cfg = HardwareConfig(dry_run=True, dry_run_closed_loop=True)
    clock = FakeClock(initial_t=1.0, auto_step=0.005)
    spy = SpyRobot(cal=cal, q_rad=wcfg.robot.home_qpos)
    wrist_cam = MockCamera(TimestampedFrame(_make_rgb(), capture_t=1.0, frame_seq=1))
    tripod_cam = MockCamera(TimestampedFrame(_make_rgb(), capture_t=1.0, frame_seq=1))
    actor = ScriptedActor(action=np.array([0.25, 0.1, -0.1, 0.05], dtype=np.float32))
    bridge = SO101BowlingHardwareBridge(
        actor=actor,
        workspace_cfg=wcfg,
        cal=cal,
        hw_cfg=hw_cfg,
        wrist_cam=wrist_cam,
        tripod_cam=tripod_cam,
        robot=spy,
        clock=clock,
    )
    bridge.connect_robot()
    home_6 = np.asarray([wcfg.robot.home_qpos[j] for j in ALL_JOINTS], dtype=np.float64)

    t1 = bridge.step()
    assert not t1.aborted
    np.testing.assert_allclose(t1.q_rad, home_6)
    assert np.linalg.norm(t1.applied_delta_rad) > 0.0
    np.testing.assert_allclose(actor.seen_obs[0]["proprio"][4:8], 0.0)

    clock.advance(0.05)
    wrist_cam.frame = TimestampedFrame(_make_rgb(), capture_t=clock.t, frame_seq=2)
    tripod_cam.frame = TimestampedFrame(_make_rgb(), capture_t=clock.t, frame_seq=2)
    t2 = bridge.step()
    assert not t2.aborted
    np.testing.assert_allclose(t2.q_rad[:4], home_6[:4] + t1.applied_delta_rad)
    np.testing.assert_allclose(t2.q_rad[4:], home_6[4:])
    assert np.linalg.norm(t2.applied_delta_rad) > 0.0
    assert np.linalg.norm(actor.seen_obs[1]["proprio"][4:8]) > 0.0

    bridge.reset()
    np.testing.assert_allclose(bridge._dry_run_q_6, home_6)
    clock.advance(0.05)
    wrist_cam.frame = TimestampedFrame(_make_rgb(), capture_t=clock.t, frame_seq=3)
    tripod_cam.frame = TimestampedFrame(_make_rgb(), capture_t=clock.t, frame_seq=3)
    t3 = bridge.step()
    assert not t3.aborted
    np.testing.assert_allclose(t3.q_rad, home_6)
    assert not np.allclose(bridge._dry_run_q_6, home_6)

    assert bridge.move_to_home() == {"homed": True, "dry_run": True, "steps": 0, "max_err_rad": 0.0}
    np.testing.assert_allclose(bridge._dry_run_q_6, home_6)

    assert spy.connect_calls == 0
    assert spy.get_observation_calls == 0
    assert spy.sent_commands == []
    assert spy.bus.enable_torque_calls == 0
    assert spy.bus.disable_torque_calls == 0

    object.__setattr__(bridge.hw_cfg, "dry_run_closed_loop", 1)
    assert bridge.step().abort_reason == "invalid_hardware_config"
    assert bridge.estopped


def test_safety_guard_rejects_unsafe_jaw_only_foul_line_and_16_corner_lag() -> None:
    wcfg = load_config()
    cal = load_calibration()
    hw_cfg = HardwareConfig(dry_run=True)
    guard = SafetyGuard(wcfg, cal, hw_cfg)

    # 1. Home pose is safe.
    home_6 = np.asarray([wcfg.robot.home_qpos[j] for j in ALL_JOINTS], dtype=np.float64)
    ok_home, reason_home, min_z_home, max_x_home = guard.check_pose(home_6)
    assert ok_home and reason_home is None
    assert min_z_home > hw_cfg.z_floor_margin_m
    assert max_x_home < hw_cfg.foul_line_x

    res_home = guard.guard_command(home_6, home_6[:4] + np.array([0.02, 0.0, 0.0, 0.0]))
    assert res_home.allowed is True
    assert res_home.reason is None

    # 2. Already-unsafe measured pose rejected with "unsafe_measured_pose:*".
    # Jaw-only Z-floor violation: moving_jaw_so101_v1 dips below z_floor_margin_m
    # while all non-jaw geoms and gripperframe stay strictly above z_floor_margin_m.
    q_jaw_only = np.array([-0.5987, -0.7884, 1.5098, -0.1819, 2.7125, 0.8146])
    jaw_bid = mujoco.mj_name2id(guard.model, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")
    guard.data.qpos[guard._all_qpos_addrs] = q_jaw_only
    mujoco.mj_kinematics(guard.model, guard.data)
    z_axes = guard.data.geom_xmat[guard._z_floor_geoms, 6:9]
    geom_min_z = guard.data.geom_xpos[guard._z_floor_geoms, 2] + np.sum(
        z_axes * guard._z_floor_aabb[:, 0] - np.abs(z_axes) * guard._z_floor_aabb[:, 1],
        axis=1,
    )
    jaw_mask = np.array([int(guard.model.geom_bodyid[g]) == jaw_bid for g in guard._z_floor_geoms])
    assert geom_min_z[jaw_mask].min() < hw_cfg.z_floor_margin_m
    assert geom_min_z[~jaw_mask].min() > hw_cfg.z_floor_margin_m
    assert guard.data.site_xpos[guard.ids.gripper_site_id, 2] > hw_cfg.z_floor_margin_m

    ok_jaw, reason_jaw, _, _ = guard.check_pose(q_jaw_only)
    assert not ok_jaw and reason_jaw == "z_floor_violation"
    res_jaw = guard.guard_command(q_jaw_only, q_jaw_only[:4])
    assert not res_jaw.allowed
    assert res_jaw.reason == "unsafe_measured_pose:z_floor_violation"
    np.testing.assert_allclose(res_jaw.target_active_4_rad, q_jaw_only[:4])

    # 3. Foul-line violation when reaching past foul_line_x = 0.44 m.
    q_foul = np.array([0.0, 0.75, -0.80, 0.0, 0.0, 0.0])
    ok_foul, reason_foul, _, max_x_foul = guard.check_pose(q_foul)
    assert not ok_foul and reason_foul == "foul_line_violation"
    assert max_x_foul > hw_cfg.foul_line_x
    res_foul = guard.guard_command(q_foul, q_foul[:4])
    assert not res_foul.allowed
    assert res_foul.reason == "unsafe_measured_pose:foul_line_violation"

    # 4. Safe endpoints whose intermediate 16-corner servo-lag pose crosses Z-floor!
    q0_corner = np.array([1.718, -1.6732, -1.1309, -0.9391, -2.2448, 0.7327])
    q_prop_corner = np.array([1.7328, -1.7023, -1.1935, -0.8893])
    ok_start, _, _, _ = guard.check_pose(q0_corner)
    assert ok_start
    delta = np.clip(q_prop_corner - q0_corner[:4], -guard.max_action_rad, guard.max_action_rad)
    q_clamped = np.clip(q0_corner[:4] + delta, guard.q_min, guard.q_max)
    pad = np.where(np.abs(q_clamped - q0_corner[:4]) > 1e-6, np.sign(q_clamped - q0_corner[:4]) * hw_cfg.stopping_margin_rad, 0.0)
    q_padded_end = q0_corner.copy()
    q_padded_end[:4] = np.clip(q_clamped + pad, guard.q_min, guard.q_max)
    ok_end, _, _, _ = guard.check_pose(q_padded_end)
    assert ok_end

    res_corner = guard.guard_command(q0_corner, q_prop_corner)
    assert not res_corner.allowed
    assert res_corner.reason == "z_floor_violation"
    np.testing.assert_allclose(res_corner.target_active_4_rad, q0_corner[:4])


def test_active_4_command_conversion_validation_and_proprio_parity() -> None:
    wcfg = load_config()
    cal = load_calibration()
    hw_cfg = HardwareConfig(dry_run=False, calibration_path="/tmp/aligned_follower.json")
    clock = FakeClock(initial_t=5.0, auto_step=0.005)

    # Reject bad robot config (use_degrees=False, max_relative_target not None, or mismatched cal).
    for kwargs in (
        {"use_degrees": False},
        {"max_relative_target": 5.0},
        {"calibration_keys": ACTIVE_JOINTS},
    ):
        bad_robot = SpyRobot(cal=cal, q_rad=wcfg.robot.home_qpos, **kwargs)
        bad_bridge = SO101BowlingHardwareBridge(
            actor=ScriptedActor(),
            workspace_cfg=wcfg,
            cal=cal,
            hw_cfg=hw_cfg,
            wrist_cam=MockCamera(),
            tripod_cam=MockCamera(),
            robot=bad_robot,
            clock=clock,
        )
        with pytest.raises(ValueError):
            bad_bridge.connect_robot()

    # Valid live robot with non-zero sent offset to verify prev_applied_delta = sent_rad - q_meas.
    spy = SpyRobot(cal=cal, q_rad=wcfg.robot.home_qpos, send_clip_offset_deg=-0.5)
    wrist_cam = MockCamera(TimestampedFrame(_make_rgb(), capture_t=clock.t, frame_seq=1))
    tripod_cam = MockCamera(TimestampedFrame(_make_rgb(), capture_t=clock.t, frame_seq=1))
    actor = ScriptedActor(action=np.array([0.5, 0.25, -0.25, 0.1], dtype=np.float32))
    bridge = SO101BowlingHardwareBridge(
        actor=actor,
        workspace_cfg=wcfg,
        cal=cal,
        hw_cfg=hw_cfg,
        wrist_cam=wrist_cam,
        tripod_cam=tripod_cam,
        robot=spy,
        clock=clock,
    )
    bridge.connect_robot()
    assert spy.connect_calls == 1
    assert spy.connect_calibrate_args == [False]

    q_before = np.asarray([wcfg.robot.home_qpos[j] for j in ACTIVE_JOINTS], dtype=np.float64)
    trace1 = bridge.step()
    assert not trace1.aborted
    assert trace1.torque_disabled_confirmed is None
    assert len(spy.sent_commands) == 1
    cmd = spy.sent_commands[0]
    assert set(cmd.keys()) == {f"{j}.pos" for j in ALL_JOINTS}
    for j in HELD_JOINTS:
        assert cmd[f"{j}.pos"] == pytest.approx(
            mjcf_rad_to_hw(j, wcfg.robot.home_qpos[j], cal)
        )

    # Verify degrees conversion (no double homing_offset) and prev_applied_delta parity.
    expected_target = q_before + np.array([0.5, 0.25, -0.25, 0.1]) * wcfg.robot.max_action_rad
    for i, j in enumerate(ACTIVE_JOINTS):
        assert cmd[f"{j}.pos"] == pytest.approx(mjcf_rad_to_hw(j, expected_target[i], cal))
    expected_sent_rad = np.asarray(
        [hw_to_mjcf_rad(j, cmd[f"{j}.pos"] - 0.5, cal) for j in ACTIVE_JOINTS]
    )
    expected_delta = expected_sent_rad - q_before
    np.testing.assert_allclose(trace1.applied_delta_rad, expected_delta, atol=1e-7)

    # Step 2: check actor receives proprio whose last 4 entries equal prev_applied_delta / max_action_rad.
    wrist_cam.frame = TimestampedFrame(_make_rgb(), capture_t=clock.t, frame_seq=2)
    tripod_cam.frame = TimestampedFrame(_make_rgb(), capture_t=clock.t, frame_seq=2)
    trace2 = bridge.step()
    assert not trace2.aborted
    prop2 = actor.seen_obs[1]["proprio"]
    expected_tail = np.clip(expected_delta / wcfg.robot.max_action_rad, -1.0, 1.0).astype(np.float32)
    np.testing.assert_allclose(prop2[8:12], expected_tail, atol=1e-6)
    assert tuple(sorted(actor.seen_obs[1].keys())) == tuple(sorted(ACTOR_OBS_KEYS))


def test_static_scene_freshness_stalled_camera_and_inference_overrun_abort() -> None:
    wcfg = load_config()
    cal = load_calibration()
    hw_cfg = HardwareConfig(
        dry_run=False,
        calibration_path="/tmp/aligned_follower.json",
        max_camera_age_s=0.20,
        max_joint_stale_s=0.15,
        max_joint_read_latency_s=0.05,
    )
    clock = FakeClock(initial_t=20.0, auto_step=0.002)
    spy = SpyRobot(cal=cal, q_rad=wcfg.robot.home_qpos)
    static_rgb = _make_rgb(value=90)
    wrist_cam = MockCamera(TimestampedFrame(static_rgb.copy(), capture_t=clock.t, frame_seq=1))
    tripod_cam = MockCamera(TimestampedFrame(static_rgb.copy(), capture_t=clock.t, frame_seq=1))
    actor = ScriptedActor()
    bridge = SO101BowlingHardwareBridge(
        actor=actor,
        workspace_cfg=wcfg,
        cal=cal,
        hw_cfg=hw_cfg,
        wrist_cam=wrist_cam,
        tripod_cam=tripod_cam,
        robot=spy,
        clock=clock,
    )
    bridge.connect_robot()

    # Pixel-identical frames with advancing timestamps remain fresh across steps.
    for seq in range(1, 4):
        wrist_cam.frame = TimestampedFrame(static_rgb.copy(), capture_t=clock.t, frame_seq=seq)
        tripod_cam.frame = TimestampedFrame(static_rgb.copy(), capture_t=clock.t, frame_seq=seq)
        clock.advance(0.03)
        trace = bridge.step()
        assert not trace.aborted
    assert len(spy.sent_commands) == 3

    # Repeated camera frame within max_camera_age_s emits fresh=0.0 without latching E-stop.
    wrist_cam.frame = TimestampedFrame(static_rgb.copy(), capture_t=clock.t, frame_seq=4)
    clock.advance(0.03)
    repeated_trace = bridge.step()
    assert not repeated_trace.aborted
    assert not bridge.estopped
    assert len(spy.sent_commands) == 4
    assert actor.seen_obs[-1]["camera_meta"][1] == pytest.approx(1.0)
    assert actor.seen_obs[-1]["camera_meta"][3] == pytest.approx(0.0)

    # Stalled single camera past max_camera_age_s triggers latched E-stop abort.
    wrist_cam.frame = TimestampedFrame(static_rgb.copy(), capture_t=clock.t, frame_seq=5)
    clock.advance(0.20)
    stalled_trace = bridge.step()
    assert stalled_trace.aborted
    assert stalled_trace.abort_reason == "stale_camera_frame"
    assert bridge.estopped
    assert len(spy.sent_commands) == 4

    # Inference overrun triggers pre-write staleness abort with ZERO bus writes.
    spy_overrun = SpyRobot(cal=cal, q_rad=wcfg.robot.home_qpos)
    overrun_clock = FakeClock(initial_t=50.0, auto_step=0.001)
    wrist_cam2 = MockCamera(TimestampedFrame(static_rgb.copy(), capture_t=50.0, frame_seq=1))
    tripod_cam2 = MockCamera(TimestampedFrame(static_rgb.copy(), capture_t=50.0, frame_seq=1))
    slow_actor = ScriptedActor(on_act=lambda: overrun_clock.advance(0.25))
    bridge2 = SO101BowlingHardwareBridge(
        actor=slow_actor,
        workspace_cfg=wcfg,
        cal=cal,
        hw_cfg=hw_cfg,
        wrist_cam=wrist_cam2,
        tripod_cam=tripod_cam2,
        robot=spy_overrun,
        clock=overrun_clock,
    )
    bridge2.connect_robot()
    overrun_trace = bridge2.step()
    assert overrun_trace.aborted
    assert overrun_trace.abort_reason == "inference_overrun_stale"
    assert bridge2.estopped
    assert spy_overrun.sent_commands == []
    assert spy_overrun.bus.disable_torque_calls == 1

    # Slow joint read triggers stale_or_slow_joint_read before camera/actor.
    slow_clock = FakeClock(initial_t=80.0, auto_step=0.0)
    spy_slow_read = SpyRobot(
        cal=cal,
        q_rad=wcfg.robot.home_qpos,
        obs_hook=lambda: slow_clock.advance(0.08),
    )
    bridge3 = SO101BowlingHardwareBridge(
        actor=ScriptedActor(),
        workspace_cfg=wcfg,
        cal=cal,
        hw_cfg=hw_cfg,
        wrist_cam=MockCamera(TimestampedFrame(static_rgb.copy(), capture_t=80.0, frame_seq=1)),
        tripod_cam=MockCamera(TimestampedFrame(static_rgb.copy(), capture_t=80.0, frame_seq=1)),
        robot=spy_slow_read,
        clock=slow_clock,
    )
    bridge3.connect_robot()
    slow_trace = bridge3.step()
    assert slow_trace.aborted
    assert slow_trace.abort_reason == "stale_or_slow_joint_read"
    assert spy_slow_read.sent_commands == []


def test_latched_estop_persists_across_steps_and_reports_bus_failure() -> None:
    wcfg = load_config()
    cal = load_calibration()
    hw_cfg = HardwareConfig(dry_run=False, calibration_path="/tmp/aligned_follower.json")
    clock = FakeClock(initial_t=100.0, auto_step=0.002)
    spy = SpyRobot(cal=cal, q_rad=wcfg.robot.home_qpos, raise_on_disable=True)
    wrist_cam = MockCamera(TimestampedFrame(_make_rgb(), capture_t=100.0, frame_seq=1))
    tripod_cam = MockCamera(TimestampedFrame(_make_rgb(), capture_t=100.0, frame_seq=1))
    bridge = SO101BowlingHardwareBridge(
        actor=ScriptedActor(),
        workspace_cfg=wcfg,
        cal=cal,
        hw_cfg=hw_cfg,
        wrist_cam=wrist_cam,
        tripod_cam=tripod_cam,
        robot=spy,
        clock=clock,
    )
    bridge.connect_robot()

    res = bridge.emergency_stop("fault_injected")
    assert res["estopped"] is True
    assert res["torque_disabled_confirmed"] is False
    assert bridge.estopped is True

    # Even with fresh cameras and reset(), E-stop stays latched until clear_estop().
    bridge.reset()
    assert bridge.estopped is True
    wrist_cam.frame = TimestampedFrame(_make_rgb(), capture_t=clock.t, frame_seq=2)
    tripod_cam.frame = TimestampedFrame(_make_rgb(), capture_t=clock.t, frame_seq=2)
    t_aborted = bridge.step()
    assert t_aborted.aborted is True
    assert t_aborted.abort_reason == "fault_injected"
    assert spy.sent_commands == []

    bridge.clear_estop()
    assert bridge.estopped is False
    wrist_cam.frame = TimestampedFrame(_make_rgb(), capture_t=clock.t, frame_seq=3)
    tripod_cam.frame = TimestampedFrame(_make_rgb(), capture_t=clock.t, frame_seq=3)
    t_ok = bridge.step()
    assert not t_ok.aborted
    assert len(spy.sent_commands) == 1

    # Confirm torque_disabled_confirmed=True when bus disable_torque succeeds.
    spy.bus.raise_on_disable = False
    res_ok = bridge.emergency_stop("clean_stop")
    assert res_ok["torque_disabled_confirmed"] is True


def test_camera_rotation_video_replay_and_dual_camera_actor_act(tmp_path: Path) -> None:
    wcfg = load_config()
    cal = load_calibration()
    hw_cfg = HardwareConfig(dry_run=True)

    # Rotation validation and values.
    sample = np.zeros((4, 6, 3), dtype=np.uint8)
    sample[0, 0] = [10, 20, 30]
    assert rotate_frame(sample, 0).shape == (4, 6, 3)
    assert rotate_frame(sample, 90).shape == (6, 4, 3)
    r180 = rotate_frame(sample, 180)
    assert r180.shape == (4, 6, 3)
    np.testing.assert_array_equal(r180[-1, -1], [10, 20, 30])
    assert rotate_frame(sample, 270).shape == (6, 4, 3)
    with pytest.raises(ValueError):
        rotate_frame(sample, 45)

    # Write deterministic synthetic .mp4 and verify VideoFileTimestampedCamera replay.
    mp4_path = tmp_path / "test_cam.mp4"
    writer = cv2.VideoWriter(
        str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (1280, 720)
    )
    try:
        assert writer.isOpened()
        for i in range(4):
            bgr = np.full((720, 1280, 3), 30 + i * 20, dtype=np.uint8)
            writer.write(bgr)
    finally:
        writer.release()

    with VideoFileTimestampedCamera(mp4_path, fps=25.0, rotation_deg=180) as vcam:
        frames = [vcam.read_latest() for _ in range(5)]
    assert frames[4] is None
    for idx in range(4):
        f = frames[idx]
        assert f is not None
        assert f.rgb.shape == (720, 1280, 3)
        assert f.frame_seq == idx
        assert f.capture_t == pytest.approx(idx / 25.0)

    # Verify OpenCVTimestampedCamera thread captures atomic BGR->RGB rotated frames.
    class FakeCapture:
        def __init__(self, _idx: int) -> None:
            self._open = True
            self._count = 0

        def isOpened(self) -> bool:
            return self._open

        def set(self, _prop: int, _val: float) -> bool:
            return True

        def read(self) -> tuple[bool, np.ndarray]:
            self._count += 1
            # BGR blue=[255, 0, 0] -> RGB=[0, 0, 255]
            bgr = np.zeros((720, 1280, 3), dtype=np.uint8)
            bgr[0, 0] = [255, 10, 20]
            return True, bgr

        def release(self) -> None:
            self._open = False

    cam_clock = FakeClock(initial_t=1.0, auto_step=0.01)
    with OpenCVTimestampedCamera(
        0,
        rotation_deg=180,
        clock=cam_clock,
        capture_factory=FakeCapture,
    ) as ocv_cam:
        assert ocv_cam.wait_ready(timeout_s=1.0)
        latest = ocv_cam.read_latest()
        assert latest is not None
        assert latest.rgb.shape == (720, 1280, 3)
        np.testing.assert_array_equal(latest.rgb[-1, -1], [20, 10, 255])

    # Real 1280x720 rotation + preprocess_camera_frame(..., 64, 112) -> DualCameraActor.act.
    actor = DualCameraActor(frame_stack=2, img_hw=(64, 112))
    actor.eval()
    raw_1280 = _make_rgb(720, 1280)
    wrist_proc = preprocess_camera_frame(rotate_frame(raw_1280, hw_cfg.wrist_rotation_deg), 64, 112)
    tripod_proc = preprocess_camera_frame(rotate_frame(raw_1280, hw_cfg.tripod_rotation_deg), 64, 112)
    assert wrist_proc.shape == (64, 112, 3)
    assert tripod_proc.shape == (64, 112, 3)

    guard = SafetyGuard(wcfg, cal, hw_cfg)
    q_home = np.asarray([wcfg.robot.home_qpos[j] for j in ACTIVE_JOINTS])
    obs = {
        "wrist_rgb": np.stack((wrist_proc, wrist_proc)),
        "tripod_rgb": np.stack((tripod_proc, tripod_proc)),
        "camera_meta": np.array([0.01, 1.0, 0.01, 1.0], dtype=np.float32),
        "proprio": scale_proprio(
            q_home,
            np.zeros(4),
            np.zeros(4),
            guard.q_min,
            guard.q_max,
            wcfg.robot.max_action_rad,
        ),
    }
    act_out = actor.act(obs, deterministic=True).cpu().numpy()
    assert act_out.shape == (4,)
    assert np.all(np.abs(act_out) <= 1.0)


def test_create_live_follower_lazy_import_uses_safe_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class StrictSO101Config:
        def __init__(
            self,
            *,
            port: str,
            use_degrees: bool,
            max_relative_target: float | None,
            cameras: dict[str, Any],
            calibration_dir: Path | None = None,
            id: str | None = None,
        ) -> None:
            captured["so_follower_kwargs"] = {
                "port": port,
                "use_degrees": use_degrees,
                "max_relative_target": max_relative_target,
                "cameras": cameras,
                "calibration_dir": calibration_dir,
                "id": id,
            }
            self.use_degrees = use_degrees
            self.max_relative_target = max_relative_target

    class BaseSOFollowerConfig:
        def __init__(
            self,
            *,
            port: str,
            use_degrees: bool = False,
            max_relative_target: float | None = None,
            cameras: dict[str, Any] | None = None,
        ) -> None:
            raise AssertionError("Base SOFollowerConfig must never be imported or instantiated")

    class DummyFollowerNew:
        def __init__(self, cfg: StrictSO101Config) -> None:
            self.config = cfg

    # 1. lerobot 0.7.0 path: lerobot.robots.so_follower exports SO101Follower + SO101FollowerConfig.
    mod_pkg = types.ModuleType("lerobot")
    mod_robots = types.ModuleType("lerobot.robots")
    mod_so_new = types.ModuleType("lerobot.robots.so_follower")
    mod_so_new.SO101Follower = DummyFollowerNew  # type: ignore[attr-defined]
    mod_so_new.SO101FollowerConfig = StrictSO101Config  # type: ignore[attr-defined]
    mod_so_new.SOFollower = DummyFollowerNew  # type: ignore[attr-defined]
    mod_so_new.SOFollowerConfig = BaseSOFollowerConfig  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lerobot", mod_pkg)
    monkeypatch.setitem(sys.modules, "lerobot.robots", mod_robots)
    monkeypatch.setitem(sys.modules, "lerobot.robots.so_follower", mod_so_new)

    hw_cfg = HardwareConfig(port="/dev/tty.usbmodem1234", calibration_path="/tmp/aligned_follower.json")
    follower_new = create_live_follower(hw_cfg)
    assert isinstance(follower_new, DummyFollowerNew)
    assert captured["so_follower_kwargs"]["port"] == "/dev/tty.usbmodem1234"
    assert captured["so_follower_kwargs"]["use_degrees"] is True
    assert captured["so_follower_kwargs"]["max_relative_target"] is None
    assert captured["so_follower_kwargs"]["cameras"] == {}
    assert captured["so_follower_kwargs"]["calibration_dir"] == Path("/tmp")
    assert captured["so_follower_kwargs"]["id"] == "aligned_follower"

    # 2. Older lerobot fallback: lerobot.robots.so_follower absent -> lerobot.robots.so101_follower.
    monkeypatch.delitem(sys.modules, "lerobot.robots.so_follower")

    class DummyConfigOld:
        def __init__(self, **kwargs: Any) -> None:
            captured["config_kwargs"] = kwargs
            self.use_degrees = kwargs["use_degrees"]
            self.max_relative_target = kwargs["max_relative_target"]

    class DummyFollowerOld:
        def __init__(self, cfg: DummyConfigOld) -> None:
            self.config = cfg

    mod_so_old = types.ModuleType("lerobot.robots.so101_follower")
    mod_so_old.SO101Follower = DummyFollowerOld  # type: ignore[attr-defined]
    mod_so_old.SO101FollowerConfig = DummyConfigOld  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lerobot.robots.so101_follower", mod_so_old)

    follower_old = create_live_follower(hw_cfg)
    assert isinstance(follower_old, DummyFollowerOld)
    assert captured["config_kwargs"]["port"] == "/dev/tty.usbmodem1234"
    assert captured["config_kwargs"]["use_degrees"] is True
    assert captured["config_kwargs"]["max_relative_target"] is None
    assert captured["config_kwargs"]["cameras"] == {}
    assert captured["config_kwargs"]["calibration_dir"] == Path("/tmp")
    assert captured["config_kwargs"]["id"] == "aligned_follower"


def test_create_live_follower_handles_live_dry_run_and_calibration_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class DummyFollower:
        def __init__(self, cfg: DummyConfig) -> None:
            self.config = cfg

    mod_pkg = types.ModuleType("lerobot")
    mod_robots = types.ModuleType("lerobot.robots")
    mod_so = types.ModuleType("lerobot.robots.so_follower")
    mod_so.SO101Follower = DummyFollower  # type: ignore[attr-defined]
    mod_so.SO101FollowerConfig = DummyConfig  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lerobot", mod_pkg)
    monkeypatch.setitem(sys.modules, "lerobot.robots", mod_robots)
    monkeypatch.setitem(sys.modules, "lerobot.robots.so_follower", mod_so)

    base_kwargs: dict[str, Any] = {
        "port": "/dev/tty.usbmodem1234",
        "use_degrees": True,
        "max_relative_target": None,
        "cameras": {},
    }
    calibration_cases: list[tuple[str | None, dict[str, Any]]] = [
        (None, base_kwargs),
        ("/tmp/cal_dir", {**base_kwargs, "calibration_dir": Path("/tmp/cal_dir")}),
    ]

    for dry_run in (False, True):
        for calibration_path, expected_kwargs in calibration_cases:
            hw_cfg = HardwareConfig(
                port="/dev/tty.usbmodem1234",
                calibration_path=calibration_path,
                dry_run=dry_run,
            )
            follower = create_live_follower(hw_cfg)
            assert isinstance(follower, DummyFollower)
            assert follower.config.kwargs == expected_kwargs


def test_guard_out_of_limit_clamping_and_last_sent_lag_corners() -> None:
    wcfg = load_config()
    cal = load_calibration()
    hw_cfg = HardwareConfig(dry_run=True)
    guard = SafetyGuard(wcfg, cal, hw_cfg)

    home_6 = np.asarray([wcfg.robot.home_qpos[j] for j in ALL_JOINTS], dtype=np.float64)

    # 1. q_meas slightly below q_min (within max_action_rad): clamps into [q_min, q_meas + delta_max]
    # and NEVER commands a single-tick move > max_action_rad.
    q_slight_oob = home_6.copy()
    q_slight_oob[0] = guard.q_min[0] - 0.03
    prop_back_in = q_slight_oob[:4].copy()
    prop_back_in[0] = guard.q_min[0] + 0.20
    res_slight = guard.guard_command(q_slight_oob, prop_back_in)
    assert res_slight.allowed is True
    assert res_slight.target_active_4_rad[0] >= guard.q_min[0]
    assert abs(res_slight.target_active_4_rad[0] - q_slight_oob[0]) <= guard.max_action_rad + 1e-12
    assert res_slight.target_active_4_rad[0] == pytest.approx(q_slight_oob[0] + guard.max_action_rad)

    # 2. q_meas further than max_action_rad outside [q_min, q_max] -> empty step interval rejected.
    q_far_oob = home_6.copy()
    q_far_oob[0] = guard.q_min[0] - 0.20
    res_far = guard.guard_command(q_far_oob, home_6[:4])
    assert res_far.allowed is False
    assert res_far.reason == "out_of_limit_step_exceeded"
    np.testing.assert_allclose(res_far.target_active_4_rad, q_far_oob[:4])

    # 3. last_sent_active_rad expands the 16-corner lag box to catch intermediate Z-floor violation.
    q0_corner = np.array([1.718, -1.6732, -1.1309, -0.9391, -2.2448, 0.7327])
    q_prop_corner = np.array([1.7328, -1.7023, -1.1935, -0.8893])
    delta = np.clip(q_prop_corner - q0_corner[:4], -guard.max_action_rad, guard.max_action_rad)
    q_clamped = np.clip(q0_corner[:4] + delta, guard.q_min, guard.q_max)
    pad = np.where(
        np.abs(q_clamped - q0_corner[:4]) > 1e-6,
        np.sign(q_clamped - q0_corner[:4]) * hw_cfg.stopping_margin_rad,
        0.0,
    )
    q_padded_end = np.clip(q_clamped + pad, guard.q_min, guard.q_max)

    # Holding still at q0_corner is safe on its own...
    res_still = guard.guard_command(q0_corner, q0_corner[:4])
    assert res_still.allowed is True
    # ...but including in-flight last_sent_active_rad triggers the 16-corner Z-floor rejection.
    res_lag = guard.guard_command(
        q0_corner,
        q0_corner[:4],
        last_sent_active_rad=q_padded_end,
    )
    assert res_lag.allowed is False
    assert res_lag.reason == "z_floor_violation"

    # 4. Bridge tracks _last_sent_active_rad across live steps and passes it to guard_command.
    clock = FakeClock(initial_t=10.0, auto_step=0.002)
    spy = SpyRobot(cal=cal, q_rad=wcfg.robot.home_qpos)
    wrist_cam = MockCamera(TimestampedFrame(_make_rgb(), capture_t=10.0, frame_seq=1))
    tripod_cam = MockCamera(TimestampedFrame(_make_rgb(), capture_t=10.0, frame_seq=1))
    bridge = SO101BowlingHardwareBridge(
        actor=ScriptedActor(action=np.array([0.2, 0.0, 0.0, 0.0], dtype=np.float32)),
        workspace_cfg=wcfg,
        cal=cal,
        hw_cfg=HardwareConfig(dry_run=False, calibration_path="/tmp/aligned_follower.json"),
        wrist_cam=wrist_cam,
        tripod_cam=tripod_cam,
        robot=spy,
        clock=clock,
    )
    bridge.connect_robot()
    seen_last_sent: list[np.ndarray | None] = []
    orig_guard = bridge.safety_guard.guard_command

    def _spy_guard(
        q_meas: np.ndarray,
        prop: np.ndarray,
        last_sent_active_rad: np.ndarray | None = None,
    ) -> Any:
        seen_last_sent.append(
            None if last_sent_active_rad is None else last_sent_active_rad.copy()
        )
        return orig_guard(q_meas, prop, last_sent_active_rad=last_sent_active_rad)

    bridge.safety_guard.guard_command = _spy_guard  # type: ignore[method-assign]
    t1 = bridge.step()
    assert not t1.aborted
    assert seen_last_sent[0] is None
    assert bridge._last_sent_active_rad is not None
    sent_after_t1 = bridge._last_sent_active_rad.copy()

    wrist_cam.frame = TimestampedFrame(_make_rgb(), capture_t=clock.t, frame_seq=2)
    tripod_cam.frame = TimestampedFrame(_make_rgb(), capture_t=clock.t, frame_seq=2)
    t2 = bridge.step()
    assert not t2.aborted
    assert seen_last_sent[1] is not None
    np.testing.assert_allclose(seen_last_sent[1], sent_after_t1)
    assert bridge._last_sent_active_rad is not None
    sent_after_t2 = bridge._last_sent_active_rad.copy()

    bridge.reset_episode()
    assert bridge._last_sent_active_rad is None
    bridge.close()
    assert bridge._last_sent_active_rad is None

    # In dry_run=True, reset() clears _last_sent_active_rad.
    dry_bridge = SO101BowlingHardwareBridge(
        actor=ScriptedActor(),
        workspace_cfg=wcfg,
        cal=cal,
        hw_cfg=HardwareConfig(dry_run=True),
        wrist_cam=MockCamera(),
        tripod_cam=MockCamera(),
        clock=clock,
    )
    dry_bridge._last_sent_active_rad = np.zeros(len(ACTIVE_JOINTS), dtype=np.float64)
    dry_bridge.reset()
    assert dry_bridge._last_sent_active_rad is None


def test_guard_pads_last_sent_direction_on_hold_and_reversal() -> None:
    wcfg = load_config()
    cal = load_calibration()
    hw_cfg = HardwareConfig(dry_run=True)
    guard = SafetyGuard(wcfg, cal, hw_cfg)

    # q_meas is safe, q_last_sent is itself safe (x=0.43760 < 0.44),
    # but q_last_sent + stopping_margin_rad along sign(q_last_sent - q_meas) violates foul_line_x.
    q_meas = np.array([0.0, 0.40, -0.80, 0.0, 0.0, 0.0], dtype=np.float64)
    q_last_sent = np.array([0.0, 0.50, -0.80, 0.0], dtype=np.float64)
    q_last_full = q_meas.copy()
    q_last_full[:4] = q_last_sent
    ok_last, _, _, max_x_last = guard.check_pose(q_last_full)
    assert ok_last is True
    assert max_x_last < hw_cfg.foul_line_x

    # Hold command (q_prop == q_meas[:4]) must still be rejected due to q_last_sent stopping overshoot.
    res_hold = guard.guard_command(q_meas, q_meas[:4], last_sent_active_rad=q_last_sent)
    assert res_hold.allowed is False
    assert res_hold.reason == "foul_line_violation"

    # Reversal command (moving away from foul line) must also be rejected due to in-flight motion.
    q_reverse = q_meas[:4] - np.array([0.0, 0.05, 0.0, 0.0], dtype=np.float64)
    res_rev = guard.guard_command(q_meas, q_reverse, last_sent_active_rad=q_last_sent)
    assert res_rev.allowed is False
    assert res_rev.reason == "foul_line_violation"


def test_live_calibration_parity_missing_path_norm_mode_and_is_calibrated() -> None:
    wcfg = load_config()
    cal = load_calibration()
    clock = FakeClock(initial_t=1.0, auto_step=0.005)

    # 1. calibration_path=None refused in live mode.
    bridge_no_cal = SO101BowlingHardwareBridge(
        actor=ScriptedActor(),
        workspace_cfg=wcfg,
        cal=cal,
        hw_cfg=HardwareConfig(dry_run=False, calibration_path=None),
        wrist_cam=MockCamera(),
        tripod_cam=MockCamera(),
        robot=SpyRobot(cal=cal, q_rad=wcfg.robot.home_qpos),
        clock=clock,
    )
    with pytest.raises(ValueError, match="calibration_path"):
        bridge_no_cal.connect_robot()

    # 2. Value mismatches on range_min / homing_offset / drive_mode / id rejected.
    hw_live = HardwareConfig(dry_run=False, calibration_path="/tmp/aligned_follower.json")
    for field_name, bad_val in (
        ("range_min", cal.joints["shoulder_pan"].range_min + 5),
        ("range_max", cal.joints["shoulder_pan"].range_max - 5),
        ("homing_offset", cal.joints["shoulder_pan"].homing_offset + 10),
        ("drive_mode", 1 - cal.joints["shoulder_pan"].drive_mode),
        ("id", 99),
    ):
        bad_robot = SpyRobot(
            cal=cal,
            q_rad=wcfg.robot.home_qpos,
            calibration_overrides={"shoulder_pan": {field_name: bad_val}},
        )
        bad_bridge = SO101BowlingHardwareBridge(
            actor=ScriptedActor(),
            workspace_cfg=wcfg,
            cal=cal,
            hw_cfg=hw_live,
            wrist_cam=MockCamera(),
            tripod_cam=MockCamera(),
            robot=bad_robot,
            clock=clock,
        )
        with pytest.raises(ValueError, match=field_name):
            bad_bridge.connect_robot()

    # 3. Non-degrees norm_mode on active joint rejected.
    bad_joints = dict(cal.joints)
    bad_joints["elbow_flex"] = replace(bad_joints["elbow_flex"], norm_mode="range_0_100")
    bad_cal = replace(cal, joints=bad_joints)
    bad_norm_bridge = SO101BowlingHardwareBridge(
        actor=ScriptedActor(),
        workspace_cfg=wcfg,
        cal=bad_cal,
        hw_cfg=hw_live,
        wrist_cam=MockCamera(),
        tripod_cam=MockCamera(),
        robot=SpyRobot(cal=cal, q_rad=wcfg.robot.home_qpos),
        clock=clock,
    )
    with pytest.raises(ValueError, match="degrees"):
        bad_norm_bridge.connect_robot()

    # 4. robot.is_calibrated=False after connect(calibrate=False) raises and latches E-stop.
    uncal_robot = SpyRobot(cal=cal, q_rad=wcfg.robot.home_qpos, is_calibrated=False)
    uncal_bridge = SO101BowlingHardwareBridge(
        actor=ScriptedActor(),
        workspace_cfg=wcfg,
        cal=cal,
        hw_cfg=hw_live,
        wrist_cam=MockCamera(),
        tripod_cam=MockCamera(),
        robot=uncal_robot,
        clock=clock,
    )
    with pytest.raises(ValueError, match="is_calibrated"):
        uncal_bridge.connect_robot()
    assert uncal_robot.connect_calibrate_args == [False]
    assert uncal_bridge.estopped is True
    assert uncal_bridge.estop_reason == "calibration_mismatch"

    # 5. Missing is_calibrated attribute or non-bool truthy value must fail connect_robot().
    missing_attr_robot = SpyRobot(cal=cal, q_rad=wcfg.robot.home_qpos)
    del missing_attr_robot.is_calibrated
    missing_attr_bridge = SO101BowlingHardwareBridge(
        actor=ScriptedActor(),
        workspace_cfg=wcfg,
        cal=cal,
        hw_cfg=hw_live,
        wrist_cam=MockCamera(),
        tripod_cam=MockCamera(),
        robot=missing_attr_robot,
        clock=clock,
    )
    with pytest.raises(ValueError, match="is_calibrated"):
        missing_attr_bridge.connect_robot()
    assert missing_attr_bridge.estopped is True

    truthy_int_robot = SpyRobot(cal=cal, q_rad=wcfg.robot.home_qpos, is_calibrated=1)  # type: ignore[arg-type]
    truthy_int_bridge = SO101BowlingHardwareBridge(
        actor=ScriptedActor(),
        workspace_cfg=wcfg,
        cal=cal,
        hw_cfg=hw_live,
        wrist_cam=MockCamera(),
        tripod_cam=MockCamera(),
        robot=truthy_int_robot,
        clock=clock,
    )
    with pytest.raises(ValueError, match="is_calibrated"):
        truthy_int_bridge.connect_robot()
    assert truthy_int_bridge.estopped is True

    # 6. Fractional float or bool calibration field values must be rejected without truncation/coercion.
    expected_id = cal.joints["shoulder_pan"].id
    for bad_typed_val in (float(expected_id) + 0.5, float(expected_id), True):
        typed_bad_robot = SpyRobot(
            cal=cal,
            q_rad=wcfg.robot.home_qpos,
            calibration_overrides={"shoulder_pan": {"id": bad_typed_val}},
        )
        typed_bad_bridge = SO101BowlingHardwareBridge(
            actor=ScriptedActor(),
            workspace_cfg=wcfg,
            cal=cal,
            hw_cfg=hw_live,
            wrist_cam=MockCamera(),
            tripod_cam=MockCamera(),
            robot=typed_bad_robot,
            clock=clock,
        )
        with pytest.raises(ValueError, match="Invalid calibration value"):
            typed_bad_bridge.connect_robot()


def test_live_step_before_connect_or_after_close_aborts_without_motor_io() -> None:
    wcfg = load_config()
    cal = load_calibration()
    clock = FakeClock(initial_t=10.0, auto_step=0.002)
    spy = SpyRobot(cal=cal, q_rad=wcfg.robot.home_qpos)
    wrist_cam = MockCamera(TimestampedFrame(_make_rgb(), capture_t=10.0, frame_seq=1))
    tripod_cam = MockCamera(TimestampedFrame(_make_rgb(), capture_t=10.0, frame_seq=1))

    def broken_clock() -> float:
        raise RuntimeError("clock boom")

    # Even without calibration_path on an injected SpyRobot, live step() before connect_robot()
    # must abort immediately with zero get_observation or send_action calls.
    bridge = SO101BowlingHardwareBridge(
        actor=ScriptedActor(),
        workspace_cfg=wcfg,
        cal=cal,
        hw_cfg=HardwareConfig(dry_run=False, calibration_path=None),
        wrist_cam=wrist_cam,
        tripod_cam=tripod_cam,
        robot=spy,
        clock=broken_clock,
    )
    trace = bridge.step()
    assert trace.aborted is True
    assert trace.abort_reason == "live_robot_not_validated"
    assert bridge.estopped is True
    assert spy.connect_calls == 0
    assert spy.get_observation_calls == 0
    assert spy.sent_commands == []
    assert spy.bus.disable_torque_calls == 0

    # After connect_robot() then close(), _live_validated must be reset to False.
    spy2 = SpyRobot(cal=cal, q_rad=wcfg.robot.home_qpos)
    bridge2 = SO101BowlingHardwareBridge(
        actor=ScriptedActor(),
        workspace_cfg=wcfg,
        cal=cal,
        hw_cfg=HardwareConfig(dry_run=False, calibration_path="/tmp/aligned_follower.json"),
        wrist_cam=MockCamera(TimestampedFrame(_make_rgb(), capture_t=10.0, frame_seq=1)),
        tripod_cam=MockCamera(TimestampedFrame(_make_rgb(), capture_t=10.0, frame_seq=1)),
        robot=spy2,
        clock=clock,
    )
    bridge2.connect_robot()
    bridge2.close()
    disable_calls_after_close = spy2.bus.disable_torque_calls
    trace_after_close = bridge2.step()
    assert trace_after_close.aborted is True
    assert trace_after_close.abort_reason == "live_robot_not_validated"
    assert spy2.get_observation_calls == 0
    assert spy2.sent_commands == []
    assert spy2.bus.disable_torque_calls == disable_calls_after_close


@pytest.mark.parametrize(
    ("invalid_timestamp", "expected_reason"),
    [
        ("entry", "non_finite_clock"),
        ("read_start", "stale_or_slow_joint_read"),
        ("read_end", "stale_or_slow_joint_read"),
        ("camera", "stale_camera_frame"),
        ("wrist_capture", "stale_camera_frame"),
        ("tripod_capture", "stale_camera_frame"),
        ("pre_write", "inference_overrun_stale"),
    ],
)
def test_live_step_rejects_non_finite_timestamp(
    invalid_timestamp: str, expected_reason: str
) -> None:
    def timestamp(stage: str) -> float:
        return float("nan") if stage == invalid_timestamp else 10.0

    wcfg = load_config()
    cal = load_calibration()
    spy = SpyRobot(cal=cal, q_rad=wcfg.robot.home_qpos)
    clock_stages = iter(("entry", "read_start", "read_end", "camera", "pre_write"))
    bridge = SO101BowlingHardwareBridge(
        actor=ScriptedActor(),
        workspace_cfg=wcfg,
        cal=cal,
        hw_cfg=HardwareConfig(dry_run=False, calibration_path="/tmp/aligned_follower.json"),
        wrist_cam=MockCamera(TimestampedFrame(_make_rgb(), timestamp("wrist_capture"), 1)),
        tripod_cam=MockCamera(TimestampedFrame(_make_rgb(), timestamp("tripod_capture"), 1)),
        robot=spy,
        clock=lambda: timestamp(next(clock_stages, "trace")),
    )
    bridge.connect_robot()
    trace = bridge.step()
    assert trace.aborted is True
    assert trace.abort_reason == expected_reason
    assert np.isfinite(trace.t)
    assert spy.sent_commands == []
    assert spy.bus.disable_torque_calls == 1
    if expected_reason == "stale_camera_frame":
        assert trace.wrist_age_s == trace.tripod_age_s == float("inf")


def test_step_exception_and_unsafe_measured_pose_latch_estop_and_surface_torque_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wcfg = load_config()
    cal = load_calibration()
    hw_cfg = HardwareConfig(dry_run=False, calibration_path="/tmp/aligned_follower.json")

    # 1. unsafe_measured_pose triggers E-stop abort with torque_disabled_confirmed=True.
    q_jaw_only = np.array([-0.5987, -0.7884, 1.5098, -0.1819, 2.7125, 0.8146])
    unsafe_q = {j: float(q_jaw_only[i]) for i, j in enumerate(ALL_JOINTS)}
    clock1 = FakeClock(initial_t=10.0, auto_step=0.002)
    spy1 = SpyRobot(cal=cal, q_rad=unsafe_q)
    bridge1 = SO101BowlingHardwareBridge(
        actor=ScriptedActor(),
        workspace_cfg=wcfg,
        cal=cal,
        hw_cfg=hw_cfg,
        wrist_cam=MockCamera(TimestampedFrame(_make_rgb(), capture_t=10.0, frame_seq=1)),
        tripod_cam=MockCamera(TimestampedFrame(_make_rgb(), capture_t=10.0, frame_seq=1)),
        robot=spy1,
        clock=clock1,
    )
    bridge1.connect_robot()
    trace_unsafe = bridge1.step()
    assert trace_unsafe.aborted is True
    assert trace_unsafe.abort_reason == "unsafe_measured_pose:z_floor_violation"
    assert trace_unsafe.torque_disabled_confirmed is True
    assert bridge1.estopped is True
    assert spy1.sent_commands == []
    assert spy1.bus.disable_torque_calls == 1

    # 2. Mid-step unexpected exception (e.g. guard_command raises) latches E-stop and surfaces
    # torque_disabled_confirmed accurately across reset_episode() and subsequent step().
    clock2 = FakeClock(initial_t=20.0, auto_step=0.002)
    spy2 = SpyRobot(cal=cal, q_rad=wcfg.robot.home_qpos, raise_on_disable=False)
    wrist2 = MockCamera(TimestampedFrame(_make_rgb(), capture_t=20.0, frame_seq=1))
    tripod2 = MockCamera(TimestampedFrame(_make_rgb(), capture_t=20.0, frame_seq=1))
    bridge2 = SO101BowlingHardwareBridge(
        actor=ScriptedActor(),
        workspace_cfg=wcfg,
        cal=cal,
        hw_cfg=hw_cfg,
        wrist_cam=wrist2,
        tripod_cam=tripod2,
        robot=spy2,
        clock=clock2,
    )
    bridge2.connect_robot()

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("unexpected fk failure")

    monkeypatch.setattr(bridge2.safety_guard, "guard_command", _boom)
    trace_exc = bridge2.step()
    assert trace_exc.aborted is True
    assert trace_exc.abort_reason == "step_exception:RuntimeError"
    assert trace_exc.torque_disabled_confirmed is True
    assert bridge2.estopped is True
    assert spy2.sent_commands == []
    assert spy2.bus.disable_torque_calls == 1

    bridge2.reset_episode()
    assert bridge2.estopped is True
    wrist2.frame = TimestampedFrame(_make_rgb(), capture_t=clock2.t, frame_seq=2)
    tripod2.frame = TimestampedFrame(_make_rgb(), capture_t=clock2.t, frame_seq=2)
    trace_after = bridge2.step()
    assert trace_after.aborted is True
    assert trace_after.abort_reason == "step_exception:RuntimeError"
    assert trace_after.torque_disabled_confirmed is True
    assert spy2.sent_commands == []

    # 3. Same with bus.disable_torque() raising -> torque_disabled_confirmed is False.
    clock3 = FakeClock(initial_t=30.0, auto_step=0.002)
    spy3 = SpyRobot(cal=cal, q_rad=wcfg.robot.home_qpos, raise_on_disable=True)
    bridge3 = SO101BowlingHardwareBridge(
        actor=ScriptedActor(action=np.array([np.nan, 0.0, 0.0, 0.0], dtype=np.float32)),
        workspace_cfg=wcfg,
        cal=cal,
        hw_cfg=hw_cfg,
        wrist_cam=MockCamera(TimestampedFrame(_make_rgb(), capture_t=30.0, frame_seq=1)),
        tripod_cam=MockCamera(TimestampedFrame(_make_rgb(), capture_t=30.0, frame_seq=1)),
        robot=spy3,
        clock=clock3,
    )
    bridge3.connect_robot()
    trace_nan = bridge3.step()
    assert trace_nan.aborted is True
    assert trace_nan.abort_reason == "invalid_actor_action"
    assert trace_nan.torque_disabled_confirmed is False
    assert bridge3.last_estop_result is not None
    assert bridge3.last_estop_result["torque_disabled_confirmed"] is False


def test_opencv_camera_close_non_hanging_on_stuck_read_and_reopen() -> None:
    entered_read = threading.Event()
    unblock_read = threading.Event()
    release_threads: list[int] = []
    control_thread_id = threading.get_ident()

    class HangingCapture:
        def __init__(self, _idx: int) -> None:
            self._open = True

        def isOpened(self) -> bool:
            return self._open

        def set(self, _prop: int, _val: float) -> bool:
            return True

        def read(self) -> tuple[bool, np.ndarray | None]:
            entered_read.set()
            unblock_read.wait(timeout=10.0)
            return False, None

        def release(self) -> None:
            release_threads.append(threading.get_ident())
            self._open = False

    class WorkingCapture:
        def __init__(self, _idx: int) -> None:
            self._open = True

        def isOpened(self) -> bool:
            return self._open

        def set(self, _prop: int, _val: float) -> bool:
            return True

        def read(self) -> tuple[bool, np.ndarray]:
            bgr = np.zeros((32, 32, 3), dtype=np.uint8)
            bgr[0, 0] = [0, 255, 0]
            return True, bgr

        def release(self) -> None:
            self._open = False

    captures: list[Any] = []
    use_working = False

    def factory(idx: int) -> Any:
        cap = WorkingCapture(idx) if use_working else HangingCapture(idx)
        captures.append(cap)
        return cap

    cam = OpenCVTimestampedCamera(0, capture_factory=factory)
    cam.open()
    assert entered_read.wait(timeout=1.0)

    t0 = time.monotonic()
    cam.close()
    elapsed = time.monotonic() - t0
    assert elapsed < 1.5
    # Control thread must NEVER call cap.release() while the capture thread is stuck in read().
    assert control_thread_id not in release_threads
    assert release_threads == []

    # Re-open with a fresh capture works cleanly without interference from the old hung thread.
    use_working = True
    cam.open()
    try:
        assert cam.wait_ready(timeout_s=1.0)
        frame = cam.read_latest()
        assert frame is not None
        assert frame.rgb.shape == (32, 32, 3)
    finally:
        cam.close()
        unblock_read.set()

    # Wait briefly for old hung thread to exit its finally block and release old cap on its own thread.
    deadline = time.monotonic() + 1.0
    while not release_threads and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(release_threads) == 1
    assert release_threads[0] != control_thread_id
    assert captures[0]._open is False


def test_bridge_close_disconnects_robot_first_before_cameras() -> None:
    wcfg = load_config()
    cal = load_calibration()
    hw_cfg = HardwareConfig(dry_run=False, calibration_path="/tmp/aligned_follower.json")
    events: list[str] = []

    class RecordingCamera(MockCamera):
        def __init__(self, name: str, raise_on_close: bool = False) -> None:
            super().__init__()
            self._name = name
            self._raise_on_close = raise_on_close

        def close(self) -> None:
            events.append(f"{self._name}_close")
            super().close()
            if self._raise_on_close:
                raise RuntimeError(f"{self._name} close failed")

    spy = SpyRobot(
        cal=cal,
        q_rad=wcfg.robot.home_qpos,
        on_disconnect=lambda: events.append("robot_disconnect"),
    )
    wrist_cam = RecordingCamera("wrist", raise_on_close=True)
    tripod_cam = RecordingCamera("tripod", raise_on_close=False)
    bridge = SO101BowlingHardwareBridge(
        actor=ScriptedActor(),
        workspace_cfg=wcfg,
        cal=cal,
        hw_cfg=hw_cfg,
        wrist_cam=wrist_cam,
        tripod_cam=tripod_cam,
        robot=spy,
    )
    bridge.connect_robot()
    bridge.close()

    assert events == ["robot_disconnect", "wrist_close", "tripod_close"]
    assert spy.disconnect_calls == 1
    assert wrist_cam.closed and tripod_cam.closed


def test_mutated_nan_threshold_fails_closed_in_guard_and_live_step() -> None:
    wcfg = load_config()
    cal = load_calibration()
    hw_cfg = HardwareConfig(dry_run=False, calibration_path="/tmp/aligned_follower.json")
    guard = SafetyGuard(wcfg, cal, hw_cfg)

    home = np.array([wcfg.robot.home_qpos[j] for j in ALL_JOINTS], dtype=np.float64)
    safe_before, reason_before, _, _ = guard.check_pose(home)
    assert safe_before is True
    assert reason_before is None

    object.__setattr__(hw_cfg, "max_joint_read_latency_s", float("nan"))
    safe_after, reason_after, min_z, max_x = guard.check_pose(home)
    assert safe_after is False
    assert reason_after == "invalid_hardware_config"
    assert np.isfinite(min_z) and np.isfinite(max_x)

    clock = FakeClock(initial_t=10.0, auto_step=0.002)
    spy = SpyRobot(cal=cal, q_rad=wcfg.robot.home_qpos)
    bridge = SO101BowlingHardwareBridge(
        actor=ScriptedActor(),
        workspace_cfg=wcfg,
        cal=cal,
        hw_cfg=hw_cfg,
        wrist_cam=MockCamera(TimestampedFrame(_make_rgb(), capture_t=10.0, frame_seq=1)),
        tripod_cam=MockCamera(TimestampedFrame(_make_rgb(), capture_t=10.0, frame_seq=1)),
        robot=spy,
        clock=clock,
    )
    bridge.connect_robot()
    trace = bridge.step()
    assert trace.aborted is True
    assert trace.abort_reason == "invalid_hardware_config"
    assert bridge.estopped is True
    assert spy.get_observation_calls == 0
    assert spy.sent_commands == []


def test_safe_hold_on_lag_corner_reject_and_negotiated_camera_logging(
    caplog: pytest.LogCaptureFixture,
) -> None:
    wcfg = load_config()
    cal = load_calibration()
    hw_cfg = HardwareConfig(dry_run=False, calibration_path="/tmp/aligned_follower.json")

    # 1. On lag-corner rejection (safe measured pose, unsafe in-flight target), live bridge
    # must command a safe hold at q_active (plus held joints) rather than dropping send_action().
    q_safe_start = {
        "shoulder_pan": 0.0,
        "shoulder_lift": 0.42,
        "elbow_flex": -0.80,
        "wrist_flex": 0.0,
        "wrist_roll": wcfg.robot.home_qpos["wrist_roll"],
        "gripper": wcfg.robot.home_qpos["gripper"],
    }

    class LaggySpyRobot(SpyRobot):
        def send_action(self, action: dict[str, float]) -> dict[str, float]:
            # Record command and echo sent target without advancing physical q_rad yet.
            self.sent_commands.append(dict(action))
            return {k: float(v) for k, v in action.items()}

    clock = FakeClock(initial_t=10.0, auto_step=0.002)
    laggy = LaggySpyRobot(cal=cal, q_rad=q_safe_start)
    wrist_cam = MockCamera(TimestampedFrame(_make_rgb(), capture_t=clock.t, frame_seq=1))
    tripod_cam = MockCamera(TimestampedFrame(_make_rgb(), capture_t=clock.t, frame_seq=1))
    # Simulate an in-flight _last_sent_active_rad = [0.0, 0.50, -0.80, 0.0] while robot stays at 0.40;
    # reversal is rejected on the 16-corner lag check and must command a safe hold at 0.40.
    bridge = SO101BowlingHardwareBridge(
        actor=ScriptedActor(action=np.array([0.0, -0.5, 0.0, 0.0], dtype=np.float32)),
        workspace_cfg=wcfg,
        cal=cal,
        hw_cfg=hw_cfg,
        wrist_cam=wrist_cam,
        tripod_cam=tripod_cam,
        robot=laggy,
        clock=clock,
    )
    bridge.connect_robot()
    laggy.q_rad["shoulder_lift"] = 0.40
    bridge._last_sent_active_rad = np.array([0.0, 0.50, -0.80, 0.0], dtype=np.float64)

    trace_hold = bridge.step()
    assert trace_hold.aborted is False
    assert trace_hold.guard_reason == "foul_line_violation"
    assert len(laggy.sent_commands) == 1
    cmd = laggy.sent_commands[0]
    assert set(cmd.keys()) == {f"{j}.pos" for j in ALL_JOINTS}
    assert cmd["shoulder_lift.pos"] == pytest.approx(mjcf_rad_to_hw("shoulder_lift", 0.40, cal))
    for j in HELD_JOINTS:
        assert cmd[f"{j}.pos"] == pytest.approx(mjcf_rad_to_hw(j, wcfg.robot.home_qpos[j], cal))
    assert bridge._last_sent_active_rad is not None
    assert bridge._last_sent_active_rad[1] == pytest.approx(0.40)

    # 2. Negotiated resolution/FPS mismatch is recorded and logged on OpenCVTimestampedCamera.open().
    class MismatchedCapture:
        def __init__(self, _idx: int) -> None:
            self._open = True

        def isOpened(self) -> bool:
            return self._open

        def set(self, _prop: int, _val: float) -> bool:
            return True

        def get(self, prop: int) -> float:
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 640.0
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 480.0
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            return 0.0

        def read(self) -> tuple[bool, np.ndarray]:
            return True, np.zeros((480, 640, 3), dtype=np.uint8)

        def release(self) -> None:
            self._open = False

    with caplog.at_level("INFO", logger="so101_bowling.hardware"):
        with OpenCVTimestampedCamera(
            0, width=1280, height=720, fps=25, rotation_deg=0, capture_factory=MismatchedCapture
        ) as cam:
            assert cam.wait_ready(timeout_s=1.0)
            assert cam.negotiated_width == 640
            assert cam.negotiated_height == 480
            assert cam.negotiated_fps == pytest.approx(30.0)
    assert any("negotiated 640x480@30.00fps differs" in rec.message for rec in caplog.records)


def _live_bridge(**q_rad: float) -> tuple[SO101BowlingHardwareBridge, SpyRobot, list[float]]:
    wcfg, cal, sleeps = load_config(), load_calibration(), []
    hw_cfg = HardwareConfig(dry_run=False, calibration_path="/tmp/aligned_follower.json")
    spy = SpyRobot(cal=cal, q_rad={**wcfg.robot.home_qpos, **q_rad})
    bridge = SO101BowlingHardwareBridge(
        ScriptedActor(), wcfg, cal, hw_cfg, MockCamera(), MockCamera(), robot=spy,
        clock=FakeClock(auto_step=0.002), sleep_fn=sleeps.append,
    )
    bridge.connect_robot()
    return bridge, spy, sleeps


_RESTORE_WRITES = [
    ("Torque_Limit", j, limit) for j, limit in zip(ALL_JOINTS, (1000,) * 5 + (500,))
]
_DERATE_WRITES = [("Torque_Limit", j, 200) for j in ALL_JOINTS]


def _record_bus(spy: SpyRobot) -> list[Any]:
    """Record goal writes, Torque_Limit writes and torque cuts in bus order."""
    events: list[Any] = []
    send = spy.send_action

    def send_action(action: dict[str, float]) -> dict[str, float]:
        events.append(("goal", action))
        return send(action)

    spy.send_action = send_action
    spy.bus.write = lambda *args: events.append(args)
    spy.bus.disable_torque = lambda: events.append("disable")
    return events


def test_move_to_home_dry_run_and_live_interpolation() -> None:
    bridge, spy, sleeps = _live_bridge(shoulder_pan=0.2)
    result = bridge.move_to_home()
    assert result["homed"] and result["steps"] == 3 and result["max_err_rad"] <= 0.03
    assert sleeps == pytest.approx([0.05] * 3)
    pans = [np.radians(cmd["shoulder_pan.pos"]) for cmd in spy.sent_commands]
    assert pans == pytest.approx([0.12, 0.04, 0.0])
    assert bridge._last_sent_active_rad == pytest.approx(bridge._home_all_6[:4])
    bridge.hw_cfg = HardwareConfig(dry_run=True)
    assert bridge.move_to_home() == {"homed": True, "dry_run": True, "steps": 0, "max_err_rad": 0.0}
    assert len(spy.sent_commands) == 3


def test_move_to_home_rate_limits_held_joints_and_checks_full_pose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = load_config().robot.home_qpos
    displaced = {j: home[j] + 0.2 for j in HELD_JOINTS}  # active joints are already home
    bridge, spy, _ = _live_bridge(**displaced)
    assert bridge.move_to_home()["steps"] == 3
    for j in HELD_JOINTS:
        sent = [hw_to_mjcf_rad(j, cmd[f"{j}.pos"], bridge.cal) for cmd in spy.sent_commands]
        assert sent == pytest.approx([home[j] + d for d in (0.12, 0.04, 0.0)])

    bridge, spy, _ = _live_bridge(**displaced)
    check_pose = bridge.safety_guard.check_pose

    def reject_commanded_roll(q: np.ndarray) -> tuple[bool, str | None, float, float]:
        # Only the 6-joint command moves wrist_roll; guard_command corners keep it measured.
        if q[ALL_JOINTS.index("wrist_roll")] < displaced["wrist_roll"] - 1e-6:
            return False, "z_floor_violation", 0.0, 0.0
        return check_pose(q)

    monkeypatch.setattr(bridge.safety_guard, "check_pose", reject_commanded_roll)
    with pytest.raises(RuntimeError, match="home_pose_rejected:z_floor_violation"):
        bridge.move_to_home()
    assert bridge.estopped and spy.sent_commands == [] and spy.bus.disable_torque_calls == 1


def test_move_to_home_rejects_6dof_mixed_corner_below_floor() -> None:
    q = np.array([-1.7995, -0.167, 1.1511, -0.6287, -1.2996, 0.795])
    bridge, spy, _ = _live_bridge(**dict(zip(ALL_JOINTS, q.tolist())))
    guard = bridge.safety_guard
    target = q + np.clip(bridge._home_all_6 - q, -0.08, 0.08)
    # Measured pose, full target and every active-only lag corner are all safe...
    assert guard.check_pose(q)[0] and guard.check_pose(target)[0]
    assert guard.guard_command(q, target[:4]).allowed
    # ...but held joints arriving while active joints lag dips below the floor.
    with pytest.raises(RuntimeError, match="home_box_rejected:z_floor_violation"):
        bridge.move_to_home()
    assert bridge.estopped and spy.sent_commands == [] and spy.bus.disable_torque_calls == 1


def test_6dof_transition_includes_padded_in_flight_goal() -> None:
    guard = SafetyGuard(load_config(), load_calibration(), HardwareConfig())
    q_meas = np.array([0.0, 0.40, -0.80, 0.0, 0.0, 0.0])
    in_flight = q_meas + np.array([0.0, 0.10, 0.0, 0.0, 0.0, 0.0])  # safe until padded
    assert guard.check_6dof_transition(q_meas, q_meas) == (True, None)
    assert guard.check_pose(in_flight)[0]
    assert guard.check_6dof_transition(q_meas, q_meas, in_flight) == (False, "foul_line_violation")


def test_move_to_home_keeps_last_commanded_goal_and_tracks_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, spy, _ = _live_bridge()
    assert bridge.move_to_home()["steps"] == 0 and bridge._last_sent_active_rad is None

    bridge, spy, _ = _live_bridge(shoulder_pan=0.2)
    send, check = spy.send_action, bridge.safety_guard.check_6dof_transition
    seen: list[np.ndarray | None] = []

    def lagging_send(action: dict[str, float]) -> dict[str, float]:
        sent = send(action)
        spy.q_rad["shoulder_pan"] += 0.025  # the servo trails its goal
        return sent

    def spy_check(q: np.ndarray, target: np.ndarray, last: np.ndarray | None) -> Any:
        seen.append(last)
        return check(q, target, last)

    spy.send_action = lagging_send
    monkeypatch.setattr(bridge.safety_guard, "check_6dof_transition", spy_check)
    assert bridge.move_to_home()["homed"]
    goals = [
        np.array([hw_to_mjcf_rad(j, cmd[f"{j}.pos"], bridge.cal) for j in ALL_JOINTS])
        for cmd in spy.sent_commands
    ]
    assert len(goals) > 1 and seen[0] is None
    np.testing.assert_allclose(seen[1:], goals[:-1])
    np.testing.assert_allclose(bridge._last_sent_active_rad, goals[-1][:4])
    # Once settled, the lagging measured pose replaces the in-flight goal.
    assert goals[-1][0] == pytest.approx(goals[-2][0] + 0.025)


def test_move_to_home_rejects_unsafe_stale_goal_when_already_settled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, spy, _ = _live_bridge()
    bridge._last_sent_active_rad = bridge._home_all_6[:4] + 0.05
    monkeypatch.setattr(
        bridge.safety_guard, "check_6dof_transition",
        lambda *args: (False, "foul_line_violation"),
    )
    with pytest.raises(RuntimeError, match="home_box_rejected:foul_line_violation"):
        bridge.move_to_home()
    assert bridge.estopped and spy.sent_commands == [] and spy.bus.disable_torque_calls == 1


@pytest.mark.parametrize("override", [
    {"max_step_rad": 0.0}, {"max_step_rad": float("nan")}, {"settle_tol_rad": float("inf")},
    {"settle_tol_rad": 0.0}, {"settle_tol_rad": 0.16}, {"max_steps": 0}, {"max_steps": 1001},
])
def test_move_to_home_rejects_invalid_overrides(override: dict[str, Any]) -> None:
    bridge, spy, _ = _live_bridge(shoulder_pan=0.2)
    with pytest.raises(ValueError):
        bridge.move_to_home(**override)
    assert spy.sent_commands == []


@pytest.mark.parametrize(("field", "value"), [
    ("max_step_interval_s", 0.0), ("max_step_interval_s", float("nan")),
    ("home_settle_tol_rad", float("inf")), ("home_settle_tol_rad", 0.16),
    ("home_max_steps", 0), ("estop_soft_hold_steps", True),
    ("estop_soft_hold_dt_s", -0.1), ("estop_relax_torque_limit", 1001),
    ("estop_soft_hold_steps", 26), ("estop_soft_hold_dt_s", 0.11), ("home_max_steps", 1001),
])
def test_invalid_safety_gate_config_fails_closed(field: str, value: Any) -> None:
    with pytest.raises(ValueError, match=field):
        HardwareConfig(**{field: value})
    HardwareConfig(  # inclusive upper bounds
        estop_soft_hold_steps=25, estop_soft_hold_dt_s=0.1, home_settle_tol_rad=0.15,
        home_max_steps=1000,
    )
    bridge, spy, _ = _live_bridge()
    object.__setattr__(bridge.hw_cfg, field, value)
    assert bridge.step().abort_reason == "invalid_hardware_config"
    assert bridge.estopped and spy.sent_commands == []


@pytest.mark.parametrize(("q_rad", "reason", "num_sent"), [
    # Jaw-only Z-floor violation is rejected by the guard before any motor write.
    (dict(zip(ALL_JOINTS, (-0.5987, -0.7884, 1.5098, -0.1819, 2.7125, 0.8146))),
     "home_guard_rejected:unsafe_measured_pose", 0),
    ({"shoulder_pan": 0.5}, "home_timeout", 1),
])
def test_move_to_home_failures_latch_estop(q_rad: dict[str, float], reason: str, num_sent: int) -> None:
    bridge, spy, _ = _live_bridge(**q_rad)
    with pytest.raises(RuntimeError, match=reason):
        bridge.move_to_home(max_steps=1)
    assert bridge.estop_reason.startswith(reason) and spy.bus.disable_torque_calls == 1
    with pytest.raises(RuntimeError, match="latched E-stop"):
        bridge.move_to_home()
    assert len(spy.sent_commands) == num_sent


def test_move_to_home_checks_measured_pose_before_reporting_settled() -> None:
    bridge, spy, _ = _live_bridge()
    object.__setattr__(bridge.hw_cfg, "z_floor_margin_m", 1.0)  # home itself is now too low
    with pytest.raises(RuntimeError, match="unsafe_measured_pose:z_floor_violation"):
        bridge.move_to_home()
    assert bridge.estopped and spy.sent_commands == [] and spy.bus.disable_torque_calls == 1


@pytest.mark.parametrize("read_end", [10.06, 9.9, float("nan"), float("inf")])
def test_move_to_home_aborts_on_bad_joint_read_latency(read_end: float) -> None:
    bridge, spy, _ = _live_bridge(shoulder_pan=0.2)
    bridge.clock = iter((10.0, read_end)).__next__
    with pytest.raises(RuntimeError, match="joint_read_latency_exceeded"):
        bridge.move_to_home()
    assert bridge.estop_reason == "joint_read_latency_exceeded" and spy.sent_commands == []


@pytest.mark.parametrize("write_fails", [False, True])
def test_estop_soft_hold_then_always_disables_torque(write_fails: bool) -> None:
    bridge, spy, sleeps = _live_bridge(shoulder_pan=0.2)
    events = _record_bus(spy)

    def failing_write(*args: Any) -> None:
        raise RuntimeError("bus write failed")

    if write_fails:
        spy.bus.write = failing_write
    goal = ("goal", pytest.approx(spy.get_observation()))
    result = bridge.emergency_stop("fault")
    assert result["soft_hold_executed"] is not write_fails and result["torque_disabled_confirmed"]
    # The hold goal is latched before the first derating write is even attempted.
    held = [goal, *([] if write_fails else [*_DERATE_WRITES, *[goal] * 5])]
    # The cut comes first; then goal latch, limit restore, and a final cut.
    cleanup = ["disable", goal, *([] if write_fails else _RESTORE_WRITES), "disable"]
    assert events == [*held, *cleanup]
    assert sleeps == [0.04] * (0 if write_fails else 5)
    # A repeat stop only re-cuts torque: no soft hold, goal write or limit restore.
    repeat = bridge.emergency_stop("again")
    assert repeat["soft_hold_executed"] is False and repeat["torque_disabled_confirmed"]
    assert events == [*held, *cleanup, "disable"]


def test_connect_latches_present_goal_before_restoring_torque_limits() -> None:
    bridge, spy, _ = _live_bridge(shoulder_pan=0.2)
    events = _record_bus(spy)
    goal = ("goal", pytest.approx(spy.get_observation()))
    bridge.connect_robot()  # recovers a limit left derated by an earlier process
    assert events == [goal, *_RESTORE_WRITES]
    assert bridge._live_validated


@pytest.mark.parametrize(("fault", "reason"), [
    (None, None),
    ("connect_raises", None),
    ("unsafe_pose", "torque_limit_restore_failed"),
    ("enable_fails", "torque_enable_failed"),
])
def test_connect_enables_torque_only_after_checks_and_latch(
    monkeypatch: pytest.MonkeyPatch, fault: str | None, reason: str | None,
) -> None:
    bridge, spy, _ = _live_bridge(shoulder_pan=0.2)
    events = _record_bus(spy)
    connect, check_pose = spy.connect, bridge.safety_guard.check_pose

    def enable_torque() -> None:
        events.append("enable")
        if fault == "enable_fails":
            raise RuntimeError("enable failed")

    def lerobot_connect(calibrate: bool = True) -> None:
        connect(calibrate)
        spy.bus.disable_torque()
        spy.bus.enable_torque()  # SOFollower.configure() leaving bus.torque_disabled()
        if fault == "connect_raises":
            raise RuntimeError("connect failed")
        events.append("connected")

    def spy_check_pose(q: np.ndarray) -> Any:
        events.append("pose")
        return check_pose(q)

    spy.bus.enable_torque = enable_torque
    spy.connect = lerobot_connect
    monkeypatch.setattr(bridge.safety_guard, "check_pose", spy_check_pose)
    if fault == "unsafe_pose":
        q_jaw_below_floor = (-0.5987, -0.7884, 1.5098, -0.1819, 2.7125, 0.8146)
        spy.q_rad.update(zip(ALL_JOINTS, q_jaw_below_floor))
    goal = ("goal", pytest.approx(spy.get_observation()))
    if fault is None:
        bridge.connect_robot()
    else:
        with pytest.raises((RuntimeError, ValueError)):
            bridge.connect_robot()

    latched = ["disable", "connected", "pose", goal, *_RESTORE_WRITES, "enable"]
    assert events == {
        None: latched,
        "connect_raises": ["disable"],
        "unsafe_pose": ["disable", "connected", "pose", "disable"],
        "enable_fails": [*latched, "disable"],
    }[fault]
    assert spy.bus.enable_torque is enable_torque  # the real method is always put back
    assert bridge._live_validated is (fault is None)
    assert bridge.estop_reason == reason


@pytest.mark.parametrize(("fault", "match"), [
    ("unsafe_pose", "Unsafe measured pose on connect: z_floor_violation"),
    ("non_finite_read", "Non-finite present joint position"),
    ("send_fails", "goal write failed"),
])
def test_connect_goal_latch_failure_fails_closed(fault: str, match: str) -> None:
    bridge, spy, _ = _live_bridge()
    events = _record_bus(spy)

    def failing_send(action: dict[str, float]) -> dict[str, float]:
        raise RuntimeError("goal write failed")

    if fault == "unsafe_pose":
        q_jaw_below_floor = (-0.5987, -0.7884, 1.5098, -0.1819, 2.7125, 0.8146)
        spy.q_rad.update(zip(ALL_JOINTS, q_jaw_below_floor))
    elif fault == "non_finite_read":
        spy.q_rad["elbow_flex"] = float("nan")
    else:
        spy.send_action = failing_send
    with pytest.raises((RuntimeError, ValueError), match=match):
        bridge.connect_robot()
    assert not bridge._live_validated and bridge.estop_reason == "torque_limit_restore_failed"
    assert events == ["disable"]  # an unvalidated robot only gets its torque cut
    assert bridge.last_estop_result["torque_disabled_confirmed"] is True


@pytest.mark.parametrize("fault", ["not_calibrated", "values_changed"])
def test_connect_calibration_mismatch_only_cuts_torque(fault: str) -> None:
    bridge, spy, _ = _live_bridge()
    events = _record_bus(spy)
    connect = spy.connect

    def mismatched_connect(calibrate: bool = True) -> None:
        connect(calibrate)
        if fault == "not_calibrated":
            spy.is_calibrated = False
        else:
            spy.calibration["elbow_flex"].homing_offset += 1

    spy.connect = mismatched_connect
    with pytest.raises(ValueError):
        bridge.connect_robot()
    assert events == ["disable"]  # no send_action and no bus.write
    assert not bridge._live_validated and bridge.estop_reason == "calibration_mismatch"


@pytest.mark.parametrize(("configured", "gripper"), [(200, 200), (800, 500)])
def test_soft_hold_caps_gripper_torque_limit(configured: int, gripper: int) -> None:
    bridge, spy, _ = _live_bridge()
    events = _record_bus(spy)
    bridge.hw_cfg = replace(bridge.hw_cfg, estop_relax_torque_limit=configured)
    bridge.emergency_stop("fault")
    expected = [configured] * 5 + [gripper]
    assert events[0][0] == "goal"
    assert events[1:7] == [("Torque_Limit", j, v) for j, v in zip(ALL_JOINTS, expected)]


def test_torque_limit_restore_failure_fails_closed_on_connect() -> None:
    bridge, spy, _ = _live_bridge()
    attempted: list[str] = []

    def write(name: str, joint: str, value: int) -> None:
        attempted.append(joint)
        raise RuntimeError("bus write failed")

    spy.bus.write = write
    with pytest.raises(RuntimeError, match="Torque_Limit restore failed"):
        bridge.connect_robot()
    assert attempted == list(ALL_JOINTS)  # no retry on a robot that never validated
    assert not bridge._live_validated and bridge.estop_reason == "torque_limit_restore_failed"
    assert bridge.last_estop_result["torque_disabled_confirmed"] is True


@pytest.mark.parametrize("failing_cut", [1, 2])
def test_unconfirmed_torque_cut_is_reported(failing_cut: int) -> None:
    bridge, spy, _ = _live_bridge()
    events = _record_bus(spy)
    cuts: list[int] = []

    def disable() -> None:
        cuts.append(len(cuts) + 1)
        if cuts[-1] == failing_cut:
            raise RuntimeError("serial dropped")

    spy.bus.disable_torque = disable
    result = bridge.emergency_stop("fault")
    assert result["torque_disabled_confirmed"] is False and result["error"] == "serial dropped"
    # Limits stay derated if the first cut fails; a failed final cut is never reported as safe.
    restored = [event for event in events if event in _RESTORE_WRITES]
    assert restored == ([] if failing_cut == 1 else _RESTORE_WRITES)


def test_comms_watchdog_trips_after_max_step_interval() -> None:
    bridge, spy, _ = _live_bridge()

    def step_after(gap_s: float) -> StepTrace:
        now = bridge.clock.advance(gap_s)
        frame = TimestampedFrame(_make_rgb(), capture_t=now, frame_seq=int(now * 1000))
        bridge.wrist_cam.frame = bridge.tripod_cam.frame = frame
        return bridge.step()

    assert not step_after(0.0).aborted and not step_after(0.1).aborted
    bridge.reset()  # an episode boundary disarms the watchdog
    assert not step_after(1.0).aborted
    assert step_after(0.3).abort_reason == "comms_watchdog_timeout"
    assert bridge.estopped and len(spy.sent_commands) == 3


@pytest.mark.hardware
def test_live_so101_connection_and_cameras() -> None:
    """Hardware integration smoke test connecting live SO-101 follower and both UVC cameras."""
    hw_cfg = load_hardware_config()
    cal_path = os.environ.get("SO101_CALIBRATION_PATH") or hw_cfg.calibration_path
    if not cal_path:
        pytest.skip("Set SO101_CALIBRATION_PATH or hardware.yaml calibration_path for live hardware test")

    workspace_cfg = load_config()
    cal = load_calibration(live_json_path=cal_path)
    live_hw_cfg = replace(hw_cfg, dry_run=False, calibration_path=cal_path)

    with (
        OpenCVTimestampedCamera(
            live_hw_cfg.wrist_cam_index,
            width=live_hw_cfg.cam_width,
            height=live_hw_cfg.cam_height,
            fps=live_hw_cfg.cam_fps,
            rotation_deg=live_hw_cfg.wrist_rotation_deg,
        ) as wrist_cam,
        OpenCVTimestampedCamera(
            live_hw_cfg.tripod_cam_index,
            width=live_hw_cfg.cam_width,
            height=live_hw_cfg.cam_height,
            fps=live_hw_cfg.cam_fps,
            rotation_deg=live_hw_cfg.tripod_rotation_deg,
        ) as tripod_cam,
    ):
        bridge = SO101BowlingHardwareBridge(
            actor=ScriptedActor(),
            workspace_cfg=workspace_cfg,
            cal=cal,
            hw_cfg=live_hw_cfg,
            wrist_cam=wrist_cam,
            tripod_cam=tripod_cam,
        )
        try:
            bridge.connect_robot()
            assert bridge.robot is not None
            obs = bridge.robot.get_observation()
            for j in ALL_JOINTS:
                assert f"{j}.pos" in obs
            assert wrist_cam.wait_ready(timeout_s=3.0)
            assert tripod_cam.wait_ready(timeout_s=3.0)
            w_frame = wrist_cam.read_latest()
            t_frame = tripod_cam.read_latest()
            assert w_frame is not None and t_frame is not None
        finally:
            bridge.close()
