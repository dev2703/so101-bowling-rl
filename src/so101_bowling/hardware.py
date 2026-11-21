"""Hardware configuration, timestamped cameras, FK safety guard, and SO-101 bridge."""

from __future__ import annotations

import logging
import math
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np
import torch
import yaml

from so101_bowling.assets import build_scene_xml, get_scene_entity_ids
from so101_bowling.config import (
    ACTIVE_JOINTS,
    ALL_JOINTS,
    CalibrationMap,
    WorkspaceConfig,
    hw_to_mjcf_rad,
    intersect_joint_limits,
    mjcf_rad_to_hw,
)
from so101_bowling.perception_utils import (
    CausalJointVelocityEstimator,
    TimestampedFrameStack,
    preprocess_camera_frame,
    scale_proprio,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_HARDWARE_YAML = _REPO_ROOT / "configs" / "hardware.yaml"
_VALID_ROTATIONS = (0, 90, 180, 270)
_BASE_MOUNT_BODIES = ("base", "shoulder")
_CALIBRATION_FIELDS = ("id", "drive_mode", "homing_offset", "range_min", "range_max")
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class HardwareConfig:
    """Real hardware bridge parameters and safety thresholds."""

    port: str = "/dev/tty.usbmodem0000"
    calibration_path: str | None = None
    wrist_cam_index: int = 0
    tripod_cam_index: int = 1
    cam_width: int = 1280
    cam_height: int = 720
    cam_fps: int = 25
    wrist_rotation_deg: int = 180
    tripod_rotation_deg: int = 0
    control_hz: float = 20.0
    z_floor_margin_m: float = 0.015
    foul_line_x: float = 0.44
    stopping_margin_rad: float = 0.015
    max_joint_stale_s: float = 0.15
    max_camera_age_s: float = 0.20
    max_joint_read_latency_s: float = 0.05
    dry_run: bool = True

    def __post_init__(self) -> None:
        if self.wrist_rotation_deg not in _VALID_ROTATIONS:
            raise ValueError(
                f"wrist_rotation_deg must be one of {_VALID_ROTATIONS}, got {self.wrist_rotation_deg}"
            )
        if self.tripod_rotation_deg not in _VALID_ROTATIONS:
            raise ValueError(
                f"tripod_rotation_deg must be one of {_VALID_ROTATIONS}, got {self.tripod_rotation_deg}"
            )
        if self.cam_width < 1 or self.cam_height < 1 or self.cam_fps < 1:
            raise ValueError("Camera dimensions and FPS must be positive")
        for field_name in (
            "control_hz",
            "z_floor_margin_m",
            "foul_line_x",
            "stopping_margin_rad",
            "max_joint_stale_s",
            "max_camera_age_s",
            "max_joint_read_latency_s",
        ):
            val = getattr(self, field_name)
            if isinstance(val, bool) or not math.isfinite(val):
                raise ValueError(f"{field_name} must be finite, got {val!r}")
        if self.control_hz <= 0.0:
            raise ValueError("control_hz must be positive")
        if self.stopping_margin_rad < 0.0:
            raise ValueError("stopping_margin_rad must be nonnegative")
        if (
            self.max_joint_stale_s <= 0.0
            or self.max_camera_age_s <= 0.0
            or self.max_joint_read_latency_s <= 0.0
        ):
            raise ValueError("Staleness and latency budgets must be positive")


def _validate_runtime_thresholds(hw_cfg: HardwareConfig) -> str | None:
    """Validate all runtime safety thresholds, failing closed on NaN or invalid values."""
    if isinstance(hw_cfg.z_floor_margin_m, bool) or not math.isfinite(hw_cfg.z_floor_margin_m):
        return "z_floor_violation"
    if isinstance(hw_cfg.foul_line_x, bool) or not math.isfinite(hw_cfg.foul_line_x):
        return "foul_line_violation"
    if isinstance(hw_cfg.stopping_margin_rad, bool) or not (
        math.isfinite(hw_cfg.stopping_margin_rad) and hw_cfg.stopping_margin_rad >= 0.0
    ):
        return "invalid_hardware_config"
    if isinstance(hw_cfg.control_hz, bool) or not (
        math.isfinite(hw_cfg.control_hz) and hw_cfg.control_hz > 0.0
    ):
        return "invalid_hardware_config"
    if isinstance(hw_cfg.max_joint_stale_s, bool) or not (
        math.isfinite(hw_cfg.max_joint_stale_s) and hw_cfg.max_joint_stale_s > 0.0
    ):
        return "invalid_hardware_config"
    if isinstance(hw_cfg.max_camera_age_s, bool) or not (
        math.isfinite(hw_cfg.max_camera_age_s) and hw_cfg.max_camera_age_s > 0.0
    ):
        return "invalid_hardware_config"
    if isinstance(hw_cfg.max_joint_read_latency_s, bool) or not (
        math.isfinite(hw_cfg.max_joint_read_latency_s) and hw_cfg.max_joint_read_latency_s > 0.0
    ):
        return "invalid_hardware_config"
    return None


def load_hardware_config(path: str | Path | None = None) -> HardwareConfig:
    """Load HardwareConfig from YAML, defaulting to configs/hardware.yaml."""
    cfg_path = Path(path) if path is not None else _DEFAULT_HARDWARE_YAML
    if not cfg_path.exists():
        raise FileNotFoundError(f"Hardware config not found: {cfg_path}")
    raw: dict[str, Any] = yaml.safe_load(cfg_path.read_text()) or {}
    return HardwareConfig(**raw)


def rotate_frame(rgb: np.ndarray, rotation_deg: int) -> np.ndarray:
    """Validate rotation angle in {0, 90, 180, 270} and rotate an HxWx3 RGB frame."""
    if rotation_deg not in _VALID_ROTATIONS:
        raise ValueError(f"rotation_deg must be one of {_VALID_ROTATIONS}, got {rotation_deg}")
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB frame, got shape {arr.shape}")
    if rotation_deg == 0:
        return np.ascontiguousarray(arr)
    if rotation_deg == 90:
        return np.ascontiguousarray(cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE))
    if rotation_deg == 180:
        return np.ascontiguousarray(cv2.rotate(arr, cv2.ROTATE_180))
    return np.ascontiguousarray(cv2.rotate(arr, cv2.ROTATE_90_COUNTERCLOCKWISE))


@dataclass(frozen=True)
class TimestampedFrame:
    """Atomic RGB frame bundled with producer monotonic capture timestamp and sequence id."""

    rgb: np.ndarray
    capture_t: float
    frame_seq: int


class TimestampedCamera(ABC):
    """Abstract camera source returning atomic TimestampedFrame samples."""

    @abstractmethod
    def open(self) -> None:
        """Open the underlying camera stream."""

    @abstractmethod
    def read_latest(self) -> TimestampedFrame | None:
        """Return the latest atomic frame and timestamp, or None if unavailable."""

    @abstractmethod
    def close(self) -> None:
        """Release camera resources."""

    def __enter__(self) -> TimestampedCamera:
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class OpenCVTimestampedCamera(TimestampedCamera):
    """Thread-backed OpenCV camera capturing atomic BGR->RGB frames with monotonic timestamps."""

    def __init__(
        self,
        device_index: int,
        width: int = 1280,
        height: int = 720,
        fps: int = 25,
        rotation_deg: int = 0,
        clock: Callable[[], float] = time.monotonic,
        capture_factory: Callable[[int], Any] | None = None,
    ) -> None:
        if rotation_deg not in _VALID_ROTATIONS:
            raise ValueError(f"rotation_deg must be one of {_VALID_ROTATIONS}, got {rotation_deg}")
        self.device_index = int(device_index)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.rotation_deg = int(rotation_deg)
        self._clock = clock
        self._capture_factory = capture_factory or cv2.VideoCapture

        self._lock = threading.Lock()
        self._stop_event: threading.Event | None = None
        self._first_frame_event = threading.Event()
        self._cap: Any = None
        self._thread: threading.Thread | None = None
        self._latest: TimestampedFrame | None = None
        self._seq: int = 0

    def open(self) -> None:
        if self._cap is not None or (self._thread is not None and self._thread.is_alive()):
            return
        cap = self._capture_factory(self.device_index)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Failed to open OpenCV camera index {self.device_index}")
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
            cap.set(cv2.CAP_PROP_FPS, float(self.fps))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1.0)
        except Exception:
            cap.release()
            raise

        stop_event = threading.Event()
        self._stop_event = stop_event
        self._cap = cap
        self._first_frame_event.clear()
        with self._lock:
            self._latest = None
            self._seq = 0
        thread = threading.Thread(
            target=self._capture_loop,
            args=(cap, stop_event),
            name=f"so101-cam-{self.device_index}",
            daemon=True,
        )
        self._thread = thread
        thread.start()

    def _capture_loop(self, cap: Any, stop_event: threading.Event) -> None:
        try:
            while not stop_event.is_set():
                ok, bgr = cap.read()
                if stop_event.is_set():
                    break
                capture_t = float(self._clock())
                if not ok or bgr is None:
                    stop_event.wait(0.002)
                    continue
                rgb = rotate_frame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), self.rotation_deg)
                if stop_event.is_set():
                    break
                with self._lock:
                    self._seq += 1
                    self._latest = TimestampedFrame(
                        rgb=rgb.copy(),
                        capture_t=capture_t,
                        frame_seq=self._seq,
                    )
                self._first_frame_event.set()
        finally:
            try:
                cap.release()
            except Exception:
                pass

    def wait_ready(self, timeout_s: float = 2.0) -> bool:
        """Wait up to timeout_s for the first captured frame."""
        return self._first_frame_event.wait(timeout=timeout_s)

    def read_latest(self) -> TimestampedFrame | None:
        with self._lock:
            if self._latest is None:
                return None
            return TimestampedFrame(
                rgb=self._latest.rgb.copy(),
                capture_t=self._latest.capture_t,
                frame_seq=self._latest.frame_seq,
            )

    def close(self) -> None:
        stop_event = self._stop_event
        thread = self._thread
        cap = self._cap
        self._stop_event = None
        self._thread = None
        self._cap = None
        if stop_event is not None:
            stop_event.set()
        if thread is not None:
            thread.join(timeout=1.0)
        elif cap is not None:
            try:
                cap.release()
            except Exception:
                pass


class VideoFileTimestampedCamera(TimestampedCamera):
    """Sequential .mp4 reader emitting RGB frames with deterministic seq / fps timestamps."""

    def __init__(
        self,
        path: str | Path,
        fps: float | None = None,
        rotation_deg: int = 0,
    ) -> None:
        if rotation_deg not in _VALID_ROTATIONS:
            raise ValueError(f"rotation_deg must be one of {_VALID_ROTATIONS}, got {rotation_deg}")
        if fps is not None and fps <= 0.0:
            raise ValueError(f"fps must be positive, got {fps}")
        self.path = Path(path)
        self._fps_override = float(fps) if fps is not None else None
        self.rotation_deg = int(rotation_deg)
        self._cap: cv2.VideoCapture | None = None
        self._effective_fps: float = 25.0
        self._seq: int = 0

    def open(self) -> None:
        if self._cap is not None:
            return
        if not self.path.exists():
            raise FileNotFoundError(f"Video file not found: {self.path}")
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Failed to open video file: {self.path}")
        if self._fps_override is not None:
            self._effective_fps = self._fps_override
        else:
            raw_fps = float(cap.get(cv2.CAP_PROP_FPS))
            self._effective_fps = raw_fps if np.isfinite(raw_fps) and raw_fps > 0.0 else 25.0
        self._cap = cap
        self._seq = 0

    def read_latest(self) -> TimestampedFrame | None:
        if self._cap is None:
            raise RuntimeError("VideoFileTimestampedCamera must be opened before reading")
        ok, bgr = self._cap.read()
        if not ok or bgr is None:
            return None
        rgb = rotate_frame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), self.rotation_deg)
        seq = self._seq
        self._seq += 1
        return TimestampedFrame(
            rgb=rgb,
            capture_t=float(seq) / self._effective_fps,
            frame_seq=seq,
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


@dataclass(frozen=True)
class GuardResult:
    """Outcome of SafetyGuard.guard_command."""

    allowed: bool
    target_active_4_rad: np.ndarray
    reason: str | None
    min_z_m: float
    max_x_m: float


class SafetyGuard:
    """Full collision-volume MuJoCo FK safety guard with 16-corner servo-lag verification."""

    def __init__(
        self,
        workspace_cfg: WorkspaceConfig,
        cal: CalibrationMap,
        hw_cfg: HardwareConfig,
    ) -> None:
        self.workspace_cfg = workspace_cfg
        self.cal = cal
        self.hw_cfg = hw_cfg

        self.model = mujoco.MjModel.from_xml_string(build_scene_xml(workspace_cfg))
        self.data = mujoco.MjData(self.model)
        self.ids = get_scene_entity_ids(self.model)

        self._all_qpos_addrs = np.asarray(
            self.model.jnt_qposadr[self.ids.all_joint_ids], dtype=np.intp
        )
        self._arm_geoms = np.array(sorted(self.ids.arm_geom_ids), dtype=np.intp)
        self._arm_aabb = self.model.geom_aabb[self._arm_geoms].reshape(-1, 2, 3)

        # Exclude fixed-Z base/shoulder pedestal mount geoms from the Z-floor check
        # while checking every articulated arm link and moving_jaw_so101_v1 collision geom.
        base_mount_body_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in _BASE_MOUNT_BODIES
        }
        assert all(bid >= 0 for bid in base_mount_body_ids), (
            f"Missing fixed-Z base mount bodies from {_BASE_MOUNT_BODIES}"
        )
        self._z_floor_geoms = np.array(
            [
                g
                for g in self._arm_geoms
                if int(self.model.geom_bodyid[g]) not in base_mount_body_ids
            ],
            dtype=np.intp,
        )
        self._z_floor_aabb = self.model.geom_aabb[self._z_floor_geoms].reshape(-1, 2, 3)

        excluded_geoms = np.array(
            [
                g
                for g in self._arm_geoms
                if int(self.model.geom_bodyid[g]) in base_mount_body_ids
            ],
            dtype=np.intp,
        )
        if excluded_geoms.size > 0:
            excluded_aabb = self.model.geom_aabb[excluded_geoms].reshape(-1, 2, 3)
            pan_jid = int(self.ids.all_joint_ids[ALL_JOINTS.index("shoulder_pan")])
            pan_qpos_adr = int(self.model.jnt_qposadr[pan_jid])
            for bid in base_mount_body_ids:
                cur_bid = int(bid)
                while cur_bid > 0:
                    j_start = int(self.model.body_jntadr[cur_bid])
                    j_count = int(self.model.body_jntnum[cur_bid])
                    for jid in range(j_start, j_start + j_count):
                        assert jid == pan_jid, (
                            f"Excluded fixed-Z body {bid} has unexpected ancestor joint {jid}"
                        )
                    cur_bid = int(self.model.body_parentid[cur_bid])
            self.data.qpos[self._all_qpos_addrs] = 0.0
            mujoco.mj_kinematics(self.model, self.data)
            assert np.allclose(
                np.abs(self.data.xaxis[pan_jid]), np.array([0.0, 0.0, 1.0]), atol=1e-6
            ), f"shoulder_pan axis must align with world Z [0,0,1], got {self.data.xaxis[pan_jid]}"
            z_axes_0 = self.data.geom_xmat[excluded_geoms, 6:9]
            min_z_0 = self.data.geom_xpos[excluded_geoms, 2] + np.sum(
                z_axes_0 * excluded_aabb[:, 0] - np.abs(z_axes_0) * excluded_aabb[:, 1],
                axis=1,
            )
            self.data.qpos[pan_qpos_adr] = 1.23
            mujoco.mj_kinematics(self.model, self.data)
            z_axes_1 = self.data.geom_xmat[excluded_geoms, 6:9]
            min_z_1 = self.data.geom_xpos[excluded_geoms, 2] + np.sum(
                z_axes_1 * excluded_aabb[:, 0] - np.abs(z_axes_1) * excluded_aabb[:, 1],
                axis=1,
            )
            assert np.allclose(min_z_0, min_z_1, atol=1e-6), (
                "Excluded base/shoulder geom Z varied with shoulder_pan"
            )
            mujoco.mj_resetData(self.model, self.data)

        limits = intersect_joint_limits(cal)
        self.q_min, self.q_max = np.asarray([limits[name] for name in ACTIVE_JOINTS]).T
        self.max_action_rad = float(workspace_cfg.robot.max_action_rad)

    def check_pose(self, q_all_6_rad: np.ndarray) -> tuple[bool, str | None, float, float]:
        """Run MuJoCo FK on all 6 joints and check Z-floor clearance and foul-line X."""
        q_arr = np.asarray(q_all_6_rad, dtype=np.float64)
        if q_arr.shape != (len(ALL_JOINTS),) or not np.isfinite(q_arr).all():
            raise ValueError(f"Expected finite 6-joint pose, got shape={q_arr.shape}")

        self.data.qpos[self._all_qpos_addrs] = q_arr
        mujoco.mj_kinematics(self.model, self.data)

        # Project each local AABB onto world Z and world X (exact env.py projection math).
        z_axes = self.data.geom_xmat[self._z_floor_geoms, 6:9]
        min_z = float(
            np.min(
                self.data.geom_xpos[self._z_floor_geoms, 2]
                + np.sum(
                    z_axes * self._z_floor_aabb[:, 0]
                    - np.abs(z_axes) * self._z_floor_aabb[:, 1],
                    axis=1,
                )
            )
        )

        x_axes = self.data.geom_xmat[self._arm_geoms, :3]
        max_x_geoms = self.data.geom_xpos[self._arm_geoms, 0] + np.sum(
            x_axes * self._arm_aabb[:, 0] + np.abs(x_axes) * self._arm_aabb[:, 1],
            axis=1,
        )
        paddle_x = float(self.data.site_xpos[self.ids.gripper_site_id, 0])
        max_x = max(float(np.max(max_x_geoms)), paddle_x)

        if (
            not (math.isfinite(min_z) and math.isfinite(self.hw_cfg.z_floor_margin_m))
            or min_z < self.hw_cfg.z_floor_margin_m
        ):
            return False, "z_floor_violation", min_z, max_x
        if (
            not (math.isfinite(max_x) and math.isfinite(self.hw_cfg.foul_line_x))
            or max_x > self.hw_cfg.foul_line_x
        ):
            return False, "foul_line_violation", min_z, max_x
        cfg_reason = _validate_runtime_thresholds(self.hw_cfg)
        if cfg_reason is not None:
            return False, cfg_reason, min_z, max_x
        return True, None, min_z, max_x

    def _pad_for_stopping(self, q_target: np.ndarray, q_meas: np.ndarray) -> np.ndarray:
        """Pad moving axes by stopping_margin_rad along their direction of motion."""
        delta = q_target - q_meas
        pad = np.where(
            np.abs(delta) > 1e-6,
            np.sign(delta) * self.hw_cfg.stopping_margin_rad,
            0.0,
        )
        return np.clip(q_target + pad, self.q_min, self.q_max)

    def guard_command(
        self,
        q_measured_all_6: np.ndarray,
        proposed_active_4_rad: np.ndarray,
        last_sent_active_rad: np.ndarray | None = None,
    ) -> GuardResult:
        """Clamp proposed active-4 target and verify all 16 axis-independent lag corners."""
        q_all = np.asarray(q_measured_all_6, dtype=np.float64)
        q_prop = np.asarray(proposed_active_4_rad, dtype=np.float64)
        if q_all.shape != (len(ALL_JOINTS),) or not np.isfinite(q_all).all():
            raise ValueError(f"Expected finite 6-joint pose, got shape={q_all.shape}")
        if q_prop.shape != (len(ACTIVE_JOINTS),) or not np.isfinite(q_prop).all():
            raise ValueError(f"Expected finite 4-joint target, got shape={q_prop.shape}")

        q_last_sent: np.ndarray | None = None
        if last_sent_active_rad is not None:
            q_last_sent = np.asarray(last_sent_active_rad, dtype=np.float64)
            if q_last_sent.shape != (len(ACTIVE_JOINTS),) or not np.isfinite(q_last_sent).all():
                raise ValueError(
                    f"Expected finite 4-joint last_sent_active_rad, got shape={q_last_sent.shape}"
                )

        q_meas_active = q_all[: len(ACTIVE_JOINTS)].copy()

        # Step 1: Reject immediately if the current measured 6-joint pose is already unsafe.
        safe_now, reason_now, min_z_now, max_x_now = self.check_pose(q_all)
        if not safe_now:
            return GuardResult(
                False,
                q_meas_active,
                f"unsafe_measured_pose:{reason_now}",
                min_z_now,
                max_x_now,
            )

        # Step 2: Intersect joint limits with ±max_action_rad around measured pose.
        step_low = np.maximum(self.q_min, q_meas_active - self.max_action_rad)
        step_high = np.minimum(self.q_max, q_meas_active + self.max_action_rad)
        if np.any(step_low > step_high):
            return GuardResult(
                False,
                q_meas_active,
                "out_of_limit_step_exceeded",
                min_z_now,
                max_x_now,
            )
        q_clamped = np.clip(q_prop, step_low, step_high)

        # Step 3: Pad moving axes by stopping_margin_rad along their direction of motion.
        q_padded = self._pad_for_stopping(q_clamped, q_meas_active)

        # Step 4: Verify all 2^4 = 16 corners formed by [axis_low, axis_high].
        axis_low = np.minimum(q_meas_active, q_padded)
        axis_high = np.maximum(q_meas_active, q_padded)
        if q_last_sent is not None:
            q_last_sent_padded = self._pad_for_stopping(q_last_sent, q_meas_active)
            axis_low = np.minimum(axis_low, np.minimum(q_last_sent, q_last_sent_padded))
            axis_high = np.maximum(axis_high, np.maximum(q_last_sent, q_last_sent_padded))

        worst_min_z = min_z_now
        worst_max_x = max_x_now
        q_corner = q_all.copy()
        n_active = len(ACTIVE_JOINTS)
        for mask in range(1 << n_active):
            for i in range(n_active):
                q_corner[i] = axis_high[i] if ((mask >> i) & 1) else axis_low[i]
            ok, reason, min_z, max_x = self.check_pose(q_corner)
            worst_min_z = min(worst_min_z, min_z)
            worst_max_x = max(worst_max_x, max_x)
            if not ok:
                return GuardResult(False, q_meas_active, reason, min_z, max_x)

        return GuardResult(True, q_clamped, None, worst_min_z, worst_max_x)


@dataclass(frozen=True)
class StepTrace:
    """Single control-tick diagnostic trace emitted by SO101BowlingHardwareBridge.step."""

    t: float
    q_rad: np.ndarray
    action: np.ndarray | None
    applied_delta_rad: np.ndarray
    guard_reason: str | None
    wrist_age_s: float
    tripod_age_s: float
    aborted: bool
    abort_reason: str | None
    wrist_rgb: np.ndarray | None = None
    tripod_rgb: np.ndarray | None = None
    torque_disabled_confirmed: bool | None = None


def create_live_follower(hw_cfg: HardwareConfig) -> Any:
    """Lazily import lerobot and instantiate SO101Follower with safe control settings."""
    try:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    except ImportError as exc:
        try:
            from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
        except ImportError:
            raise ImportError(
                "lerobot is required for live SO-101 hardware; install with `uv sync --extra hardware`"
            ) from exc

    cfg_kwargs: dict[str, Any] = {
        "port": hw_cfg.port,
        "use_degrees": True,
        "max_relative_target": None,
        "cameras": {},
    }
    if hw_cfg.calibration_path is not None:
        cal_path = Path(hw_cfg.calibration_path).expanduser()
        if cal_path.suffix == ".json":
            cfg_kwargs["calibration_dir"] = cal_path.parent
            cfg_kwargs["id"] = cal_path.stem
        else:
            cfg_kwargs["calibration_dir"] = cal_path
    return SO101Follower(SO101FollowerConfig(**cfg_kwargs))


class SO101BowlingHardwareBridge:
    """Safe real-world control loop bridging dual cameras, actor policy, and SO-101 follower."""

    def __init__(
        self,
        actor: Any,
        workspace_cfg: WorkspaceConfig,
        cal: CalibrationMap,
        hw_cfg: HardwareConfig,
        wrist_cam: TimestampedCamera,
        tripod_cam: TimestampedCamera,
        robot: Any | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.actor = actor
        self.workspace_cfg = workspace_cfg
        self.cal = cal
        self.hw_cfg = hw_cfg
        self.wrist_cam = wrist_cam
        self.tripod_cam = tripod_cam
        self.robot = robot
        self.clock = clock

        self.safety_guard = SafetyGuard(workspace_cfg, cal, hw_cfg)
        self._velocity = CausalJointVelocityEstimator()
        self._frame_stack = int(getattr(actor, "frame_stack", 2))
        self._wrist_stack = TimestampedFrameStack(k=self._frame_stack)
        self._tripod_stack = TimestampedFrameStack(k=self._frame_stack)

        self._q_min = self.safety_guard.q_min
        self._q_max = self.safety_guard.q_max
        self._max_action_rad = float(workspace_cfg.robot.max_action_rad)
        self._render_h = int(workspace_cfg.camera.render_height)
        self._render_w = int(workspace_cfg.camera.render_width)
        self._home_all_6 = np.asarray(
            [workspace_cfg.robot.home_qpos[j] for j in ALL_JOINTS], dtype=np.float64
        )
        self._prev_applied_delta = np.zeros(len(ACTIVE_JOINTS), dtype=np.float64)
        self._last_sent_active_rad: np.ndarray | None = None

        self._estopped = False
        self._estop_reason: str | None = None
        self.last_estop_result: dict[str, Any] | None = None
        self._live_validated: bool = False

    @property
    def estopped(self) -> bool:
        return self._estopped

    @property
    def estop_reason(self) -> str | None:
        return self._estop_reason

    def clear_estop(self) -> None:
        """Explicitly clear a latched emergency stop."""
        self._estopped = False
        self._estop_reason = None
        self.last_estop_result = None

    def reset(self) -> None:
        """Reset temporal estimators without clearing a latched emergency stop."""
        self._velocity.reset()
        self._wrist_stack.reset()
        self._tripod_stack.reset()
        self._prev_applied_delta = np.zeros(len(ACTIVE_JOINTS), dtype=np.float64)
        if self.hw_cfg.dry_run:
            self._last_sent_active_rad = None

    def reset_episode(self) -> None:
        """Alias for reset() to satisfy TrialBridge protocol."""
        self.reset()

    def _validate_robot_config(self, robot: Any) -> None:
        cfg = getattr(robot, "config", None)
        if cfg is None:
            raise ValueError("Robot instance must expose a config attribute")
        if getattr(cfg, "use_degrees", None) is not True:
            raise ValueError("SO101Follower config must enforce use_degrees=True")
        if getattr(cfg, "max_relative_target", "UNSET") is not None:
            raise ValueError("SO101Follower config must enforce max_relative_target=None")

    def _validate_active_joint_norm_modes(self) -> None:
        for jname in ACTIVE_JOINTS:
            jc = self.cal.joints.get(jname)
            if jc is None or jc.norm_mode != "degrees":
                actual = None if jc is None else jc.norm_mode
                raise ValueError(
                    f"Active joint {jname!r} must have norm_mode=='degrees', got {actual!r}"
                )

    def _validate_calibration(self, robot: Any) -> None:
        robot_cal = getattr(robot, "calibration", None)
        if not isinstance(robot_cal, dict) or set(robot_cal.keys()) != set(self.cal.joints.keys()):
            raise ValueError(
                "Robot calibration keys must match CalibrationMap joints "
                f"{tuple(self.cal.joints.keys())}"
            )
        for jname, jc in self.cal.joints.items():
            entry = robot_cal[jname]
            for field in _CALIBRATION_FIELDS:
                if isinstance(entry, dict):
                    if field not in entry:
                        raise ValueError(
                            f"Robot calibration for {jname!r} missing field {field!r}"
                        )
                    raw_val = entry[field]
                elif hasattr(entry, field):
                    raw_val = getattr(entry, field)
                else:
                    raise ValueError(
                        f"Robot calibration for {jname!r} missing field {field!r}"
                    )
                if not isinstance(raw_val, (int, np.integer)) or isinstance(raw_val, bool):
                    raise ValueError(
                        f"Invalid calibration value for {jname!r}.{field}: expected int, got {raw_val!r}"
                    )
                actual = int(raw_val)
                expected = int(getattr(jc, field))
                if actual != expected:
                    raise ValueError(
                        f"Robot calibration mismatch for {jname!r}.{field}: "
                        f"got {actual}, expected {expected}"
                    )

    def connect_robot(self) -> None:
        """Connect live robot after verifying safety settings; strict no-op in dry_run."""
        if self.hw_cfg.dry_run:
            return
        self._live_validated = False
        if not self.hw_cfg.calibration_path:
            raise ValueError(
                "HardwareConfig.calibration_path must be set when dry_run=False"
            )
        self._validate_active_joint_norm_modes()
        if self.robot is None:
            self.robot = create_live_follower(self.hw_cfg)
        self._validate_robot_config(self.robot)
        if getattr(self.robot, "calibration", None) is not None:
            self._validate_calibration(self.robot)
        self.robot.connect(calibrate=False)
        try:
            if getattr(self.robot, "is_calibrated", False) is not True:
                raise ValueError(
                    "Live robot must report is_calibrated is True after connect(calibrate=False)"
                )
            self._validate_calibration(self.robot)
        except Exception:
            self._live_validated = False
            self.emergency_stop("calibration_mismatch")
            raise
        self._live_validated = True

    def emergency_stop(self, reason: str) -> dict[str, Any]:
        """Latch E-stop immediately and attempt to disable motor torque when live."""
        self._estopped = True
        self._estop_reason = reason
        torque_disabled_confirmed = False
        error: str | None = None

        if self.robot is not None and not self.hw_cfg.dry_run:
            try:
                self.robot.bus.disable_torque()
                torque_disabled_confirmed = True
            except Exception as exc:
                torque_disabled_confirmed = False
                error = str(exc)

        result = {
            "estopped": True,
            "reason": reason,
            "dry_run": self.hw_cfg.dry_run,
            "torque_disabled_confirmed": torque_disabled_confirmed,
            "error": error,
        }
        self.last_estop_result = result
        return dict(result)

    def close(self) -> None:
        """Disconnect live robot first (disables torque on disconnect in LeRobot), then close cameras."""
        self._live_validated = False
        self._last_sent_active_rad = None
        if self.robot is not None and not self.hw_cfg.dry_run:
            if getattr(self.robot, "is_connected", False):
                try:
                    self.robot.disconnect()
                except Exception:
                    _LOGGER.exception("Error disconnecting SO-101 robot during bridge.close()")
        for name, cam in (("wrist_cam", self.wrist_cam), ("tripod_cam", self.tripod_cam)):
            try:
                cam.close()
            except Exception:
                _LOGGER.exception("Error closing %s during bridge.close()", name)

    def _aborted_trace(
        self,
        t: float,
        reason: str,
        q_rad: np.ndarray | None = None,
        action: np.ndarray | None = None,
        guard_reason: str | None = None,
        wrist_age_s: float = float("inf"),
        tripod_age_s: float = float("inf"),
        wrist_rgb: np.ndarray | None = None,
        tripod_rgb: np.ndarray | None = None,
    ) -> StepTrace:
        torque_disabled_confirmed = (
            self.last_estop_result.get("torque_disabled_confirmed")
            if self.last_estop_result is not None
            else None
        )
        return StepTrace(
            t=t,
            q_rad=(
                q_rad.copy()
                if q_rad is not None
                else np.zeros(len(ALL_JOINTS), dtype=np.float64)
            ),
            action=action.copy() if action is not None else None,
            applied_delta_rad=np.zeros(len(ACTIVE_JOINTS), dtype=np.float64),
            guard_reason=guard_reason,
            wrist_age_s=wrist_age_s,
            tripod_age_s=tripod_age_s,
            aborted=True,
            abort_reason=reason,
            wrist_rgb=wrist_rgb,
            tripod_rgb=tripod_rgb,
            torque_disabled_confirmed=torque_disabled_confirmed,
        )

    def _read_joints_rad(self) -> tuple[float, float, np.ndarray]:
        t_start = float(self.clock())
        if self.robot is None:
            if not self.hw_cfg.dry_run:
                raise RuntimeError("No robot connected for live control")
            q_rad = self._home_all_6.copy()
        else:
            raw_obs = self.robot.get_observation()
            q_rad = np.asarray(
                [
                    hw_to_mjcf_rad(j, float(raw_obs[f"{j}.pos"]), self.cal)
                    for j in ALL_JOINTS
                ],
                dtype=np.float64,
            )
        t_end = float(self.clock())
        return t_start, t_end, q_rad

    @staticmethod
    def _stack_frames(stack: TimestampedFrameStack, k: int) -> np.ndarray:
        frames = stack.frames
        return np.stack((frames[0],) * (k - len(frames)) + frames)

    def _safe_clock(self, fallback: float = 0.0) -> float:
        try:
            timestamp = float(self.clock())
            return timestamp if math.isfinite(timestamp) else fallback
        except Exception:
            return fallback

    def _latch_unvalidated_stop(self, reason: str) -> None:
        self._estopped = True
        self._estop_reason = reason
        self.last_estop_result = {
            "estopped": True,
            "reason": reason,
            "dry_run": False,
            "torque_disabled_confirmed": False,
            "error": "robot_not_connected_or_validated",
        }

    def step(self) -> StepTrace:
        """Execute one safe perception -> actor -> FK-guard -> actuation cycle."""
        now_entry = 0.0
        q_all_6: np.ndarray | None = None
        action: np.ndarray | None = None
        guard_reason: str | None = None
        wrist_age_s = float("inf")
        tripod_age_s = float("inf")
        wrist_proc: np.ndarray | None = None
        tripod_proc: np.ndarray | None = None

        try:
            if self._estopped:
                return self._aborted_trace(
                    self._safe_clock(),
                    self._estop_reason or "estopped",
                )
            if not self.hw_cfg.dry_run and not self._live_validated:
                self._latch_unvalidated_stop("live_robot_not_validated")
                return self._aborted_trace(self._safe_clock(), "live_robot_not_validated")

            now_entry = float(self.clock())
            if not math.isfinite(now_entry):
                self.emergency_stop("non_finite_clock")
                return self._aborted_trace(0.0, "non_finite_clock")

            cfg_reason = _validate_runtime_thresholds(self.hw_cfg)
            if cfg_reason is not None:
                self.emergency_stop(cfg_reason)
                return self._aborted_trace(now_entry, cfg_reason)

            try:
                t_start, t_end, q_all_6 = self._read_joints_rad()
            except Exception as exc:
                reason = f"joint_read_failed:{exc}"
                self.emergency_stop(reason)
                return self._aborted_trace(self._safe_clock(now_entry), reason)

            if not (math.isfinite(t_start) and math.isfinite(t_end)):
                self.emergency_stop("stale_or_slow_joint_read")
                return self._aborted_trace(now_entry, "stale_or_slow_joint_read", q_rad=q_all_6)

            read_latency = t_end - t_start
            if (
                not (
                    math.isfinite(read_latency)
                    and math.isfinite(self.hw_cfg.max_joint_read_latency_s)
                    and 0.0 <= read_latency <= self.hw_cfg.max_joint_read_latency_s
                )
                or not np.isfinite(q_all_6).all()
            ):
                self.emergency_stop("stale_or_slow_joint_read")
                return self._aborted_trace(t_end, "stale_or_slow_joint_read", q_rad=q_all_6)

            q_active = q_all_6[: len(ACTIVE_JOINTS)]
            dq_active = self._velocity.update(t_start, q_active)
            if (
                not math.isfinite(self.hw_cfg.max_joint_stale_s)
                or self._velocity.is_stale(t_end, self.hw_cfg.max_joint_stale_s)
            ):
                self.emergency_stop("stale_or_slow_joint_read")
                return self._aborted_trace(t_end, "stale_or_slow_joint_read", q_rad=q_all_6)

            try:
                wrist_frame = self.wrist_cam.read_latest()
                tripod_frame = self.tripod_cam.read_latest()
            except Exception as exc:
                reason = f"camera_read_failed:{exc}"
                self.emergency_stop(reason)
                return self._aborted_trace(self._safe_clock(t_end), reason, q_rad=q_all_6)

            now_cam = float(self.clock())
            if (
                wrist_frame is None
                or tripod_frame is None
                or not math.isfinite(now_cam)
                or not math.isfinite(wrist_frame.capture_t)
                or not math.isfinite(tripod_frame.capture_t)
            ):
                self.emergency_stop("stale_camera_frame")
                return self._aborted_trace(self._safe_clock(t_end), "stale_camera_frame", q_rad=q_all_6)

            wrist_proc = preprocess_camera_frame(
                wrist_frame.rgb, out_h=self._render_h, out_w=self._render_w
            )
            tripod_proc = preprocess_camera_frame(
                tripod_frame.rgb, out_h=self._render_h, out_w=self._render_w
            )
            self._wrist_stack.push(wrist_proc, wrist_frame.capture_t)
            self._tripod_stack.push(tripod_proc, tripod_frame.capture_t)

            wrist_age_s = self._wrist_stack.age_s(now_cam)
            tripod_age_s = self._tripod_stack.age_s(now_cam)
            wrist_fresh = (
                math.isfinite(wrist_age_s)
                and math.isfinite(self.hw_cfg.max_camera_age_s)
                and 0.0 <= wrist_age_s <= self.hw_cfg.max_camera_age_s
                and self._wrist_stack.is_fresh(now_cam, self.hw_cfg.max_camera_age_s)
            )
            tripod_fresh = (
                math.isfinite(tripod_age_s)
                and math.isfinite(self.hw_cfg.max_camera_age_s)
                and 0.0 <= tripod_age_s <= self.hw_cfg.max_camera_age_s
                and self._tripod_stack.is_fresh(now_cam, self.hw_cfg.max_camera_age_s)
            )
            if not (wrist_fresh and tripod_fresh):
                self.emergency_stop("stale_camera_frame")
                return self._aborted_trace(
                    now_cam,
                    "stale_camera_frame",
                    q_rad=q_all_6,
                    wrist_age_s=wrist_age_s,
                    tripod_age_s=tripod_age_s,
                    wrist_rgb=wrist_proc,
                    tripod_rgb=tripod_proc,
                )

            obs = {
                "wrist_rgb": self._stack_frames(self._wrist_stack, self._frame_stack),
                "tripod_rgb": self._stack_frames(self._tripod_stack, self._frame_stack),
                "camera_meta": np.asarray(
                    (
                        min(10.0, wrist_age_s),
                        float(wrist_fresh),
                        min(10.0, tripod_age_s),
                        float(tripod_fresh),
                    ),
                    dtype=np.float32,
                ),
                "proprio": scale_proprio(
                    q_active,
                    dq_active,
                    self._prev_applied_delta,
                    self._q_min,
                    self._q_max,
                    self._max_action_rad,
                ),
            }

            try:
                raw_action = self.actor.act(obs, deterministic=True)
                if isinstance(raw_action, torch.Tensor):
                    action = raw_action.detach().cpu().numpy().astype(np.float64).reshape(-1)
                else:
                    action = np.asarray(raw_action, dtype=np.float64).reshape(-1)
            except Exception as exc:
                reason = f"actor_failed:{exc}"
                self.emergency_stop(reason)
                return self._aborted_trace(
                    self._safe_clock(now_cam),
                    reason,
                    q_rad=q_all_6,
                    wrist_age_s=wrist_age_s,
                    tripod_age_s=tripod_age_s,
                    wrist_rgb=wrist_proc,
                    tripod_rgb=tripod_proc,
                )

            if action.shape != (len(ACTIVE_JOINTS),) or not np.isfinite(action).all():
                self.emergency_stop("invalid_actor_action")
                return self._aborted_trace(
                    self._safe_clock(now_cam),
                    "invalid_actor_action",
                    q_rad=q_all_6,
                    wrist_age_s=wrist_age_s,
                    tripod_age_s=tripod_age_s,
                    wrist_rgb=wrist_proc,
                    tripod_rgb=tripod_proc,
                )

            proposed = q_active + np.clip(action, -1.0, 1.0) * self._max_action_rad
            guard = self.safety_guard.guard_command(
                q_all_6,
                proposed,
                last_sent_active_rad=self._last_sent_active_rad,
            )
            guard_reason = guard.reason
            if guard.reason is not None and guard.reason.startswith("unsafe_measured_pose:"):
                self.emergency_stop(guard.reason)
                return self._aborted_trace(
                    self._safe_clock(now_cam),
                    guard.reason,
                    q_rad=q_all_6,
                    action=action,
                    guard_reason=guard.reason,
                    wrist_age_s=wrist_age_s,
                    tripod_age_s=tripod_age_s,
                    wrist_rgb=wrist_proc,
                    tripod_rgb=tripod_proc,
                )

            # P0 Pre-Write Freshness Recheck after inference and FK verification.
            now_pre_write = float(self.clock())
            if not math.isfinite(now_pre_write):
                self.emergency_stop("inference_overrun_stale")
                return self._aborted_trace(
                    self._safe_clock(now_cam),
                    "inference_overrun_stale",
                    q_rad=q_all_6,
                    action=action,
                    guard_reason=guard.reason,
                    wrist_rgb=wrist_proc,
                    tripod_rgb=tripod_proc,
                )

            wrist_age_pre = self._wrist_stack.age_s(now_pre_write)
            tripod_age_pre = self._tripod_stack.age_s(now_pre_write)
            wrist_age_s = wrist_age_pre
            tripod_age_s = tripod_age_pre
            if (
                not math.isfinite(self.hw_cfg.max_joint_stale_s)
                or self._velocity.is_stale(now_pre_write, self.hw_cfg.max_joint_stale_s)
                or not (
                    math.isfinite(wrist_age_pre)
                    and math.isfinite(self.hw_cfg.max_camera_age_s)
                    and 0.0 <= wrist_age_pre <= self.hw_cfg.max_camera_age_s
                )
                or not (
                    math.isfinite(tripod_age_pre)
                    and math.isfinite(self.hw_cfg.max_camera_age_s)
                    and 0.0 <= tripod_age_pre <= self.hw_cfg.max_camera_age_s
                )
            ):
                self.emergency_stop("inference_overrun_stale")
                return self._aborted_trace(
                    now_pre_write,
                    "inference_overrun_stale",
                    q_rad=q_all_6,
                    action=action,
                    guard_reason=guard.reason,
                    wrist_age_s=wrist_age_pre,
                    tripod_age_s=tripod_age_pre,
                    wrist_rgb=wrist_proc,
                    tripod_rgb=tripod_proc,
                )

            applied_delta = np.zeros(len(ACTIVE_JOINTS), dtype=np.float64)
            if guard.allowed and not self.hw_cfg.dry_run:
                assert self.robot is not None
                self._validate_active_joint_norm_modes()
                cmd = {
                    f"{j}.pos": mjcf_rad_to_hw(j, float(guard.target_active_4_rad[i]), self.cal)
                    for i, j in enumerate(ACTIVE_JOINTS)
                }
                try:
                    sent = self.robot.send_action(cmd)
                    sent_rad = np.asarray(
                        [
                            hw_to_mjcf_rad(j, float(sent[f"{j}.pos"]), self.cal)
                            for j in ACTIVE_JOINTS
                        ],
                        dtype=np.float64,
                    )
                    if not np.isfinite(sent_rad).all():
                        raise ValueError("Non-finite joint value returned by send_action")
                except Exception as exc:
                    reason = f"send_action_failed:{exc}"
                    self.emergency_stop(reason)
                    return self._aborted_trace(
                        self._safe_clock(now_pre_write),
                        reason,
                        q_rad=q_all_6,
                        action=action,
                        guard_reason=guard.reason,
                        wrist_age_s=wrist_age_pre,
                        tripod_age_s=tripod_age_pre,
                        wrist_rgb=wrist_proc,
                        tripod_rgb=tripod_proc,
                    )
                self._last_sent_active_rad = sent_rad.copy()
                applied_delta = sent_rad - q_active

            self._prev_applied_delta = applied_delta.copy()
            return StepTrace(
                t=now_pre_write,
                q_rad=q_all_6.copy(),
                action=action,
                applied_delta_rad=applied_delta,
                guard_reason=guard.reason,
                wrist_age_s=wrist_age_pre,
                tripod_age_s=tripod_age_pre,
                aborted=False,
                abort_reason=None,
                wrist_rgb=wrist_proc,
                tripod_rgb=tripod_proc,
                torque_disabled_confirmed=None,
            )
        except Exception as exc:
            reason = f"step_exception:{type(exc).__name__}"
            if not self.hw_cfg.dry_run and not self._live_validated:
                self._latch_unvalidated_stop(reason)
            else:
                self.emergency_stop(reason)
            return self._aborted_trace(
                self._safe_clock(now_entry),
                reason,
                q_rad=q_all_6,
                action=action,
                guard_reason=guard_reason,
                wrist_age_s=wrist_age_s,
                tripod_age_s=tripod_age_s,
                wrist_rgb=wrist_proc,
                tripod_rgb=tripod_proc,
            )
