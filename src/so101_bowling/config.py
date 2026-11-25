"""Workspace configuration, calibration loading, and hardware-sim conversions."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ALL_JOINTS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
ACTIVE_JOINTS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
)
HELD_JOINTS: tuple[str, ...] = ("wrist_roll", "gripper")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_WORKSPACE_YAML = _REPO_ROOT / "configs" / "workspace.yaml"
_DEFAULT_CALIBRATION_YAML = _REPO_ROOT / "configs" / "so101_calibration.yaml"


@dataclass(frozen=True)
class RobotConfig:
    mjcf_path: str = "assets/so101/so101_new_calib_camera.xml"
    base_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    home_qpos: dict[str, float] = field(
        default_factory=lambda: {
            "shoulder_pan": 0.0,
            "shoulder_lift": -1.55,
            "elbow_flex": 1.05,
            "wrist_flex": 1.05,
            "wrist_roll": 0.0,
            "gripper": -0.17453297762778586,
        }
    )
    plastic_rgba: tuple[float, float, float, float] = (0.93, 0.92, 0.89, 1.0)
    servo_rgba: tuple[float, float, float, float] = (0.10, 0.10, 0.10, 1.0)
    control_hz: float = 20.0
    physics_dt: float = 0.002
    max_action_rad: float = 0.08


@dataclass(frozen=True)
class BallConfig:
    radius: float = 0.0485
    mass: float = 0.065
    spawn_pos: tuple[float, float] = (0.28, 0.0)
    rgba: tuple[float, float, float, float] = (0.08, 0.34, 0.86, 1.0)


@dataclass(frozen=True)
class PinConfig:
    count: int = 3
    height: float = 0.1778
    belly_radius: float = 0.03435
    neck_radius: float = 0.01615
    base_radius: float = 0.0255
    foot_radius: float = 0.015
    mass: float = 0.032
    positions: tuple[tuple[float, float], ...] = (
        (0.62, 0.0),
        (0.695, -0.044),
        (0.695, 0.044),
    )
    body_rgba: tuple[float, float, float, float] = (0.96, 0.94, 0.88, 1.0)
    stripe_rgba: tuple[float, float, float, float] = (0.82, 0.12, 0.12, 1.0)
    mesh_dir: str = "assets/bowling"
    num_radial_segments: int = 32
    num_height_segments: int = 48


@dataclass(frozen=True)
class TableConfig:
    desk_x_range: tuple[float, float] = (-0.15, 0.80)
    desk_y_range: tuple[float, float] = (-0.35, 0.35)
    desk_thickness: float = 0.03
    desk_rgba: tuple[float, float, float, float] = (0.80, 0.73, 0.62, 1.0)
    pad_x_range: tuple[float, float] = (0.08, 0.42)
    pad_y_range: tuple[float, float] = (-0.18, 0.18)
    pad_thickness: float = 0.0035
    pad_rgba: tuple[float, float, float, float] = (0.16, 0.17, 0.19, 1.0)
    pad_ramp_length: float = 0.015
    pad_friction: tuple[float, float, float, float, float] = (
        0.50,
        0.50,
        0.005,
        0.0025,
        0.0025,
    )
    desk_friction: tuple[float, float, float, float, float] = (
        0.28,
        0.28,
        0.003,
        0.0008,
        0.0008,
    )
    pin_desk_friction: tuple[float, float, float] = (0.45, 0.005, 0.0005)


@dataclass(frozen=True)
class CameraConfig:
    render_height: int = 64
    render_width: int = 112
    wrist_pos: tuple[float, float, float] = (0.045, 0.0, -0.028)
    wrist_xyaxes: tuple[float, float, float, float, float, float] = (
        0.0,
        -1.0,
        0.0,
        0.990,
        0.0,
        -0.140,
    )
    wrist_fovy: float = 72.0
    tripod_pos: tuple[float, float, float] = (0.36, -0.58, 0.44)
    tripod_target: tuple[float, float, float] = (0.38, 0.0, 0.06)
    tripod_fovy: float = 48.0
    overview_pos: tuple[float, float, float] = (0.32, 0.0, 0.95)
    overview_target: tuple[float, float, float] = (0.35, 0.0, 0.05)
    overview_fovy: float = 55.0


@dataclass(frozen=True)
class DeliveryConfig:
    max_steps: int = 80
    staging_clearance_m: float = 0.055
    foul_line_x: float = 0.44
    release_line_x: float = 0.44
    min_push_disp_m: float = 0.025
    min_launch_speed_mps: float = 0.25
    target_launch_speed_mps: float = 0.95
    aim_sigma_rad: float = 0.14
    sep_dist_m: float = 0.015
    sep_duration_s: float = 0.025
    wobble_tilt_deg: float = 12.0
    full_fall_tilt_deg: float = 45.0
    wobble_duration_s: float = 0.02
    off_table_z_m: float = -0.05
    quiet_vel_mps: float = 0.02
    quiet_dwell_s: float = 0.25
    max_coast_s: float = 2.5


@dataclass(frozen=True)
class RewardConfig:
    gamma: float = 0.99
    w_stage_phi: float = 0.40
    w_push_phi: float = 0.60
    w_contact_speed: float = 0.60
    w_launch_angle: float = 0.60
    w_action_jerk: float = 0.005
    r_pin_credit: float = 5.0
    r_strike_bonus: float = 5.0
    r_illegal_penalty: float = -2.0


@dataclass(frozen=True)
class DRConfig:
    enabled: bool = False
    ball_spawn_dx: float = 0.03
    ball_spawn_dy: float = 0.05
    ball_mass_range: tuple[float, float] = (0.030, 0.120)
    pin_mass_range: tuple[float, float] = (0.015, 0.060)
    pin_pos_jitter: float = 0.006
    pad_roll_friction_scale: tuple[float, float] = (0.6, 1.8)
    desk_roll_friction_scale: tuple[float, float] = (0.6, 1.8)
    slide_friction_scale: tuple[float, float] = (0.7, 1.4)
    joint_damping_scale: tuple[float, float] = (0.7, 1.4)
    actuator_kp_scale: tuple[float, float] = (0.85, 1.15)
    tripod_pos_jitter: float = 0.025
    tripod_target_jitter: float = 0.015
    tripod_fovy_jitter: float = 2.5
    wrist_pos_jitter: float = 0.004
    wrist_fovy_jitter: float = 2.5
    light_pos_jitter: float = 0.20
    light_intensity_scale: tuple[float, float] = (0.75, 1.25)
    color_jitter: float = 0.08
    num_distractors: int = 6
    brightness_delta: float = 0.15
    contrast_scale: tuple[float, float] = (0.85, 1.15)
    gamma_range: tuple[float, float] = (0.88, 1.12)
    saturation_scale: tuple[float, float] = (0.85, 1.15)
    noise_std: float = 4.0
    max_shift_px: int = 3


@dataclass(frozen=True)
class WorkspaceConfig:
    robot: RobotConfig = field(default_factory=RobotConfig)
    ball: BallConfig = field(default_factory=BallConfig)
    pin: PinConfig = field(default_factory=PinConfig)
    table: TableConfig = field(default_factory=TableConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    delivery: DeliveryConfig = field(default_factory=DeliveryConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    dr: DRConfig = field(default_factory=DRConfig)


@dataclass(frozen=True)
class JointCalibration:
    name: str
    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int
    norm_mode: str
    mjcf_min: float
    mjcf_max: float


@dataclass(frozen=True)
class CalibrationMap:
    joints: dict[str, JointCalibration]
    safety_margin_rad: float = 0.02
    counts_per_rev: int = 4096


def _to_tuples(raw: dict[str, Any], tuple_keys: set[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if k in tuple_keys and isinstance(v, list):
            out[k] = tuple(
                tuple(item) if isinstance(item, list) else item for item in v
            )
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None) -> WorkspaceConfig:
    """Load WorkspaceConfig from YAML, defaulting to configs/workspace.yaml."""
    cfg_path = Path(path) if path is not None else _DEFAULT_WORKSPACE_YAML
    if not cfg_path.exists():
        raise FileNotFoundError(f"Workspace config not found: {cfg_path}")

    raw: dict[str, Any] = yaml.safe_load(cfg_path.read_text()) or {}

    robot_raw = _to_tuples(
        raw.get("robot", {}), {"base_pos", "plastic_rgba", "servo_rgba"}
    )
    ball_raw = _to_tuples(raw.get("ball", {}), {"spawn_pos", "rgba"})
    pin_raw = _to_tuples(
        raw.get("pin", {}), {"positions", "body_rgba", "stripe_rgba"}
    )
    table_raw = _to_tuples(
        raw.get("table", {}),
        {
            "desk_x_range",
            "desk_y_range",
            "desk_rgba",
            "pad_x_range",
            "pad_y_range",
            "pad_rgba",
            "pad_friction",
            "desk_friction",
            "pin_desk_friction",
        },
    )
    cam_raw = _to_tuples(
        raw.get("camera", {}),
        {
            "wrist_pos",
            "wrist_xyaxes",
            "tripod_pos",
            "tripod_target",
            "overview_pos",
            "overview_target",
        },
    )
    delivery_raw = raw.get("delivery", {})
    reward_raw = raw.get("reward", {})
    dr_raw = _to_tuples(
        raw.get("dr", {}),
        {
            "ball_mass_range",
            "pin_mass_range",
            "pad_roll_friction_scale",
            "desk_roll_friction_scale",
            "slide_friction_scale",
            "joint_damping_scale",
            "actuator_kp_scale",
            "light_intensity_scale",
            "contrast_scale",
            "gamma_range",
            "saturation_scale",
        },
    )

    return WorkspaceConfig(
        robot=RobotConfig(**robot_raw),
        ball=BallConfig(**ball_raw),
        pin=PinConfig(**pin_raw),
        table=TableConfig(**table_raw),
        camera=CameraConfig(**cam_raw),
        delivery=DeliveryConfig(**delivery_raw),
        reward=RewardConfig(**reward_raw),
        dr=DRConfig(**dr_raw),
    )


def load_calibration(
    yaml_path: str | Path | None = None,
    live_json_path: str | Path | None = None,
) -> CalibrationMap:
    """Load calibration from YAML and optionally overlay a live LeRobot JSON file."""
    cal_path = Path(yaml_path) if yaml_path is not None else _DEFAULT_CALIBRATION_YAML
    if not cal_path.exists():
        raise FileNotFoundError(f"Calibration YAML not found: {cal_path}")

    raw = yaml.safe_load(cal_path.read_text()) or {}
    counts_per_rev = int(raw.get("counts_per_rev", 4096))
    safety_margin_rad = float(raw.get("safety_margin_rad", 0.02))
    raw_joints: dict[str, dict[str, Any]] = raw.get("joints", {})

    live_data: dict[str, dict[str, Any]] = {}
    if live_json_path is not None:
        live_p = Path(live_json_path)
        if not live_p.exists():
            raise FileNotFoundError(f"Live calibration JSON not found: {live_p}")
        live_data = json.loads(live_p.read_text())

    joints: dict[str, JointCalibration] = {}
    for jname in ALL_JOINTS:
        if jname not in raw_joints:
            raise KeyError(f"Missing joint {jname!r} in {cal_path}")
        entry = dict(raw_joints[jname])
        if jname in live_data:
            for field_name in (
                "id",
                "drive_mode",
                "homing_offset",
                "range_min",
                "range_max",
            ):
                if field_name in live_data[jname]:
                    entry[field_name] = live_data[jname][field_name]
        joints[jname] = JointCalibration(
            name=jname,
            id=int(entry["id"]),
            drive_mode=int(entry["drive_mode"]),
            homing_offset=int(entry["homing_offset"]),
            range_min=int(entry["range_min"]),
            range_max=int(entry["range_max"]),
            norm_mode=str(entry["norm_mode"]),
            mjcf_min=float(entry["mjcf_min"]),
            mjcf_max=float(entry["mjcf_max"]),
        )

    return CalibrationMap(
        joints=joints,
        safety_margin_rad=safety_margin_rad,
        counts_per_rev=counts_per_rev,
    )


def raw_ticks_to_hw(joint_name: str, raw_ticks: float, cal: CalibrationMap) -> float:
    """Convert raw STS3215 counts (post-EEPROM homing_offset) to LeRobot calibrated hw units.

    Matches LeRobot motors_bus.py:
    - DEGREES: deg = (raw - mid) * 360.0 / (counts_per_rev - 1), negated if drive_mode == 1.
    - RANGE_0_100: norm = (bounded_raw - range_min) / (range_max - range_min) * 100.0.
    """
    jc = cal.joints[joint_name]
    if jc.norm_mode == "degrees":
        mid = 0.5 * (jc.range_min + jc.range_max)
        deg = (raw_ticks - mid) * 360.0 / float(cal.counts_per_rev - 1)
        return -deg if jc.drive_mode else deg
    if jc.norm_mode == "range_0_100":
        bounded = min(max(raw_ticks, float(jc.range_min)), float(jc.range_max))
        span = float(jc.range_max - jc.range_min)
        if span <= 0.0:
            raise ValueError(f"Invalid range for {joint_name}: {jc.range_min}..{jc.range_max}")
        norm = (bounded - jc.range_min) / span * 100.0
        return 100.0 - norm if jc.drive_mode else norm
    raise ValueError(f"Unsupported norm_mode {jc.norm_mode!r} for {joint_name}")


def hw_to_mjcf_rad(joint_name: str, hw_val: float, cal: CalibrationMap) -> float:
    """Convert calibrated LeRobot joint value (degrees or 0..100 gripper) to MJCF radians.

    Note: homing_offset is already stored in motor EEPROM by LeRobot and is never reapplied.
    """
    jc = cal.joints[joint_name]
    if jc.norm_mode == "degrees":
        return math.radians(hw_val)
    if jc.norm_mode == "range_0_100":
        frac = hw_val / 100.0
        return jc.mjcf_min + frac * (jc.mjcf_max - jc.mjcf_min)
    raise ValueError(f"Unsupported norm_mode {jc.norm_mode!r} for {joint_name}")


def mjcf_rad_to_hw(joint_name: str, rad_val: float, cal: CalibrationMap) -> float:
    """Convert MJCF radians to calibrated LeRobot joint value (degrees or 0..100 gripper)."""
    jc = cal.joints[joint_name]
    if jc.norm_mode == "degrees":
        return math.degrees(rad_val)
    if jc.norm_mode == "range_0_100":
        span = jc.mjcf_max - jc.mjcf_min
        return (rad_val - jc.mjcf_min) / span * 100.0
    raise ValueError(f"Unsupported norm_mode {jc.norm_mode!r} for {joint_name}")


def intersect_joint_limits(cal: CalibrationMap) -> dict[str, tuple[float, float]]:
    """Intersect hardware calibrated limits and MJCF joint limits with safety margin."""
    limits: dict[str, tuple[float, float]] = {}
    denom = float(cal.counts_per_rev - 1)
    margin = cal.safety_margin_rad

    for jname, jc in cal.joints.items():
        if jc.norm_mode == "degrees":
            mid = 0.5 * (jc.range_min + jc.range_max)
            hw_min_rad = math.radians((jc.range_min - mid) * 360.0 / denom)
            hw_max_rad = math.radians((jc.range_max - mid) * 360.0 / denom)
            lo = max(hw_min_rad, jc.mjcf_min) + margin
            hi = min(hw_max_rad, jc.mjcf_max) - margin
        elif jc.norm_mode == "range_0_100":
            lo = jc.mjcf_min + margin
            hi = jc.mjcf_max - margin
        else:
            raise ValueError(f"Unsupported norm_mode {jc.norm_mode!r} for {jname}")

        if lo >= hi:
            raise ValueError(
                f"Degenerate intersected limit for {jname}: [{lo}, {hi}]"
            )
        limits[jname] = (lo, hi)

    return limits
