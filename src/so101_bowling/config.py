"""Workspace and joint-calibration configuration for SO-101 bowling."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

import yaml

T = TypeVar("T")

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
    pad_friction: tuple[float, float, float, float, float] = (0.50, 0.50, 0.005, 0.0025, 0.0025)
    desk_friction: tuple[float, float, float, float, float] = (0.28, 0.28, 0.003, 0.0008, 0.0008)
    pin_desk_friction: tuple[float, float, float] = (0.45, 0.005, 0.0005)


@dataclass(frozen=True)
class CameraConfig:
    wrist_pos: tuple[float, float, float] = (0.045, 0.0, -0.028)
    wrist_xyaxes: tuple[float, float, float, float, float, float] = (0.0, -1.0, 0.0, 0.990, 0.0, -0.140)
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
    full_fall_tilt_deg: float = 46.0
    wobble_duration_s: float = 0.02
    off_table_z_m: float = -0.05
    quiet_vel_mps: float = 0.02
    quiet_dwell_s: float = 0.25
    max_coast_s: float = 2.5


@dataclass(frozen=True)
class RewardConfig:
    gamma: float = 0.99
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


@dataclass(frozen=True)
class DRConfig:
    """Per-episode physics domain randomization ranges."""

    enabled: bool = True
    ball_spawn_dx: float = 0.03
    ball_spawn_dy: float = 0.05
    ball_mass_range: tuple[float, float] = (0.030, 0.120)
    pin_mass_range: tuple[float, float] = (0.015, 0.060)
    rack_shift_x: float = 0.04
    rack_shift_y: float = 0.08
    pin_pos_jitter: float = 0.008
    pin_init_tilt_deg: float = 3.0
    pad_roll_friction_scale: tuple[float, float] = (0.6, 1.8)
    desk_roll_friction_scale: tuple[float, float] = (0.6, 1.8)
    slide_friction_scale: tuple[float, float] = (0.7, 1.4)


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


def _build_section(cls: type, raw: dict[str, Any] | None):
    if raw is None:  # a section header with no keys keeps all defaults
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"{cls.__name__} section must be a YAML mapping")
    valid = {f.name for f in fields(cls)}
    unknown = sorted(map(str, set(raw) - valid))
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {', '.join(unknown)}")
    kwargs = {}
    for k, v in raw.items():
        if isinstance(v, list):
            kwargs[k] = tuple(tuple(x) if isinstance(x, list) else x for x in v)
        else:
            kwargs[k] = v
    return cls(**kwargs)


def load_config(path: str | Path | None = None) -> WorkspaceConfig:
    """Load WorkspaceConfig from YAML, defaulting to configs/workspace.yaml."""
    cfg_path = Path(path) if path is not None else _DEFAULT_WORKSPACE_YAML
    if not cfg_path.exists():
        raise FileNotFoundError(f"Workspace config not found: {cfg_path}")
    raw: dict[str, Any] = yaml.safe_load(cfg_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Workspace config must be a YAML mapping: {cfg_path}")
    unknown = sorted(map(str, set(raw) - {f.name for f in fields(WorkspaceConfig)}))
    if unknown:
        raise ValueError(f"Unknown WorkspaceConfig keys in {cfg_path}: {', '.join(unknown)}")
    return WorkspaceConfig(
        robot=_build_section(RobotConfig, raw.get("robot", {})),
        ball=_build_section(BallConfig, raw.get("ball", {})),
        pin=_build_section(PinConfig, raw.get("pin", {})),
        table=_build_section(TableConfig, raw.get("table", {})),
        camera=_build_section(CameraConfig, raw.get("camera", {})),
        delivery=_build_section(DeliveryConfig, raw.get("delivery", {})),
        reward=_build_section(RewardConfig, raw.get("reward", {})),
        dr=_build_section(DRConfig, raw.get("dr", {})),
    )


def load_calibration(yaml_path: str | Path | None = None) -> CalibrationMap:
    """Load joint calibration limits from configs/so101_calibration.yaml."""
    cal_path = Path(yaml_path) if yaml_path is not None else _DEFAULT_CALIBRATION_YAML
    if not cal_path.exists():
        raise FileNotFoundError(f"Calibration YAML not found: {cal_path}")
    raw = yaml.safe_load(cal_path.read_text()) or {}
    raw_joints: dict[str, dict[str, Any]] = raw.get("joints", {})
    joints = {
        name: JointCalibration(
            name=name,
            range_min=int(raw_joints[name]["range_min"]),
            range_max=int(raw_joints[name]["range_max"]),
            norm_mode=str(raw_joints[name]["norm_mode"]),
            mjcf_min=float(raw_joints[name]["mjcf_min"]),
            mjcf_max=float(raw_joints[name]["mjcf_max"]),
        )
        for name in ALL_JOINTS
    }
    return CalibrationMap(
        joints=joints,
        safety_margin_rad=float(raw.get("safety_margin_rad", 0.02)),
        counts_per_rev=int(raw.get("counts_per_rev", 4096)),
    )


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
            raise ValueError(f"Degenerate intersected limit for {jname}: [{lo}, {hi}]")
        limits[jname] = (lo, hi)
    return limits


def _coerce_scalar(value: Any, typ: type) -> Any:
    if typ is bool:
        if isinstance(value, bool):
            return value
        raise TypeError(f"Expected bool, got {type(value).__name__}: {value!r}")
    if typ in (int, float, str):
        return typ(value)
    return value


def load_policy_config(
    cls: type[T],
    yaml_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> T:
    """Load a policy config dataclass: defaults -> YAML file -> CLI overrides."""
    hints = get_type_hints(cls)
    valid = {f.name for f in fields(cls)}
    merged: dict[str, Any] = {}

    if yaml_path is not None:
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Policy config not found: {path}")
        raw = yaml.safe_load(path.read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Policy config must be a YAML mapping: {path}")
        unknown = sorted(set(raw) - valid)
        if unknown:
            raise ValueError(f"Unknown {cls.__name__} keys in {path}: {', '.join(unknown)}")
        merged.update(raw)

    if overrides:
        unknown = sorted(set(overrides) - valid)
        if unknown:
            raise ValueError(f"Unknown {cls.__name__} override keys: {', '.join(unknown)}")
        for k, v in overrides.items():
            if v is not None:
                merged[k] = v

    coerced = {k: _coerce_scalar(v, hints[k]) for k, v in merged.items()}
    return cls(**coerced)


def parse_policy_cli(
    cls: type[T],
    argv: list[str] | None = None,
    *,
    description: str = "",
) -> T:
    """Parse --config <yaml> plus auto-generated CLI flags for every dataclass field."""
    hints = get_type_hints(cls)
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, default=None, help="Path to policy YAML config.")

    for f in fields(cls):
        flag = f"--{f.name.replace('_', '-')}"
        typ = hints[f.name]
        help_text = f"(default: {f.default})"
        if typ is bool:
            parser.add_argument(flag, dest=f.name, action=argparse.BooleanOptionalAction, default=None, help=help_text)
        elif typ in (int, float, str):
            parser.add_argument(flag, dest=f.name, type=typ, default=None, help=help_text)
        else:
            raise TypeError(f"Unsupported CLI field type for {cls.__name__}.{f.name}: {typ}")

    ns = parser.parse_args(argv)
    overrides = {f.name: getattr(ns, f.name) for f in fields(cls) if getattr(ns, f.name) is not None}
    return load_policy_config(cls, yaml_path=ns.config, overrides=overrides)
