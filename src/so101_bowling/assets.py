"""MuJoCo scene XML builder and entity ID lookup for SO-101 bowling."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from so101_bowling.config import (
    ACTIVE_JOINTS,
    ALL_JOINTS,
    HELD_JOINTS,
    PinConfig,
    WorkspaceConfig,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_UPSTREAM_YELLOW_RGBA = "1 0.82 0.12 1"


@dataclass(frozen=True)
class SceneEntityIds:
    active_joint_ids: list[int]
    all_joint_ids: list[int]
    active_qpos_addrs: list[int]
    active_qvel_addrs: list[int]
    active_actuator_ids: list[int]
    held_actuator_ids: list[int]
    gripper_site_id: int
    ball_body_id: int
    ball_geom_id: int
    ball_joint_qvel_addr: int
    pin_body_ids: list[int]
    pin_geom_ids: list[set[int]]
    pin_joint_qvel_addrs: list[int]
    pusher_geom_ids: set[int]
    arm_geom_ids: set[int]
    table_geom_ids: set[int]
    ball_joint_qpos_addr: int
    pin_joint_qpos_addrs: list[int]
    pad_pair_ids: list[int]
    desk_pair_ids: list[int]


def pin_radius_profile(z: np.ndarray, pin_cfg: PinConfig) -> np.ndarray:
    """Evaluate smooth axisymmetric pin radius r(z) for z in [0, pin_cfg.height]."""
    h = float(pin_cfg.height)
    r_base, r_belly, r_neck = float(pin_cfg.base_radius), float(pin_cfg.belly_radius), float(pin_cfg.neck_radius)
    u_ctrl = np.array([0.00, 0.06, 0.25, 0.43, 0.62, 0.72, 0.84, 0.94, 1.00])
    r_ctrl = np.array([
        r_base,
        0.80 * r_base + 0.20 * r_belly,
        r_belly,
        0.50 * (r_belly + r_neck),
        r_neck,
        1.08 * r_neck,
        min(1.45 * r_neck, 0.75 * r_belly),
        1.15 * r_neck,
        0.0,
    ])
    u = np.clip(np.asarray(z, dtype=np.float64) / h, 0.0, 1.0)
    r = np.zeros_like(u)
    for i in range(len(u_ctrl) - 1):
        u0, u1, r0, r1 = u_ctrl[i], u_ctrl[i + 1], r_ctrl[i], r_ctrl[i + 1]
        mask = (u >= u0) & (u <= u1 if i == len(u_ctrl) - 2 else u < u1)
        if np.any(mask):
            t = (u[mask] - u0) / (u1 - u0)
            r[mask] = r0 + (r1 - r0) * 0.5 * (1.0 - np.cos(np.pi * t))
    return r


def compute_hollow_pin_inertial(
    mass: float, pin_cfg: PinConfig
) -> tuple[float, tuple[float, float, float]]:
    """Compute (com_z, (ixx, iyy, izz)) for a uniform thin-shell hollow bowling pin."""
    z = np.linspace(0.0, float(pin_cfg.height), 401)
    r = pin_radius_profile(z, pin_cfg)
    r_mid = 0.5 * (r[:-1] + r[1:])
    z_mid = 0.5 * (z[:-1] + z[1:])
    d_area = 2.0 * np.pi * r_mid * np.hypot(r[1:] - r[:-1], z[1:] - z[:-1])
    base_area = math.pi * float(pin_cfg.base_radius) ** 2
    total_area = base_area + float(np.sum(d_area))
    dm_shell = mass * (d_area / total_area)
    m_base = mass * (base_area / total_area)
    com_z = float(np.sum(dm_shell * z_mid) / mass)
    izz = float(np.sum(dm_shell * (r_mid**2))) + 0.5 * m_base * (float(pin_cfg.base_radius) ** 2)
    ixx = (
        float(np.sum(dm_shell * (0.5 * r_mid**2 + (z_mid - com_z) ** 2)))
        + 0.25 * m_base * (float(pin_cfg.base_radius) ** 2)
        + m_base * (com_z**2)
    )
    return com_z, (ixx, ixx, izz)


def _lookat_xyaxes(pos: tuple[float, float, float], target: tuple[float, float, float]) -> str:
    fwd = np.asarray(target, dtype=np.float64) - np.asarray(pos, dtype=np.float64)
    fwd /= np.linalg.norm(fwd)
    up_hint = np.array([1.0, 0.0, 0.0] if abs(float(fwd[2])) > 0.99 else [0.0, 0.0, 1.0])
    right = np.cross(fwd, up_hint)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return " ".join(f"{v:.6f}" for v in (*right, *up))


def _fmt(vals: tuple[float, ...] | list[float]) -> str:
    return " ".join(f"{float(v):.6g}" for v in vals)


def _ensure(parent: ET.Element, tag: str) -> ET.Element:
    elem = parent.find(tag)
    return elem if elem is not None else ET.SubElement(parent, tag)


def build_scene_xml(cfg: WorkspaceConfig, project_root: str | Path | None = None) -> str:
    """Assemble the complete MuJoCo scene XML string combining SO-101 and bowling lane."""
    root_dir = Path(project_root) if project_root is not None else _REPO_ROOT
    mjcf_path = root_dir / cfg.robot.mjcf_path
    pin_mesh_dir = root_dir / cfg.pin.mesh_dir
    body_obj_path = pin_mesh_dir / "bowling_pin_body.obj"
    stripe_obj_path = pin_mesh_dir / "bowling_pin_stripe.obj"

    tree = ET.parse(mjcf_path)
    mj = tree.getroot()
    mj.set("model", "so101_bowling")

    compiler = _ensure(mj, "compiler")
    compiler.set("angle", "radian")
    compiler.set("meshdir", str((root_dir / "assets" / "so101" / "meshes").resolve()))
    compiler.set("autolimits", "true")

    option = mj.find("option")
    if option is None:
        option = ET.Element("option")
        mj.insert(0, option)
    option.attrib.update(
        timestep=f"{cfg.robot.physics_dt:.6g}",
        gravity="0 0 -9.81",
        integrator="implicitfast",
        cone="elliptic",
        impratio="10",
    )

    visual = _ensure(mj, "visual")
    _ensure(visual, "global").attrib.update(offwidth="640", offheight="480")
    _ensure(visual, "map").set("znear", "0.002")

    for geom in mj.findall(".//geom[@class='collision']"):
        geom.attrib.pop("material", None)
        geom.set("rgba", "0 0 0 0")

    plastic_str, servo_str = _fmt(cfg.robot.plastic_rgba), _fmt(cfg.robot.servo_rgba)
    for mat in mj.findall("./asset/material"):
        rgba = (mat.get("rgba") or "").strip()
        if rgba == _UPSTREAM_YELLOW_RGBA:
            mat.set("rgba", plastic_str)
        elif rgba.startswith("0.1 0.1 0.1"):
            mat.set("rgba", servo_str)

    worldbody = mj.find("worldbody")
    if worldbody is None:
        raise ValueError("Missing <worldbody> in SO-101 MJCF")
    base_body = worldbody.find("./body[@name='base']")
    if base_body is not None:
        base_body.set("pos", _fmt(cfg.robot.base_pos))

    gripper_body = worldbody.find(".//body[@name='gripper']")
    if gripper_body is not None and gripper_body.find("./camera[@name='wrist_cam']") is None:
        ET.SubElement(
            gripper_body,
            "camera",
            name="wrist_cam",
            pos=_fmt(cfg.camera.wrist_pos),
            xyaxes=_fmt(cfg.camera.wrist_xyaxes),
            fovy=f"{cfg.camera.wrist_fovy:.4g}",
        )

    asset = _ensure(mj, "asset")
    ET.SubElement(
        asset, "texture", name="studio_sky", type="skybox", builtin="gradient",
        rgb1="0.28 0.30 0.34", rgb2="0.11 0.12 0.14", width="256", height="256",
    )
    ET.SubElement(asset, "mesh", name="bowling_pin_body", file=str(body_obj_path.resolve()))
    ET.SubElement(asset, "mesh", name="bowling_pin_stripe", file=str(stripe_obj_path.resolve()))
    for name, rgba, spec, shin in (
        ("desk_mat", cfg.table.desk_rgba, "0.15", "0.2"),
        ("pad_mat", cfg.table.pad_rgba, "0.02", "0.05"),
        ("ball_mat", cfg.ball.rgba, "0.35", "0.6"),
        ("pin_body_mat", cfg.pin.body_rgba, "0.25", "0.4"),
        ("pin_stripe_mat", cfg.pin.stripe_rgba, "0.25", "0.4"),
    ):
        ET.SubElement(asset, "material", name=name, rgba=_fmt(rgba), specular=spec, shininess=shin)

    ET.SubElement(
        worldbody, "light", name="main_light", pos="0.40 -0.30 1.40",
        dir="-0.05 0.20 -0.98", diffuse="0.68 0.66 0.63", ambient="0.28 0.28 0.28", castshadow="true",
    )
    ET.SubElement(
        worldbody, "light", name="fill_light", pos="0.25 0.50 1.10",
        dir="0.0 -0.40 -0.91", diffuse="0.38 0.38 0.40", castshadow="false",
    )
    for cname, cpos, ctgt, cfovy in (
        ("tripod_cam", cfg.camera.tripod_pos, cfg.camera.tripod_target, cfg.camera.tripod_fovy),
        ("overview_cam", cfg.camera.overview_pos, cfg.camera.overview_target, cfg.camera.overview_fovy),
    ):
        ET.SubElement(
            worldbody, "camera", name=cname, pos=_fmt(cpos),
            xyaxes=_lookat_xyaxes(cpos, ctgt), fovy=f"{cfovy:.4g}",
        )

    dx0, dx1 = cfg.table.desk_x_range
    dy0, dy1 = cfg.table.desk_y_range
    px0, px1 = cfg.table.pad_x_range
    py0, py1 = cfg.table.pad_y_range
    dt, pt = float(cfg.table.desk_thickness), float(cfg.table.pad_thickness)
    pin_fric = _fmt(cfg.table.pin_desk_friction)
    for name, x0, x1, y0, y1 in (
        ("desk_back", dx0, px0, dy0, dy1),
        ("desk_front", px1, dx1, dy0, dy1),
        ("desk_left", px0, px1, py1, dy1),
        ("desk_right", px0, px1, dy0, py0),
    ):
        ET.SubElement(
            worldbody, "geom", name=name, type="box",
            pos=f"{0.5*(x0+x1):.6g} {0.5*(y0+y1):.6g} {-0.5*dt:.6g}",
            size=f"{0.5*(x1-x0):.6g} {0.5*(y1-y0):.6g} {0.5*dt:.6g}",
            material="desk_mat", condim="6", friction=pin_fric,
            priority="1", solref="0.02 1", contype="1", conaffinity="1",
        )

    pad_hz = 0.5 * (dt + pt)
    ET.SubElement(
        worldbody, "geom", name="mousepad", type="box",
        pos=f"{0.5*(px0+px1):.6g} {0.5*(py0+py1):.6g} {pt - pad_hz:.6g}",
        size=f"{0.5*(px1-px0):.6g} {0.5*(py1-py0):.6g} {pad_hz:.6g}",
        material="pad_mat", friction=pin_fric, priority="1", solref="0.02 1", contype="1", conaffinity="1",
    )

    ramp_len = float(cfg.table.pad_ramp_length)
    theta = math.atan2(pt, ramp_len)
    ramp_hz = 0.0015
    ET.SubElement(
        worldbody, "geom", name="pad_ramp", type="box",
        pos=f"{px1 + 0.5*ramp_len - ramp_hz*math.sin(theta):.6g} {0.5*(py0+py1):.6g} {0.5*pt - ramp_hz*math.cos(theta):.6g}",
        euler=f"0 {theta:.6g} 0",
        size=f"{0.5*math.hypot(ramp_len, pt):.6g} {0.5*(py1-py0):.6g} {ramp_hz:.6g}",
        material="pad_mat", priority="1", solref="0.02 1", contype="1", conaffinity="1",
    )

    br, bm = float(cfg.ball.radius), float(cfg.ball.mass)
    b_i = (2.0 / 3.0) * bm * (br**2)
    bx, by = cfg.ball.spawn_pos
    ball_body = ET.SubElement(worldbody, "body", name="ball", pos=f"{bx:.6g} {by:.6g} {pt + br:.6g}")
    ET.SubElement(ball_body, "freejoint", name="ball_joint")
    ET.SubElement(ball_body, "inertial", pos="0 0 0", mass=f"{bm:.6g}", diaginertia=f"{b_i:.8g} {b_i:.8g} {b_i:.8g}")
    ET.SubElement(
        ball_body, "geom", name="ball_geom", type="sphere", size=f"{br:.6g}",
        material="ball_mat", contype="1", conaffinity="1", solref="0.014 0.15", friction="0.12 0.005 0.0001",
    )

    com_z, (ixx, iyy, izz) = compute_hollow_pin_inertial(cfg.pin.mass, cfg.pin)
    h, foot_r, belly_r, neck_r = float(cfg.pin.height), float(cfg.pin.foot_radius), float(cfg.pin.belly_radius), float(cfg.pin.neck_radius)
    head_r = min(1.42 * neck_r, 0.72 * belly_r)
    pin_col = dict(rgba="0 0 0 0", group="3", contype="1", conaffinity="1", solref="0.014 0.15", friction="0.12 0.005 0.0001")

    for idx, (px, py) in enumerate(cfg.pin.positions):
        pb = ET.SubElement(worldbody, "body", name=f"pin_{idx}", pos=f"{px:.6g} {py:.6g} 0")
        ET.SubElement(pb, "freejoint", name=f"pin_joint_{idx}")
        ET.SubElement(pb, "inertial", pos=f"0 0 {com_z:.6g}", mass=f"{cfg.pin.mass:.6g}", diaginertia=f"{ixx:.8g} {iyy:.8g} {izz:.8g}")
        ET.SubElement(pb, "geom", name=f"pin_{idx}_vis_body", type="mesh", mesh="bowling_pin_body", material="pin_body_mat", contype="0", conaffinity="0", group="2")
        ET.SubElement(pb, "geom", name=f"pin_{idx}_vis_stripe", type="mesh", mesh="bowling_pin_stripe", material="pin_stripe_mat", contype="0", conaffinity="0", group="2")
        for suffix, zpos, sz in (
            ("base", "0 0 0.008", f"{foot_r:.6g} 0.008"),
            ("belly", "0 0 0.044", f"{belly_r:.6g} 0.024"),
            ("taper", "0 0 0.0815", f"{0.5*(belly_r+neck_r):.6g} 0.0135"),
            ("neck", "0 0 0.1125", f"{neck_r:.6g} 0.0175"),
        ):
            ET.SubElement(pb, "geom", name=f"pin_{idx}_col_{suffix}", type="cylinder", pos=zpos, size=sz, **pin_col)
        ET.SubElement(pb, "geom", name=f"pin_{idx}_col_head", type="capsule", fromto=f"0 0 0.145 0 0 {h - head_r:.6g}", size=f"{head_r:.6g}", **pin_col)

    contact = _ensure(mj, "contact")
    pad_f, desk_f = _fmt(cfg.table.pad_friction), _fmt(cfg.table.desk_friction)
    for g2, fric in (
        ("mousepad", pad_f),
        ("pad_ramp", pad_f),
        ("desk_back", desk_f),
        ("desk_front", desk_f),
        ("desk_left", desk_f),
        ("desk_right", desk_f),
    ):
        ET.SubElement(contact, "pair", name=f"ball_{g2}_pair", geom1="ball_geom", geom2=g2, condim="6", friction=fric)

    return ET.tostring(mj, encoding="unicode")


def get_scene_entity_ids(model: mujoco.MjModel) -> SceneEntityIds:
    """Resolve and return all MuJoCo entity IDs and addresses for the bowling scene."""
    jnt = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
    act = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    body = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    geom = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
    pair = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_PAIR, n)

    active_joint_ids = [jnt(n) for n in ACTIVE_JOINTS]
    all_joint_ids = [jnt(n) for n in ALL_JOINTS]
    pin_body_ids, pin_geom_ids, pin_qpos_addrs, pin_qvel_addrs = [], [], [], []
    for idx in range(3):
        bid = body(f"pin_{idx}")
        pin_body_ids.append(bid)
        pin_qpos_addrs.append(int(model.jnt_qposadr[jnt(f"pin_joint_{idx}")]))
        pin_qvel_addrs.append(int(model.jnt_dofadr[jnt(f"pin_joint_{idx}")]))
        pin_geom_ids.append({g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == bid})

    arm_bodies = {
        body(n) for n in (
            "base", "shoulder", "upper_arm", "lower_arm", "wrist",
            "gripper", "wrist_camera_mount", "wrist_camera", "moving_jaw_so101_v1",
        )
    }
    pusher_bodies = {body("gripper"), body("moving_jaw_so101_v1")}
    arm_geom_ids = {g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in arm_bodies and int(model.geom_contype[g]) != 0}
    pusher_geom_ids = {g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in pusher_bodies and int(model.geom_contype[g]) != 0}
    table_geom_ids = {geom(n) for n in ("mousepad", "pad_ramp", "desk_back", "desk_front", "desk_left", "desk_right")}

    return SceneEntityIds(
        active_joint_ids=active_joint_ids,
        all_joint_ids=all_joint_ids,
        active_qpos_addrs=[int(model.jnt_qposadr[j]) for j in active_joint_ids],
        active_qvel_addrs=[int(model.jnt_dofadr[j]) for j in active_joint_ids],
        active_actuator_ids=[act(n) for n in ACTIVE_JOINTS],
        held_actuator_ids=[act(n) for n in HELD_JOINTS],
        gripper_site_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe"),
        ball_body_id=body("ball"),
        ball_geom_id=geom("ball_geom"),
        ball_joint_qvel_addr=int(model.jnt_dofadr[jnt("ball_joint")]),
        pin_body_ids=pin_body_ids,
        pin_geom_ids=pin_geom_ids,
        pin_joint_qvel_addrs=pin_qvel_addrs,
        pusher_geom_ids=pusher_geom_ids,
        arm_geom_ids=arm_geom_ids,
        table_geom_ids=table_geom_ids,
        ball_joint_qpos_addr=int(model.jnt_qposadr[jnt("ball_joint")]),
        pin_joint_qpos_addrs=pin_qpos_addrs,
        pad_pair_ids=[pair(f"ball_{n}_pair") for n in ("mousepad", "pad_ramp")],
        desk_pair_ids=[pair(f"ball_{n}_pair") for n in ("desk_back", "desk_front", "desk_left", "desk_right")],
    )
