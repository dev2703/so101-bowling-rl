"""Procedural bowling pin assets, MuJoCo scene XML builder, and entity ID lookup."""

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
    held_joint_ids: list[int]
    all_joint_ids: list[int]
    active_qpos_addrs: list[int]
    active_qvel_addrs: list[int]
    active_actuator_ids: list[int]
    held_actuator_ids: list[int]
    gripper_site_id: int
    ball_body_id: int
    ball_geom_id: int
    ball_joint_qpos_addr: int
    ball_joint_qvel_addr: int
    pin_body_ids: list[int]
    pin_geom_ids: list[set[int]]
    pin_joint_qpos_addrs: list[int]
    pin_joint_qvel_addrs: list[int]
    pusher_geom_ids: set[int]
    arm_geom_ids: set[int]
    arm_body_ids: list[int]
    pad_geom_ids: set[int]
    desk_geom_ids: set[int]
    wrist_cam_id: int
    tripod_cam_id: int
    distractor_body_ids: list[int]
    distractor_geom_ids: list[int]


def pin_radius_profile(z: np.ndarray, pin_cfg: PinConfig) -> np.ndarray:
    """Evaluate smooth axisymmetric pin radius r(z) for z in [0, pin_cfg.height]."""
    h = float(pin_cfg.height)
    r_base = float(pin_cfg.base_radius)
    r_belly = float(pin_cfg.belly_radius)
    r_neck = float(pin_cfg.neck_radius)

    u_ctrl = np.array([0.00, 0.06, 0.25, 0.43, 0.62, 0.72, 0.84, 0.94, 1.00])
    r_ctrl = np.array(
        [
            r_base,
            0.80 * r_base + 0.20 * r_belly,
            r_belly,
            0.50 * (r_belly + r_neck),
            r_neck,
            1.08 * r_neck,
            min(1.45 * r_neck, 0.75 * r_belly),
            1.15 * r_neck,
            0.0,
        ]
    )

    u = np.clip(np.asarray(z, dtype=np.float64) / h, 0.0, 1.0)
    r = np.zeros_like(u)
    for i in range(len(u_ctrl) - 1):
        u0, u1 = u_ctrl[i], u_ctrl[i + 1]
        r0, r1 = r_ctrl[i], r_ctrl[i + 1]
        mask = (u >= u0) & (u <= u1 if i == len(u_ctrl) - 2 else u < u1)
        if np.any(mask):
            t = (u[mask] - u0) / (u1 - u0)
            # Cosine blend preserves exact local extrema at control points.
            blend = 0.5 * (1.0 - np.cos(np.pi * t))
            r[mask] = r0 + (r1 - r0) * blend
    return r


def _build_lathe_obj(
    z_levels: np.ndarray,
    radii: np.ndarray,
    n_rad: int,
    cap_bottom: bool,
    cap_top: bool,
) -> str:
    """Build a Wavefront OBJ string from axisymmetric (z, radius) rings."""
    lines: list[str] = []
    angles = np.linspace(0.0, 2.0 * np.pi, n_rad, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    ring_start_idx: list[int] = []
    v_count = 0

    for z_val, rad in zip(z_levels, radii):
        if rad <= 1e-8:
            ring_start_idx.append(-1)
            continue
        ring_start_idx.append(v_count + 1)
        for j in range(n_rad):
            x = rad * cos_a[j]
            y = rad * sin_a[j]
            lines.append(f"v {x:.6f} {y:.6f} {z_val:.6f}")
            v_count += 1

    bottom_center_idx = None
    if cap_bottom and radii[0] > 1e-8:
        lines.append(f"v 0.000000 0.000000 {z_levels[0]:.6f}")
        v_count += 1
        bottom_center_idx = v_count

    top_center_idx = None
    if cap_top:
        lines.append(f"v 0.000000 0.000000 {z_levels[-1]:.6f}")
        v_count += 1
        top_center_idx = v_count

    # Side quads split into triangles
    for i in range(len(z_levels) - 1):
        i0 = ring_start_idx[i]
        i1 = ring_start_idx[i + 1]
        if i0 > 0 and i1 > 0:
            for j in range(n_rad):
                jn = (j + 1) % n_rad
                a, b = i0 + j, i0 + jn
                c, d = i1 + jn, i1 + j
                lines.append(f"f {a} {b} {c}")
                lines.append(f"f {a} {c} {d}")
        elif i0 > 0 and i1 < 0 and top_center_idx is not None:
            for j in range(n_rad):
                jn = (j + 1) % n_rad
                lines.append(f"f {i0 + j} {i0 + jn} {top_center_idx}")

    if bottom_center_idx is not None and radii[0] > 1e-8:
        # Duplicate bottom edge ring so flat cap normals do not bleed onto side walls.
        cap_ring_start = v_count + 1
        for j in range(n_rad):
            x = radii[0] * cos_a[j]
            y = radii[0] * sin_a[j]
            lines.append(f"v {x:.6f} {y:.6f} {z_levels[0]:.6f}")
            v_count += 1
        for j in range(n_rad):
            jn = (j + 1) % n_rad
            lines.append(f"f {bottom_center_idx} {cap_ring_start + jn} {cap_ring_start + j}")

    if top_center_idx is not None and ring_start_idx[-1] > 0:
        cap_top_start = v_count + 1
        for j in range(n_rad):
            x = radii[-1] * cos_a[j]
            y = radii[-1] * sin_a[j]
            lines.append(f"v {x:.6f} {y:.6f} {z_levels[-1]:.6f}")
            v_count += 1
        for j in range(n_rad):
            jn = (j + 1) % n_rad
            lines.append(f"f {cap_top_start + j} {cap_top_start + jn} {top_center_idx}")

    return "\n".join(lines) + "\n"


def generate_pin_meshes(out_dir: str | Path, pin_cfg: PinConfig) -> tuple[Path, Path]:
    """Generate cream body and red neck-stripe OBJ meshes for the bowling pin.

    Returns:
        (body_obj_path, stripe_obj_path)
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    h = float(pin_cfg.height)
    n_rad = int(pin_cfg.num_radial_segments)
    n_h = int(pin_cfg.num_height_segments)

    # Ensure exact control heights (0.25*H belly, 0.62*H neck) and smooth dome are sampled.
    u_grid = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 1.0, n_h + 1),
                np.array([0.06, 0.25, 0.43, 0.62, 0.72, 0.84, 0.94, 0.97, 0.99]),
            ]
        )
    )
    z_body = u_grid * h
    r_body = pin_radius_profile(z_body, pin_cfg)
    body_obj = _build_lathe_obj(z_body, r_body, n_rad, cap_bottom=True, cap_top=True)

    # Neck stripe sleeve (0.57*H .. 0.69*H) offset by 0.25 mm radially to avoid z-fighting.
    z_stripe = np.linspace(0.57 * h, 0.69 * h, 9)
    r_stripe = pin_radius_profile(z_stripe, pin_cfg) + 0.00025
    stripe_obj = _build_lathe_obj(
        z_stripe, r_stripe, n_rad, cap_bottom=False, cap_top=False
    )

    body_path = out_path / "bowling_pin_body.obj"
    stripe_path = out_path / "bowling_pin_stripe.obj"
    body_path.write_text(body_obj)
    stripe_path.write_text(stripe_obj)
    return body_path, stripe_path


def compute_hollow_pin_inertial(
    mass: float, pin_cfg: PinConfig
) -> tuple[float, tuple[float, float, float]]:
    """Compute (com_z, (ixx, iyy, izz)) for a uniform thin-shell hollow bowling pin."""
    if mass <= 0.0:
        raise ValueError(f"Pin mass must be positive, got {mass}")

    n_slices = 400
    z = np.linspace(0.0, float(pin_cfg.height), n_slices + 1)
    r = pin_radius_profile(z, pin_cfg)

    r0 = r[:-1]
    r1 = r[1:]
    z0 = z[:-1]
    z1 = z[1:]
    r_mid = 0.5 * (r0 + r1)
    z_mid = 0.5 * (z0 + z1)
    ds = np.hypot(r1 - r0, z1 - z0)
    d_area = 2.0 * np.pi * r_mid * ds

    base_area = math.pi * float(pin_cfg.base_radius) ** 2
    total_area = base_area + float(np.sum(d_area))

    dm_shell = mass * (d_area / total_area)
    m_base = mass * (base_area / total_area)

    com_z = float(np.sum(dm_shell * z_mid) / mass)

    izz_shell = float(np.sum(dm_shell * (r_mid**2)))
    izz_base = 0.5 * m_base * (float(pin_cfg.base_radius) ** 2)
    izz = izz_shell + izz_base

    ixx_shell = float(np.sum(dm_shell * (0.5 * r_mid**2 + (z_mid - com_z) ** 2)))
    ixx_base = 0.25 * m_base * (float(pin_cfg.base_radius) ** 2) + m_base * (com_z**2)
    ixx = ixx_shell + ixx_base
    return com_z, (ixx, ixx, izz)


def compute_pin_com_z(pin_cfg: PinConfig) -> float:
    """Return center-of-mass height (m) above pin base for a uniform hollow shell."""
    com_z, _ = compute_hollow_pin_inertial(pin_cfg.mass, pin_cfg)
    return com_z


def compute_hollow_pin_inertia(
    mass: float, pin_cfg: PinConfig
) -> tuple[float, float, float]:
    """Compute diagonal principal moments (ixx, iyy, izz) about CoM for a hollow pin."""
    _, diag = compute_hollow_pin_inertial(mass, pin_cfg)
    return diag


def lookat_axes(
    pos: tuple[float, float, float] | np.ndarray,
    target: tuple[float, float, float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute orthonormal camera (right, up, forward) basis vectors looking from pos to target."""
    p = np.asarray(pos, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    fwd = t - p
    fwd /= np.linalg.norm(fwd)
    up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(fwd, up_hint))) > 0.99:
        up_hint = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    right = np.cross(fwd, up_hint)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return right, up, fwd


def _lookat_xyaxes(
    pos: tuple[float, float, float], target: tuple[float, float, float]
) -> str:
    """Compute MuJoCo camera xyaxes string from world position and look-at target."""
    right, up, _ = lookat_axes(pos, target)
    return " ".join(f"{v:.6f}" for v in (*right, *up))


def _fmt(vals: tuple[float, ...] | list[float]) -> str:
    return " ".join(f"{float(v):.6g}" for v in vals)


def build_scene_xml(
    cfg: WorkspaceConfig, project_root: str | Path | None = None
) -> str:
    """Assemble the complete MuJoCo scene XML string combining SO-101 and bowling lane."""
    root_dir = Path(project_root) if project_root is not None else _REPO_ROOT
    mjcf_path = root_dir / cfg.robot.mjcf_path
    if not mjcf_path.exists():
        raise FileNotFoundError(f"SO-101 MJCF not found: {mjcf_path}")

    pin_mesh_dir = root_dir / cfg.pin.mesh_dir
    body_obj_path = pin_mesh_dir / "bowling_pin_body.obj"
    stripe_obj_path = pin_mesh_dir / "bowling_pin_stripe.obj"
    if not body_obj_path.exists() or not stripe_obj_path.exists():
        body_obj_path, stripe_obj_path = generate_pin_meshes(pin_mesh_dir, cfg.pin)

    tree = ET.parse(mjcf_path)
    mj = tree.getroot()
    mj.set("model", "so101_bowling")

    # Point compiler meshdir to vendored STL directory.
    compiler = mj.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(mj, "compiler")
    compiler.set("angle", "radian")
    compiler.set("meshdir", str((root_dir / "assets" / "so101" / "meshes").resolve()))
    compiler.set("autolimits", "true")

    # Simulation physics options (elliptic cone + implicitfast for accurate condim=6 rolling).
    option = mj.find("option")
    if option is None:
        option = ET.Element("option")
        mj.insert(0, option)
    option.set("timestep", f"{cfg.robot.physics_dt:.6g}")
    option.set("gravity", "0 0 -9.81")
    option.set("integrator", "implicitfast")
    option.set("cone", "elliptic")
    option.set("impratio", "10")

    visual = mj.find("visual")
    if visual is None:
        visual = ET.SubElement(mj, "visual")
    glob = visual.find("global")
    if glob is None:
        glob = ET.SubElement(visual, "global")
    glob.set("offwidth", "640")
    glob.set("offheight", "480")
    vmap = visual.find("map")
    if vmap is None:
        vmap = ET.SubElement(visual, "map")
    vmap.set("znear", "0.002")

    # Make all SO-101 arm collision geoms fully transparent so only visual meshes render.
    for geom in mj.findall(".//geom[@class='collision']"):
        geom.attrib.pop("material", None)
        geom.set("rgba", "0 0 0 0")

    # Recolor upstream yellow plastic to white/off-white and servos to matte charcoal.
    plastic_str = _fmt(cfg.robot.plastic_rgba)
    servo_str = _fmt(cfg.robot.servo_rgba)
    for mat in mj.findall("./asset/material"):
        rgba = (mat.get("rgba") or "").strip()
        if rgba == _UPSTREAM_YELLOW_RGBA:
            mat.set("rgba", plastic_str)
        elif rgba.startswith("0.1 0.1 0.1"):
            mat.set("rgba", servo_str)

    # Set base position and inject wrist_cam inside gripper body.
    worldbody = mj.find("worldbody")
    if worldbody is None:
        raise ValueError("Missing <worldbody> in SO-101 MJCF")
    base_body = worldbody.find("./body[@name='base']")
    if base_body is not None:
        base_body.set("pos", _fmt(cfg.robot.base_pos))

    gripper_body = worldbody.find(".//body[@name='gripper']")
    if gripper_body is None:
        raise ValueError("Missing <body name='gripper'> in SO-101 MJCF")
    if gripper_body.find("./camera[@name='wrist_cam']") is None:
        ET.SubElement(
            gripper_body,
            "camera",
            name="wrist_cam",
            pos=_fmt(cfg.camera.wrist_pos),
            xyaxes=_fmt(cfg.camera.wrist_xyaxes),
            fovy=f"{cfg.camera.wrist_fovy:.4g}",
        )

    # Assets: skybox, bowling pin meshes, and scene materials.
    asset = mj.find("asset")
    if asset is None:
        asset = ET.SubElement(mj, "asset")
    ET.SubElement(
        asset,
        "texture",
        name="studio_sky",
        type="skybox",
        builtin="gradient",
        rgb1="0.28 0.30 0.34",
        rgb2="0.11 0.12 0.14",
        width="256",
        height="256",
    )
    ET.SubElement(
        asset,
        "mesh",
        name="bowling_pin_body",
        file=str(body_obj_path.resolve()),
    )
    ET.SubElement(
        asset,
        "mesh",
        name="bowling_pin_stripe",
        file=str(stripe_obj_path.resolve()),
    )
    ET.SubElement(
        asset, "material", name="desk_mat", rgba=_fmt(cfg.table.desk_rgba), specular="0.15", shininess="0.2"
    )
    ET.SubElement(
        asset, "material", name="pad_mat", rgba=_fmt(cfg.table.pad_rgba), specular="0.02", shininess="0.05"
    )
    ET.SubElement(
        asset, "material", name="ball_mat", rgba=_fmt(cfg.ball.rgba), specular="0.35", shininess="0.6"
    )
    ET.SubElement(
        asset, "material", name="pin_body_mat", rgba=_fmt(cfg.pin.body_rgba), specular="0.25", shininess="0.4"
    )
    ET.SubElement(
        asset, "material", name="pin_stripe_mat", rgba=_fmt(cfg.pin.stripe_rgba), specular="0.25", shininess="0.4"
    )
    for idx in range(cfg.dr.num_distractors):
        ET.SubElement(
            asset,
            "material",
            name=f"distractor_mat_{idx}",
            rgba="0.5 0.5 0.5 1",
            specular="0.1",
        )

    # Lighting & fixed cameras.
    ET.SubElement(
        worldbody,
        "light",
        name="main_light",
        pos="0.40 -0.30 1.40",
        dir="-0.05 0.20 -0.98",
        diffuse="0.68 0.66 0.63",
        ambient="0.28 0.28 0.28",
        castshadow="true",
    )
    ET.SubElement(
        worldbody,
        "light",
        name="fill_light",
        pos="0.25 0.50 1.10",
        dir="0.0 -0.40 -0.91",
        diffuse="0.38 0.38 0.40",
        castshadow="false",
    )
    ET.SubElement(
        worldbody,
        "camera",
        name="tripod_cam",
        pos=_fmt(cfg.camera.tripod_pos),
        xyaxes=_lookat_xyaxes(cfg.camera.tripod_pos, cfg.camera.tripod_target),
        fovy=f"{cfg.camera.tripod_fovy:.4g}",
    )
    ET.SubElement(
        worldbody,
        "camera",
        name="overview_cam",
        pos=_fmt(cfg.camera.overview_pos),
        xyaxes=_lookat_xyaxes(cfg.camera.overview_pos, cfg.camera.overview_target),
        fovy=f"{cfg.camera.overview_fovy:.4g}",
    )

    # Partitioned desk top around mousepad (zero coplanar overlapping surfaces).
    dx0, dx1 = cfg.table.desk_x_range
    dy0, dy1 = cfg.table.desk_y_range
    px0, px1 = cfg.table.pad_x_range
    py0, py1 = cfg.table.pad_y_range
    dt = float(cfg.table.desk_thickness)
    pt = float(cfg.table.pad_thickness)
    pin_fric = _fmt(cfg.table.pin_desk_friction)

    desk_slabs = [
        ("desk_back", dx0, px0, dy0, dy1),
        ("desk_front", px1, dx1, dy0, dy1),
        ("desk_left", px0, px1, py1, dy1),
        ("desk_right", px0, px1, dy0, py0),
    ]
    for name, x0, x1, y0, y1 in desk_slabs:
        ET.SubElement(
            worldbody,
            "geom",
            name=name,
            type="box",
            pos=f"{0.5*(x0+x1):.6g} {0.5*(y0+y1):.6g} {-0.5*dt:.6g}",
            size=f"{0.5*(x1-x0):.6g} {0.5*(y1-y0):.6g} {0.5*dt:.6g}",
            material="desk_mat",
            condim="6",
            friction=pin_fric,
            priority="1",
            solref="0.02 1",
            contype="1",
            conaffinity="1",
        )

    # Mousepad fills [px0, px1] x [py0, py1] from z=-dt to z=+pt (raised 3.5 mm).
    pad_hz = 0.5 * (dt + pt)
    pad_cz = pt - pad_hz
    ET.SubElement(
        worldbody,
        "geom",
        name="mousepad",
        type="box",
        pos=f"{0.5*(px0+px1):.6g} {0.5*(py0+py1):.6g} {pad_cz:.6g}",
        size=f"{0.5*(px1-px0):.6g} {0.5*(py1-py0):.6g} {pad_hz:.6g}",
        material="pad_mat",
        friction=pin_fric,
        priority="1",
        solref="0.02 1",
        contype="1",
        conaffinity="1",
    )

    # Smooth bevel transition ramp from mousepad front edge (px1, z=pt) down to desk (z=0).
    ramp_len = float(cfg.table.pad_ramp_length)
    theta = math.atan2(pt, ramp_len)
    ramp_hx = 0.5 * math.hypot(ramp_len, pt)
    ramp_hz = 0.0015
    ramp_mx = px1 + 0.5 * ramp_len
    ramp_mz = 0.5 * pt
    ramp_cx = ramp_mx - ramp_hz * math.sin(theta)
    ramp_cz = ramp_mz - ramp_hz * math.cos(theta)
    ET.SubElement(
        worldbody,
        "geom",
        name="pad_ramp",
        type="box",
        pos=f"{ramp_cx:.6g} {0.5*(py0+py1):.6g} {ramp_cz:.6g}",
        euler=f"0 {theta:.6g} 0",
        size=f"{ramp_hx:.6g} {0.5*(py1-py0):.6g} {ramp_hz:.6g}",
        material="pad_mat",
        priority="1",
        solref="0.02 1",
        contype="1",
        conaffinity="1",
    )

    # Blue hollow plastic bowling ball.
    br = float(cfg.ball.radius)
    bm = float(cfg.ball.mass)
    b_inertia = (2.0 / 3.0) * bm * (br**2)
    bx, by = cfg.ball.spawn_pos
    ball_body = ET.SubElement(
        worldbody,
        "body",
        name="ball",
        pos=f"{bx:.6g} {by:.6g} {pt + br:.6g}",
    )
    ET.SubElement(ball_body, "freejoint", name="ball_joint")
    ET.SubElement(
        ball_body,
        "inertial",
        pos="0 0 0",
        mass=f"{bm:.6g}",
        diaginertia=f"{b_inertia:.8g} {b_inertia:.8g} {b_inertia:.8g}",
    )
    ET.SubElement(
        ball_body,
        "geom",
        name="ball_geom",
        type="sphere",
        size=f"{br:.6g}",
        material="ball_mat",
        contype="1",
        conaffinity="1",
        solref="0.014 0.15",
        friction="0.12 0.005 0.0001",
    )

    # 3 Hollow unweighted cream-white plastic bowling pins with compound collision primitives.
    com_z, (ixx, iyy, izz) = compute_hollow_pin_inertial(cfg.pin.mass, cfg.pin)
    h = float(cfg.pin.height)
    base_r = float(cfg.pin.base_radius)
    belly_r = float(cfg.pin.belly_radius)
    neck_r = float(cfg.pin.neck_radius)

    for idx, (px, py) in enumerate(cfg.pin.positions):
        pin_body = ET.SubElement(
            worldbody,
            "body",
            name=f"pin_{idx}",
            pos=f"{px:.6g} {py:.6g} 0",
        )
        ET.SubElement(pin_body, "freejoint", name=f"pin_joint_{idx}")
        ET.SubElement(
            pin_body,
            "inertial",
            pos=f"0 0 {com_z:.6g}",
            mass=f"{cfg.pin.mass:.6g}",
            diaginertia=f"{ixx:.8g} {iyy:.8g} {izz:.8g}",
        )
        # Visual lathe OBJ meshes.
        ET.SubElement(
            pin_body,
            "geom",
            name=f"pin_{idx}_vis_body",
            type="mesh",
            mesh="bowling_pin_body",
            material="pin_body_mat",
            contype="0",
            conaffinity="0",
            group="2",
        )
        ET.SubElement(
            pin_body,
            "geom",
            name=f"pin_{idx}_vis_stripe",
            type="mesh",
            mesh="bowling_pin_stripe",
            material="pin_stripe_mat",
            contype="0",
            conaffinity="0",
            group="2",
        )
        # Compound primitive collision profile: recessed foot cylinder + belly + taper + neck + head.
        # Kept transparent (rgba="0 0 0 0", group="3") so only cream/red visual meshes render.
        foot_r = float(cfg.pin.foot_radius)
        pin_col_attrs = {
            "rgba": "0 0 0 0",
            "group": "3",
            "contype": "1",
            "conaffinity": "1",
            "solref": "0.014 0.15",
            "friction": "0.12 0.005 0.0001",
        }
        ET.SubElement(
            pin_body,
            "geom",
            name=f"pin_{idx}_col_base",
            type="cylinder",
            pos="0 0 0.008",
            size=f"{foot_r:.6g} 0.008",
            **pin_col_attrs,
        )
        ET.SubElement(
            pin_body,
            "geom",
            name=f"pin_{idx}_col_belly",
            type="cylinder",
            pos="0 0 0.044",
            size=f"{belly_r:.6g} 0.024",
            **pin_col_attrs,
        )
        ET.SubElement(
            pin_body,
            "geom",
            name=f"pin_{idx}_col_taper",
            type="cylinder",
            pos="0 0 0.0815",
            size=f"{0.5 * (belly_r + neck_r):.6g} 0.0135",
            **pin_col_attrs,
        )
        ET.SubElement(
            pin_body,
            "geom",
            name=f"pin_{idx}_col_neck",
            type="cylinder",
            pos="0 0 0.1125",
            size=f"{neck_r:.6g} 0.0175",
            **pin_col_attrs,
        )
        head_r = min(1.42 * neck_r, 0.72 * belly_r)
        ET.SubElement(
            pin_body,
            "geom",
            name=f"pin_{idx}_col_head",
            type="capsule",
            fromto=f"0 0 0.145 0 0 {h - head_r:.6g}",
            size=f"{head_r:.6g}",
            **pin_col_attrs,
        )

    # Lightweight visual-only distractors (hidden at z=-5 in nominal mode).
    for idx in range(cfg.dr.num_distractors):
        db = ET.SubElement(
            worldbody,
            "body",
            name=f"distractor_{idx}",
            pos="0 0 -5",
        )
        gtype = ("box", "cylinder", "sphere")[idx % 3]
        size_str = "0.02 0.02 0.02" if gtype == "box" else ("0.02 0.025" if gtype == "cylinder" else "0.02")
        ET.SubElement(
            db,
            "geom",
            name=f"distractor_geom_{idx}",
            type=gtype,
            size=size_str,
            material=f"distractor_mat_{idx}",
            contype="0",
            conaffinity="0",
        )

    # Explicit condim="6" rolling/torsional/sliding contact pairs for ball-pad and ball-desk.
    contact = mj.find("contact")
    if contact is None:
        contact = ET.SubElement(mj, "contact")
    pad_f = _fmt(cfg.table.pad_friction)
    desk_f = _fmt(cfg.table.desk_friction)
    ET.SubElement(
        contact,
        "pair",
        name="ball_mousepad_pair",
        geom1="ball_geom",
        geom2="mousepad",
        condim="6",
        friction=pad_f,
    )
    ET.SubElement(
        contact,
        "pair",
        name="ball_pad_ramp_pair",
        geom1="ball_geom",
        geom2="pad_ramp",
        condim="6",
        friction=pad_f,
    )
    for dname in ("desk_back", "desk_front", "desk_left", "desk_right"):
        ET.SubElement(
            contact,
            "pair",
            name=f"ball_{dname}_pair",
            geom1="ball_geom",
            geom2=dname,
            condim="6",
            friction=desk_f,
        )

    return ET.tostring(mj, encoding="unicode")


def get_scene_entity_ids(model: mujoco.MjModel) -> SceneEntityIds:
    """Resolve and return all MuJoCo entity IDs and addresses for the bowling scene."""

    def _jnt_id(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

    def _act_id(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    def _body_id(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

    def _geom_id(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)

    def _cam_id(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)

    active_joint_ids = [_jnt_id(n) for n in ACTIVE_JOINTS]
    held_joint_ids = [_jnt_id(n) for n in HELD_JOINTS]
    all_joint_ids = [_jnt_id(n) for n in ALL_JOINTS]

    active_qpos_addrs = [int(model.jnt_qposadr[j]) for j in active_joint_ids]
    active_qvel_addrs = [int(model.jnt_dofadr[j]) for j in active_joint_ids]
    active_actuator_ids = [_act_id(n) for n in ACTIVE_JOINTS]
    held_actuator_ids = [_act_id(n) for n in HELD_JOINTS]

    gripper_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")

    ball_body_id = _body_id("ball")
    ball_geom_id = _geom_id("ball_geom")
    ball_jnt_id = _jnt_id("ball_joint")
    ball_joint_qpos_addr = int(model.jnt_qposadr[ball_jnt_id])
    ball_joint_qvel_addr = int(model.jnt_dofadr[ball_jnt_id])

    pin_body_ids: list[int] = []
    pin_geom_ids: list[set[int]] = []
    pin_joint_qpos_addrs: list[int] = []
    pin_joint_qvel_addrs: list[int] = []
    for idx in range(3):
        bid = _body_id(f"pin_{idx}")
        jid = _jnt_id(f"pin_joint_{idx}")
        pin_body_ids.append(bid)
        pin_joint_qpos_addrs.append(int(model.jnt_qposadr[jid]))
        pin_joint_qvel_addrs.append(int(model.jnt_dofadr[jid]))
        geoms = {
            g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == bid
        }
        pin_geom_ids.append(geoms)

    arm_non_base_names = (
        "shoulder",
        "upper_arm",
        "lower_arm",
        "wrist",
        "gripper",
        "wrist_camera_mount",
        "wrist_camera",
        "moving_jaw_so101_v1",
    )
    arm_body_ids = [_body_id(n) for n in arm_non_base_names]
    all_arm_bodies = {_body_id("base"), *arm_body_ids}
    arm_geom_ids = {
        g
        for g in range(model.ngeom)
        if int(model.geom_bodyid[g]) in all_arm_bodies
        and int(model.geom_contype[g]) != 0
    }

    pusher_bodies = {_body_id("gripper"), _body_id("moving_jaw_so101_v1")}
    pusher_geom_ids = {
        g
        for g in range(model.ngeom)
        if int(model.geom_bodyid[g]) in pusher_bodies
        and int(model.geom_contype[g]) != 0
    }

    pad_geom_ids = {_geom_id("mousepad"), _geom_id("pad_ramp")}
    desk_geom_ids = {
        _geom_id(n) for n in ("desk_back", "desk_front", "desk_left", "desk_right")
    }

    wrist_cam_id = _cam_id("wrist_cam")
    tripod_cam_id = _cam_id("tripod_cam")

    distractor_body_ids: list[int] = []
    distractor_geom_ids: list[int] = []
    idx = 0
    while True:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"distractor_{idx}")
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"distractor_geom_{idx}")
        if bid < 0 or gid < 0:
            break
        distractor_body_ids.append(bid)
        distractor_geom_ids.append(gid)
        idx += 1

    return SceneEntityIds(
        active_joint_ids=active_joint_ids,
        held_joint_ids=held_joint_ids,
        all_joint_ids=all_joint_ids,
        active_qpos_addrs=active_qpos_addrs,
        active_qvel_addrs=active_qvel_addrs,
        active_actuator_ids=active_actuator_ids,
        held_actuator_ids=held_actuator_ids,
        gripper_site_id=gripper_site_id,
        ball_body_id=ball_body_id,
        ball_geom_id=ball_geom_id,
        ball_joint_qpos_addr=ball_joint_qpos_addr,
        ball_joint_qvel_addr=ball_joint_qvel_addr,
        pin_body_ids=pin_body_ids,
        pin_geom_ids=pin_geom_ids,
        pin_joint_qpos_addrs=pin_joint_qpos_addrs,
        pin_joint_qvel_addrs=pin_joint_qvel_addrs,
        pusher_geom_ids=pusher_geom_ids,
        arm_geom_ids=arm_geom_ids,
        arm_body_ids=arm_body_ids,
        pad_geom_ids=pad_geom_ids,
        desk_geom_ids=desk_geom_ids,
        wrist_cam_id=wrist_cam_id,
        tripod_cam_id=tripod_cam_id,
        distractor_body_ids=distractor_body_ids,
        distractor_geom_ids=distractor_geom_ids,
    )
