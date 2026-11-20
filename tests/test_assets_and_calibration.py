"""Unit and physics validation tests for config, calibration, assets, and DR."""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from so101_bowling.assets import (
    build_scene_xml,
    compute_hollow_pin_inertia,
    compute_pin_com_z,
    generate_pin_meshes,
    get_scene_entity_ids,
)
from so101_bowling.config import (
    ACTIVE_JOINTS,
    ALL_JOINTS,
    HELD_JOINTS,
    hw_to_mjcf_rad,
    intersect_joint_limits,
    load_calibration,
    load_config,
    mjcf_rad_to_hw,
    raw_ticks_to_hw,
)
from so101_bowling.randomization import DomainRandomizer, augment_rgb

_LIVE_CAL_JSON = Path(
    "/Users/deveshvyas/.cache/huggingface/lerobot/calibration/robots/so_follower/aligned_follower.json"
)


def _load_obj_vertices(path: Path) -> np.ndarray:
    verts = []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            _, xs, ys, zs = line.split()
            verts.append((float(xs), float(ys), float(zs)))
    return np.asarray(verts, dtype=np.float64)


def test_calibration_roundtrip_and_no_double_homing() -> None:
    cal = load_calibration(
        live_json_path=_LIVE_CAL_JSON if _LIVE_CAL_JSON.exists() else None
    )
    assert tuple(cal.joints.keys()) == ALL_JOINTS

    # Verify round-trip for all 6 joints across range.
    for jname in ALL_JOINTS:
        jc = cal.joints[jname]
        if jc.norm_mode == "degrees":
            test_vals = [-90.0, -35.5, 0.0, 22.25, 85.0]
        else:
            test_vals = [0.0, 25.0, 50.0, 75.0, 100.0]
        for hw in test_vals:
            rad = hw_to_mjcf_rad(jname, hw, cal)
            rt = mjcf_rad_to_hw(jname, rad, cal)
            assert rt == pytest.approx(hw, abs=1e-9)

    # Verify raw_ticks_to_hw does NOT double-apply homing_offset:
    # midpoint raw tick must map to 0.0 degrees regardless of large homing_offset.
    for jname in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"):
        jc = cal.joints[jname]
        assert jc.homing_offset != 0
        mid_raw = 0.5 * (jc.range_min + jc.range_max)
        deg = raw_ticks_to_hw(jname, mid_raw, cal)
        assert deg == pytest.approx(0.0, abs=1e-9)
        assert hw_to_mjcf_rad(jname, deg, cal) == pytest.approx(0.0, abs=1e-9)

    # Verify gripper 0 = closed (mjcf_min) and 100 = open (mjcf_max).
    g_jc = cal.joints["gripper"]
    assert raw_ticks_to_hw("gripper", g_jc.range_min, cal) == pytest.approx(0.0, abs=1e-9)
    assert raw_ticks_to_hw("gripper", g_jc.range_max, cal) == pytest.approx(100.0, abs=1e-9)
    assert hw_to_mjcf_rad("gripper", 0.0, cal) == pytest.approx(g_jc.mjcf_min, abs=1e-9)
    assert hw_to_mjcf_rad("gripper", 100.0, cal) == pytest.approx(g_jc.mjcf_max, abs=1e-9)


def test_intersect_joint_limits_respects_margin() -> None:
    cal = load_calibration()
    limits = intersect_joint_limits(cal)
    assert set(limits.keys()) == set(ALL_JOINTS)
    margin = cal.safety_margin_rad
    for jname, (lo, hi) in limits.items():
        jc = cal.joints[jname]
        assert lo < hi
        assert lo >= jc.mjcf_min + margin - 1e-12
        assert hi <= jc.mjcf_max - margin + 1e-12
        if jc.norm_mode == "degrees":
            mid = 0.5 * (jc.range_min + jc.range_max)
            hw_lo = math.radians((jc.range_min - mid) * 360.0 / 4095.0)
            hw_hi = math.radians((jc.range_max - mid) * 360.0 / 4095.0)
            assert lo >= hw_lo + margin - 1e-12
            assert hi <= hw_hi - margin + 1e-12


def test_pin_mesh_dimensions_and_hollow_inertia(tmp_path: Path) -> None:
    cfg = load_config()
    body_path, stripe_path = generate_pin_meshes(tmp_path, cfg.pin)
    assert body_path.name == "bowling_pin_body.obj"
    assert stripe_path.name == "bowling_pin_stripe.obj"

    verts = _load_obj_vertices(body_path)
    z = verts[:, 2]
    r = np.hypot(verts[:, 0], verts[:, 1])

    assert z.min() == pytest.approx(0.0, abs=1e-4)
    assert z.max() == pytest.approx(0.1778, abs=1.5e-3)
    assert r.max() == pytest.approx(0.03435, abs=1.5e-3)

    base_r = r[np.isclose(z, 0.0, atol=1e-4) & (r > 1e-4)]
    assert np.mean(base_r) == pytest.approx(0.0255, abs=1.5e-3)

    neck_mask = (z > 0.09) & (z < 0.13)
    assert r[neck_mask].min() == pytest.approx(0.01615, abs=1.5e-3)

    # Hollow thin-shell inertia sanity checks.
    ixx, iyy, izz = compute_hollow_pin_inertia(cfg.pin.mass, cfg.pin)
    com_z = compute_pin_com_z(cfg.pin)
    assert 0.04 < com_z < 0.10
    assert ixx == pytest.approx(iyy)
    assert ixx > izz > 0.0


def test_scene_xml_compiles_and_entity_ids() -> None:
    cfg = load_config()
    xml_str = build_scene_xml(cfg)
    model = mujoco.MjModel.from_xml_string(xml_str)
    ids = get_scene_entity_ids(model)

    assert len(ids.active_joint_ids) == len(ACTIVE_JOINTS) == 4
    assert len(ids.held_joint_ids) == len(HELD_JOINTS) == 2
    assert len(ids.all_joint_ids) == len(ALL_JOINTS) == 6
    assert len(ids.active_qpos_addrs) == 4
    assert len(ids.active_qvel_addrs) == 4
    assert len(ids.active_actuator_ids) == 4
    assert len(ids.held_actuator_ids) == 2
    assert ids.gripper_site_id >= 0
    assert ids.ball_body_id >= 0
    assert ids.ball_geom_id >= 0
    assert len(ids.pin_body_ids) == 3
    assert len(ids.pin_geom_ids) == 3
    assert all(len(gset) >= 5 for gset in ids.pin_geom_ids)
    assert len(ids.pin_joint_qpos_addrs) == 3
    assert len(ids.pin_joint_qvel_addrs) == 3
    assert len(ids.pusher_geom_ids) > 0
    assert ids.pusher_geom_ids.issubset(ids.arm_geom_ids)
    assert len(ids.arm_body_ids) >= 6
    assert len(ids.pad_geom_ids) == 2
    assert len(ids.desk_geom_ids) == 4
    assert ids.wrist_cam_id >= 0
    assert ids.tripod_cam_id >= 0
    assert len(ids.distractor_body_ids) == cfg.dr.num_distractors
    assert len(ids.distractor_geom_ids) == cfg.dr.num_distractors


def test_pins_rest_upright_stability() -> None:
    cfg = load_config()
    model = mujoco.MjModel.from_xml_string(build_scene_xml(cfg))
    data = mujoco.MjData(model)
    ids = get_scene_entity_ids(model)

    # Set arm to home pose so it does not collide or drift.
    for jname, qpos_val in cfg.robot.home_qpos.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
        data.qpos[model.jnt_qposadr[jid]] = qpos_val
        data.ctrl[aid] = qpos_val
    mujoco.mj_forward(model, data)

    n_steps = int(round(1.0 / model.opt.timestep))
    for _ in range(n_steps):
        mujoco.mj_step(model, data)

    for bid in ids.pin_body_ids:
        xmat = data.xmat[bid].reshape(3, 3)
        z_dot = float(np.clip(xmat[2, 2], -1.0, 1.0))
        tilt_deg = math.degrees(math.acos(z_dot))
        assert tilt_deg < 0.5, f"Pin {bid} tilted {tilt_deg:.3f} deg"


def test_condim6_rolling_friction_and_pad_desk_seam() -> None:
    cfg = load_config()
    model = mujoco.MjModel.from_xml_string(build_scene_xml(cfg))
    data = mujoco.MjData(model)
    ids = get_scene_entity_ids(model)

    for jname, qpos_val in cfg.robot.home_qpos.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
        data.qpos[model.jnt_qposadr[jid]] = qpos_val
        data.ctrl[aid] = qpos_val
    mujoco.mj_forward(model, data)

    # Give ball initial +x velocity on mousepad; move pins aside to isolate rolling friction.
    for p_q in ids.pin_joint_qpos_addrs:
        data.qpos[p_q + 1] += 0.25
    data.qvel[ids.ball_joint_qvel_addr] = 0.80
    # Match initial rolling angular velocity (v / r) around +Y for clean rolling.
    data.qvel[ids.ball_joint_qvel_addr + 4] = 0.80 / cfg.ball.radius
    mujoco.mj_forward(model, data)

    vx_hist = []
    z_hist = []
    x_hist = []
    n_steps = int(round(0.6 / model.opt.timestep))
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
        x_hist.append(float(data.xpos[ids.ball_body_id][0]))
        z_hist.append(float(data.xpos[ids.ball_body_id][2]))
        vx_hist.append(float(data.qvel[ids.ball_joint_qvel_addr]))

    # Ball crosses mousepad edge (x=0.42) onto desk smoothly without bouncing or exploding.
    assert max(x_hist) > 0.45
    assert min(z_hist) > 0.045
    assert max(z_hist) < 0.056
    # Measurable deceleration due to condim=6 rolling friction.
    assert vx_hist[-1] < 0.78
    assert vx_hist[-1] > 0.30


def test_domain_randomizer_and_rgb_augmentation() -> None:
    cfg = load_config()
    model = mujoco.MjModel.from_xml_string(build_scene_xml(cfg))
    data = mujoco.MjData(model)
    ids = get_scene_entity_ids(model)

    nom_mass = model.body_mass.copy()
    nom_pair_fric = model.pair_friction.copy()
    nom_cam_pos = model.cam_pos.copy()

    dr = DomainRandomizer(model, ids, cfg)
    rng = np.random.default_rng(42)

    info_on = dr.apply_episode(data, rng, dr_enabled=True)
    assert info_on["dr_enabled"] is True
    assert not np.allclose(model.body_mass, nom_mass)
    assert not np.allclose(model.pair_friction, nom_pair_fric)
    assert not np.allclose(model.cam_pos, nom_cam_pos)

    # Verify distractors placed outside lane and non-colliding.
    for bid, gid in zip(ids.distractor_body_ids, ids.distractor_geom_ids):
        assert model.body_pos[bid, 2] > -0.1
        assert model.geom_contype[gid] == 0

    info_off = dr.apply_episode(data, rng, dr_enabled=False)
    assert info_off["dr_enabled"] is False
    np.testing.assert_allclose(model.body_mass, nom_mass)
    np.testing.assert_allclose(model.pair_friction, nom_pair_fric)
    np.testing.assert_allclose(model.cam_pos, nom_cam_pos)

    # RGB augmentation preserves shape & dtype and leaves image unchanged when disabled.
    img = np.full((64, 112, 3), 120, dtype=np.uint8)
    img[:, :56, 0] = 200
    unchanged = augment_rgb(img, rng, dr_enabled=False)
    assert unchanged.dtype == np.uint8 and unchanged.shape == (64, 112, 3)
    np.testing.assert_array_equal(unchanged, img)

    aug = augment_rgb(img, rng, dr_enabled=True, shift_xy=(2, -1))
    assert aug.dtype == np.uint8 and aug.shape == (64, 112, 3)
    assert not np.array_equal(aug, img)


def test_render_wrist_and_tripod_cameras() -> None:
    cfg = load_config()
    model = mujoco.MjModel.from_xml_string(build_scene_xml(cfg))
    data = mujoco.MjData(model)

    for jname, qpos_val in cfg.robot.home_qpos.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
        data.qpos[model.jnt_qposadr[jid]] = qpos_val
        data.ctrl[aid] = qpos_val
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(
        model, height=cfg.camera.render_height, width=cfg.camera.render_width
    )
    try:
        renderer.update_scene(data, camera="wrist_cam")
        wrist_img = renderer.render()
        renderer.update_scene(data, camera="tripod_cam")
        tripod_img = renderer.render()
    finally:
        renderer.close()

    for img in (wrist_img, tripod_img):
        assert img.shape == (64, 112, 3)
        assert img.dtype == np.uint8
        assert img.mean() > 15.0
        assert img.std() > 10.0

    assert not np.allclose(wrist_img, tripod_img)
