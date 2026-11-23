"""Tests for 4-DOF IK, legal scripted 3/3 strike, strike corridor, and dt invariance."""

from __future__ import annotations

from dataclasses import replace

import mujoco
import numpy as np
import pytest

from so101_bowling.assets import build_scene_xml, get_scene_entity_ids
from so101_bowling.config import ACTIVE_JOINTS, load_calibration, load_config, intersect_joint_limits
from so101_bowling.env import SO101BowlingEnv
from so101_bowling.kinematics import PushParams, ScriptedPushController, solve_4dof_ik
from so101_bowling.scorer import DeliveryScorer, EpisodeOutcome, ScorerConfig, SubstepSnapshot


def _run_env_episode(physics_dt: float, params: PushParams) -> tuple[EpisodeOutcome, float]:
    """Run one full SO101BowlingEnv episode with ScriptedPushController."""
    cfg = load_config()
    cfg = replace(cfg, robot=replace(cfg.robot, physics_dt=physics_dt))
    max_arm_x_seen = -np.inf

    with SO101BowlingEnv(cfg=cfg, render_rgb=False, seed=42) as env:
        env.reset(seed=42)
        ctrl = ScriptedPushController(
            env.model, env.ids, cfg, env.q_min, env.q_max, params=params
        )
        ctrl.reset(
            env.data.xpos[env.ids.ball_body_id].copy(),
            env.data.xpos[env.ids.pin_body_ids].copy(),
        )
        max_arm_x_seen = max(max_arm_x_seen, float(env._last_snapshot.max_arm_x))

        info = {}
        for _ in range(cfg.delivery.max_steps):
            q_meas = env.data.qpos[env.ids.active_qpos_addrs].copy()
            action = ctrl.step(q_meas)
            _, _, done, _, info = env.step(action)
            max_arm_x_seen = max(max_arm_x_seen, float(env._last_snapshot.max_arm_x))
            if done:
                break

        raw = info["outcome"]
        outcome = EpisodeOutcome(**raw)
    return outcome, max_arm_x_seen


def test_solve_4dof_ik_reaches_staging_and_push_targets() -> None:
    cfg = load_config()
    model = mujoco.MjModel.from_xml_string(build_scene_xml(cfg))
    data = mujoco.MjData(model)
    ids = get_scene_entity_ids(model)
    limits = intersect_joint_limits(load_calibration())
    q_min, q_max = np.asarray([limits[j] for j in ACTIVE_JOINTS]).T

    for jname, val in cfg.robot.home_qpos.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        data.qpos[model.jnt_qposadr[jid]] = val
    mujoco.mj_forward(model, data)

    q_seed = np.array([0.0, -0.45, 0.75, 1.05], dtype=np.float64)
    for tgt, fwd_proj in (
        (np.array([0.195, 0.0, 0.052]), 0.20),
        (np.array([0.380, 0.0, 0.072]), 0.85),
    ):
        q_sol = solve_4dof_ik(
            model,
            data,
            ids,
            tgt,
            q_min,
            q_max,
            q_init=q_seed,
            prefer_fwd_vec=np.array([1.0, 0.0, fwd_proj]),
        )
        assert q_sol.shape == (4,)
        assert np.all(q_sol >= q_min - 1e-9)
        assert np.all(q_sol <= q_max + 1e-9)
        data.qpos[ids.active_qpos_addrs] = q_sol
        mujoco.mj_kinematics(model, data)
        err = float(np.linalg.norm(data.site_xpos[ids.gripper_site_id] - tgt))
        assert err < 2e-3, f"IK error {err:.5f} m for target {tgt}"
        q_seed = q_sol


def test_nominal_legal_scripted_strike_with_direct_scorer() -> None:
    """Verify M1 blocker #1: legal 4-DOF 3/3 strike with zero fouls/slams/double-taps."""
    outcome, max_arm_x = _run_env_episode(0.002, PushParams())
    assert outcome.legal_delivery is True
    assert outcome.invalid_reason is None
    assert outcome.credited_pins == 3
    assert outcome.legal_strike is True
    assert outcome.contact_forward_disp_m >= 0.025
    assert outcome.launch_speed >= 0.50
    assert max_arm_x <= 0.44

    # Also verify directly against DeliveryScorer + SubstepSnapshot without env wrapper.
    cfg = load_config()
    model = mujoco.MjModel.from_xml_string(build_scene_xml(cfg))
    data = mujoco.MjData(model)
    ids = get_scene_entity_ids(model)
    limits = intersect_joint_limits(load_calibration())
    q_min, q_max = np.asarray([limits[j] for j in ACTIVE_JOINTS]).T
    for jname, val in cfg.robot.home_qpos.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
        data.qpos[model.jnt_qposadr[jid]] = val
        data.ctrl[aid] = val
    mujoco.mj_forward(model, data)
    mujoco.mj_step(model, data, nstep=20)
    data.time = 0.0
    mujoco.mj_forward(model, data)

    scorer = DeliveryScorer(
        ScorerConfig(
            sim_dt=cfg.robot.physics_dt,
            min_wobble_substeps=max(2, round(0.02 / cfg.robot.physics_dt)),
        )
    )
    scorer.reset(
        data.xpos[ids.ball_body_id].copy(),
        data.xpos[ids.pin_body_ids].copy(),
        data.xmat[ids.pin_body_ids, 2::3].copy(),
        paddle_init_pos=data.site_xpos[ids.gripper_site_id].copy(),
    )
    ctrl = ScriptedPushController(model, ids, cfg, q_min, q_max, PushParams())
    ctrl.reset(data.xpos[ids.ball_body_id].copy(), data.xpos[ids.pin_body_ids].copy())
    substeps = round(1.0 / (cfg.robot.control_hz * cfg.robot.physics_dt))
    arm_geoms = np.array(sorted(ids.arm_geom_ids), dtype=np.intp)
    arm_aabb = model.geom_aabb[arm_geoms].reshape(-1, 2, 3)
    pin_by_geom = {g: i for i, gset in enumerate(ids.pin_geom_ids) for g in gset}

    def snap() -> SubstepSnapshot:
        paddle = data.site_xpos[ids.gripper_site_id]
        x_axes = data.geom_xmat[arm_geoms, :3]
        max_x = data.geom_xpos[arm_geoms, 0] + np.sum(
            x_axes * arm_aabb[:, 0] + np.abs(x_axes) * arm_aabb[:, 1], axis=1
        )
        pb = ob = ap = False
        bp = [False, False, False]
        pp = np.zeros((3, 3), dtype=bool)
        for ci in range(data.ncon):
            g1, g2 = int(data.contact[ci].geom1), int(data.contact[ci].geom2)
            if ids.ball_geom_id in (g1, g2):
                o = g2 if g1 == ids.ball_geom_id else g1
                pb |= o in ids.pusher_geom_ids
                ob |= o in ids.arm_geom_ids and o not in ids.pusher_geom_ids
                if o in pin_by_geom:
                    bp[pin_by_geom[o]] = True
                continue
            p1, p2 = pin_by_geom.get(g1), pin_by_geom.get(g2)
            if g1 in ids.arm_geom_ids or g2 in ids.arm_geom_ids:
                other = g2 if g1 in ids.arm_geom_ids else g1
                ap |= other in pin_by_geom
            elif p1 is not None and p2 is not None:
                pp[p1, p2] = pp[p2, p1] = True
        return SubstepSnapshot(
            t=float(data.time),
            paddle_pos=paddle.copy(),
            ball_pos=data.xpos[ids.ball_body_id].copy(),
            ball_vel=data.qvel[ids.ball_joint_qvel_addr : ids.ball_joint_qvel_addr + 3].copy(),
            pin_positions=data.xpos[ids.pin_body_ids].copy(),
            pin_up_vectors=data.xmat[ids.pin_body_ids, 2::3].copy(),
            pin_speeds=np.array([np.linalg.norm(data.qvel[pa : pa + 3]) for pa in ids.pin_joint_qvel_addrs]),
            max_arm_x=max(float(max_x.max()), float(paddle[0])),
            pusher_ball_contact=pb,
            other_arm_ball_contact=ob,
            arm_pin_contact=ap,
            table_slam=False,
            ball_pin_contacts=tuple(bp),
            pin_pin_contacts=pp,
            paddle_ball_surface_dist=min(
                mujoco.mj_geomDistance(model, data, g, ids.ball_geom_id, 0.5, None)
                for g in ids.pusher_geom_ids
            ),
        )

    for step in range(cfg.delivery.max_steps):
        qm = data.qpos[ids.active_qpos_addrs].copy()
        act = ctrl.step(qm)
        data.ctrl[ids.active_actuator_ids] = np.clip(qm + act * cfg.robot.max_action_rad, q_min, q_max)
        for _ in range(substeps):
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            scorer.update_substep(snap())
        done, _ = scorer.should_end_delivery((step + 1) / cfg.robot.control_hz, 4.0)
        if done:
            break
    data.ctrl[ids.active_actuator_ids] = data.qpos[ids.active_qpos_addrs]
    cs = 0
    settled, unres = scorer.is_coast_settled(0.0)
    while not settled:
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        scorer.update_substep(snap())
        cs += 1
        settled, unres = scorer.is_coast_settled(cs * cfg.robot.physics_dt)
    direct_out = scorer.finalize_episode(snap(), unres)
    assert direct_out.legal_strike is True
    assert direct_out.invalid_reason is None


@pytest.mark.parametrize("yaw_rad", [0.0, 0.03, -0.03])
def test_strike_corridor_centered_hits(yaw_rad: float) -> None:
    outcome, max_arm_x = _run_env_episode(0.002, PushParams(aim_yaw_offset_rad=yaw_rad))
    assert outcome.legal_delivery is True
    assert outcome.invalid_reason is None
    assert outcome.credited_pins == 3
    assert outcome.legal_strike is True
    assert max_arm_x <= 0.44


@pytest.mark.parametrize("yaw_rad", [0.15, -0.15])
def test_strike_corridor_off_center_aim(yaw_rad: float) -> None:
    outcome, max_arm_x = _run_env_episode(0.002, PushParams(aim_yaw_offset_rad=yaw_rad))
    assert outcome.legal_delivery is True
    assert outcome.invalid_reason is None
    assert outcome.launch_angle_err_rad > 0.05
    assert outcome.credited_pins < 3
    assert max_arm_x <= 0.44


@pytest.mark.parametrize("yaw_rad", [0.35, -0.35])
def test_strike_corridor_wide_miss_zero_pins(yaw_rad: float) -> None:
    outcome, max_arm_x = _run_env_episode(0.002, PushParams(aim_yaw_offset_rad=yaw_rad))
    assert outcome.legal_delivery is True
    assert outcome.invalid_reason is None
    assert outcome.credited_pins == 0
    assert outcome.legal_strike is False
    assert max_arm_x <= 0.44


def test_timestep_invariance_002_vs_004() -> None:
    """Verify M1 blocker #2: same 20 Hz push strikes legally at dt=0.002 and dt=0.004."""
    out_002, max_x_002 = _run_env_episode(0.002, PushParams())
    out_004, max_x_004 = _run_env_episode(0.004, PushParams())

    for out, mx in ((out_002, max_x_002), (out_004, max_x_004)):
        assert out.legal_delivery is True
        assert out.invalid_reason is None
        assert out.credited_pins == 3
        assert out.legal_strike is True
        assert mx <= 0.44

    rel_speed_diff = abs(out_004.launch_speed - out_002.launch_speed) / out_002.launch_speed
    assert rel_speed_diff <= 0.15, (
        f"Launch speed diverged across dt: {out_002.launch_speed:.3f} vs {out_004.launch_speed:.3f}"
    )


def test_scripted_push_dr_legality_and_ball_jitter() -> None:
    """Verify DR genuinely jitters ball/pins and ScriptedPushController stays >=85% legal."""
    cfg = load_config()
    nominal_ball = np.array([cfg.ball.spawn_pos[0], cfg.ball.spawn_pos[1], cfg.table.pad_thickness + cfg.ball.radius])
    saw_ball_jitter = False
    saw_pin_jitter = False
    legal_count = 0
    saw_lags: set[int] = set()
    episodes = 20

    with SO101BowlingEnv(cfg=cfg, dr_enabled=True, render_rgb=False) as env:
        ctrl = ScriptedPushController(env.model, env.ids, cfg, env.q_min, env.q_max)
        for seed in range(episodes):
            _, reset_info = env.reset(seed=seed)
            saw_lags.add(int(env._action_lag))
            ball_pos = env.data.xpos[env.ids.ball_body_id].copy()
            pin_pos = env.data.xpos[env.ids.pin_body_ids].copy()
            if np.linalg.norm(ball_pos[:2] - nominal_ball[:2]) > 1e-4:
                saw_ball_jitter = True
            if np.linalg.norm(pin_pos[:, :2] - np.asarray(cfg.pin.positions)) > 1e-4:
                saw_pin_jitter = True
            assert np.allclose(ball_pos[:2], np.asarray(reset_info["randomization"]["ball_spawn"])[:2], atol=1e-4)

            ctrl.reset(ball_pos, pin_pos)
            info = {}
            for _ in range(cfg.delivery.max_steps):
                action = ctrl.step(env._measured_q())
                _, _, done, _, info = env.step(action)
                if done:
                    break
            if info["outcome"]["legal_delivery"]:
                legal_count += 1

    assert saw_ball_jitter, "DomainRandomizer failed to apply ball spawn jitter"
    assert saw_pin_jitter, "DomainRandomizer failed to apply pin position jitter"
    assert saw_lags == {0, 1, 2}, f"Expected all action lags {{0,1,2}}, saw {saw_lags}"
    assert legal_count / episodes >= 0.85, f"DR legal rate too low: {legal_count}/{episodes}"
