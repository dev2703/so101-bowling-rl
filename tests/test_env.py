"""Integration checks for timing, observation boundaries, and nominal safety."""

from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest

from so101_bowling import SO101BowlingEnv, load_config
from so101_bowling.vec_env import make_vec_env

ACTOR_KEYS = {"wrist_rgb", "tripod_rgb", "proprio", "camera_meta"}
ZERO = np.zeros(4, dtype=np.float32)


def assert_observation(env, obs):
    assert env.observation_space.contains(obs)
    assert {key for key in obs if not key.startswith("privileged_")} == ACTOR_KEYS
    for key in ("wrist_rgb", "tripod_rgb"):
        assert obs[key].shape == (2, 64, 112, 3)
        assert obs[key].dtype == np.uint8
    for key, size in {
        "proprio": 12, "camera_meta": 4, "privileged_state": 46,
        "privileged_aux_targets": 6, "privileged_aux_mask": 3,
    }.items():
        assert obs[key].shape == (size,)
        assert obs[key].dtype == np.float32
        assert np.isfinite(obs[key]).all()


def test_observation_and_single_renderer():
    with SO101BowlingEnv() as env:
        obs, _ = env.reset(seed=13)
        assert_observation(env, obs)
        assert env.action_space.contains(ZERO)
        assert env._scene_option.geomgroup[3] == 0
        renderer = env._renderer
        for key in ("wrist_rgb", "tripod_rgb"):
            assert np.ptp(obs[key][0]) > 0
            np.testing.assert_array_equal(obs[key][0], obs[key][1])
        action = np.full(4, 0.25, dtype=np.float32)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        assert_observation(env, next_obs)
        np.testing.assert_allclose(next_obs["proprio"][-4:], action)
        assert np.isfinite(reward)
        assert not terminated and not truncated
        assert env.render_camera("overview_cam", 128, 224).shape == (128, 224, 3)
        assert env._renderer is renderer
    assert env._renderer is None
    env.close()


@pytest.mark.parametrize("dr_enabled", [False, True])
def test_seeded_rollout(dr_enabled):
    with SO101BowlingEnv(dr_enabled=dr_enabled) as env:
        def rollout():
            initial, info = env.reset(seed=99)
            return info, [initial, *(env.step(ZERO)[0] for _ in range(5))]

        info1, first = rollout()
        info2, second = rollout()
        assert info1 == info2
        for obs1, obs2 in zip(first, second):
            for key in obs1:
                np.testing.assert_array_equal(obs1[key], obs2[key])


def test_home_clearance_and_idle_timeout():
    with SO101BowlingEnv(render_rgb=False) as env:
        env.reset(seed=7)
        table = env.ids.desk_geom_ids | env.ids.pad_geom_ids
        for _ in range(env.cfg.delivery.max_steps):
            assert env._last_snapshot.max_arm_x < min(0.35, env.scorer.config.foul_line_x)
            assert not any(
                (g1 in env.ids.arm_geom_ids and g2 in table)
                or (g2 in env.ids.arm_geom_ids and g1 in table)
                for g1, g2 in (contact.geom for contact in env.data.contact)
            )
            _, _, done, truncated, info = env.step(ZERO)
            assert not truncated
            if done:
                break
        assert env._step_count == env.cfg.delivery.max_steps
        assert info["end_reason"] == "delivery_timeout"
        assert env.data.time > env.delivery_timeout_s
        outcome = info["outcome"]
        assert outcome["credited_pins"] == outcome["wobble_only_pins"] == 0
        assert outcome["penalty_reward"] == 0.0
        assert not env.scorer.pin_wobbled.any()
        assert not env._last_snapshot.table_slam
        with pytest.raises(gym.error.ResetNeeded, match="reset"):
            env.step(ZERO)


def test_domain_randomization_and_nominal_restoration():
    with SO101BowlingEnv() as env:
        nominal, _ = env.reset(seed=3)
        masses = env.model.body_mass.copy()
        env.dr_enabled = True
        first, _ = env.reset(seed=3)
        first_masses = env.model.body_mass.copy()
        second, _ = env.reset(seed=4)
        assert not np.array_equal(masses, first_masses)
        assert not np.array_equal(first_masses, env.model.body_mass)
        for key in ("wrist_rgb", "tripod_rgb", "privileged_state"):
            assert not np.array_equal(nominal[key], first[key])
            assert not np.array_equal(first[key], second[key])
        env.dr_enabled = False
        restored, _ = env.reset(seed=26)
        np.testing.assert_array_equal(masses, env.model.body_mass)
        for key in nominal:
            np.testing.assert_array_equal(nominal[key], restored[key])


def test_physics_only_and_custom_timestep():
    cfg = load_config()
    cfg = replace(cfg, robot=replace(cfg.robot, physics_dt=0.004))
    with SO101BowlingEnv(cfg=cfg, render_rgb=False, seed=17) as env:
        obs, _ = env.reset()
        assert_observation(env, obs)
        assert env.scorer.config.sim_dt == 0.004
        assert env.scorer.config.min_wobble_substeps == 5
        obs, *_ = env.step(ZERO)
        assert_observation(env, obs)
        assert not obs["wrist_rgb"].any()
        assert not obs["tripod_rgb"].any()
        assert not obs["privileged_aux_mask"].any()
        np.testing.assert_array_equal(obs["camera_meta"], [10, 0, 10, 0])
        assert env._renderer is None
    assert env._renderer is None


def test_config_fields_propagate_to_scorer_and_dr_default():
    cfg = load_config()
    custom = replace(
        cfg,
        delivery=replace(cfg.delivery, foul_line_x=0.41, min_push_disp_m=0.031, wobble_tilt_deg=15.0),
        reward=replace(cfg.reward, w_push_phi=0.85, r_pin_credit=6.5, r_strike_bonus=4.0, r_illegal_penalty=-3.0, gamma=0.97),
        dr=replace(cfg.dr, enabled=True),
    )
    with SO101BowlingEnv(cfg=custom, render_rgb=False, seed=5) as env:
        assert env.dr_enabled is True
        sc = env.scorer.config
        assert sc.foul_line_x == 0.41
        assert sc.min_push_disp_m == 0.031
        assert sc.wobble_tilt_deg == 15.0
        assert sc.w_push_phi == 0.85
        assert sc.r_pin_credit == 6.5
        assert sc.r_strike_bonus == 4.0
        assert sc.r_illegal_penalty == -3.0
        assert sc.gamma == 0.97


def test_stationary_scene_stays_fresh_and_per_camera_shifts_independent():
    with SO101BowlingEnv(dr_enabled=False, render_rgb=True) as env:
        obs0, _ = env.reset(seed=11)
        np.testing.assert_allclose(obs0["camera_meta"], [0.0, 1.0, 0.0, 1.0])
        for _ in range(3):
            obs, _, done, _, _ = env.step(ZERO)
            assert not done
            # Live stationary renders advance capture time and stay fresh (not rejected by pixel equality).
            assert obs["camera_meta"][0] == pytest.approx(0.0, abs=1e-6)
            assert obs["camera_meta"][1] == 1.0
            assert obs["camera_meta"][2] == pytest.approx(0.0, abs=1e-6)
            assert obs["camera_meta"][3] == 1.0

    with SO101BowlingEnv(dr_enabled=True, render_rgb=False) as env:
        saw_distinct = False
        for seed in range(10):
            env.reset(seed=seed)
            assert len(env._shifts) == 2
            if env._shifts[0] != env._shifts[1]:
                saw_distinct = True
                break
        assert saw_distinct


def test_env_detects_deliberate_foul_and_table_slam():
    # Deliberate foul-line crossing via tightening foul_line_x right in front of home reach.
    cfg = load_config()
    foul_cfg = replace(cfg, delivery=replace(cfg.delivery, foul_line_x=0.25))
    with SO101BowlingEnv(cfg=foul_cfg, render_rgb=False) as env:
        env.reset(seed=2)
        done = False
        info = {}
        for _ in range(25):
            # Extend shoulder_lift & elbow forward along +X past foul_line_x=0.25.
            _, _, done, _, info = env.step(np.array([0.0, 1.0, -0.5, 0.0], dtype=np.float32))
            if done:
                break
        assert done
        assert info["outcome"]["legal_delivery"] is False
        assert info["outcome"]["invalid_reason"] == "foul_line_crossing"
        assert info["outcome"]["penalty_reward"] == cfg.reward.r_illegal_penalty

    # Deliberate downward table slam into mousepad/desk.
    with SO101BowlingEnv(cfg=cfg, render_rgb=False) as env:
        env.reset(seed=2)
        done = False
        info = {}
        for _ in range(40):
            _, _, done, _, info = env.step(np.array([0.0, 1.0, 0.5, 1.0], dtype=np.float32))
            if done:
                break
        assert done
        assert info["outcome"]["legal_delivery"] is False
        assert info["outcome"]["invalid_reason"] in ("table_slam", "foul_line_crossing", "non_pusher_ball_contact")


def test_pin_speeds_ignore_angular_velocity_during_coast():
    """Pin quietness checks must measure 3D linear speed (m/s), not 6D angular spin."""
    with SO101BowlingEnv(render_rgb=False, seed=3) as env:
        env.reset(seed=3)
        for addr in env.ids.pin_joint_qvel_addrs:
            env.data.qvel[addr : addr + 3] = [0.0, 0.0, 0.0]
            env.data.qvel[addr + 3 : addr + 6] = [2.5, -1.8, 3.0]
        snap = env._snapshot()
        np.testing.assert_allclose(snap.pin_speeds, 0.0, atol=1e-12)


def test_vectorized_snapshot_matches_contact_flags():
    """Vectorized contact scanner must detect ball-pin, pin-pin, and pusher-ball contacts."""
    with SO101BowlingEnv(render_rgb=False, seed=42) as env:
        env.reset(seed=42)
        from so101_bowling.kinematics import ScriptedPushController

        ctrl = ScriptedPushController(env.model, env.ids, env.cfg, env.q_min, env.q_max)
        ctrl.reset(env.data.xpos[env.ids.ball_body_id].copy(), env.data.xpos[env.ids.pin_body_ids].copy())
        saw_pusher = False
        for _ in range(env.cfg.delivery.max_steps):
            _, _, done, _, _ = env.step(ctrl.step(env._measured_q()))
            saw_pusher |= env.scorer.phase.value >= 1
            if done:
                break
        assert saw_pusher
        assert env.scorer.pin_impacted.all()


def test_action_lag_queue_executes_delayed_deltas_without_collapse():
    """Queued action deltas under lag > 0 must execute sequentially without stale-qpos collapse."""
    with SO101BowlingEnv(dr_enabled=True, render_rgb=False) as env:
        for seed in range(20):
            env.reset(seed=seed)
            if env._action_lag == 2:
                break
        assert env._action_lag == 2
        cmd = np.array([0.6, 0.0, 0.0, 0.0], dtype=np.float32)
        obs0, *_ = env.step(cmd)
        obs1, *_ = env.step(cmd)
        obs2, *_ = env.step(ZERO)
        obs3, *_ = env.step(ZERO)
        # First two steps execute seeded zero deltas; steps 2 and 3 both execute 0.6.
        assert abs(float(obs0["proprio"][8])) < 0.05
        assert abs(float(obs1["proprio"][8])) < 0.05
        assert float(obs2["proprio"][8]) == pytest.approx(0.6, abs=0.05)
        assert float(obs3["proprio"][8]) == pytest.approx(0.6, abs=0.05)


def test_make_vec_env_inherits_cfg_dr_enabled_by_default():
    """make_vec_env(dr_enabled=None) must match SO101BowlingEnv by inheriting cfg.dr.enabled."""
    cfg = replace(load_config(), dr=replace(load_config().dr, enabled=True))
    with make_vec_env(1, cfg=cfg, render_rgb=False, base_seed=0) as sync_env:
        obs_a = sync_env.reset(seed=0)
        obs_b = sync_env.reset(seed=1)
        assert not np.array_equal(obs_a["privileged_state"], obs_b["privileged_state"])

    with make_vec_env(2, cfg=cfg, render_rgb=False, base_seed=0) as subproc_env:
        obs_2a = subproc_env.reset(seed=0)
        obs_2b = subproc_env.reset(seed=10)
        assert not np.array_equal(obs_2a["privileged_state"], obs_2b["privileged_state"])
