"""Integration checks for timing, observation boundaries, and nominal safety."""

from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest

from so101_bowling import SO101BowlingEnv, load_config

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
