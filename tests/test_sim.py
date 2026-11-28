"""Focused tests for SO101BowlingEnv, DeliveryScorer, train.py, and eval.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium.utils.env_checker import check_env

from so101_bowling import DeliveryPhase, DeliveryScorer, SO101BowlingEnv, ScorerConfig
from so101_bowling.scorer import SubstepSnapshot

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_snap(
    t: float,
    *,
    paddle_pos=(0.15, 0.0, 0.06),
    ball_pos=(0.28, 0.0, 0.052),
    ball_vel=(0.0, 0.0, 0.0),
    pin_tilts_deg=(0.0, 0.0, 0.0),
    pusher_ball_contact=False,
    other_arm_ball_contact=False,
    table_slam=False,
    ball_pin_contacts=(False, False, False),
    paddle_ball_surface_dist=0.10,
) -> SubstepSnapshot:
    pins = np.array([[0.62, 0.0, 0.0], [0.695, -0.044, 0.0], [0.695, 0.044, 0.0]], dtype=float)
    ups = np.array(
        [[np.sin(np.radians(deg)), 0.0, np.cos(np.radians(deg))] for deg in pin_tilts_deg],
        dtype=float,
    )
    return SubstepSnapshot(
        t=t,
        paddle_pos=np.array(paddle_pos, dtype=float),
        ball_pos=np.array(ball_pos, dtype=float),
        ball_vel=np.array(ball_vel, dtype=float),
        pin_positions=pins,
        pin_up_vectors=ups,
        pin_speeds=np.zeros(3, dtype=float),
        max_arm_x=float(paddle_pos[0]),
        pusher_ball_contact=pusher_ball_contact,
        other_arm_ball_contact=other_arm_ball_contact,
        arm_pin_contact=False,
        table_slam=table_slam,
        ball_pin_contacts=ball_pin_contacts,
        pin_pin_contacts=np.zeros((3, 3), dtype=bool),
        paddle_ball_surface_dist=paddle_ball_surface_dist,
    )


def _advance_to_released(scorer: DeliveryScorer) -> None:
    ball0 = np.array([0.28, 0.0, 0.052])
    pins0 = np.array([[0.62, 0.0, 0.0], [0.695, -0.044, 0.0], [0.695, 0.044, 0.0]])
    ups0 = np.tile(np.array([0.0, 0.0, 1.0]), (3, 1))
    scorer.reset(ball0, pins0, ups0, paddle_init_pos=np.array([0.15, 0.0, 0.06]))
    scorer.update_substep(_make_snap(0.01, pusher_ball_contact=True, paddle_ball_surface_dist=0.0))
    scorer.update_substep(
        _make_snap(0.02, ball_pos=(0.35, 0.0, 0.052), ball_vel=(0.8, 0.0, 0.0), pusher_ball_contact=True, paddle_ball_surface_dist=0.0)
    )
    t = 0.022
    for _ in range(15):
        scorer.update_substep(
            _make_snap(t, ball_pos=(0.45, 0.0, 0.052), ball_vel=(0.8, 0.0, 0.0), paddle_ball_surface_dist=0.05)
        )
        t += 0.002
    assert scorer.phase == DeliveryPhase.RELEASED


def test_env_reset_step_and_determinism() -> None:
    env1 = SO101BowlingEnv(seed=7)
    env2 = SO101BowlingEnv(seed=7)

    obs1, info1 = env1.reset(seed=7)
    obs2, info2 = env2.reset(seed=7)
    assert info1 == {} and info2 == {}
    assert obs1.shape == (53,) and obs1.dtype == np.float32
    assert np.all(np.isfinite(obs1))
    assert np.array_equal(obs1, obs2)

    assert env1.action_space.shape == (4,)
    assert env1.action_space.dtype == np.float32

    rng = np.random.default_rng(123)
    for _ in range(10):
        act = rng.uniform(-0.5, 0.5, size=(4,)).astype(np.float32)
        o1, r1, t1, tr1, _ = env1.step(act)
        o2, r2, t2, tr2, _ = env2.step(act)
        assert o1.shape == (53,) and o1.dtype == np.float32
        assert np.all(np.isfinite(o1))
        assert np.array_equal(o1, o2)
        assert r1 == r2 and t1 == t2 and tr1 == tr2
        if t1 or tr1:
            break


def test_env_termination_and_scorer_info_keys() -> None:
    env = SO101BowlingEnv(seed=0)
    env.reset(seed=0)
    zero = np.zeros(4, dtype=np.float32)

    info: dict = {}
    terminated = False
    for _ in range(env.cfg.delivery.max_steps + 5):
        _, _, terminated, truncated, info = env.step(zero)
        assert truncated is False
        if terminated:
            break

    assert terminated is True
    for key in ("outcome", "end_reason", "credited_pins", "legal_strike", "legal_delivery"):
        assert key in info
    assert info["end_reason"] == "delivery_timeout"
    assert info["legal_delivery"] is False
    assert info["credited_pins"] == 0
    assert info["legal_strike"] is False
    assert isinstance(info["outcome"], dict)

    with pytest.raises(Exception):
        env.step(zero)


def test_scorer_phase_and_critic_state() -> None:
    scorer = DeliveryScorer(ScorerConfig())
    ball = np.array([0.28, 0.0, 0.052])
    pins = np.array([[0.62, 0.0, 0.0], [0.695, -0.044, 0.0], [0.695, 0.044, 0.0]])
    ups = np.tile(np.array([0.0, 0.0, 1.0]), (3, 1))
    scorer.reset(ball, pins, ups, paddle_init_pos=np.array([0.15, 0.0, 0.06]))
    assert scorer.phase == DeliveryPhase.APPROACH
    cs = scorer.critic_state(0.75)
    assert cs.shape == (12,) and cs.dtype == np.float32
    assert cs[DeliveryPhase.APPROACH] == 1.0
    assert np.isclose(cs[4], 0.75)


def test_train_and_eval_cli_smoke(tmp_path: Path) -> None:
    ckpt = tmp_path / "ppo_smoke.pt"
    train_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train.py"),
        "--total-steps",
        "128",
        "--rollout-steps",
        "64",
        "--epochs",
        "2",
        "--batch-size",
        "32",
        "--seed",
        "0",
        "--out",
        str(ckpt),
    ]
    res_train = subprocess.run(train_cmd, capture_output=True, text=True, check=True)
    assert ckpt.exists(), res_train.stderr

    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    for key in ("actor_state_dict", "obs_dim", "act_dim", "obs_mean", "obs_std", "args"):
        assert key in payload
    assert payload["obs_dim"] == 53
    assert payload["act_dim"] == 4
    assert payload["obs_mean"].shape == (53,)
    assert payload["obs_std"].shape == (53,)

    eval_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval.py"),
        "--checkpoint",
        str(ckpt),
        "--episodes",
        "2",
        "--seed",
        "10",
    ]
    res_eval = subprocess.run(eval_cmd, capture_output=True, text=True, check=True)
    assert "SUMMARY" in res_eval.stdout

    rand_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval.py"),
        "--random",
        "--episodes",
        "2",
        "--seed",
        "10",
    ]
    res_rand = subprocess.run(rand_cmd, capture_output=True, text=True, check=True)
    assert "SUMMARY (random)" in res_rand.stdout


def test_policy_yaml_and_cli_override(tmp_path: Path) -> None:
    from dataclasses import dataclass
    from so101_bowling import load_policy_config, parse_policy_cli

    @dataclass(frozen=True)
    class DummyCfg:
        lr: float = 3e-4
        epochs: int = 4
        anneal_lr: bool = True
        device: str = "auto"

    yaml_file = tmp_path / "dummy.yaml"
    yaml_file.write_text("lr: 1e-4\nepochs: 8\nanneal_lr: false\n")

    # YAML overrides defaults; CLI overrides YAML
    cfg = parse_policy_cli(DummyCfg, ["--config", str(yaml_file), "--epochs", "2", "--anneal-lr"])
    assert cfg.lr == pytest.approx(1e-4)
    assert cfg.epochs == 2
    assert cfg.anneal_lr is True
    assert cfg.device == "auto"

    # Unknown YAML keys fail fast
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("lr: 1e-4\nunknown_knob: 123\n")
    with pytest.raises(ValueError, match="Unknown DummyCfg keys"):
        load_policy_config(DummyCfg, bad_yaml)

    # End-to-end smoke with configs/ppo.yaml + CLI overrides including custom hidden size
    ckpt = tmp_path / "ppo_yaml.pt"
    train_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train.py"),
        "--config",
        str(REPO_ROOT / "configs" / "ppo.yaml"),
        "--total-steps",
        "64",
        "--rollout-steps",
        "64",
        "--epochs",
        "1",
        "--batch-size",
        "32",
        "--hidden",
        "64",
        "--no-anneal-lr",
        "--out",
        str(ckpt),
    ]
    subprocess.run(train_cmd, capture_output=True, text=True, check=True)
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert payload["args"]["hidden"] == 64
    assert payload["args"]["anneal_lr"] is False

    eval_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval.py"),
        "--checkpoint",
        str(ckpt),
        "--episodes",
        "1",
    ]
    res_eval = subprocess.run(eval_cmd, capture_output=True, text=True, check=True)
    assert "SUMMARY" in res_eval.stdout


def test_scorer_strict_knock_threshold_and_partial_tilt_bins() -> None:
    scorer = DeliveryScorer(ScorerConfig())
    _advance_to_released(scorer)
    final_snap = _make_snap(
        0.10,
        ball_pos=(0.62, 0.0, 0.052),
        ball_pin_contacts=(True, True, False),
        pin_tilts_deg=(45.0, 46.5, 0.0),
    )
    out = scorer.finalize_episode(final_snap, unresolved_motion=False)
    assert out.legal_delivery is True
    assert out.credited_pins == 1
    assert out.full_knock_pins == 1
    assert out.partial_pin_reward == pytest.approx(1.0)
    assert out.pin_reward == pytest.approx(6.0)

    scorer2 = DeliveryScorer(ScorerConfig())
    _advance_to_released(scorer2)
    out2 = scorer2.finalize_episode(
        _make_snap(0.10, ball_pos=(0.62, 0.0, 0.052), ball_pin_contacts=(True, True, False), pin_tilts_deg=(4.9, 5.1, 0.0)),
        unresolved_motion=False,
    )
    assert out2.credited_pins == 0
    assert out2.partial_pin_reward == pytest.approx(1.0 / 9.0)


def test_scorer_first_contact_bonus_and_step_penalty() -> None:
    cfg = ScorerConfig(gamma=1.0)
    scorer = DeliveryScorer(cfg)
    ball0 = np.array([0.28, 0.0, 0.052])
    pins0 = np.array([[0.62, 0.0, 0.0], [0.695, -0.044, 0.0], [0.695, 0.044, 0.0]])
    ups0 = np.tile(np.array([0.0, 0.0, 1.0]), (3, 1))
    paddle = np.array([0.15, 0.0, 0.06])
    scorer.reset(ball0, pins0, ups0, paddle_init_pos=paddle)

    zero = np.zeros(4)
    # Stationary approach step -> only -w_step_penalty
    r_idle = scorer.step_control_reward(paddle, zero, zero, terminated=False)
    assert r_idle == pytest.approx(-cfg.w_step_penalty)

    # First legal pusher-ball contact awards r_first_contact (+1.0) once
    scorer.update_substep(_make_snap(0.01, paddle_pos=paddle, pusher_ball_contact=True, paddle_ball_surface_dist=0.0))
    r_contact = scorer.step_control_reward(paddle, zero, zero, terminated=False)
    assert r_contact > cfg.r_first_contact - cfg.w_step_penalty

    # Second pushing step with zero displacement does not award r_first_contact again
    scorer.update_substep(_make_snap(0.02, paddle_pos=paddle, pusher_ball_contact=True, paddle_ball_surface_dist=0.0))
    r_push_again = scorer.step_control_reward(paddle, zero, zero, terminated=False)
    assert r_push_again == pytest.approx(-cfg.w_step_penalty)


def test_scorer_tiered_penalties_and_timeout() -> None:
    ball0 = np.array([0.28, 0.0, 0.052])
    pins0 = np.array([[0.62, 0.0, 0.0], [0.695, -0.044, 0.0], [0.695, 0.044, 0.0]])
    ups0 = np.tile(np.array([0.0, 0.0, 1.0]), (3, 1))

    s_hard = DeliveryScorer(ScorerConfig())
    s_hard.reset(ball0, pins0, ups0)
    s_hard.update_substep(_make_snap(0.01, table_slam=True))
    assert s_hard.finalize_episode(_make_snap(0.01, table_slam=True), False).penalty_reward == pytest.approx(-2.0)

    s_soft = DeliveryScorer(ScorerConfig())
    s_soft.reset(ball0, pins0, ups0)
    s_soft.update_substep(_make_snap(0.01, other_arm_ball_contact=True))
    assert s_soft.finalize_episode(_make_snap(0.01, other_arm_ball_contact=True), False).penalty_reward == pytest.approx(-0.5)

    s_timeout = DeliveryScorer(ScorerConfig())
    s_timeout.reset(ball0, pins0, ups0)
    out_to = s_timeout.finalize_episode(_make_snap(1.0), False)
    assert out_to.invalid_reason == "no_legal_release"
    assert out_to.penalty_reward == pytest.approx(-0.5)


def test_scorer_ball_closest_approach_reward() -> None:
    s_hit = DeliveryScorer(ScorerConfig())
    _advance_to_released(s_hit)
    out_hit = s_hit.finalize_episode(_make_snap(0.10, ball_pos=(0.62, 0.0, 0.052)), False)

    s_miss = DeliveryScorer(ScorerConfig())
    _advance_to_released(s_miss)
    out_miss = s_miss.finalize_episode(_make_snap(0.10, ball_pos=(0.62, 0.25, 0.052)), False)

    assert out_hit.approach_reward == pytest.approx(1.5)
    assert out_hit.approach_reward > out_miss.approach_reward


def test_train_checkpointing_and_resume(tmp_path: Path) -> None:
    ckpt = tmp_path / "ppo.pt"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train.py"),
            "--total-steps",
            "128",
            "--rollout-steps",
            "64",
            "--save-every",
            "64",
            "--epochs",
            "1",
            "--batch-size",
            "32",
            "--seed",
            "0",
            "--out",
            str(ckpt),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    step64 = tmp_path / "ppo_step_64.pt"
    step128 = tmp_path / "ppo_step_128.pt"
    latest = tmp_path / "ppo_latest.pt"
    assert step64.exists() and step128.exists() and latest.exists() and ckpt.exists()

    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    for k in ("critic_state_dict", "optimizer_state_dict", "obs_var", "obs_count", "global_step"):
        assert k in payload
    assert payload["global_step"] == 128

    ckpt_resumed = tmp_path / "ppo_resumed.pt"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train.py"),
            "--resume",
            str(step64),
            "--total-steps",
            "192",
            "--rollout-steps",
            "64",
            "--save-every",
            "64",
            "--epochs",
            "1",
            "--batch-size",
            "32",
            "--seed",
            "0",
            "--out",
            str(ckpt_resumed),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    resumed_payload = torch.load(ckpt_resumed, map_location="cpu", weights_only=True)
    assert resumed_payload["global_step"] == 192


def test_workspace_config_rejects_unknown_keys(tmp_path: Path) -> None:
    from so101_bowling import load_config

    bad_section = tmp_path / "bad_section.yaml"
    bad_section.write_text("vision:\n  width: 64\n")
    with pytest.raises(ValueError, match="Unknown WorkspaceConfig keys.*vision"):
        load_config(bad_section)

    bad_key = tmp_path / "bad_key.yaml"
    bad_key.write_text("dr:\n  num_distractors: 6\n")
    with pytest.raises(ValueError, match="Unknown DRConfig keys: num_distractors"):
        load_config(bad_key)


def _dr_state(env: SO101BowlingEnv, seed: int) -> dict[str, np.ndarray]:
    obs, _ = env.reset(seed=seed)
    ids = env.ids
    ball = ids.ball_joint_qpos_addr
    return {
        "obs": obs,
        "ball_mass": env.model.body_mass[[ids.ball_body_id]],
        "pin_mass": env.model.body_mass[ids.pin_body_ids],
        "pair_friction": env.model.pair_friction.copy(),
        "ball_spawn": env.data.qpos[ball : ball + 2].copy(),
        "pin_pos": env.data.xpos[ids.pin_body_ids].copy(),
        "pin_up": env.data.xmat[ids.pin_body_ids, 2::3].copy(),
    }


def test_dr_varies_across_seeds_and_is_deterministic() -> None:
    env = SO101BowlingEnv(dr=True)
    first = _dr_state(env, 1)
    other = _dr_state(env, 999)
    # Same env after another seed, and a fresh env, both reproduce seed 1 exactly.
    for again in (_dr_state(env, 1), _dr_state(SO101BowlingEnv(dr=True), 1)):
        for key, value in first.items():
            assert np.array_equal(value, again[key]), key
    for key, value in first.items():
        assert not np.allclose(value, other[key]), key
    for state in (first, other):
        tilt = np.degrees(np.arccos(np.clip(state["pin_up"][:, 2], -1.0, 1.0)))
        assert np.all((tilt > 0.0) & (tilt <= env.cfg.dr.pin_init_tilt_deg + 0.2))

    nominal = SO101BowlingEnv(dr=False)
    assert np.array_equal(nominal.reset(seed=1)[0], nominal.reset(seed=999)[0])


def test_dr_pins_stay_upright_without_impact() -> None:
    env = SO101BowlingEnv(dr=True)
    zero = np.zeros(4, dtype=np.float32)
    for seed in range(100):
        env.reset(seed=seed)
        assert not env._last_snapshot.pin_pin_contacts.any(), seed
    for seed in (*range(20), 74):
        env.reset(seed=seed)
        for _ in range(env.cfg.delivery.max_steps):
            _, _, terminated, _, info = env.step(zero)
            tilt = np.degrees(np.arccos(np.clip(env.data.xmat[env.ids.pin_body_ids, 8], -1.0, 1.0)))
            assert tilt.max() < 15.0, (seed, tilt)
            if terminated:
                break
        assert terminated and info["end_reason"] == "delivery_timeout", (seed, info)
        assert info["credited_pins"] == 0, seed


def test_scorer_late_release_tracks_shifted_head_pin() -> None:
    ups0 = np.tile(np.array([0.0, 0.0, 1.0]), (3, 1))
    ball0 = np.array([0.28, 0.0, 0.052])

    # Head pin shifted back to x=0.66: separating at ball_x=0.58 (> nominal 0.57, < 0.61) is legal
    s_back = DeliveryScorer(ScorerConfig())
    pins_back = np.array([[0.66, 0.0, 0.0], [0.735, -0.044, 0.0], [0.735, 0.044, 0.0]])
    s_back.reset(ball0, pins_back, ups0, paddle_init_pos=np.array([0.15, 0.0, 0.06]))
    s_back.update_substep(_make_snap(0.01, pusher_ball_contact=True, paddle_ball_surface_dist=0.0))
    s_back.update_substep(
        _make_snap(0.02, ball_pos=(0.58, 0.0, 0.052), ball_vel=(0.8, 0.0, 0.0), pusher_ball_contact=True, paddle_ball_surface_dist=0.0)
    )
    t = 0.022
    for _ in range(15):
        s_back.update_substep(_make_snap(t, ball_pos=(0.58, 0.0, 0.052), ball_vel=(0.8, 0.0, 0.0), paddle_ball_surface_dist=0.05))
        t += 0.002
    assert s_back.phase == DeliveryPhase.RELEASED

    # Head pin shifted forward to x=0.58: pushing at ball_x=0.54 (> 0.53) triggers late_release_past_rack
    s_fwd = DeliveryScorer(ScorerConfig())
    pins_fwd = np.array([[0.58, 0.0, 0.0], [0.655, -0.044, 0.0], [0.655, 0.044, 0.0]])
    s_fwd.reset(ball0, pins_fwd, ups0, paddle_init_pos=np.array([0.15, 0.0, 0.06]))
    s_fwd.update_substep(_make_snap(0.01, pusher_ball_contact=True, paddle_ball_surface_dist=0.0))
    s_fwd.update_substep(
        _make_snap(0.02, ball_pos=(0.54, 0.0, 0.052), ball_vel=(0.8, 0.0, 0.0), pusher_ball_contact=True, paddle_ball_surface_dist=0.0)
    )
    assert s_fwd.phase == DeliveryPhase.INVALID
    assert s_fwd.invalid_reason == "late_release_past_rack"


def test_env_passes_gymnasium_checker() -> None:
    check_env(SO101BowlingEnv(seed=0), skip_render_check=True)


REWARD_PARTS = (
    "shaping_reward",
    "contact_reward",
    "cost_reward",
    "launch_reward",
    "approach_reward",
    "pin_reward",
    "penalty_reward",
)


def test_reward_components_sum_to_episode_return() -> None:
    env = SO101BowlingEnv(dr=False)
    # Same env twice (timeout, then touch-and-fault) so per-episode totals must also reset.
    outcomes = []
    for action in ((0.0, 0.0, 0.0, 0.0), (0.1, 0.2, 0.7, -0.7)):
        env.reset(seed=0)
        ep_ret, terminated = 0.0, False
        while not terminated:
            _, reward, terminated, _, info = env.step(np.array(action, dtype=np.float32))
            ep_ret += reward
        out = info["outcome"]
        assert sum(out[k] for k in REWARD_PARTS) == pytest.approx(ep_ret), info["end_reason"]
        outcomes.append((info["end_reason"], out))
    (timeout_reason, timeout), (_, fault) = outcomes
    assert timeout_reason == "delivery_timeout"
    assert timeout["contact_reward"] == 0.0 and timeout["cost_reward"] < 0.0
    assert fault["contact_reward"] > 0.0 and fault["penalty_reward"] < 0.0

    scorer = DeliveryScorer(ScorerConfig())
    _advance_to_released(scorer)
    ep_ret = scorer.step_control_reward((0.3, 0.0, 0.06), np.full(4, 0.5), np.zeros(4), terminated=False)
    ep_ret += scorer.step_control_reward((0.35, 0.0, 0.06), np.zeros(4), np.full(4, 0.5), terminated=True)
    legal = scorer.finalize_episode(
        _make_snap(0.10, ball_pos=(0.62, 0.0, 0.052), ball_pin_contacts=(True, True, False), pin_tilts_deg=(60.0, 20.0, 0.0)),
        unresolved_motion=False,
    )
    ep_ret += legal.total_terminal_reward
    assert legal.legal_delivery is True
    assert all(getattr(legal, k) != 0.0 for k in REWARD_PARTS if k != "penalty_reward")
    assert sum(getattr(legal, k) for k in REWARD_PARTS) == pytest.approx(ep_ret)


def test_vectorized_train_saves_deterministic_best(tmp_path: Path) -> None:
    ckpt = tmp_path / "ppo_vec.pt"
    train_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train.py"),
        "--num-envs",
        "2",
        "--eval-every",
        "64",
        "--eval-episodes",
        "2",
        "--total-steps",
        "128",
        "--rollout-steps",
        "64",
        "--epochs",
        "1",
        "--batch-size",
        "32",
        "--out",
        str(ckpt),
    ]
    res_train = subprocess.run(train_cmd, capture_output=True, text=True, check=True)
    best = tmp_path / "ppo_vec_best.pt"
    assert best.exists(), res_train.stderr
    assert "R[" in res_train.stdout
    assert "EVAL" in res_train.stdout
    best_payload = torch.load(best, map_location="cpu", weights_only=True)
    assert best_payload["best_return_kind"] == "eval"

    # Resuming a legacy checkpoint (no best_return_kind) must reset best_ret so deterministic eval replaces _best.pt.
    legacy = tmp_path / "legacy.pt"
    legacy_payload = dict(best_payload)
    legacy_payload.pop("best_return_kind")
    legacy_payload["best_return"] = 1e6
    legacy_payload["global_step"] = 64
    torch.save(legacy_payload, legacy)
    resumed_out = tmp_path / "legacy_resumed.pt"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train.py"),
            "--resume",
            str(legacy),
            "--num-envs",
            "2",
            "--eval-every",
            "64",
            "--eval-episodes",
            "2",
            "--total-steps",
            "128",
            "--rollout-steps",
            "64",
            "--epochs",
            "1",
            "--batch-size",
            "32",
            "--out",
            str(resumed_out),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    resumed_best = tmp_path / "legacy_resumed_best.pt"
    assert resumed_best.exists()
    resumed_best_payload = torch.load(resumed_best, map_location="cpu", weights_only=True)
    assert resumed_best_payload["best_return_kind"] == "eval"
    assert resumed_best_payload["best_return"] < 1e5

    eval_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval.py"),
        "--checkpoint",
        str(best),
        "--episodes",
        "2",
        "--no-dr",
    ]
    res_eval = subprocess.run(eval_cmd, capture_output=True, text=True, check=True)
    assert "SUMMARY" in res_eval.stdout


def test_eval_import_is_headless() -> None:
    code = "import sys; sys.path.insert(0, sys.argv[1]); import eval; assert 'mujoco.viewer' not in sys.modules"
    subprocess.run(
        [sys.executable, "-c", code, str(REPO_ROOT / "scripts")], capture_output=True, text=True, check=True
    )
