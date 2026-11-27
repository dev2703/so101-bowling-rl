"""Focused tests for SO101BowlingEnv, DeliveryScorer, train.py, and eval.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from so101_bowling import DeliveryPhase, DeliveryScorer, SO101BowlingEnv, ScorerConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


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
