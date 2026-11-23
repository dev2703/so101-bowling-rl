"""Regression tests for bug fixes and edge cases across scorer, PPO, and SAC."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from so101_bowling import PPOConfig, SO101BowlingEnv, load_config
from so101_bowling.ppo import evaluate_actor
from so101_bowling.ppo_models import DualCameraActor
from so101_bowling.scorer import DeliveryPhase, DeliveryScorer, ScorerConfig


def test_approach_timeout_settles_at_quiet_dwell_without_unresolved_motion() -> None:
    """Idle APPROACH timeout should settle at quiet_dwell_s (0.25s), not max_coast_s (2.5s)."""
    scorer = DeliveryScorer(ScorerConfig(quiet_dwell_s=0.25, max_coast_s=2.5))
    scorer.reset(
        ball_init_pos=np.array([0.28, 0.0, 0.052]),
        pin_init_positions=np.array([[0.62, 0.0, 0.0], [0.695, -0.044, 0.0], [0.695, 0.044, 0.0]]),
        pin_init_up_vectors=np.tile([0.0, 0.0, 1.0], (3, 1)),
    )
    assert scorer.phase == DeliveryPhase.APPROACH
    assert scorer.is_coast_settled(0.0) == (False, False)
    assert scorer.is_coast_settled(0.24) == (False, False)
    assert scorer.is_coast_settled(0.25) == (True, False)


def test_env_idle_timeout_does_not_waste_max_coast_substeps() -> None:
    """Idle episode in SO101BowlingEnv finishes shortly after delivery_timeout_s with unresolved_motion=False."""
    with SO101BowlingEnv(render_rgb=False) as env:
        env.reset(seed=1)
        zero = np.zeros(4, dtype=np.float32)
        done = False
        info = {}
        for _ in range(env.cfg.delivery.max_steps):
            _, _, done, _, info = env.step(zero)
            if done:
                break
        assert done
        assert info["end_reason"] == "delivery_timeout"
        assert env.data.time > env.delivery_timeout_s
        assert env.data.time < env.delivery_timeout_s + env.cfg.delivery.quiet_dwell_s + 0.05
        assert info["outcome"]["unresolved_motion"] is False
        assert info["outcome"]["invalid_reason"] == "no_legal_release"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_epochs": 0},
        {"lr": 0.0},
        {"lr": -1e-4},
        {"lr": float("nan")},
        {"gamma": 0.0},
        {"gamma": 1.5},
        {"gae_lambda": 0.0},
        {"clip_coef": 0.0},
        {"max_grad_norm": -1.0},
        {"vf_coef": -0.1},
        {"ent_coef": -0.1},
        {"aux_coef": -0.1},
        {"target_kl": 0.0},
        {"target_kl": float("inf")},
        {"eval_episodes": 0},
        {"encoder_type": "invalid_enc"},
        {"num_envs": True},
        {"dr_enabled": 1},
    ],
)
def test_ppo_config_validation_rejects_invalid_values(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        PPOConfig(**kwargs)


def test_evaluate_actor_restores_training_mode_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """evaluate_actor must restore actor.training even if env creation fails."""
    actor = DualCameraActor(frame_stack=2)
    actor.train()

    class _BoomEnv:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr("so101_bowling.ppo.SO101BowlingEnv", _BoomEnv)
    with pytest.raises(RuntimeError, match="boom"):
        evaluate_actor(actor=actor, cfg=load_config(), episodes=1, device="cpu")
    assert actor.training is True
