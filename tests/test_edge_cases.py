"""Regression tests for bug fixes and edge cases across scorer, PPO, and SAC."""

from __future__ import annotations

from pathlib import Path

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


def test_runtime_nan_abort_ppo_and_sac(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    from so101_bowling import ppo as ppo_mod
    from so101_bowling import sac as sac_mod
    from so101_bowling.ppo import train_ppo
    from so101_bowling.sac import SACConfig, train_sac

    wcfg = load_config()

    monkeypatch.setattr(
        ppo_mod,
        "masked_aux_loss",
        lambda pred, tgt, mask: pred.sum() * 0.0 + float("nan"),
    )
    ppo_dir = tmp_path / "ppo_nan"
    ppo_cfg = PPOConfig(
        num_envs=1,
        rollout_steps=4,
        total_timesteps=4,
        minibatch_size=4,
        eval_episodes=1,
        save_video=False,
        device="cpu",
    )
    with pytest.raises(FloatingPointError, match="Non-finite"):
        train_ppo(ppo_cfg=ppo_cfg, workspace_cfg=wcfg, run_dir=ppo_dir)
    ppo_summary = json.loads((ppo_dir / "summary.json").read_text())
    assert ppo_summary["nan_inf_detected"] is True
    assert "Non-finite" in ppo_summary["nan_inf_error"]

    monkeypatch.setattr(
        sac_mod,
        "masked_aux_loss",
        lambda pred, tgt, mask: pred.sum() * 0.0 + float("inf"),
    )
    sac_dir = tmp_path / "sac_nan"
    sac_cfg = SACConfig(
        num_envs=1,
        capacity=16,
        batch_size=4,
        learning_starts=0,
        utd_ratio=1.0,
        total_timesteps=4,
        eval_interval_steps=4,
        checkpoint_interval_steps=4,
        eval_episodes=1,
        save_video=False,
        device="cpu",
    )
    with pytest.raises(FloatingPointError, match="Non-finite"):
        train_sac(cfg=sac_cfg, workspace_cfg=wcfg, run_dir=sac_dir)
    sac_summary = json.loads((sac_dir / "summary.json").read_text())
    assert sac_summary["nan_inf_detected"] is True
    assert "Non-finite" in sac_summary["nan_inf_error"]


def test_actor_nan_inf_overflow_raises_floating_point_and_persists_status(
    tmp_path: Path,
) -> None:
    import importlib.util
    import json

    # 1. Direct DualCameraActor overflow raises FloatingPointError (not ValueError from Normal).
    actor = DualCameraActor(frame_stack=2)
    with torch.no_grad():
        actor.mean_head.weight.fill_(float("nan"))
    obs = {
        "wrist_rgb": torch.zeros((1, 2, 64, 112, 3), dtype=torch.uint8),
        "tripod_rgb": torch.zeros((1, 2, 64, 112, 3), dtype=torch.uint8),
        "camera_meta": torch.zeros((1, 4), dtype=torch.float32),
        "proprio": torch.zeros((1, 12), dtype=torch.float32),
    }
    with pytest.raises(FloatingPointError, match="Non-finite actor distribution"):
        actor.forward_actor(obs)

    # 2. End-to-end colab_train.py with lr=1e35 / actor-lr=1e35 records nan_inf_detected=True.
    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "colab_train_nan", repo_root / "scripts" / "colab_train.py"
    )
    assert spec is not None and spec.loader is not None
    ct_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ct_mod)

    ppo_pf_dir = tmp_path / "pf_ppo_nan"
    with pytest.raises(SystemExit) as exc_ppo:
        ct_mod.main(
            [
                "--algo",
                "ppo",
                "--lr",
                "1e35",
                "--preflight",
                "--preflight-steps",
                "5",
                "--preflight-batch-size",
                "4",
                "--device",
                "cpu",
                "--mujoco-gl",
                "",
                "--run-dir",
                str(ppo_pf_dir),
            ]
        )
    assert exc_ppo.value.code == 1
    ppo_pf_report = json.loads((ppo_pf_dir / "preflight_report.json").read_text())
    ppo_pf_status = json.loads((ppo_pf_dir / "status.json").read_text())
    assert ppo_pf_report["passed"] is False
    assert ppo_pf_report["nan_inf_detected"] is True
    assert ppo_pf_report["checks"]["finite_ok"] is False
    assert ppo_pf_status["state"] == "failed"
    assert ppo_pf_status["nan_inf_detected"] is True

    sac_pf_dir = tmp_path / "pf_sac_nan"
    with pytest.raises(SystemExit) as exc_sac:
        ct_mod.main(
            [
                "--algo",
                "sac",
                "--actor-lr",
                "1e35",
                "--preflight",
                "--preflight-steps",
                "5",
                "--preflight-batch-size",
                "4",
                "--device",
                "cpu",
                "--mujoco-gl",
                "",
                "--run-dir",
                str(sac_pf_dir),
            ]
        )
    assert exc_sac.value.code == 1
    sac_pf_report = json.loads((sac_pf_dir / "preflight_report.json").read_text())
    sac_pf_status = json.loads((sac_pf_dir / "status.json").read_text())
    assert sac_pf_report["passed"] is False
    assert sac_pf_report["nan_inf_detected"] is True
    assert sac_pf_report["checks"]["finite_ok"] is False
    assert sac_pf_status["state"] == "failed"
    assert sac_pf_status["nan_inf_detected"] is True

    ppo_rt_dir = tmp_path / "rt_ppo_nan"
    with pytest.raises(FloatingPointError, match="Non-finite"):
        ct_mod.main(
            [
                "--algo",
                "ppo",
                "--lr",
                "1e35",
                "--num-envs",
                "1",
                "--rollout-steps",
                "4",
                "--minibatch-size",
                "4",
                "--total-timesteps",
                "8",
                "--eval-interval-updates",
                "1",
                "--eval-episodes",
                "1",
                "--no-save-video",
                "--device",
                "cpu",
                "--mujoco-gl",
                "",
                "--run-dir",
                str(ppo_rt_dir),
            ]
        )
    ppo_rt_summary = json.loads((ppo_rt_dir / "summary.json").read_text())
    ppo_rt_status = json.loads((ppo_rt_dir / "status.json").read_text())
    assert ppo_rt_summary["nan_inf_detected"] is True
    assert ppo_rt_status["state"] == "failed"
    assert ppo_rt_status["nan_inf_detected"] is True
