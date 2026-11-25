"""Unit tests for continuous PUSHING potential, Stage 1 teacher, and Stage 2 DAgger."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from so101_bowling.config import load_config
from so101_bowling.scorer import DeliveryPhase, DeliveryScorer, ScorerConfig
from so101_bowling.teacher import (
    PrivilegedTeacherAgent,
    TeacherPPOConfig,
    load_teacher,
    train_teacher_ppo,
)


def test_continuous_pushing_potential_eliminates_contact_cliff() -> None:
    scorer = DeliveryScorer(ScorerConfig(w_stage_phi=0.40, w_push_phi=0.60))
    ball = np.array([0.28, 0.0, 0.052])
    pins = np.array([[0.62, 0.0, 0.0], [0.695, -0.044, 0.0], [0.695, 0.044, 0.0]])
    ups = np.tile([0.0, 0.0, 1.0], (3, 1))
    scorer.reset(ball, pins, ups, paddle_init_pos=ball - np.array([0.1035, 0.0, 0.0]))

    assert scorer.compute_potential(scorer.p_stage) == pytest.approx(0.40)
    # Entering PUSHING at 0 forward displacement stays continuous at 0.40 (no -0.40 cliff).
    scorer.phase = DeliveryPhase.PUSHING
    scorer.max_contact_fwd_disp = 0.0
    assert scorer.compute_potential(scorer.p_stage) == pytest.approx(0.40)
    # Forward ball displacement increases potential monotonically up to w_stage_phi + w_push_phi.
    scorer.max_contact_fwd_disp = 0.05
    assert scorer.compute_potential(scorer.p_stage) == pytest.approx(0.70)
    scorer.max_contact_fwd_disp = 0.12
    assert scorer.compute_potential(scorer.p_stage) == pytest.approx(1.00)
    # INVALID or terminated drops potential to 0.0.
    scorer.phase = DeliveryPhase.INVALID
    assert scorer.compute_potential(scorer.p_stage) == 0.0


def test_teacher_ppo_smoke_train_save_and_load(tmp_path: Path) -> None:
    cfg = TeacherPPOConfig(
        seed=0,
        dr_enabled=False,
        num_envs=1,
        rollout_steps=32,
        total_timesteps=64,
        hidden_dim=64,
        device="cpu",
        num_epochs=2,
        minibatch_size=16,
        bc_warmup_episodes=2,
        bc_warmup_epochs=2,
        eval_interval_updates=2,
        eval_episodes=2,
        checkpoint_interval_updates=2,
    )
    run_dir = tmp_path / "teacher_smoke"
    summary = train_teacher_ppo(cfg=cfg, workspace_cfg=load_config(), run_dir=run_dir)
    assert summary["global_step"] == 64
    assert summary["nan_inf_detected"] is False
    assert (run_dir / "best_teacher.pt").is_file()
    assert (run_dir / "final_teacher.pt").is_file()
    assert (run_dir / "summary.json").is_file()

    loaded = load_teacher(run_dir / "best_teacher.pt", map_location="cpu")
    assert isinstance(loaded, PrivilegedTeacherAgent)
    obs = {
        "privileged_state": np.zeros(46, dtype=np.float32),
        "proprio": np.zeros(12, dtype=np.float32),
    }
    act = loaded.act(obs, deterministic=True).numpy()
    assert act.shape == (4,)
    assert np.isfinite(act).all()
    assert np.all(np.abs(act) <= 1.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rounds": 0},
        {"steps_per_round": 0},
        {"batch_size": 0},
        {"lr": 0.0},
        {"lr": float("nan")},
        {"aux_coef": -0.1},
        {"max_grad_norm": 0.0},
        {"beta_start": 1.2},
        {"beta_end": -0.1},
        {"drq_pad_px": -1},
        {"encoder_type": "bad"},
        {"rounds": True},
        {"squashed": 1},
    ],
)
def test_dagger_config_validation_rejects_invalid_values(kwargs: dict) -> None:
    from so101_bowling import DAggerConfig

    with pytest.raises(ValueError):
        DAggerConfig(**kwargs)


def test_dagger_per_episode_mixing_and_summary_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import so101_bowling.dagger as dagger_mod
    from so101_bowling import DAggerConfig, train_dagger

    # Train a tiny teacher first.
    t_dir = tmp_path / "teacher"
    train_teacher_ppo(
        cfg=TeacherPPOConfig(
            seed=0,
            dr_enabled=False,
            num_envs=1,
            rollout_steps=8,
            total_timesteps=8,
            hidden_dim=32,
            device="cpu",
            num_epochs=1,
            minibatch_size=4,
            bc_warmup_episodes=0,
            bc_warmup_epochs=0,
            eval_interval_updates=1,
            eval_episodes=1,
            checkpoint_interval_updates=1,
        ),
        workspace_cfg=load_config(),
        run_dir=t_dir,
    )

    # Fake 1-env vec_env where every episode lasts 4 steps and teacher/student emit distinct actions.
    class _FakeVecEnv:
        def __init__(self) -> None:
            self._step = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def _obs(self) -> dict[str, np.ndarray]:
            return {
                "wrist_rgb": np.zeros((1, 2, 64, 112, 3), dtype=np.uint8),
                "tripod_rgb": np.zeros((1, 2, 64, 112, 3), dtype=np.uint8),
                "camera_meta": np.zeros((1, 4), dtype=np.float32),
                "proprio": np.zeros((1, 12), dtype=np.float32),
                "privileged_state": np.zeros((1, 46), dtype=np.float32),
                "privileged_aux_targets": np.zeros((1, 6), dtype=np.float32),
                "privileged_aux_mask": np.ones((1, 3), dtype=np.float32),
            }

        def reset(self, seed=None):
            self._step = 0
            return self._obs()

        def step(self, actions):
            executed.append(float(actions[0, 0]))
            self._step += 1
            done = (self._step % 4) == 0
            return (
                self._obs(),
                np.zeros((1,), dtype=np.float32),
                np.asarray([done], dtype=bool),
                np.asarray([False], dtype=bool),
                [{"outcome": {"legal_strike": False, "credited_pins": 0}}] if done else [{}],
            )

    import torch

    executed: list[float] = []

    class _ConstTeacher(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))

        def act(self, obs, deterministic=True):
            return torch.ones((1, 4), dtype=torch.float32)

    class _ConstActor(dagger_mod.DualCameraActor):
        def act(self, obs, deterministic=False):
            return -torch.ones((1, 4), dtype=torch.float32)

    monkeypatch.setattr(dagger_mod, "load_teacher", lambda *a, **kw: _ConstTeacher())
    monkeypatch.setattr(dagger_mod, "DualCameraActor", _ConstActor)
    monkeypatch.setattr(dagger_mod, "make_vec_env", lambda **kw: _FakeVecEnv())
    monkeypatch.setattr(
        dagger_mod,
        "evaluate_actor",
        lambda **kw: {
            "episodes": 1,
            "legal_strike_rate": 0.5,
            "mean_credited_pins": 2.0,
            "mean_full_knock_pins": 2.0,
            "mean_wobble_only_pins": 0.0,
            "illegal_rate": 0.0,
            "mean_return": 10.0,
            "mean_episode_steps": 40.0,
            "eval_aux_loss": 0.012,
        },
    )

    d_dir = tmp_path / "dagger"
    summary = train_dagger(
        cfg=DAggerConfig(
            seed=3,
            dr_enabled=False,
            num_envs=1,
            rounds=1,
            steps_per_round=16,
            capacity=32,
            epochs_per_round=1,
            batch_size=8,
            beta_start=0.5,
            beta_end=0.5,
            squashed=True,
            save_video=False,
            device="cpu",
        ),
        teacher_path=t_dir / "best_teacher.pt",
        workspace_cfg=load_config(),
        run_dir=d_dir,
    )
    # Per-episode mixing: within each 4-step episode, every step must use the same policy (+1 or -1).
    for ep_start in range(0, 16, 4):
        ep_actions = executed[ep_start : ep_start + 4]
        assert len(set(ep_actions)) == 1
    assert summary["eval_aux_loss"] == pytest.approx(0.012)
    assert "bc_loss" in summary["final_losses"]
    assert "aux_loss" in summary["final_losses"]
    assert "validation_loss_note" in summary
