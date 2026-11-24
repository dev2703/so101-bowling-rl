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
