"""Tests for episode dataset collection, replay-buffer warm-start, and checkpoint resume."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from so101_bowling import AsymmetricSACAgent, RawFrameReplayBuffer, SACConfig, load_config
from so101_bowling.dataset import collect_dataset, load_dataset_into_replay_buffer
from so101_bowling.sac import update_sac_step


def test_collect_dataset_and_load_into_replay_buffer(tmp_path: Path) -> None:
    """Collect scripted 4-DOF episodes (physics-only), inspect schema, and warm-start SAC buffer."""
    ds_dir = tmp_path / "bowling_ds"
    info = collect_dataset(
        out_dir=ds_dir,
        num_episodes=3,
        cfg=load_config(),
        dr_enabled=False,
        render_rgb=False,
        frame_stack=2,
        seed=42,
        jitter_scripted=False,
    )
    assert info["num_episodes"] == 3
    assert info["total_transitions"] > 10
    assert info["legal_delivery_rate"] == 1.0
    assert (ds_dir / "dataset_info.json").exists()
    lines = (ds_dir / "episodes.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3
    first_meta = json.loads(lines[0])
    assert (ds_dir / first_meta["file"]).exists()

    # Load into RawFrameReplayBuffer with num_envs=2 and run a real SAC update step!
    buf = RawFrameReplayBuffer(capacity=512, num_envs=2, frame_stack=2)
    loaded = load_dataset_into_replay_buffer(ds_dir, buf)
    assert loaded >= 10
    assert loaded % 2 == 0
    assert len(buf) == loaded

    batch = buf.sample(batch_size=8, device="cpu", rng=np.random.default_rng(7))
    assert batch["obs"]["wrist_rgb"].shape == (8, 2, 64, 112, 3)
    assert batch["actions"].shape == (8, 4)

    cfg = SACConfig(num_envs=2, capacity=512, batch_size=8, learning_starts=0, drq_pad_px=2)
    agent = AsymmetricSACAgent(frame_stack=2)
    actor_opt = torch.optim.Adam(
        list(agent.actor.parameters()) + list(agent.aux_head.parameters()), lr=cfg.actor_lr
    )
    critic_opt = torch.optim.Adam(agent.critic.parameters(), lr=cfg.critic_lr)
    alpha_opt = torch.optim.Adam([agent.log_alpha], lr=cfg.alpha_lr)
    metrics = update_sac_step(
        agent=agent,
        batch=batch,
        actor_opt=actor_opt,
        critic_opt=critic_opt,
        alpha_opt=alpha_opt,
        cfg=cfg,
        do_actor_update=True,
    )
    assert all(np.isfinite(v) for v in metrics.values())


def test_load_dataset_missing_episodes_raises(tmp_path: Path) -> None:
    buf = RawFrameReplayBuffer(capacity=64, num_envs=2, frame_stack=2)
    with pytest.raises(FileNotFoundError):
        load_dataset_into_replay_buffer(tmp_path / "nonexistent", buf)


def test_train_ppo_and_sac_reject_missing_resume_checkpoint(tmp_path: Path) -> None:
    from so101_bowling.ppo import PPOConfig, train_ppo
    from so101_bowling.sac import train_sac

    ppo_cfg = PPOConfig(num_envs=1, rollout_steps=4, total_timesteps=4, minibatch_size=4, eval_episodes=1)
    with pytest.raises(FileNotFoundError):
        train_ppo(ppo_cfg, run_dir=tmp_path / "ppo", resume_from=tmp_path / "missing.pt")

    sac_cfg = SACConfig(num_envs=1, capacity=16, batch_size=4, learning_starts=0, total_timesteps=4, eval_episodes=1)
    with pytest.raises(FileNotFoundError):
        train_sac(sac_cfg, run_dir=tmp_path / "sac", resume_from=tmp_path / "missing.pt")
