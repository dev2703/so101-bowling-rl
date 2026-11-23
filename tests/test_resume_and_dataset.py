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


def test_collect_dataset_clears_stale_npz(tmp_path: Path) -> None:
    ds_dir = tmp_path / "ds_stale"
    ep_dir = ds_dir / "episodes"
    ep_dir.mkdir(parents=True)
    stale_0 = ep_dir / "ep_000000.npz"
    stale_5 = ep_dir / "ep_000005.npz"
    stale_0.write_bytes(b"old")
    stale_5.write_bytes(b"old")

    info = collect_dataset(
        out_dir=ds_dir,
        num_episodes=2,
        cfg=load_config(),
        dr_enabled=False,
        render_rgb=False,
        seed=0,
        jitter_scripted=False,
    )
    assert info["num_episodes"] == 2
    saved_files = sorted(p.name for p in ep_dir.glob("ep_*.npz"))
    assert saved_files == ["ep_000000.npz", "ep_000001.npz"]
    assert not stale_5.exists()


def test_collect_dataset_only_legal_raises_on_shortfall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from so101_bowling.env import SO101BowlingEnv

    orig_step = SO101BowlingEnv.step

    def always_illegal_step(self, action):
        obs, rew, term, trunc, info = orig_step(self, action)
        if term or trunc:
            info = dict(info)
            outcome = dict(info.get("outcome", {}))
            outcome["legal_delivery"] = False
            outcome["invalid_reason"] = "forced_illegal_test"
            info["outcome"] = outcome
        return obs, rew, term, trunc, info

    monkeypatch.setattr(SO101BowlingEnv, "step", always_illegal_step)
    with pytest.raises(RuntimeError, match="collected only 0/2 legal episodes"):
        collect_dataset(
            out_dir=tmp_path / "ds_shortfall",
            num_episodes=2,
            cfg=load_config(),
            dr_enabled=False,
            render_rgb=False,
            only_legal=True,
            max_attempts_multiplier=2,
        )


def test_ppo_and_sac_resume_preserves_best_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from so101_bowling import ppo as ppo_mod
    from so101_bowling import sac as sac_mod
    from so101_bowling.ppo import PPOConfig, train_ppo
    from so101_bowling.ppo_models import load_actor
    from so101_bowling.sac import train_sac

    wcfg = load_config()
    eval_scores = iter(
        [
            {
                "episodes": 1,
                "legal_strike_rate": 1.0,
                "mean_credited_pins": 3.0,
                "mean_full_knock_pins": 3.0,
                "mean_wobble_only_pins": 0.0,
                "illegal_rate": 0.0,
                "mean_return": 25.0,
                "mean_episode_steps": 40.0,
                "eval_aux_loss": 0.01,
            },
            {
                "episodes": 1,
                "legal_strike_rate": 0.0,
                "mean_credited_pins": 0.0,
                "mean_full_knock_pins": 0.0,
                "mean_wobble_only_pins": 0.0,
                "illegal_rate": 1.0,
                "mean_return": -5.0,
                "mean_episode_steps": 80.0,
                "eval_aux_loss": 0.25,
            },
        ]
    )
    monkeypatch.setattr(ppo_mod, "evaluate_actor", lambda **kwargs: next(eval_scores))
    monkeypatch.setattr(sac_mod, "evaluate_actor", lambda **kwargs: next(eval_scores))

    # 1. PPO resume test
    ppo_dir = tmp_path / "ppo_resume"
    ppo_cfg1 = PPOConfig(
        num_envs=1,
        rollout_steps=4,
        total_timesteps=4,
        minibatch_size=4,
        eval_interval_updates=1,
        checkpoint_interval_updates=1,
        eval_episodes=1,
        save_video=False,
        device="cpu",
    )
    train_ppo(ppo_cfg1, workspace_cfg=wcfg, run_dir=ppo_dir)
    best_before = {k: v.clone() for k, v in load_actor(ppo_dir / "best_actor.pt").state_dict().items()}

    ppo_cfg2 = PPOConfig(
        num_envs=1,
        rollout_steps=4,
        total_timesteps=8,
        minibatch_size=4,
        eval_interval_updates=1,
        checkpoint_interval_updates=1,
        eval_episodes=1,
        save_video=False,
        device="cpu",
    )
    ppo_summary2 = train_ppo(
        ppo_cfg2, workspace_cfg=wcfg, run_dir=ppo_dir, resume_from=ppo_dir / "latest.pt"
    )
    assert ppo_summary2["best_eval"]["legal_strike_rate"] == 1.0
    assert ppo_summary2["best_eval"]["mean_return"] == 25.0
    best_after = load_actor(ppo_dir / "best_actor.pt").state_dict()
    for k, v in best_before.items():
        assert torch.equal(v, best_after[k])

    # 2. SAC resume test (including legacy checkpoint without best_score falling back to summary.json)
    eval_scores = iter(
        [
            {
                "episodes": 1,
                "legal_strike_rate": 1.0,
                "mean_credited_pins": 3.0,
                "mean_full_knock_pins": 3.0,
                "mean_wobble_only_pins": 0.0,
                "illegal_rate": 0.0,
                "mean_return": 22.0,
                "mean_episode_steps": 42.0,
                "eval_aux_loss": 0.02,
            },
            {
                "episodes": 1,
                "legal_strike_rate": 0.0,
                "mean_credited_pins": 0.0,
                "mean_full_knock_pins": 0.0,
                "mean_wobble_only_pins": 0.0,
                "illegal_rate": 1.0,
                "mean_return": -4.0,
                "mean_episode_steps": 80.0,
                "eval_aux_loss": 0.30,
            },
        ]
    )
    sac_dir = tmp_path / "sac_resume"
    sac_cfg1 = SACConfig(
        num_envs=1,
        capacity=32,
        batch_size=4,
        learning_starts=0,
        total_timesteps=4,
        eval_interval_steps=4,
        checkpoint_interval_steps=4,
        eval_episodes=1,
        save_video=False,
        device="cpu",
    )
    train_sac(sac_cfg1, workspace_cfg=wcfg, run_dir=sac_dir)
    sac_best_before = {
        k: v.clone() for k, v in load_actor(sac_dir / "actor_best.pt").state_dict().items()
    }
    # Simulate legacy checkpoint missing best_score so summary.json fallback is also exercised.
    legacy_ckpt = torch.load(sac_dir / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
    legacy_ckpt.pop("best_score", None)
    legacy_ckpt.pop("best_eval", None)
    torch.save(legacy_ckpt, sac_dir / "checkpoint_latest.pt")

    sac_cfg2 = SACConfig(
        num_envs=1,
        capacity=32,
        batch_size=4,
        learning_starts=0,
        total_timesteps=8,
        eval_interval_steps=4,
        checkpoint_interval_steps=4,
        eval_episodes=1,
        save_video=False,
        device="cpu",
    )
    sac_summary2 = train_sac(
        sac_cfg2, workspace_cfg=wcfg, run_dir=sac_dir, resume_from=sac_dir / "checkpoint_latest.pt"
    )
    assert sac_summary2["best_eval"]["legal_strike_rate"] == 1.0
    assert sac_summary2["best_eval"]["mean_return"] == 22.0
    sac_best_after = load_actor(sac_dir / "actor_best.pt").state_dict()
    for k, v in sac_best_before.items():
        assert torch.equal(v, sac_best_after[k])


def test_sac_demo_pretrain_steps_and_immediate_updates(tmp_path: Path) -> None:
    from so101_bowling.sac import train_sac

    ds_dir = tmp_path / "demos"
    collect_dataset(
        out_dir=ds_dir,
        num_episodes=1,
        cfg=load_config(),
        dr_enabled=False,
        render_rgb=False,
        seed=0,
        jitter_scripted=False,
    )
    sac_cfg = SACConfig(
        num_envs=1,
        capacity=128,
        batch_size=4,
        learning_starts=1000,
        utd_ratio=1.0,
        total_timesteps=4,
        eval_interval_steps=4,
        checkpoint_interval_steps=4,
        eval_episodes=1,
        save_video=False,
        demo_pretrain_steps=3,
        device="cpu",
    )
    summary = train_sac(
        cfg=sac_cfg,
        workspace_cfg=load_config(),
        run_dir=tmp_path / "sac_warmstart",
        demo_dataset=ds_dir,
    )
    assert summary["demo_transitions_loaded"] >= 4
    assert summary["demo_pretrain_steps_run"] == 3
    # 3 pretrain updates + 4 immediate online updates despite learning_starts=1000.
    assert summary["total_critic_updates"] == 7
