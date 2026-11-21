"""Unit and integration tests for dual-camera Asymmetric SAC."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.distributions import Normal

from so101_bowling import (
    AsymmetricSACAgent,
    DualCameraActor,
    PrivilegedTwinQ,
    SACConfig,
    load_actor,
    load_config,
    load_sac_config,
)
from so101_bowling.ppo_models import ACTOR_OBS_KEYS
from so101_bowling.sac import compute_utd_updates, train_sac, update_sac_step


def _synthetic_obs(batch_size: int = 4, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "wrist_rgb": torch.randint(0, 256, (batch_size, 2, 64, 112, 3), dtype=torch.uint8, generator=g),
        "tripod_rgb": torch.randint(0, 256, (batch_size, 2, 64, 112, 3), dtype=torch.uint8, generator=g),
        "camera_meta": torch.rand((batch_size, 4), dtype=torch.float32, generator=g),
        "proprio": torch.randn((batch_size, 12), dtype=torch.float32, generator=g).clamp(-1.0, 1.0),
        "privileged_state": torch.randn((batch_size, 46), dtype=torch.float32, generator=g),
        "privileged_aux_targets": torch.randn((batch_size, 6), dtype=torch.float32, generator=g),
        "privileged_aux_mask": torch.ones((batch_size, 3), dtype=torch.float32),
    }


def _synthetic_sac_batch(batch_size: int = 4, seed: int = 0) -> dict[str, object]:
    g = torch.Generator().manual_seed(seed)
    obs = _synthetic_obs(batch_size=batch_size, seed=seed)
    next_full = _synthetic_obs(batch_size=batch_size, seed=seed + 100)
    next_obs = {
        k: next_full[k]
        for k in ("wrist_rgb", "tripod_rgb", "camera_meta", "proprio", "privileged_state")
    }
    return {
        "obs": obs,
        "next_obs": next_obs,
        "actions": torch.rand((batch_size, 4), dtype=torch.float32, generator=g) * 2.0 - 1.0,
        "rewards": torch.randn((batch_size,), dtype=torch.float32, generator=g),
        "terminated": torch.zeros((batch_size,), dtype=torch.bool),
        "truncated": torch.zeros((batch_size,), dtype=torch.bool),
    }


def test_actor_pre_m4_payload_compat_and_squashed_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(10)
    obs = _synthetic_obs(batch_size=2, seed=11)
    actor_obs_batched = {k: obs[k] for k in ACTOR_OBS_KEYS}
    actor_obs_single = {k: obs[k][0] for k in ACTOR_OBS_KEYS}

    # Legacy pre-M4 payload (no "squashed" key) must load as squashed=False and clamp raw mean.
    legacy_actor = DualCameraActor(frame_stack=2, encoder_type="spatial_grid")
    with torch.no_grad():
        legacy_actor.mean_head.bias.fill_(1.5)  # forces raw mean outside [-1, 1] to distinguish clamp vs tanh
    legacy_actor.eval()
    payload = legacy_actor.export_payload()
    assert payload["config"]["squashed"] is False
    del payload["config"]["squashed"]
    legacy_path = tmp_path / "legacy_actor.pt"
    torch.save(payload, legacy_path)

    loaded_legacy = load_actor(legacy_path, map_location="cpu")
    assert loaded_legacy.squashed is False
    with torch.no_grad():
        dist, _ = loaded_legacy.forward_actor(actor_obs_batched)
        assert isinstance(dist, Normal)
        expected_clamped = torch.clamp(dist.mean, -1.0, 1.0)
        assert torch.allclose(loaded_legacy.act(actor_obs_batched, deterministic=True), expected_clamped)
        assert torch.allclose(
            loaded_legacy.act(actor_obs_batched, deterministic=True),
            legacy_actor.act(actor_obs_batched, deterministic=True),
        )

    # Squashed M4 actor roundtrip: exact tanh(mean) parity for batched and unbatched inputs.
    sac_actor = DualCameraActor(frame_stack=2, encoder_type="spatial_grid", squashed=True)
    with torch.no_grad():
        sac_actor.mean_head.bias.copy_(torch.tensor([1.5, -1.5, 0.3, -0.3]))
    sac_actor.eval()
    sac_path = sac_actor.save_actor(tmp_path / "sac_actor.pt")
    loaded_sac = load_actor(sac_path, map_location="cpu")
    assert loaded_sac.squashed is True

    with torch.no_grad():
        dist_sac, _ = loaded_sac.forward_actor(actor_obs_batched)
        assert isinstance(dist_sac, Normal)
        expected_squashed = torch.tanh(dist_sac.mean)
        act_batched = loaded_sac.act(actor_obs_batched, deterministic=True)
        act_single = loaded_sac.act(actor_obs_single, deterministic=True)
        assert act_batched.shape == (2, 4)
        assert act_single.shape == (4,)
        assert torch.allclose(act_batched, expected_squashed, atol=1e-6)
        assert torch.allclose(act_single, expected_squashed[0], atol=1e-6)
        assert not torch.allclose(act_batched, torch.clamp(dist_sac.mean, -1.0, 1.0), atol=1e-3)


def test_sample_squashed_saturated_finite_grads() -> None:
    torch.manual_seed(20)
    actor = DualCameraActor(frame_stack=2, encoder_type="spatial_grid", squashed=True)
    with torch.no_grad():
        actor.mean_head.weight.zero_()
        actor.mean_head.bias.copy_(torch.tensor([20.0, -20.0, 20.0, -20.0]))
        actor.log_std.fill_(-1.0)

    obs = _synthetic_obs(batch_size=4, seed=21)
    action, log_prob, z_vis = actor.sample_squashed(obs)
    assert action.shape == (4, 4)
    assert log_prob.shape == (4,)
    assert z_vis.shape == (4, 256)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)
    assert torch.isfinite(log_prob).all()

    loss = (action.sum() - 0.2 * log_prob.mean())
    loss.backward()
    assert actor.mean_head.bias.grad is not None and torch.isfinite(actor.mean_head.bias.grad).all()
    assert actor.mean_head.weight.grad is not None and torch.isfinite(actor.mean_head.weight.grad).all()
    assert actor.log_std.grad is not None and torch.isfinite(actor.log_std.grad).all()


def test_sac_actor_step_grads_both_encoders_with_aux_zero() -> None:
    torch.manual_seed(30)
    cfg = SACConfig(
        num_envs=2,
        capacity=32,
        batch_size=4,
        learning_starts=0,
        aux_coef=0.0,
        drq_pad_px=4,
    )
    agent = AsymmetricSACAgent(frame_stack=2, encoder_type="spatial_grid", init_alpha=cfg.init_alpha)
    actor_opt = torch.optim.Adam(
        list(agent.actor.parameters()) + list(agent.aux_head.parameters()), lr=cfg.actor_lr
    )
    critic_opt = torch.optim.Adam(agent.critic.parameters(), lr=cfg.critic_lr)
    alpha_opt = torch.optim.Adam([agent.log_alpha], lr=cfg.alpha_lr)

    batch = _synthetic_sac_batch(batch_size=4, seed=31)
    metrics = update_sac_step(
        agent=agent,
        batch=batch,
        actor_opt=actor_opt,
        critic_opt=critic_opt,
        alpha_opt=alpha_opt,
        cfg=cfg,
        do_actor_update=True,
        torch_gen=torch.Generator().manual_seed(32),
    )
    assert metrics["aux_loss"] == 0.0
    assert all(np.isfinite(v) for v in metrics.values())

    for enc_name, enc in (
        ("wrist_encoder", agent.actor.wrist_encoder),
        ("tripod_encoder", agent.actor.tripod_encoder),
    ):
        grads = [p.grad for p in enc.parameters() if p.grad is not None]
        assert grads, f"Expected gradients on {enc_name}"
        assert all(torch.isfinite(g).all() for g in grads)
        total_abs = sum(float(g.abs().sum()) for g in grads)
        assert total_abs > 0.0, f"Expected non-zero gradients on {enc_name}"

    for param in agent.critic.parameters():
        assert param.grad is None
        assert param.requires_grad is True


def test_privileged_twin_q_input_dim_66() -> None:
    torch.manual_seed(40)
    critic = PrivilegedTwinQ(action_dim=4)
    assert critic.input_dim == 66

    obs = _synthetic_obs(batch_size=5, seed=41)
    action = torch.randn((5, 4), dtype=torch.float32, requires_grad=True)
    q1, q2 = critic(obs, action)
    assert q1.shape == (5,)
    assert q2.shape == (5,)
    assert not torch.allclose(q1, q2)

    (q1.sum() + q2.sum()).backward()
    assert action.grad is not None
    assert torch.isfinite(action.grad).all()
    assert float(action.grad.abs().sum()) > 0.0


def test_sac_utd_accounting_warmup_and_fractional_carry() -> None:
    cfg_warm = SACConfig(num_envs=4, capacity=64, batch_size=16, learning_starts=100, utd_ratio=0.5)
    assert compute_utd_updates(global_step=99, buffer_size=64, cfg=cfg_warm, utd_carry=0.0) == (0, 0.0)
    assert compute_utd_updates(global_step=100, buffer_size=15, cfg=cfg_warm, utd_carry=0.0) == (0, 0.0)
    assert compute_utd_updates(global_step=100, buffer_size=16, cfg=cfg_warm, utd_carry=0.0) == (2, 0.0)

    cfg_frac = SACConfig(num_envs=1, capacity=32, batch_size=4, learning_starts=0, utd_ratio=0.3)
    carry = 0.0
    counts: list[int] = []
    for step in range(1, 11):
        n_upd, carry = compute_utd_updates(global_step=step, buffer_size=4, cfg=cfg_frac, utd_carry=carry)
        counts.append(n_upd)
    assert counts == [0, 0, 0, 1, 0, 0, 1, 0, 0, 1]
    assert sum(counts) == 3
    assert carry == pytest.approx(0.0, abs=1e-9)


def test_sac_truncation_bootstraps_and_termination_zeroes_target() -> None:
    def _critic_loss_for_final_obs(final_state_val: float, terminated_flag: bool) -> float:
        torch.manual_seed(50)
        cfg = SACConfig(
            num_envs=1,
            capacity=16,
            batch_size=1,
            learning_starts=0,
            gamma=0.9,
            tau=1e-6,
            init_alpha=0.2,
            autotune_alpha=False,
            aux_coef=0.0,
            drq_pad_px=0,
        )
        agent = AsymmetricSACAgent(frame_stack=2, encoder_type="spatial_grid", init_alpha=1e-4)
        # Zero out critic so q1 == q2 == 0 -> critic_loss == backup^2.
        with torch.no_grad():
            for p in agent.critic.parameters():
                p.zero_()

        class _ProbeQ(torch.nn.Module):
            def forward(self, obs_in, act_in):
                v = obs_in["privileged_state"][:, 0].float() * 10.0
                return v, v

        agent.critic_target = _ProbeQ()  # type: ignore[assignment]
        agent.soft_update_target = lambda tau: None  # type: ignore[method-assign]

        actor_opt = torch.optim.Adam(agent.actor.parameters(), lr=cfg.actor_lr)
        critic_opt = torch.optim.Adam(agent.critic.parameters(), lr=0.0)
        batch = _synthetic_sac_batch(batch_size=1, seed=51)
        batch["rewards"] = torch.tensor([2.0], dtype=torch.float32)
        batch["terminated"] = torch.tensor([terminated_flag], dtype=torch.bool)
        batch["truncated"] = torch.tensor([not terminated_flag], dtype=torch.bool)
        batch["next_obs"]["privileged_state"].zero_()
        batch["next_obs"]["privileged_state"][0, 0] = final_state_val

        metrics = update_sac_step(
            agent=agent,
            batch=batch,
            actor_opt=actor_opt,
            critic_opt=critic_opt,
            alpha_opt=None,
            cfg=cfg,
            do_actor_update=False,
        )
        return metrics["critic_loss"]

    # Terminated: backup = reward = 2.0 -> MSE(0, 2.0) = 4.0 regardless of final_obs.
    loss_term_a = _critic_loss_for_final_obs(final_state_val=1.0, terminated_flag=True)
    loss_term_b = _critic_loss_for_final_obs(final_state_val=5.0, terminated_flag=True)
    assert loss_term_a == pytest.approx(4.0, abs=1e-4)
    assert loss_term_b == pytest.approx(4.0, abs=1e-4)

    # Truncated (terminated=False): backup = 2.0 + 0.9 * (10.0 * final_state_val - alpha*logp),
    # so changing final_obs shifts critic loss substantially!
    loss_trunc_a = _critic_loss_for_final_obs(final_state_val=1.0, terminated_flag=False)
    loss_trunc_b = _critic_loss_for_final_obs(final_state_val=5.0, terminated_flag=False)
    assert loss_trunc_a == pytest.approx((2.0 + 9.0) ** 2, rel=0.05)
    assert loss_trunc_b == pytest.approx((2.0 + 45.0) ** 2, rel=0.05)
    assert loss_trunc_b > loss_trunc_a + 1000.0


def test_train_sac_smoke_and_compare_baselines_cli(tmp_path: Path) -> None:
    # 1. Verify lazy package imports do not pull torch or tensorboard.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, so101_bowling; "
            "assert 'torch' not in sys.modules; assert 'tensorboard' not in sys.modules",
        ],
        check=True,
    )

    # Verify default YAML loads cleanly.
    default_cfg = load_sac_config()
    assert default_cfg.capacity == 50000
    assert default_cfg.utd_ratio == 0.5

    # 2. Tiny CPU train_sac smoke run.
    run_dir = tmp_path / "sac_smoke"
    sac_cfg = SACConfig(
        seed=0,
        dr_enabled=False,
        frame_stack=2,
        num_envs=2,
        total_timesteps=16,
        capacity=32,
        learning_starts=4,
        batch_size=4,
        utd_ratio=0.5,
        eval_interval_steps=16,
        eval_episodes=1,
        checkpoint_interval_steps=16,
        save_video=False,
        device="cpu",
    )
    wcfg = load_config()
    summary = train_sac(cfg=sac_cfg, workspace_cfg=wcfg, run_dir=run_dir)
    assert summary["global_step"] == 16
    assert summary["total_critic_updates"] > 0
    for fname in ("actor_latest.pt", "actor_best.pt", "checkpoint_latest.pt", "summary.json", "metrics.jsonl"):
        assert (run_dir / fname).exists()

    # Reload actor_latest.pt and confirm squashed=True.
    actor = load_actor(run_dir / "actor_latest.pt", map_location="cpu")
    assert actor.squashed is True

    # 3. Run scripts/compare_baselines.py against actor_latest.pt.
    repo_root = Path(__file__).resolve().parents[1]
    out_json = tmp_path / "compare.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "compare_baselines.py"),
            "--episodes",
            "1",
            "--seed",
            "42",
            "--sac-checkpoint",
            str(run_dir / "actor_latest.pt"),
            "--device",
            "cpu",
            "--out-json",
            str(out_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(proc.stdout)
    assert out_json.exists()
    assert report["scripted_4dof"]["privileged_state"] is True
    assert report["scripted_4dof"]["episodes"] == 1
    assert report["sac"]["privileged_state"] is False
    assert report["sac"]["episodes"] == 1


def test_sac_config_rejects_batch_size_above_usable_capacity_and_bool_ints() -> None:
    with pytest.raises(ValueError, match="usable replay capacity"):
        SACConfig(num_envs=4, capacity=19, batch_size=17)

    cfg_ok = SACConfig(num_envs=4, capacity=19, batch_size=16)
    assert cfg_ok.batch_size == 16

    with pytest.raises(ValueError, match="usable replay capacity"):
        SACConfig(num_envs=4, capacity=16, batch_size=17)

    for field_name in (
        "seed",
        "num_envs",
        "frame_stack",
        "capacity",
        "batch_size",
        "learning_starts",
        "total_timesteps",
        "policy_delay",
        "drq_pad_px",
        "eval_interval_steps",
        "eval_episodes",
        "checkpoint_interval_steps",
    ):
        with pytest.raises(ValueError, match=field_name):
            SACConfig(**{field_name: True})
        with pytest.raises(ValueError, match=field_name):
            SACConfig(**{field_name: 1.5})

    for field_name in (
        "utd_ratio",
        "gamma",
        "tau",
        "actor_lr",
        "critic_lr",
        "alpha_lr",
        "init_alpha",
        "aux_coef",
        "max_grad_norm",
    ):
        with pytest.raises(ValueError, match=field_name):
            SACConfig(**{field_name: True})
        with pytest.raises(ValueError, match=field_name):
            SACConfig(**{field_name: float("nan")})

    for field_name in ("dr_enabled", "autotune_alpha", "save_video"):
        with pytest.raises(ValueError, match=field_name):
            SACConfig(**{field_name: 1})

