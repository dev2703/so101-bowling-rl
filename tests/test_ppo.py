"""Unit and integration invariants for dual-camera Asymmetric PPO."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, Mock

import cv2
import numpy as np
import pytest
import torch

from so101_bowling import (
    AsymmetricPPOAgent,
    DualCameraActor,
    PPOConfig,
    SO101BowlingEnv,
    load_actor,
    load_config,
)
from so101_bowling import ppo, vec_env
from so101_bowling.ppo import (
    RolloutBuffer,
    _batch_slice,
    compute_gae,
    evaluate_actor,
    evaluate_camera_ablation,
    train_ppo,
)
from so101_bowling.ppo_models import ACTOR_OBS_KEYS, apply_view_shift, masked_aux_loss


def _synthetic_obs(batch_size: int = 4, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "wrist_rgb": torch.randint(0, 256, (batch_size, 2, 64, 112, 3), dtype=torch.uint8, generator=g),
        "tripod_rgb": torch.randint(0, 256, (batch_size, 2, 64, 112, 3), dtype=torch.uint8, generator=g),
        "camera_meta": torch.rand((batch_size, 4), dtype=torch.float32, generator=g),
        "proprio": torch.randn((batch_size, 12), dtype=torch.float32, generator=g).clamp(-1, 1),
        "privileged_state": torch.randn((batch_size, 46), dtype=torch.float32, generator=g),
        "privileged_aux_targets": torch.randn((batch_size, 6), dtype=torch.float32, generator=g),
        "privileged_aux_mask": torch.ones((batch_size, 3), dtype=torch.float32),
    }


def test_actor_privileged_isolation() -> None:
    torch.manual_seed(1)
    actor = DualCameraActor(frame_stack=2, encoder_type="spatial_grid")
    actor.eval()

    obs_clean = _synthetic_obs(batch_size=3, seed=7)
    obs_corrupt = dict(obs_clean)
    obs_corrupt["privileged_state"] = torch.full_like(obs_clean["privileged_state"], float("nan"))
    obs_corrupt["privileged_aux_targets"] = torch.full_like(obs_clean["privileged_aux_targets"], 1e9)
    obs_corrupt["privileged_aux_mask"] = torch.zeros_like(obs_clean["privileged_aux_mask"])

    with torch.no_grad():
        clean_dist, clean_z_vis = actor.forward_actor(obs_clean)
        corrupt_dist, corrupt_z_vis = actor.forward_actor(obs_corrupt)

    assert torch.equal(clean_dist.mean, corrupt_dist.mean)
    assert torch.equal(clean_dist.stddev, corrupt_dist.stddev)
    assert torch.equal(clean_z_vis, corrupt_z_vis)


@torch.no_grad()
def test_ppo_drq_shift_ratio_exactness() -> None:
    torch.manual_seed(2)
    num_steps, num_envs, minibatch_size, drq_pad_px = 3, 2, 4, 4
    rng = np.random.default_rng(2)
    agent = AsymmetricPPOAgent()
    agent.eval()
    buffer = RolloutBuffer(num_steps=num_steps, num_envs=num_envs, frame_stack=2)
    rewards = np.zeros(num_envs, dtype=np.float32)
    done = np.zeros(num_envs, dtype=bool)

    for step in range(num_steps):
        obs = _synthetic_obs(batch_size=num_envs, seed=11 + step)
        w_shift = torch.from_numpy(rng.integers(-drq_pad_px, drq_pad_px + 1, (num_envs, 2)))
        t_shift = torch.from_numpy(rng.integers(-drq_pad_px, drq_pad_px + 1, (num_envs, 2)))
        action, logprob, _, value, _ = agent.get_action_and_value(
            obs, wrist_shifts=w_shift, tripod_shifts=t_shift
        )
        buffer.insert(
            step=step,
            obs={key: value.numpy() for key, value in obs.items()},
            action=action,
            logprob=logprob,
            value=value,
            reward=rewards,
            terminated=done,
            truncated=done,
            final_value=torch.zeros(num_envs),
            wrist_shift=w_shift,
            tripod_shift=t_shift,
        )

    zeros = torch.zeros_like(buffer.values)
    flat = buffer.flat_view(zeros, zeros)
    agent.train()
    batch_size = num_steps * num_envs
    permutation = rng.permutation(batch_size)
    for start in range(0, batch_size, minibatch_size):
        indices = permutation[start : start + minibatch_size]
        mb_obs, mb_act, old_logp, _, _, _, mb_w_shift, mb_t_shift = _batch_slice(
            flat, indices, torch.device("cpu")
        )
        _, new_logp, _, _, _ = agent.get_action_and_value(
            mb_obs, wrist_shifts=mb_w_shift, tripod_shifts=mb_t_shift, action=mb_act
        )
        logratio = new_logp - old_logp
        ratio = logratio.exp()
        assert logratio.abs().max().item() < 1e-6
        assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-6, rtol=0)
        assert abs(((ratio - 1.0) - logratio).mean().item()) < 1e-6

    # Also confirm view shift matches numpy edge padding.
    img = torch.arange(1 * 6 * 8 * 10, dtype=torch.float32).reshape(1, 6, 8, 10)
    shifted = apply_view_shift(img, torch.tensor([[2, -1]]))
    assert shifted.shape == img.shape
    np_padded = np.pad(img[0].numpy(), ((0, 0), (2, 2), (1, 1)), mode="edge")
    expected = np_padded[:, 0:8, 2:12]
    np.testing.assert_allclose(shifted[0].numpy(), expected)


@pytest.mark.parametrize("encoder_type", ["spatial_grid", "spatial_softmax"])
@pytest.mark.parametrize("legacy", [False, True])
def test_standalone_actor_export_parity(tmp_path: Path, encoder_type: str, legacy: bool) -> None:
    torch.manual_seed(3)
    img_hw = (64, 112) if legacy else (49, 81)
    cfg = load_config()
    cfg = replace(cfg, camera=replace(cfg.camera, render_height=img_hw[0], render_width=img_hw[1]))
    agent = AsymmetricPPOAgent(frame_stack=2, encoder_type=encoder_type, img_hw=img_hw)
    ckpt_path = tmp_path / "actor.pt"
    agent.actor.save_actor(ckpt_path)

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert not any(
        k.startswith("critic") or k.startswith("aux_head") for k in payload["state_dict"]
    )
    assert payload["config"]["img_hw"] == list(img_hw)
    if legacy:
        del payload["config"]["img_hw"]
        torch.save(payload, ckpt_path)
    exported = load_actor(ckpt_path, map_location="cpu")
    assert exported.img_hw == img_hw

    with SO101BowlingEnv(cfg=cfg, render_rgb=True, frame_stack=2) as env:
        obs, _ = env.reset(seed=5)
        agent.actor.eval()
        with torch.no_grad():
            a_train = agent.actor.act(obs, deterministic=True)
            a_export = exported.act({k: obs[k] for k in ACTOR_OBS_KEYS}, deterministic=True)

    assert a_train.shape == (4,)
    assert torch.allclose(a_train, a_export, atol=1e-6)


@pytest.mark.parametrize("encoder_type", ["spatial_grid", "spatial_softmax"])
def test_encoders_and_masked_aux_gradients(encoder_type: str) -> None:
    torch.manual_seed(4)
    agent = AsymmetricPPOAgent(frame_stack=2, encoder_type=encoder_type)
    obs = _synthetic_obs(batch_size=4, seed=19)
    obs["privileged_aux_mask"] = torch.tensor(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        dtype=torch.float32,
    )

    targets = obs["privileged_aux_targets"]
    mask = obs["privileged_aux_mask"]
    corrupted_targets = targets.masked_fill(~mask.bool().repeat_interleave(2, dim=-1), 1e6)

    _, _, _, _, aux_pred = agent.get_action_and_value(obs)
    aux_loss = masked_aux_loss(aux_pred, targets, mask)
    corrupted_loss = masked_aux_loss(aux_pred, corrupted_targets, mask)
    assert torch.isfinite(aux_loss)
    assert torch.equal(aux_loss, corrupted_loss)

    zero_mask_loss = masked_aux_loss(aux_pred, corrupted_targets, torch.zeros_like(mask))
    assert float(zero_mask_loss.detach()) == 0.0 and torch.isfinite(zero_mask_loss)

    aux_loss.backward()
    for param in agent.critic.parameters():
        assert param.grad is None or torch.count_nonzero(param.grad) == 0
    for enc in (agent.actor.wrist_encoder, agent.actor.tripod_encoder):
        grad_norm = sum(float(p.grad.abs().sum()) for p in enc.parameters() if p.grad is not None)
        assert grad_norm > 0.0


def test_gae_terminated_vs_truncated() -> None:
    rewards = torch.tensor([[1.0, 1.0], [5.0, 5.0], [3.0, 3.0], [4.0, 4.0]])
    values = torch.tensor([[0.5, 0.5], [2.0, 2.0], [1.0, 1.0], [4.0, 4.0]])
    terminated = torch.zeros_like(rewards, dtype=torch.bool)
    truncated = torch.zeros_like(terminated)
    terminated[0, 0] = True
    truncated[0, 1] = True
    next_value = torch.tensor([3.0, 3.0])
    final_values = torch.zeros_like(rewards)
    final_values[0] = torch.tensor([999.0, 10.0])

    adv, ret = compute_gae(
        rewards=rewards,
        values=values,
        terminated=terminated,
        truncated=truncated,
        next_value=next_value,
        final_values=final_values,
        gamma=0.9,
        gae_lambda=0.95,
    )
    # Step 0 env 0: terminated -> 1.0 + 0 - 0.5 = 0.5
    assert adv[0, 0] == pytest.approx(0.5)
    # Step 0 env 1: truncated -> 1.0 + 0.9 * 10.0 - 0.5 = 9.5 (ignores step 1 adv!)
    assert adv[0, 1] == pytest.approx(9.5)
    delta1 = 5.0 + 0.9 * 1.0 - 2.0
    delta2 = 3.0 + 0.9 * 4.0 - 1.0
    delta3 = 4.0 + 0.9 * 3.0 - 4.0
    decay = 0.9 * 0.95
    expected = torch.tensor([
        delta1 + decay * delta2 + decay**2 * delta3,
        delta2 + decay * delta3,
        delta3,
    ])
    torch.testing.assert_close(adv[1:], expected[:, None].expand(-1, 2))
    assert torch.allclose(ret, adv + values)


def test_ppo_end_to_end_smoke_train_eval_and_ablation(tmp_path: Path) -> None:
    ppo_cfg = PPOConfig(
        seed=0,
        dr_enabled=True,
        frame_stack=2,
        num_envs=2,
        rollout_steps=16,
        total_timesteps=64,
        device="cpu",
        num_epochs=2,
        minibatch_size=16,
        drq_pad_px=4,
        encoder_type="spatial_grid",
        eval_interval_updates=2,
        eval_episodes=1,
        checkpoint_interval_updates=2,
        save_video=False,
    )
    wcfg = load_config()
    run_dir = tmp_path / "ppo_smoke"
    summary = train_ppo(ppo_cfg=ppo_cfg, workspace_cfg=wcfg, run_dir=run_dir)

    assert summary["global_step"] == 64
    for key in ("policy_loss", "value_loss", "aux_loss", "approx_kl", "post_update_kl"):
        assert np.isfinite(summary["last_metrics"][key])
    assert (run_dir / "best_actor.pt").exists()
    assert (run_dir / "final_actor.pt").exists()
    assert (run_dir / "latest.pt").exists()

    actor = load_actor(run_dir / "best_actor.pt", map_location="cpu")
    ablation = evaluate_camera_ablation(
        actor=actor,
        cfg=wcfg,
        dr_enabled=False,
        frame_stack=2,
        gamma=ppo_cfg.gamma,
        episodes=1,
        seed=42,
        device="cpu",
    )
    assert set(ablation.keys()) == {"both", "no_wrist", "no_tripod"}
    for metrics in ablation.values():
        assert metrics["episodes"] == 1
        assert np.isfinite(metrics["mean_return"])


def test_sync_ppo_smoke_train_saves_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = load_config()
    cfg = replace(cfg, camera=replace(cfg.camera, render_height=49, render_width=81))
    ppo_cfg = PPOConfig(
        num_envs=1,
        rollout_steps=1,
        total_timesteps=1,
        device="cpu",
        num_epochs=1,
        minibatch_size=1,
        eval_episodes=1,
        save_video=True,
    )
    def make_sync_env(**kwargs):
        env = vec_env.make_vec_env(**kwargs)
        assert isinstance(env, vec_env.SyncVecBowlingEnv)
        return env

    factory = Mock(side_effect=make_sync_env)
    monkeypatch.setattr(ppo, "make_vec_env", factory)
    summary = train_ppo(ppo_cfg, workspace_cfg=cfg, run_dir=tmp_path)
    factory.assert_called_once()
    assert summary["global_step"] == 1
    video = tmp_path / "videos" / "eval_step_0000001.mp4"
    capture = cv2.VideoCapture(str(video))
    try:
        ok, frame = capture.read()
        assert ok
        assert frame.shape == (98, 324, 3)
    finally:
        capture.release()


@pytest.mark.parametrize("module", ["so101_bowling", "so101_bowling.vec_env"])
def test_package_import_does_not_load_training_dependencies(module: str) -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; import {module}; "
            "assert 'torch' not in sys.modules; assert 'tensorboard' not in sys.modules",
        ],
        check=True,
    )


@pytest.mark.parametrize("failure_stage", ["spawn", "handshake"])
def test_subproc_startup_failure_cleans_up(
    monkeypatch: pytest.MonkeyPatch, failure_stage: str
) -> None:
    thread_vars = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
    monkeypatch.setenv(thread_vars[0], "6")
    for name in thread_vars[1:]:
        monkeypatch.delenv(name, raising=False)

    pipes = [(Mock(), MagicMock()) for _ in range(2)]
    procs = [Mock(pid=None), Mock(pid=None)]
    startup_error = EOFError("worker startup failed")

    def start_worker(index: int) -> None:
        assert [os.environ[name] for name in thread_vars] == ["1"] * 4
        if index == 1 and failure_stage == "spawn":
            raise startup_error
        procs[index].pid = 100 + index

    for index, proc in enumerate(procs):
        proc.start.side_effect = lambda index=index: start_worker(index)
    procs[0].is_alive.side_effect = [True, True, False]
    procs[1].is_alive.return_value = False
    pipes[1][0].recv.side_effect = startup_error
    ctx = Mock()
    ctx.Pipe.side_effect = pipes
    ctx.Process.side_effect = procs
    monkeypatch.setattr(vec_env.mp, "get_context", Mock(return_value=ctx))

    with pytest.raises(EOFError, match="worker startup failed"):
        vec_env.SubprocVecBowlingEnv(num_envs=2, cfg=load_config())

    for parent, child in pipes:
        parent.close.assert_called_once_with()
        child.__exit__.assert_called_once()
    for proc in procs:
        proc.close.assert_called_once_with()
    procs[0].terminate.assert_called_once_with()
    procs[0].kill.assert_called_once_with()
    assert [call.kwargs["timeout"] for call in procs[0].join.call_args_list] == [5.0, 2.0, 2.0]
    if failure_stage == "spawn":
        procs[1].join.assert_not_called()
    else:
        procs[1].join.assert_called_once_with(timeout=5.0)
    assert [os.environ.get(name) for name in thread_vars] == ["6", None, None, None]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_envs", 0),
        ("rollout_steps", 0),
        ("total_timesteps", 0),
        ("minibatch_size", 0),
        ("eval_interval_updates", 0),
        ("checkpoint_interval_updates", 0),
        ("drq_pad_px", -1),
        ("frame_stack", 0),
    ],
)
def test_train_ppo_rejects_invalid_config(tmp_path: Path, field: str, value: int) -> None:
    run_dir = tmp_path / "invalid"
    with pytest.raises(ValueError, match=field):
        train_ppo(PPOConfig(**{field: value}), run_dir=run_dir)
    assert not run_dir.exists()


def test_evaluate_actor_rejects_zero_episodes() -> None:
    with pytest.raises(ValueError, match="episodes"):
        evaluate_actor(actor=DualCameraActor(), cfg=load_config(), episodes=0)


def test_train_ppo_closes_writer_if_vec_env_startup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = Mock()
    monkeypatch.setattr(ppo, "SummaryWriter", Mock(return_value=writer))
    monkeypatch.setattr(ppo, "make_vec_env", Mock(side_effect=EOFError("startup failed")))
    cfg = PPOConfig(num_envs=1, rollout_steps=1, device="cpu")
    with pytest.raises(EOFError, match="startup failed"):
        train_ppo(cfg, run_dir=tmp_path)
    writer.close.assert_called_once_with()


def test_preflight_ppo_passes_and_detects_nan_or_zero_aux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from so101_bowling.ppo import preflight_ppo

    cfg = PPOConfig(
        num_envs=1,
        rollout_steps=8,
        minibatch_size=4,
        lr=1.0e-3,
        aux_coef=1.0,
        device="cpu",
    )
    report = preflight_ppo(cfg, workspace_cfg=load_config(), steps=100, batch_size=4)
    assert report["algo"] == "ppo"
    assert report["passed"] is True
    assert report["nan_inf_detected"] is False
    assert report["aux_mask_active_elements"] > 0.0
    assert report["final_losses"]["aux_loss"] < 0.08
    assert report["final_losses"]["aux_loss"] < report["initial_losses"]["aux_loss"]

    # Inject NaN into masked_aux_loss to verify preflight catches NaN/Inf cleanly.
    monkeypatch.setattr(
        ppo,
        "masked_aux_loss",
        lambda pred, tgt, mask: pred.sum() * 0.0 + float("nan"),
    )
    nan_report = preflight_ppo(cfg, workspace_cfg=load_config(), steps=10, batch_size=4)
    assert nan_report["passed"] is False
    assert nan_report["nan_inf_detected"] is True
    assert nan_report["checks"]["finite_ok"] is False


def test_evaluate_actor_steps_eval_aux_and_eval_ppo_cli(tmp_path: Path) -> None:
    import json

    wcfg = load_config()
    agent = AsymmetricPPOAgent(frame_stack=2, encoder_type="spatial_grid")
    ev = evaluate_actor(
        actor=agent.actor,
        cfg=wcfg,
        dr_enabled=False,
        frame_stack=2,
        episodes=1,
        seed=42,
        device="cpu",
        aux_head=agent.aux_head,
    )
    assert ev["mean_episode_steps"] > 0.0
    assert ev["eval_aux_loss"] is not None
    assert np.isfinite(ev["eval_aux_loss"])

    ppo_cfg = PPOConfig(
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
    run_dir = tmp_path / "ppo_eval_cli"
    summary = train_ppo(ppo_cfg=ppo_cfg, workspace_cfg=wcfg, run_dir=run_dir)
    assert summary["nan_inf_detected"] is False
    assert "validation_loss_note" in summary
    assert "final_losses" in summary
    assert summary["eval_aux_loss"] is not None
    assert summary["final_eval"]["mean_episode_steps"] > 0.0
    metrics_row = json.loads((run_dir / "metrics.jsonl").read_text().strip().splitlines()[-1])
    assert "eval_aux_loss" in metrics_row
    assert "eval_eval_aux_loss" not in metrics_row

    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "eval_ppo.py"),
            "--run-dir",
            str(run_dir),
            "--episodes",
            "1",
            "--device",
            "cpu",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    eval_out = json.loads(proc.stdout)
    assert eval_out["mean_episode_steps"] > 0.0
    assert "training_metrics" in eval_out
    assert eval_out["training_metrics"]["nan_inf_detected"] is False
    assert "final_losses" in eval_out["training_metrics"]
