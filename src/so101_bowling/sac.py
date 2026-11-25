"""Asymmetric dual-camera SAC training, DrQ-v2 view shifts, UTD accounting, and helpers."""

from __future__ import annotations

import json
import math
import random
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter

from so101_bowling.config import WorkspaceConfig, load_config
from so101_bowling.env import SO101BowlingEnv
from so101_bowling.ppo import (
    VALIDATION_LOSS_NOTE,
    _assert_finite_mapping,
    _assert_finite_params,
    _assert_finite_tensor,
    _module_grad_and_param_stats,
    evaluate_actor,
    resolve_device,
)
from so101_bowling.ppo_models import (
    ACTION_DIM,
    AUX_MASK_DIM,
    AUX_TARGET_DIM,
    CAMERA_META_DIM,
    PRIVILEGED_STATE_DIM,
    PROPRIO_DIM,
    masked_aux_loss,
)
from so101_bowling.sac_models import AsymmetricSACAgent
from so101_bowling.tracking import WandbConfig, WandbTracker
from so101_bowling.vec_env import make_vec_env

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SAC_YAML = _REPO_ROOT / "configs" / "sac.yaml"


@dataclass(frozen=True)
class SACConfig:
    seed: int = 0
    dr_enabled: bool = False
    frame_stack: int = 2
    num_envs: int = 4
    total_timesteps: int = 100000
    capacity: int = 50000
    learning_starts: int = 2000
    batch_size: int = 128
    utd_ratio: float = 0.5
    policy_delay: int = 1
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    alpha_lr: float = 3.0e-4
    init_alpha: float = 0.2
    autotune_alpha: bool = True
    aux_coef: float = 0.5
    max_grad_norm: float = 1.0
    drq_pad_px: int = 4
    encoder_type: str = "spatial_grid"
    device: str = "auto"
    eval_interval_steps: int = 5000
    eval_episodes: int = 10
    checkpoint_interval_steps: int = 10000
    save_video: bool = True
    demo_pretrain_steps: int = 0

    def __post_init__(self) -> None:
        for name in (
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
            "demo_pretrain_steps",
        ):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, int):
                raise ValueError(f"{name} must be an int, got {type(val).__name__}")
        for name in ("dr_enabled", "autotune_alpha", "save_video"):
            val = getattr(self, name)
            if not isinstance(val, bool):
                raise ValueError(f"{name} must be a bool, got {type(val).__name__}")
        for name in (
            "num_envs",
            "frame_stack",
            "total_timesteps",
            "eval_interval_steps",
            "eval_episodes",
            "checkpoint_interval_steps",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.policy_delay < 1:
            raise ValueError("policy_delay must be >= 1")
        if self.learning_starts < 0:
            raise ValueError("learning_starts must be nonnegative")
        if self.demo_pretrain_steps < 0:
            raise ValueError("demo_pretrain_steps must be nonnegative")
        if self.drq_pad_px < 0:
            raise ValueError("drq_pad_px must be nonnegative")
        min_capacity = self.num_envs * (self.frame_stack + 2)
        if self.capacity < min_capacity:
            raise ValueError(
                f"capacity must be >= num_envs * (frame_stack + 2) ({min_capacity}), got {self.capacity}"
            )
        usable_capacity = (self.capacity // self.num_envs) * self.num_envs
        if self.batch_size > usable_capacity:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be <= usable replay capacity "
                f"({usable_capacity} = ({self.capacity} // {self.num_envs}) * {self.num_envs})"
            )
        if self.encoder_type not in ("spatial_grid", "spatial_softmax"):
            raise ValueError(f"Unsupported encoder_type: {self.encoder_type!r}")

        for name in (
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
            raw_val = getattr(self, name)
            if isinstance(raw_val, bool) or not isinstance(raw_val, (int, float)):
                raise ValueError(f"{name} must be a real number, got {type(raw_val).__name__}")
            val = float(raw_val)
            if not math.isfinite(val):
                raise ValueError(f"{name} must be finite, got {val}")

        if self.utd_ratio <= 0.0:
            raise ValueError("utd_ratio must be positive")
        if not (0.0 < self.gamma <= 1.0):
            raise ValueError("gamma must be in (0, 1]")
        if not (0.0 < self.tau <= 1.0):
            raise ValueError("tau must be in (0, 1]")
        for name in ("actor_lr", "critic_lr", "alpha_lr", "init_alpha", "max_grad_norm"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.aux_coef < 0.0:
            raise ValueError("aux_coef must be nonnegative")


def load_sac_config(path: str | Path | None = None) -> SACConfig:
    """Load SACConfig from YAML, defaulting to configs/sac.yaml."""
    cfg_path = Path(path) if path is not None else _DEFAULT_SAC_YAML
    if not cfg_path.exists():
        raise FileNotFoundError(f"SAC config not found: {cfg_path}")
    raw = yaml.safe_load(cfg_path.read_text()) or {}
    return SACConfig(**raw)


def compute_utd_updates(
    global_step: int,
    buffer_size: int,
    cfg: SACConfig,
    utd_carry: float,
    demo_warmstart: bool = False,
) -> tuple[int, float]:
    """Compute integer gradient updates for the current env step with fractional carry."""
    if (not demo_warmstart and global_step < cfg.learning_starts) or buffer_size < cfg.batch_size:
        return 0, utd_carry
    budget = utd_carry + cfg.num_envs * cfg.utd_ratio
    n_updates = int(budget + 1e-12)
    new_carry = max(0.0, budget - n_updates)
    return n_updates, new_carry


def _sample_drq_shifts(
    batch_size: int,
    pad_px: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Tensor | None:
    """Draw independent per-sample (dy, dx) shifts in [-pad_px, +pad_px]."""
    if pad_px <= 0:
        return None
    if generator is not None and generator.device != device:
        return torch.randint(
            -pad_px,
            pad_px + 1,
            (batch_size, 2),
            device=generator.device,
            dtype=torch.long,
            generator=generator,
        ).to(device)
    return torch.randint(
        -pad_px,
        pad_px + 1,
        (batch_size, 2),
        device=device,
        dtype=torch.long,
        generator=generator,
    )


def update_sac_step(
    agent: AsymmetricSACAgent,
    batch: Mapping[str, Any],
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    alpha_opt: torch.optim.Optimizer | None,
    cfg: SACConfig,
    do_actor_update: bool = True,
    torch_gen: torch.Generator | None = None,
) -> dict[str, float]:
    """Execute one SAC critic update and optional actor/aux/alpha update."""
    actions = batch["actions"]
    rewards = batch["rewards"]
    terminated = batch["terminated"]
    batch_size = int(actions.shape[0])
    device = actions.device

    next_w_shifts = _sample_drq_shifts(batch_size, cfg.drq_pad_px, device, generator=torch_gen)
    next_t_shifts = _sample_drq_shifts(batch_size, cfg.drq_pad_px, device, generator=torch_gen)

    with torch.no_grad():
        next_action, next_log_prob, _ = agent.actor.sample_squashed(
            batch["next_obs"],
            wrist_shifts=next_w_shifts,
            tripod_shifts=next_t_shifts,
        )
        next_q1, next_q2 = agent.critic_target(batch["next_obs"], next_action)
        target_q = torch.minimum(next_q1, next_q2) - agent.alpha.detach() * next_log_prob
        not_term = (~terminated).to(dtype=rewards.dtype)
        backup = rewards + not_term * cfg.gamma * target_q

    q1, q2 = agent.critic(batch["obs"], actions)
    critic_loss = 0.5 * (F.mse_loss(q1, backup) + F.mse_loss(q2, backup))

    _assert_finite_mapping(
        "sac.critic_step",
        {
            "next_action": next_action,
            "next_log_prob": next_log_prob,
            "next_q1": next_q1,
            "next_q2": next_q2,
            "target_q": target_q,
            "backup": backup,
            "q1": q1,
            "q2": q2,
            "critic_loss": critic_loss,
        },
    )

    critic_opt.zero_grad(set_to_none=True)
    critic_loss.backward()
    critic_grad_norm = nn.utils.clip_grad_norm_(agent.critic.parameters(), cfg.max_grad_norm)
    _assert_finite_tensor("sac.critic_grad_norm", critic_grad_norm)
    critic_opt.step()
    critic_opt.zero_grad(set_to_none=True)
    agent.soft_update_target(cfg.tau)
    _assert_finite_params("sac.critic", agent.critic.parameters())

    q_all = torch.cat((q1.detach(), q2.detach()), dim=0)
    metrics: dict[str, float] = {
        "critic_loss": float(critic_loss.detach().cpu()),
        "actor_loss": 0.0,
        "aux_loss": 0.0,
        "alpha_loss": 0.0,
        "alpha": float(agent.alpha.detach().cpu()),
        "q_mean": float(q_all.mean().cpu()),
        "q_min": float(q_all.min().cpu()),
        "q_max": float(q_all.max().cpu()),
        "critic_grad_norm": float(critic_grad_norm.detach().cpu()),
        "actor_grad_norm": 0.0,
        "log_prob_mean": 0.0,
    }

    if not do_actor_update:
        return metrics

    w_shifts = _sample_drq_shifts(batch_size, cfg.drq_pad_px, device, generator=torch_gen)
    t_shifts = _sample_drq_shifts(batch_size, cfg.drq_pad_px, device, generator=torch_gen)
    actor_params = list(agent.actor.parameters()) + list(agent.aux_head.parameters())
    critic_params = list(agent.critic.parameters())

    actor_opt.zero_grad(set_to_none=True)
    prev_requires_grad = [p.requires_grad for p in critic_params]
    try:
        for p in critic_params:
            p.requires_grad_(False)
        pi_action, log_prob, z_vis = agent.actor.sample_squashed(
            batch["obs"],
            wrist_shifts=w_shifts,
            tripod_shifts=t_shifts,
        )
        q1_pi, q2_pi = agent.critic(batch["obs"], pi_action)
        q_pi = torch.minimum(q1_pi, q2_pi)
        policy_loss = (agent.alpha.detach() * log_prob - q_pi).mean()

        if cfg.aux_coef > 0.0:
            aux_pred = agent.aux_head(z_vis)
            aux_loss = masked_aux_loss(
                aux_pred,
                batch["obs"]["privileged_aux_targets"].to(dtype=torch.float32),
                batch["obs"]["privileged_aux_mask"],
            )
            total_actor_loss = policy_loss + cfg.aux_coef * aux_loss
        else:
            aux_loss = torch.zeros((), device=device, dtype=policy_loss.dtype)
            total_actor_loss = policy_loss

        _assert_finite_mapping(
            "sac.actor_step",
            {
                "pi_action": pi_action,
                "log_prob": log_prob,
                "q1_pi": q1_pi,
                "q2_pi": q2_pi,
                "policy_loss": policy_loss,
                "aux_loss": aux_loss,
                "total_actor_loss": total_actor_loss,
                "log_alpha": agent.log_alpha,
            },
        )
        total_actor_loss.backward()
    finally:
        for p, req in zip(critic_params, prev_requires_grad, strict=True):
            p.requires_grad_(req)

    actor_grad_norm = nn.utils.clip_grad_norm_(actor_params, cfg.max_grad_norm)
    _assert_finite_tensor("sac.actor_grad_norm", actor_grad_norm)
    actor_opt.step()
    _assert_finite_params("sac.actor", actor_params)

    if cfg.autotune_alpha and alpha_opt is not None:
        alpha_opt.zero_grad(set_to_none=True)
        alpha_loss = -(agent.log_alpha * (log_prob.detach() + agent.target_entropy)).mean()
        _assert_finite_tensor("sac.alpha_loss", alpha_loss)
        alpha_loss.backward()
        if agent.log_alpha.grad is not None:
            _assert_finite_tensor("sac.log_alpha.grad", agent.log_alpha.grad)
        alpha_opt.step()
        with torch.no_grad():
            agent.log_alpha.clamp_(-10.0, 2.0)
        _assert_finite_tensor("sac.log_alpha", agent.log_alpha)
    else:
        alpha_loss = torch.zeros((), device=device, dtype=policy_loss.dtype)

    metrics.update(
        {
            "actor_loss": float(policy_loss.detach().cpu()),
            "aux_loss": float(aux_loss.detach().cpu()),
            "alpha_loss": float(alpha_loss.detach().cpu()),
            "alpha": float(agent.alpha.detach().cpu()),
            "actor_grad_norm": float(actor_grad_norm.detach().cpu()),
            "log_prob_mean": float(log_prob.detach().mean().cpu()),
        }
    )
    return metrics


def _set_seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def _save_sac_checkpoint(
    out_dir: Path,
    agent: AsymmetricSACAgent,
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    alpha_opt: torch.optim.Optimizer | None,
    global_step: int,
    total_critic_updates: int,
    cfg: SACConfig,
    best_score: tuple[float, float, float] | None = None,
    best_eval: dict[str, Any] | None = None,
    utd_carry: float = 0.0,
    last_log: dict[str, Any] | None = None,
    last_eval: dict[str, Any] | None = None,
) -> None:
    torch.save(
        {
            "agent": agent.state_dict(),
            "actor_opt": actor_opt.state_dict(),
            "critic_opt": critic_opt.state_dict(),
            "alpha_opt": alpha_opt.state_dict() if alpha_opt is not None else None,
            "global_step": global_step,
            "total_critic_updates": total_critic_updates,
            "best_score": list(best_score) if best_score is not None else None,
            "best_eval": dict(best_eval) if best_eval is not None else {},
            "last_eval": dict(last_eval) if last_eval is not None else {},
            "utd_carry": float(utd_carry),
            "last_log": dict(last_log) if last_log is not None else {},
            "sac_cfg": asdict(cfg),
        },
        out_dir / "checkpoint_latest.pt",
    )


def _extract_sac_final_losses(last_log: Mapping[str, Any]) -> dict[str, float]:
    keys = ("critic_loss", "actor_loss", "aux_loss", "alpha_loss", "alpha", "q_mean", "log_prob_mean")
    return {
        k: float(last_log[k])
        for k in keys
        if k in last_log and isinstance(last_log[k], (int, float)) and not isinstance(last_log[k], bool)
    }


def _write_sac_summary(
    out_dir: Path,
    *,
    global_step: int,
    total_timesteps: int,
    total_critic_updates: int,
    buffer_size: int,
    demo_transitions_loaded: int,
    demo_pretrain_steps_run: int,
    best_eval: dict[str, Any],
    last_eval: dict[str, Any],
    last_log: dict[str, Any],
    nan_inf_detected: bool,
    nan_inf_error: str | None = None,
    interrupted: bool = False,
    wandb_url: str | None = None,
) -> dict[str, Any]:
    eval_aux_loss = last_eval.get("eval_aux_loss")
    if eval_aux_loss is None and best_eval:
        eval_aux_loss = best_eval.get("eval_aux_loss")
    summary: dict[str, Any] = {
        "run_dir": str(out_dir),
        "global_step": int(global_step),
        "total_timesteps": int(total_timesteps),
        "total_critic_updates": int(total_critic_updates),
        "buffer_size": int(buffer_size),
        "demo_transitions_loaded": int(demo_transitions_loaded),
        "demo_pretrain_steps_run": int(demo_pretrain_steps_run),
        "best_eval": best_eval,
        "final_eval": last_eval,
        "last_metrics": last_log,
        "final_losses": _extract_sac_final_losses(last_log),
        "eval_aux_loss": eval_aux_loss,
        "nan_inf_detected": bool(nan_inf_detected),
        "validation_loss_note": VALIDATION_LOSS_NOTE,
    }
    if wandb_url is not None:
        summary["wandb_url"] = wandb_url
    if nan_inf_error is not None:
        summary["nan_inf_error"] = nan_inf_error
    if interrupted:
        summary["interrupted"] = True
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def preflight_sac(
    cfg: SACConfig,
    workspace_cfg: WorkspaceConfig | None = None,
    *,
    steps: int = 150,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Run deterministic single-batch overfit and numerical stability verification for Asymmetric SAC."""
    from so101_bowling.replay_buffer import RawFrameReplayBuffer

    if steps <= 0:
        raise ValueError("steps must be positive")
    wcfg = workspace_cfg if workspace_cfg is not None else load_config()
    rng = _set_seed(cfg.seed)
    torch_gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    img_hw = (wcfg.camera.render_height, wcfg.camera.render_width)

    target_batch = min(batch_size or cfg.batch_size, 32)
    if target_batch < 1:
        raise ValueError("batch_size must be positive")
    cap = max(64, target_batch * 2, cfg.frame_stack + 4)

    agent = AsymmetricSACAgent(
        frame_stack=cfg.frame_stack,
        action_dim=ACTION_DIM,
        encoder_type=cfg.encoder_type,
        img_hw=img_hw,
        init_alpha=cfg.init_alpha,
    ).to(device)
    actor_params = list(agent.actor.parameters()) + list(agent.aux_head.parameters())
    actor_opt = torch.optim.Adam(actor_params, lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(agent.critic.parameters(), lr=cfg.critic_lr)
    alpha_opt = (
        torch.optim.Adam([agent.log_alpha], lr=cfg.alpha_lr) if cfg.autotune_alpha else None
    )

    buffer = RawFrameReplayBuffer(
        capacity=cap,
        num_envs=1,
        frame_stack=cfg.frame_stack,
        img_hw=img_hw,
        action_dim=ACTION_DIM,
        state_dim=PRIVILEGED_STATE_DIM,
        proprio_dim=PROPRIO_DIM,
        camera_meta_dim=CAMERA_META_DIM,
        aux_target_dim=AUX_TARGET_DIM,
        aux_mask_dim=AUX_MASK_DIM,
    )

    with SO101BowlingEnv(
        cfg=wcfg,
        dr_enabled=cfg.dr_enabled,
        render_rgb=True,
        frame_stack=cfg.frame_stack,
        gamma=cfg.gamma,
    ) as env:
        obs, _ = env.reset(seed=cfg.seed)
        active_mask_count = 0.0
        max_collect = max(target_batch, 32)
        for step_idx in range(max_collect):
            action = rng.uniform(-1.0, 1.0, size=(ACTION_DIM,)).astype(np.float32)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            infos = [{"final_obs": next_obs}] if (truncated and not terminated) else [{}]
            buffer.add(
                obs={k: v[None] for k, v in obs.items()},
                actions=action[None],
                rewards=np.array([reward], dtype=np.float32),
                next_obs={k: v[None] for k, v in next_obs.items()},
                terminated=np.array([terminated], dtype=bool),
                truncated=np.array([truncated], dtype=bool),
                infos=infos,
            )
            active_mask_count += float(np.sum(obs["privileged_aux_mask"]))
            if terminated or truncated:
                obs, _ = env.reset(seed=cfg.seed + step_idx + 1)
            else:
                obs = next_obs
            if len(buffer) >= target_batch and active_mask_count > 0.0:
                break

    frozen_batch = buffer.sample(target_batch, device=device, rng=rng)
    for _ in range(4):
        if float(frozen_batch["obs"]["privileged_aux_mask"].sum().cpu().item()) > 0.0:
            break
        frozen_batch = buffer.sample(target_batch, device=device, rng=rng)

    aux_mask_active = float(frozen_batch["obs"]["privileged_aux_mask"].sum().cpu().item())
    preflight_cfg = replace(cfg, drq_pad_px=0)
    effective_steps = max(steps, 300 if min(cfg.actor_lr, cfg.critic_lr) < 1e-4 else steps)

    nan_inf_detected = False
    failure_reasons: list[str] = []
    initial_losses: dict[str, float] = {}
    final_losses: dict[str, float] = {}
    q_min = float("inf")
    q_max = -float("inf")
    log_alpha_min = float("inf")
    log_alpha_max = -float("inf")
    actor_std_min = float("inf")
    actor_std_max = 0.0
    max_grad_norm = 0.0
    max_abs_param = 0.0

    if aux_mask_active <= 0.0:
        failure_reasons.append("No active privileged_aux_mask elements in preflight batch")
    if cfg.aux_coef <= 0.0:
        failure_reasons.append("aux_coef must be > 0.0 for SAC preflight aux_loss verification")

    agent.train()
    try:
        for step_idx in range(effective_steps):
            step_metrics = update_sac_step(
                agent=agent,
                batch=frozen_batch,
                actor_opt=actor_opt,
                critic_opt=critic_opt,
                alpha_opt=alpha_opt,
                cfg=preflight_cfg,
                do_actor_update=True,
                torch_gen=torch_gen,
            )
            with torch.no_grad():
                std_t = torch.exp(agent.actor.log_std.clamp(-5.0, 2.0))
                _assert_finite_tensor(f"preflight_sac.step_{step_idx}.actor_std", std_t)
                actor_std_min = min(actor_std_min, float(std_t.min().cpu().item()))
                actor_std_max = max(actor_std_max, float(std_t.max().cpu().item()))
                log_alpha_val = float(agent.log_alpha.detach().cpu().item())
                log_alpha_min = min(log_alpha_min, log_alpha_val)
                log_alpha_max = max(log_alpha_max, log_alpha_val)
                _, step_max_param = _module_grad_and_param_stats(agent.parameters())
                max_abs_param = max(max_abs_param, step_max_param)

            q_min = min(q_min, step_metrics["q_min"])
            q_max = max(q_max, step_metrics["q_max"])
            max_grad_norm = max(
                max_grad_norm,
                step_metrics["critic_grad_norm"],
                step_metrics["actor_grad_norm"],
            )

            loss_snapshot = {
                "critic_loss": step_metrics["critic_loss"],
                "actor_loss": step_metrics["actor_loss"],
                "aux_loss": step_metrics["aux_loss"],
                "alpha_loss": step_metrics["alpha_loss"],
                "alpha": step_metrics["alpha"],
            }
            if step_idx == 0:
                initial_losses = loss_snapshot
            final_losses = loss_snapshot
    except FloatingPointError as exc:
        nan_inf_detected = True
        failure_reasons.append(str(exc))

    aux_overfit_ok = (
        cfg.aux_coef > 0.0
        and bool(final_losses)
        and bool(initial_losses)
        and final_losses["aux_loss"] < 0.08
        and final_losses["aux_loss"] < initial_losses["aux_loss"]
    )
    critic_overfit_ok = (
        bool(final_losses)
        and bool(initial_losses)
        and final_losses["critic_loss"] < 0.08
        and (
            final_losses["critic_loss"] < initial_losses["critic_loss"]
            or initial_losses["critic_loss"] < 0.02
        )
    )
    checks = {
        "finite_ok": not nan_inf_detected,
        "aux_mask_active_ok": aux_mask_active > 0.0,
        "non_zero_grads_ok": max_grad_norm > 0.0,
        "aux_overfit_ok": aux_overfit_ok,
        "critic_overfit_ok": critic_overfit_ok,
    }
    for cname, ok in checks.items():
        if not ok and not any(cname in r for r in failure_reasons):
            if cname not in ("finite_ok", "aux_mask_active_ok"):
                failure_reasons.append(f"Preflight check failed: {cname}")

    return {
        "algo": "sac",
        "passed": all(checks.values()),
        "device": str(device),
        "batch_size": target_batch,
        "steps": effective_steps,
        "nan_inf_detected": nan_inf_detected,
        "aux_mask_active_elements": aux_mask_active,
        "initial_losses": initial_losses,
        "final_losses": final_losses,
        "numerical_stats": {
            "q_min": q_min if math.isfinite(q_min) else None,
            "q_max": q_max if math.isfinite(q_max) else None,
            "log_alpha_min": log_alpha_min if math.isfinite(log_alpha_min) else None,
            "log_alpha_max": log_alpha_max if math.isfinite(log_alpha_max) else None,
            "actor_std_min": actor_std_min if math.isfinite(actor_std_min) else None,
            "actor_std_max": actor_std_max,
            "max_grad_norm": max_grad_norm,
            "max_abs_param": max_abs_param,
        },
        "checks": checks,
        "failure_reasons": failure_reasons,
    }


def train_sac(
    cfg: SACConfig,
    workspace_cfg: WorkspaceConfig | None = None,
    run_dir: str | Path = "runs/sac",
    resume_from: str | Path | None = None,
    demo_dataset: str | Path | None = None,
    wandb_cfg: WandbConfig | None = None,
) -> dict[str, Any]:
    """Train AsymmetricSACAgent with RawFrameReplayBuffer and export standalone actors."""
    from so101_bowling.replay_buffer import RawFrameReplayBuffer

    wcfg = workspace_cfg if workspace_cfg is not None else load_config()
    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sac_config.json").write_text(json.dumps(asdict(cfg), indent=2))

    rng = _set_seed(cfg.seed)
    torch_gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    img_hw = (wcfg.camera.render_height, wcfg.camera.render_width)

    agent = AsymmetricSACAgent(
        frame_stack=cfg.frame_stack,
        action_dim=ACTION_DIM,
        encoder_type=cfg.encoder_type,
        img_hw=img_hw,
        init_alpha=cfg.init_alpha,
    ).to(device)
    actor_params = list(agent.actor.parameters()) + list(agent.aux_head.parameters())
    actor_opt = torch.optim.Adam(actor_params, lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(agent.critic.parameters(), lr=cfg.critic_lr)
    alpha_opt = (
        torch.optim.Adam([agent.log_alpha], lr=cfg.alpha_lr) if cfg.autotune_alpha else None
    )

    buffer = RawFrameReplayBuffer(
        capacity=cfg.capacity,
        num_envs=cfg.num_envs,
        frame_stack=cfg.frame_stack,
        img_hw=img_hw,
        action_dim=ACTION_DIM,
        state_dim=PRIVILEGED_STATE_DIM,
        proprio_dim=PROPRIO_DIM,
        camera_meta_dim=CAMERA_META_DIM,
        aux_target_dim=AUX_TARGET_DIM,
        aux_mask_dim=AUX_MASK_DIM,
    )

    demo_transitions_loaded = 0
    if demo_dataset is not None:
        from so101_bowling.dataset import load_dataset_into_replay_buffer

        demo_transitions_loaded = load_dataset_into_replay_buffer(demo_dataset, buffer)

    has_demo_warmstart = demo_transitions_loaded >= cfg.batch_size
    effective_learning_starts = 0 if has_demo_warmstart else cfg.learning_starts
    demo_pretrain_steps_run = 0

    metrics_path = out_dir / "metrics.jsonl"
    best_score: tuple[float, float, float] = (-1.0, -1.0, -float("inf"))
    best_eval: dict[str, Any] = {}
    last_eval: dict[str, Any] = {}
    last_log: dict[str, Any] = {}
    global_step = 0
    total_critic_updates = 0
    utd_carry = 0.0

    if resume_from is not None:
        ckpt_path = Path(resume_from)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"SAC resume checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        agent.load_state_dict(ckpt["agent"])
        actor_opt.load_state_dict(ckpt["actor_opt"])
        critic_opt.load_state_dict(ckpt["critic_opt"])
        if alpha_opt is not None and ckpt.get("alpha_opt") is not None:
            alpha_opt.load_state_dict(ckpt["alpha_opt"])
        global_step = int(ckpt.get("global_step", 0))
        total_critic_updates = int(ckpt.get("total_critic_updates", 0))
        utd_carry = float(ckpt.get("utd_carry", 0.0))
        prev_summary: dict[str, Any] = {}
        summary_candidate = ckpt_path.parent / "summary.json"
        if not summary_candidate.exists() and (out_dir / "summary.json").exists():
            summary_candidate = out_dir / "summary.json"
        if summary_candidate.exists():
            prev_summary = json.loads(summary_candidate.read_text())
        if "best_score" in ckpt and ckpt["best_score"] is not None:
            best_score = tuple(float(x) for x in ckpt["best_score"])
            best_eval = dict(ckpt.get("best_eval", {}))
        elif prev_summary:
            prev_be = prev_summary.get("best_eval") or {}
            if prev_be:
                best_eval = dict(prev_be)
                best_score = (
                    float(prev_be.get("legal_strike_rate", -1.0)),
                    float(prev_be.get("mean_credited_pins", -1.0)),
                    float(prev_be.get("mean_return", -float("inf"))),
                )
        if "last_eval" in ckpt and ckpt["last_eval"]:
            last_eval = dict(ckpt["last_eval"])
        elif prev_summary and prev_summary.get("final_eval"):
            last_eval = dict(prev_summary["final_eval"])
        elif best_eval:
            last_eval = dict(best_eval)
        if "last_log" in ckpt and ckpt["last_log"]:
            last_log = dict(ckpt["last_log"])
        elif prev_summary and prev_summary.get("last_metrics"):
            last_log = dict(prev_summary["last_metrics"])
    elif metrics_path.exists():
        metrics_path.unlink()

    next_eval_step = ((global_step // cfg.eval_interval_steps) + 1) * cfg.eval_interval_steps
    next_ckpt_step = (
        (global_step // cfg.checkpoint_interval_steps) + 1
    ) * cfg.checkpoint_interval_steps
    log_interval_steps = max(cfg.num_envs, min(1000, cfg.eval_interval_steps))
    next_log_step = ((global_step // log_interval_steps) + 1) * log_interval_steps

    tracker = WandbTracker(
        wandb_cfg,
        run_dir=out_dir,
        algo="sac",
        train_config={**asdict(cfg), "demo_transitions_loaded": demo_transitions_loaded},
    )
    if tracker.run_url:
        print(f"[sac] wandb_url={tracker.run_url}", flush=True)

    try:
        with ExitStack() as stack:
            writer = SummaryWriter(log_dir=str(out_dir / "tb"))
            stack.callback(writer.close)
            vec_env = stack.enter_context(
                make_vec_env(
                    num_envs=cfg.num_envs,
                    cfg=wcfg,
                    dr_enabled=cfg.dr_enabled,
                    render_rgb=True,
                    frame_stack=cfg.frame_stack,
                    gamma=cfg.gamma,
                    base_seed=cfg.seed,
                )
            )
            if abs(vec_env.scorer_gamma - cfg.gamma) > 1e-6:
                raise ValueError(
                    f"Scorer gamma {vec_env.scorer_gamma} != SAC gamma {cfg.gamma}"
                )

            obs = vec_env.reset(seed=cfg.seed)
            ep_outcomes: list[dict[str, Any]] = []
            recent_step_metrics: list[dict[str, float]] = []
            recent_actor_metrics: list[dict[str, float]] = []
            t_start = perf_counter()

            if has_demo_warmstart and cfg.demo_pretrain_steps > 0 and resume_from is None:
                pretrain_log_every = max(1, cfg.demo_pretrain_steps // 5)
                agent.train()
                for pre_idx in range(1, cfg.demo_pretrain_steps + 1):
                    batch = buffer.sample(cfg.batch_size, device=device, rng=rng)
                    do_actor = (total_critic_updates + 1) % cfg.policy_delay == 0
                    step_metrics = update_sac_step(
                        agent=agent,
                        batch=batch,
                        actor_opt=actor_opt,
                        critic_opt=critic_opt,
                        alpha_opt=alpha_opt,
                        cfg=cfg,
                        do_actor_update=do_actor,
                        torch_gen=torch_gen,
                    )
                    total_critic_updates += 1
                    demo_pretrain_steps_run += 1
                    recent_step_metrics.append(step_metrics)
                    if do_actor:
                        recent_actor_metrics.append(step_metrics)
                    if pre_idx % pretrain_log_every == 0 or pre_idx == cfg.demo_pretrain_steps:
                        print(
                            f"[sac] demo_pretrain {pre_idx}/{cfg.demo_pretrain_steps} "
                            f"critic={step_metrics['critic_loss']:.4f} "
                            f"aux={step_metrics['aux_loss']:.4f} "
                            f"alpha={step_metrics['alpha']:.4f}",
                            flush=True,
                        )

            while global_step < cfg.total_timesteps:
                if global_step < effective_learning_starts:
                    actions = rng.uniform(-1.0, 1.0, size=(cfg.num_envs, ACTION_DIM)).astype(
                        np.float32
                    )
                else:
                    agent.actor.eval()
                    with torch.no_grad():
                        actions = np.clip(
                            agent.actor.act(obs, deterministic=False).cpu().numpy(),
                            -1.0,
                            1.0,
                        ).astype(np.float32)

                _assert_finite_tensor(
                    f"sac.rollout.step_{global_step}.actions", torch.as_tensor(actions)
                )
                next_obs, rewards, terminated, truncated, infos = vec_env.step(actions)
                _assert_finite_tensor(
                    f"sac.rollout.step_{global_step}.rewards", torch.as_tensor(rewards)
                )
                buffer.add(obs, actions, rewards, next_obs, terminated, truncated, infos)
                for info in infos:
                    if "outcome" in info:
                        ep_outcomes.append(info["outcome"])

                obs = next_obs
                global_step += cfg.num_envs

                n_updates, utd_carry = compute_utd_updates(
                    global_step,
                    len(buffer),
                    cfg,
                    utd_carry,
                    demo_warmstart=has_demo_warmstart,
                )
                if n_updates > 0:
                    agent.train()
                    for _ in range(n_updates):
                        batch = buffer.sample(cfg.batch_size, device=device, rng=rng)
                        do_actor = (total_critic_updates + 1) % cfg.policy_delay == 0
                        step_metrics = update_sac_step(
                            agent=agent,
                            batch=batch,
                            actor_opt=actor_opt,
                            critic_opt=critic_opt,
                            alpha_opt=alpha_opt,
                            cfg=cfg,
                            do_actor_update=do_actor,
                            torch_gen=torch_gen,
                        )
                        total_critic_updates += 1
                        recent_step_metrics.append(step_metrics)
                        if do_actor:
                            recent_actor_metrics.append(step_metrics)

                is_final = global_step >= cfg.total_timesteps
                should_eval = global_step >= next_eval_step or is_final
                should_ckpt = global_step >= next_ckpt_step or is_final
                should_log = global_step >= next_log_step or should_eval or should_ckpt

                ev: dict[str, Any] | None = None
                if should_eval:
                    while next_eval_step <= global_step:
                        next_eval_step += cfg.eval_interval_steps
                    vid_path = (
                        out_dir / "videos" / f"eval_step_{global_step:07d}.mp4"
                        if cfg.save_video
                        else None
                    )
                    ev = evaluate_actor(
                        actor=agent.actor,
                        cfg=wcfg,
                        dr_enabled=cfg.dr_enabled,
                        frame_stack=cfg.frame_stack,
                        gamma=cfg.gamma,
                        episodes=cfg.eval_episodes,
                        seed=cfg.seed + 10000,
                        device=device,
                        video_path=vid_path,
                        aux_head=agent.aux_head,
                    )
                    last_eval = ev
                    score = (ev["legal_strike_rate"], ev["mean_credited_pins"], ev["mean_return"])
                    if score > best_score or not (out_dir / "actor_best.pt").exists():
                        best_score = score
                        best_eval = ev
                        agent.actor.save_actor(out_dir / "actor_best.pt")
                    print(
                        f"[sac] eval step={global_step}/{cfg.total_timesteps} "
                        f"strike={ev['legal_strike_rate']:.3f} pins={ev['mean_credited_pins']:.3f} "
                        f"steps={ev['mean_episode_steps']:.1f} return={ev['mean_return']:.3f} "
                        f"eval_aux={ev.get('eval_aux_loss')}",
                        flush=True,
                    )
                    if vid_path is not None:
                        tracker.log_video(
                            "sac/eval_video",
                            vid_path,
                            step=global_step,
                            fps=int(round(wcfg.robot.control_hz)),
                        )

                if should_log:
                    while next_log_step <= global_step:
                        next_log_step += log_interval_steps
                    elapsed = max(1e-6, perf_counter() - t_start)
                    recent_ep = ep_outcomes[-50:]
                    log_row: dict[str, Any] = {
                        "global_step": global_step,
                        "total_critic_updates": total_critic_updates,
                        "buffer_size": len(buffer),
                        "transitions_per_sec": global_step / elapsed,
                        "rollout_episodes": len(ep_outcomes),
                        "alpha": float(agent.alpha.detach().cpu()),
                        "critic_loss": (
                            float(np.mean([m["critic_loss"] for m in recent_step_metrics]))
                            if recent_step_metrics
                            else 0.0
                        ),
                        "q_mean": (
                            float(np.mean([m["q_mean"] for m in recent_step_metrics]))
                            if recent_step_metrics
                            else 0.0
                        ),
                        "actor_loss": (
                            float(np.mean([m["actor_loss"] for m in recent_actor_metrics]))
                            if recent_actor_metrics
                            else 0.0
                        ),
                        "aux_loss": (
                            float(np.mean([m["aux_loss"] for m in recent_actor_metrics]))
                            if recent_actor_metrics
                            else 0.0
                        ),
                        "alpha_loss": (
                            float(np.mean([m["alpha_loss"] for m in recent_actor_metrics]))
                            if recent_actor_metrics
                            else 0.0
                        ),
                        "log_prob_mean": (
                            float(np.mean([m["log_prob_mean"] for m in recent_actor_metrics]))
                            if recent_actor_metrics
                            else 0.0
                        ),
                        "train_legal_strike_rate": (
                            float(np.mean([int(bool(o.get("legal_strike", False))) for o in recent_ep]))
                            if recent_ep
                            else 0.0
                        ),
                        "train_mean_credited_pins": (
                            float(np.mean([int(o.get("credited_pins", 0)) for o in recent_ep]))
                            if recent_ep
                            else 0.0
                        ),
                        "train_illegal_rate": (
                            float(
                                np.mean([int(not bool(o.get("legal_delivery", False))) for o in recent_ep])
                            )
                            if recent_ep
                            else 0.0
                        ),
                    }
                    if ev is not None:
                        log_row.update(
                            {
                                (k if k.startswith("eval_") else f"eval_{k}"): v
                                for k, v in ev.items()
                                if k != "episodes"
                            }
                        )

                    for key, val in log_row.items():
                        if (
                            isinstance(val, (int, float))
                            and not isinstance(val, bool)
                            and key != "global_step"
                        ):
                            writer.add_scalar(f"sac/{key}", val, global_step)
                    tracker.log_metrics("sac", log_row, step=global_step)
                    with metrics_path.open("a") as f:
                        f.write(json.dumps(log_row) + "\n")
                    last_log = log_row
                    recent_step_metrics.clear()
                    recent_actor_metrics.clear()
                    print(
                        f"[sac] step={global_step}/{cfg.total_timesteps} "
                        f"upd={total_critic_updates} critic={log_row['critic_loss']:.4f} "
                        f"actor={log_row['actor_loss']:.4f} aux={log_row['aux_loss']:.4f} "
                        f"alpha={log_row['alpha']:.4f} tps={log_row['transitions_per_sec']:.1f}",
                        flush=True,
                    )

                if should_ckpt:
                    while next_ckpt_step <= global_step:
                        next_ckpt_step += cfg.checkpoint_interval_steps
                    agent.actor.save_actor(out_dir / "actor_latest.pt")
                    _save_sac_checkpoint(
                        out_dir=out_dir,
                        agent=agent,
                        actor_opt=actor_opt,
                        critic_opt=critic_opt,
                        alpha_opt=alpha_opt,
                        global_step=global_step,
                        total_critic_updates=total_critic_updates,
                        cfg=cfg,
                        best_score=best_score,
                        best_eval=best_eval,
                        utd_carry=utd_carry,
                        last_log=last_log,
                        last_eval=last_eval,
                    )

            agent.actor.save_actor(out_dir / "actor_latest.pt")
            if not (out_dir / "actor_best.pt").exists():
                agent.actor.save_actor(out_dir / "actor_best.pt")
    except FloatingPointError as exc:
        summary = _write_sac_summary(
            out_dir,
            global_step=global_step,
            total_timesteps=cfg.total_timesteps,
            total_critic_updates=total_critic_updates,
            buffer_size=len(buffer),
            demo_transitions_loaded=demo_transitions_loaded,
            demo_pretrain_steps_run=demo_pretrain_steps_run,
            best_eval=best_eval,
            last_eval=last_eval,
            last_log=last_log,
            nan_inf_detected=True,
            nan_inf_error=str(exc),
            wandb_url=tracker.run_url,
        )
        tracker.update_summary(summary)
        tracker.finish(exit_code=1)
        raise
    except KeyboardInterrupt:
        agent.actor.save_actor(out_dir / "actor_latest.pt")
        if not (out_dir / "actor_best.pt").exists():
            agent.actor.save_actor(out_dir / "actor_best.pt")
        _save_sac_checkpoint(
            out_dir=out_dir,
            agent=agent,
            actor_opt=actor_opt,
            critic_opt=critic_opt,
            alpha_opt=alpha_opt,
            global_step=global_step,
            total_critic_updates=total_critic_updates,
            cfg=cfg,
            best_score=best_score,
            best_eval=best_eval,
            utd_carry=utd_carry,
            last_log=last_log,
            last_eval=last_eval,
        )
        summary = _write_sac_summary(
            out_dir,
            global_step=global_step,
            total_timesteps=cfg.total_timesteps,
            total_critic_updates=total_critic_updates,
            buffer_size=len(buffer),
            demo_transitions_loaded=demo_transitions_loaded,
            demo_pretrain_steps_run=demo_pretrain_steps_run,
            best_eval=best_eval,
            last_eval=last_eval,
            last_log=last_log,
            nan_inf_detected=False,
            interrupted=True,
            wandb_url=tracker.run_url,
        )
        tracker.update_summary(summary)
        tracker.finish(exit_code=130)
        raise
    except Exception:
        tracker.finish(exit_code=1)
        raise

    summary = _write_sac_summary(
        out_dir,
        global_step=global_step,
        total_timesteps=cfg.total_timesteps,
        total_critic_updates=total_critic_updates,
        buffer_size=len(buffer),
        demo_transitions_loaded=demo_transitions_loaded,
        demo_pretrain_steps_run=demo_pretrain_steps_run,
        best_eval=best_eval,
        last_eval=last_eval,
        last_log=last_log,
        nan_inf_detected=False,
        wandb_url=tracker.run_url,
    )
    tracker.update_summary(summary)
    tracker.finish(exit_code=0)
    return summary
