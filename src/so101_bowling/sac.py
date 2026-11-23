"""Asymmetric dual-camera SAC training, DrQ-v2 view shifts, UTD accounting, and helpers."""

from __future__ import annotations

import json
import math
import random
from contextlib import ExitStack
from dataclasses import asdict, dataclass
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
from so101_bowling.ppo import evaluate_actor, resolve_device
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
) -> tuple[int, float]:
    """Compute integer gradient updates for the current env step with fractional carry."""
    if global_step < cfg.learning_starts or buffer_size < cfg.batch_size:
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

    critic_opt.zero_grad(set_to_none=True)
    critic_loss.backward()
    nn.utils.clip_grad_norm_(agent.critic.parameters(), cfg.max_grad_norm)
    critic_opt.step()
    critic_opt.zero_grad(set_to_none=True)
    agent.soft_update_target(cfg.tau)

    metrics: dict[str, float] = {
        "critic_loss": float(critic_loss.detach().cpu()),
        "actor_loss": 0.0,
        "aux_loss": 0.0,
        "alpha_loss": 0.0,
        "alpha": float(agent.alpha.detach().cpu()),
        "q_mean": float((0.5 * (q1.detach() + q2.detach())).mean().cpu()),
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

        total_actor_loss.backward()
    finally:
        for p, req in zip(critic_params, prev_requires_grad, strict=True):
            p.requires_grad_(req)

    nn.utils.clip_grad_norm_(actor_params, cfg.max_grad_norm)
    actor_opt.step()

    if cfg.autotune_alpha and alpha_opt is not None:
        alpha_opt.zero_grad(set_to_none=True)
        alpha_loss = -(agent.log_alpha * (log_prob.detach() + agent.target_entropy)).mean()
        alpha_loss.backward()
        alpha_opt.step()
        with torch.no_grad():
            agent.log_alpha.clamp_(-10.0, 2.0)
    else:
        alpha_loss = torch.zeros((), device=device, dtype=policy_loss.dtype)

    metrics.update(
        {
            "actor_loss": float(policy_loss.detach().cpu()),
            "aux_loss": float(aux_loss.detach().cpu()),
            "alpha_loss": float(alpha_loss.detach().cpu()),
            "alpha": float(agent.alpha.detach().cpu()),
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
) -> None:
    torch.save(
        {
            "agent": agent.state_dict(),
            "actor_opt": actor_opt.state_dict(),
            "critic_opt": critic_opt.state_dict(),
            "alpha_opt": alpha_opt.state_dict() if alpha_opt is not None else None,
            "global_step": global_step,
            "total_critic_updates": total_critic_updates,
            "sac_cfg": asdict(cfg),
        },
        out_dir / "checkpoint_latest.pt",
    )


def train_sac(
    cfg: SACConfig,
    workspace_cfg: WorkspaceConfig | None = None,
    run_dir: str | Path = "runs/sac",
    resume_from: str | Path | None = None,
    demo_dataset: str | Path | None = None,
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

    metrics_path = out_dir / "metrics.jsonl"
    best_score: tuple[float, float, float] = (-1.0, -1.0, -float("inf"))
    best_eval: dict[str, Any] = {}
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
    elif metrics_path.exists():
        metrics_path.unlink()

    next_eval_step = ((global_step // cfg.eval_interval_steps) + 1) * cfg.eval_interval_steps
    next_ckpt_step = (
        (global_step // cfg.checkpoint_interval_steps) + 1
    ) * cfg.checkpoint_interval_steps
    log_interval_steps = max(cfg.num_envs, min(1000, cfg.eval_interval_steps))
    next_log_step = ((global_step // log_interval_steps) + 1) * log_interval_steps

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

        while global_step < cfg.total_timesteps:
            if global_step < cfg.learning_starts:
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

            next_obs, rewards, terminated, truncated, infos = vec_env.step(actions)
            buffer.add(obs, actions, rewards, next_obs, terminated, truncated, infos)
            for info in infos:
                if "outcome" in info:
                    ep_outcomes.append(info["outcome"])

            obs = next_obs
            global_step += cfg.num_envs

            n_updates, utd_carry = compute_utd_updates(global_step, len(buffer), cfg, utd_carry)
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
                )

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
                )
                score = (ev["legal_strike_rate"], ev["mean_credited_pins"], ev["mean_return"])
                if score > best_score or not (out_dir / "actor_best.pt").exists():
                    best_score = score
                    best_eval = ev
                    agent.actor.save_actor(out_dir / "actor_best.pt")

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
                    log_row.update({f"eval_{k}": v for k, v in ev.items() if k != "episodes"})

                for key, val in log_row.items():
                    if isinstance(val, (int, float)) and key != "global_step":
                        writer.add_scalar(f"sac/{key}", val, global_step)
                with metrics_path.open("a") as f:
                    f.write(json.dumps(log_row) + "\n")
                last_log = log_row
                recent_step_metrics.clear()
                recent_actor_metrics.clear()

        agent.actor.save_actor(out_dir / "actor_latest.pt")
        if not (out_dir / "actor_best.pt").exists():
            agent.actor.save_actor(out_dir / "actor_best.pt")

    summary = {
        "run_dir": str(out_dir),
        "global_step": global_step,
        "total_critic_updates": total_critic_updates,
        "buffer_size": len(buffer),
        "demo_transitions_loaded": demo_transitions_loaded,
        "best_eval": best_eval,
        "last_metrics": last_log,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
