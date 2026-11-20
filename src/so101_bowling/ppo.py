"""Asymmetric dual-camera PPO training, GAE, DrQ shifts, and evaluation."""

from __future__ import annotations

import json
import random
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import cv2
import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter

from so101_bowling.config import WorkspaceConfig, load_config
from so101_bowling.env import SO101BowlingEnv
from so101_bowling.ppo_models import (
    ACTOR_OBS_KEYS,
    ACTION_DIM,
    AUX_MASK_DIM,
    AUX_TARGET_DIM,
    AsymmetricPPOAgent,
    CAMERA_META_DIM,
    DualCameraActor,
    PRIVILEGED_STATE_DIM,
    PROPRIO_DIM,
    masked_aux_loss,
)
from so101_bowling.vec_env import make_vec_env

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PPO_YAML = _REPO_ROOT / "configs" / "ppo.yaml"


@dataclass(frozen=True)
class PPOConfig:
    seed: int = 0
    dr_enabled: bool = False
    frame_stack: int = 2
    num_envs: int = 4
    rollout_steps: int = 128
    total_timesteps: int = 200000
    device: str = "auto"
    lr: float = 3.0e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_epochs: int = 4
    minibatch_size: int = 64
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    aux_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03
    drq_pad_px: int = 4
    encoder_type: str = "spatial_grid"
    eval_interval_updates: int = 10
    eval_episodes: int = 10
    checkpoint_interval_updates: int = 25
    save_video: bool = True


def load_ppo_config(path: str | Path | None = None) -> PPOConfig:
    """Load PPOConfig from YAML, defaulting to configs/ppo.yaml."""
    cfg_path = Path(path) if path is not None else _DEFAULT_PPO_YAML
    if not cfg_path.exists():
        raise FileNotFoundError(f"PPO config not found: {cfg_path}")
    raw = yaml.safe_load(cfg_path.read_text()) or {}
    return PPOConfig(**raw)


def resolve_device(device_str: str) -> torch.device:
    """Resolve 'auto' to CUDA if available else CPU; honor explicit devices."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    next_value: Tensor,
    final_values: Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    """Compute Generalized Advantage Estimation distinguishing terminated vs truncated."""
    num_steps, _ = rewards.shape
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros_like(next_value)
    done = terminated | truncated
    zeros = torch.zeros_like(next_value)

    for t in reversed(range(num_steps)):
        base_next_val = next_value if t == num_steps - 1 else values[t + 1]
        next_v = torch.where(
            terminated[t],
            zeros,
            torch.where(truncated[t], final_values[t], base_next_val),
        )
        cont = (~done[t]).to(dtype=rewards.dtype)
        delta = rewards[t] + gamma * next_v - values[t]
        lastgaelam = delta + gamma * gae_lambda * cont * lastgaelam
        advantages[t] = lastgaelam

    return advantages, advantages + values


class RolloutBuffer:
    """Stores CPU uint8 dual-camera rollouts plus exact DrQ shifts for PPO updates."""

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        frame_stack: int,
        img_hw: tuple[int, int] = (64, 112),
    ) -> None:
        self.num_steps = num_steps
        self.num_envs = num_envs
        h, w = img_hw

        self.wrist_rgb = torch.empty((num_steps, num_envs, frame_stack, h, w, 3), dtype=torch.uint8)
        self.tripod_rgb = torch.empty((num_steps, num_envs, frame_stack, h, w, 3), dtype=torch.uint8)
        self.camera_meta = torch.empty((num_steps, num_envs, CAMERA_META_DIM), dtype=torch.float32)
        self.proprio = torch.empty((num_steps, num_envs, PROPRIO_DIM), dtype=torch.float32)
        self.privileged_state = torch.empty(
            (num_steps, num_envs, PRIVILEGED_STATE_DIM), dtype=torch.float32
        )
        self.privileged_aux_targets = torch.empty(
            (num_steps, num_envs, AUX_TARGET_DIM), dtype=torch.float32
        )
        self.privileged_aux_mask = torch.empty(
            (num_steps, num_envs, AUX_MASK_DIM), dtype=torch.float32
        )

        self.actions = torch.empty((num_steps, num_envs, ACTION_DIM), dtype=torch.float32)
        self.logprobs = torch.empty((num_steps, num_envs), dtype=torch.float32)
        self.rewards = torch.empty((num_steps, num_envs), dtype=torch.float32)
        self.terminated = torch.empty((num_steps, num_envs), dtype=torch.bool)
        self.truncated = torch.empty((num_steps, num_envs), dtype=torch.bool)
        self.values = torch.empty((num_steps, num_envs), dtype=torch.float32)
        self.final_values = torch.zeros((num_steps, num_envs), dtype=torch.float32)

        self.wrist_shifts = torch.zeros((num_steps, num_envs, 2), dtype=torch.long)
        self.tripod_shifts = torch.zeros((num_steps, num_envs, 2), dtype=torch.long)

    def insert(
        self,
        step: int,
        obs: Mapping[str, np.ndarray],
        action: Tensor,
        logprob: Tensor,
        value: Tensor,
        reward: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        final_value: Tensor,
        wrist_shift: Tensor,
        tripod_shift: Tensor,
    ) -> None:
        self.wrist_rgb[step].copy_(torch.from_numpy(obs["wrist_rgb"]))
        self.tripod_rgb[step].copy_(torch.from_numpy(obs["tripod_rgb"]))
        self.camera_meta[step].copy_(torch.from_numpy(obs["camera_meta"]))
        self.proprio[step].copy_(torch.from_numpy(obs["proprio"]))
        self.privileged_state[step].copy_(torch.from_numpy(obs["privileged_state"]))
        self.privileged_aux_targets[step].copy_(torch.from_numpy(obs["privileged_aux_targets"]))
        self.privileged_aux_mask[step].copy_(torch.from_numpy(obs["privileged_aux_mask"]))

        self.actions[step].copy_(action.detach().cpu())
        self.logprobs[step].copy_(logprob.detach().cpu())
        self.values[step].copy_(value.detach().cpu())
        self.rewards[step].copy_(torch.from_numpy(reward))
        self.terminated[step].copy_(torch.from_numpy(terminated))
        self.truncated[step].copy_(torch.from_numpy(truncated))
        self.final_values[step].copy_(final_value.detach().cpu())
        self.wrist_shifts[step].copy_(wrist_shift.detach().cpu())
        self.tripod_shifts[step].copy_(tripod_shift.detach().cpu())

    def flat_view(self, advantages: Tensor, returns: Tensor) -> dict[str, Tensor]:
        n = self.num_steps * self.num_envs
        return {
            "wrist_rgb": self.wrist_rgb.reshape(n, *self.wrist_rgb.shape[2:]),
            "tripod_rgb": self.tripod_rgb.reshape(n, *self.tripod_rgb.shape[2:]),
            "camera_meta": self.camera_meta.reshape(n, CAMERA_META_DIM),
            "proprio": self.proprio.reshape(n, PROPRIO_DIM),
            "privileged_state": self.privileged_state.reshape(n, PRIVILEGED_STATE_DIM),
            "privileged_aux_targets": self.privileged_aux_targets.reshape(n, AUX_TARGET_DIM),
            "privileged_aux_mask": self.privileged_aux_mask.reshape(n, AUX_MASK_DIM),
            "actions": self.actions.reshape(n, ACTION_DIM),
            "logprobs": self.logprobs.reshape(n),
            "values": self.values.reshape(n),
            "advantages": advantages.reshape(n),
            "returns": returns.reshape(n),
            "wrist_shifts": self.wrist_shifts.reshape(n, 2),
            "tripod_shifts": self.tripod_shifts.reshape(n, 2),
        }


def _obs_to_device(obs: Mapping[str, np.ndarray], device: torch.device) -> dict[str, Tensor]:
    return {k: torch.as_tensor(v, device=device) for k, v in obs.items()}


def _batch_slice(
    flat: Mapping[str, Tensor], indices: np.ndarray | Tensor, device: torch.device
) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    obs = {
        k: flat[k][indices].to(device=device)
        for k in (
            *ACTOR_OBS_KEYS,
            "privileged_state",
            "privileged_aux_targets",
            "privileged_aux_mask",
        )
    }
    return (
        obs,
        flat["actions"][indices].to(device=device),
        flat["logprobs"][indices].to(device=device),
        flat["advantages"][indices].to(device=device),
        flat["returns"][indices].to(device=device),
        flat["values"][indices].to(device=device),
        flat["wrist_shifts"][indices].to(device=device),
        flat["tripod_shifts"][indices].to(device=device),
    )


def _apply_occlusion(obs: dict[str, np.ndarray], occlude_camera: str | None) -> dict[str, np.ndarray]:
    if occlude_camera is None:
        return obs
    out = dict(obs)
    if occlude_camera == "wrist":
        out["wrist_rgb"] = np.zeros_like(obs["wrist_rgb"])
    elif occlude_camera == "tripod":
        out["tripod_rgb"] = np.zeros_like(obs["tripod_rgb"])
    else:
        raise ValueError(f"Unsupported occlude_camera={occlude_camera!r}")
    return out


@torch.no_grad()
def evaluate_actor(
    actor: DualCameraActor,
    cfg: WorkspaceConfig,
    dr_enabled: bool = False,
    frame_stack: int = 2,
    gamma: float = 0.99,
    episodes: int = 10,
    seed: int = 1000,
    device: torch.device | str = "cpu",
    video_path: str | Path | None = None,
    occlude_camera: str | None = None,
) -> dict[str, Any]:
    """Evaluate standalone DualCameraActor deterministically without privileged inputs."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")

    was_training = actor.training
    actor.eval()
    target_device = torch.device(device)
    actor.to(target_device)

    strikes = 0
    illegals = 0
    credited_list: list[int] = []
    full_knock_list: list[int] = []
    wobble_only_list: list[int] = []
    returns: list[float] = []

    writer = None
    vid_h, vid_w = 2 * cfg.camera.render_height, 2 * cfg.camera.render_width
    try:
        with SO101BowlingEnv(
            cfg=cfg,
            dr_enabled=dr_enabled,
            render_rgb=True,
            frame_stack=frame_stack,
            gamma=gamma,
        ) as env:
            if video_path is not None:
                vp = Path(video_path)
                vp.parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(vp), fourcc, cfg.robot.control_hz, (vid_w * 2, vid_h))

            for ep in range(episodes):
                obs, _ = env.reset(seed=seed + ep)
                ep_ret = 0.0
                done = False
                info: dict[str, Any] = {}
                while not done:
                    if writer is not None and ep == 0:
                        w_img = env.render_camera("wrist_cam", vid_h, vid_w)
                        t_img = env.render_camera("tripod_cam", vid_h, vid_w)
                        panel = np.concatenate((w_img, t_img), axis=1)
                        writer.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

                    actor_obs = {k: obs[k] for k in ACTOR_OBS_KEYS}
                    actor_obs = _apply_occlusion(actor_obs, occlude_camera)
                    action = actor.act(actor_obs, deterministic=True).cpu().numpy()
                    obs, reward, terminated, truncated, info = env.step(action)
                    ep_ret += float(reward)
                    done = terminated or truncated

                if writer is not None and ep == 0:
                    w_img = env.render_camera("wrist_cam", vid_h, vid_w)
                    t_img = env.render_camera("tripod_cam", vid_h, vid_w)
                    panel = np.concatenate((w_img, t_img), axis=1)
                    writer.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
                    writer.release()
                    writer = None

                outcome = info.get("outcome", {})
                legal = bool(outcome.get("legal_delivery", False))
                strikes += int(bool(outcome.get("legal_strike", False)))
                illegals += int(not legal)
                credited_list.append(int(outcome.get("credited_pins", 0)))
                full_knock_list.append(int(outcome.get("full_knock_pins", 0)))
                wobble_only_list.append(int(outcome.get("wobble_only_pins", 0)))
                returns.append(ep_ret)
    finally:
        if writer is not None:
            writer.release()
        if was_training:
            actor.train()

    n = len(returns)
    return {
        "episodes": len(returns),
        "legal_strike_rate": strikes / n,
        "mean_credited_pins": float(np.mean(credited_list)),
        "mean_full_knock_pins": float(np.mean(full_knock_list)),
        "mean_wobble_only_pins": float(np.mean(wobble_only_list)),
        "illegal_rate": illegals / n,
        "mean_return": float(np.mean(returns)),
    }


def evaluate_camera_ablation(
    actor: DualCameraActor,
    cfg: WorkspaceConfig,
    dr_enabled: bool = False,
    frame_stack: int = 2,
    gamma: float = 0.99,
    episodes: int = 10,
    seed: int = 1000,
    device: torch.device | str = "cpu",
) -> dict[str, dict[str, Any]]:
    """Run paired evaluation with both cameras, wrist occluded, and tripod occluded."""
    return {
        label: evaluate_actor(
            actor=actor,
            cfg=cfg,
            dr_enabled=dr_enabled,
            frame_stack=frame_stack,
            gamma=gamma,
            episodes=episodes,
            seed=seed,
            device=device,
            occlude_camera=occlude,
        )
        for label, occlude in (("both", None), ("no_wrist", "wrist"), ("no_tripod", "tripod"))
    }


def _set_seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def _explained_variance(y_pred: Tensor, y_true: Tensor) -> float:
    if y_true.numel() < 2:
        return 0.0
    var_y = torch.var(y_true)
    if float(var_y) < 1e-8:
        return 0.0
    return float(1.0 - torch.var(y_true - y_pred) / var_y)


def _validate_ppo_config(cfg: PPOConfig) -> None:
    for name in (
        "num_envs",
        "rollout_steps",
        "total_timesteps",
        "minibatch_size",
        "eval_interval_updates",
        "checkpoint_interval_updates",
        "frame_stack",
    ):
        if getattr(cfg, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if cfg.drq_pad_px < 0:
        raise ValueError("drq_pad_px must be nonnegative")


def train_ppo(
    ppo_cfg: PPOConfig,
    workspace_cfg: WorkspaceConfig | None = None,
    run_dir: str | Path = "runs/ppo",
) -> dict[str, Any]:
    """Train DualCameraActor + AsymmetricPPOAgent and export standalone actor checkpoints."""
    _validate_ppo_config(ppo_cfg)
    wcfg = workspace_cfg if workspace_cfg is not None else load_config()
    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ppo_config.json").write_text(json.dumps(asdict(ppo_cfg), indent=2))

    rng = _set_seed(ppo_cfg.seed)
    device = resolve_device(ppo_cfg.device)
    img_hw = (wcfg.camera.render_height, wcfg.camera.render_width)

    agent = AsymmetricPPOAgent(
        frame_stack=ppo_cfg.frame_stack,
        action_dim=ACTION_DIM,
        encoder_type=ppo_cfg.encoder_type,
        img_hw=img_hw,
    ).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=ppo_cfg.lr, eps=1e-5)

    batch_size = ppo_cfg.num_envs * ppo_cfg.rollout_steps
    num_updates = max(1, ppo_cfg.total_timesteps // batch_size)
    minibatch_size = min(ppo_cfg.minibatch_size, batch_size)

    buffer = RolloutBuffer(
        num_steps=ppo_cfg.rollout_steps,
        num_envs=ppo_cfg.num_envs,
        frame_stack=ppo_cfg.frame_stack,
        img_hw=img_hw,
    )
    metrics_path = out_dir / "metrics.jsonl"

    best_score: tuple[float, float, float] = (-1.0, -1.0, -float("inf"))
    best_eval: dict[str, Any] = {}
    last_log: dict[str, Any] = {}
    global_step = 0

    with ExitStack() as stack:
        writer = SummaryWriter(log_dir=str(out_dir / "tb"))
        stack.callback(writer.close)
        vec_env = stack.enter_context(
            make_vec_env(
                num_envs=ppo_cfg.num_envs,
                cfg=wcfg,
                dr_enabled=ppo_cfg.dr_enabled,
                render_rgb=True,
                frame_stack=ppo_cfg.frame_stack,
                gamma=ppo_cfg.gamma,
                base_seed=ppo_cfg.seed,
            )
        )
        if abs(vec_env.scorer_gamma - ppo_cfg.gamma) > 1e-6:
            raise ValueError(
                f"Scorer gamma {vec_env.scorer_gamma} != PPO gamma {ppo_cfg.gamma}"
            )

        next_obs = vec_env.reset(seed=ppo_cfg.seed)
        ep_outcomes: list[dict[str, Any]] = []
        t_start = perf_counter()

        for update in range(1, num_updates + 1):
            if ppo_cfg.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                optimizer.param_groups[0]["lr"] = frac * ppo_cfg.lr

            agent.eval()
            for step in range(ppo_cfg.rollout_steps):
                global_step += ppo_cfg.num_envs
                pad = ppo_cfg.drq_pad_px
                if pad > 0:
                    w_shift = torch.from_numpy(
                        rng.integers(-pad, pad + 1, size=(ppo_cfg.num_envs, 2), dtype=np.int64)
                    )
                    t_shift = torch.from_numpy(
                        rng.integers(-pad, pad + 1, size=(ppo_cfg.num_envs, 2), dtype=np.int64)
                    )
                else:
                    w_shift = torch.zeros((ppo_cfg.num_envs, 2), dtype=torch.long)
                    t_shift = torch.zeros((ppo_cfg.num_envs, 2), dtype=torch.long)

                with torch.no_grad():
                    obs_t = _obs_to_device(next_obs, device)
                    action, logprob, _, value, _ = agent.get_action_and_value(
                        obs_t,
                        wrist_shifts=w_shift.to(device),
                        tripod_shifts=t_shift.to(device),
                    )

                clipped_action = np.clip(action.cpu().numpy(), -1.0, 1.0)
                new_obs, rewards, terminated, truncated, infos = vec_env.step(clipped_action)

                final_values = torch.zeros(ppo_cfg.num_envs, dtype=torch.float32)
                for idx, info in enumerate(infos):
                    if "outcome" in info:
                        ep_outcomes.append(info["outcome"])
                    if truncated[idx] and not terminated[idx] and "final_obs" in info:
                        with torch.no_grad():
                            f_obs_t = {
                                k: torch.as_tensor(v, device=device).unsqueeze(0)
                                for k, v in info["final_obs"].items()
                            }
                            final_values[idx] = agent.get_value(f_obs_t).squeeze(0).cpu()

                buffer.insert(
                    step=step,
                    obs=next_obs,
                    action=action,
                    logprob=logprob,
                    value=value,
                    reward=rewards,
                    terminated=terminated,
                    truncated=truncated,
                    final_value=final_values,
                    wrist_shift=w_shift,
                    tripod_shift=t_shift,
                )
                next_obs = new_obs

            with torch.no_grad():
                next_value = agent.get_value(_obs_to_device(next_obs, device)).cpu()
            advantages, returns = compute_gae(
                rewards=buffer.rewards,
                values=buffer.values,
                terminated=buffer.terminated,
                truncated=buffer.truncated,
                next_value=next_value,
                final_values=buffer.final_values,
                gamma=ppo_cfg.gamma,
                gae_lambda=ppo_cfg.gae_lambda,
            )

            flat = buffer.flat_view(advantages, returns)
            adv = flat["advantages"]
            if adv.numel() > 1:
                flat["advantages"] = (adv - adv.mean()) / (adv.std() + 1e-8)

            agent.train()
            clipfracs: list[float] = []
            pg_losses: list[float] = []
            v_losses: list[float] = []
            ent_losses: list[float] = []
            aux_losses: list[float] = []
            pre_kls: list[float] = []

            stop_early = False
            for _ in range(ppo_cfg.num_epochs):
                perm = rng.permutation(batch_size)
                for start in range(0, batch_size, minibatch_size):
                    mb_inds = perm[start : start + minibatch_size]
                    (
                        mb_obs,
                        mb_act,
                        mb_logp,
                        mb_adv,
                        mb_ret,
                        _,
                        mb_w_shift,
                        mb_t_shift,
                    ) = _batch_slice(flat, mb_inds, device)

                    _, newlogp, entropy, newvalue, aux_pred = agent.get_action_and_value(
                        mb_obs,
                        wrist_shifts=mb_w_shift,
                        tripod_shifts=mb_t_shift,
                        action=mb_act,
                    )
                    logratio = newlogp - mb_logp
                    ratio = logratio.exp()

                    with torch.no_grad():
                        approx_kl = float(((ratio - 1.0) - logratio).mean().cpu())
                        pre_kls.append(approx_kl)
                        clipfracs.append(float(((ratio - 1.0).abs() > ppo_cfg.clip_coef).float().mean().cpu()))

                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(
                        ratio, 1.0 - ppo_cfg.clip_coef, 1.0 + ppo_cfg.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    v_loss = 0.5 * ((newvalue - mb_ret) ** 2).mean()
                    ent_loss = entropy.mean()
                    aux_loss = masked_aux_loss(
                        aux_pred,
                        mb_obs["privileged_aux_targets"].to(dtype=torch.float32),
                        mb_obs["privileged_aux_mask"],
                    )

                    loss = (
                        pg_loss
                        + ppo_cfg.vf_coef * v_loss
                        - ppo_cfg.ent_coef * ent_loss
                        + ppo_cfg.aux_coef * aux_loss
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), ppo_cfg.max_grad_norm)
                    optimizer.step()

                    pg_losses.append(float(pg_loss.detach().cpu()))
                    v_losses.append(float(v_loss.detach().cpu()))
                    ent_losses.append(float(ent_loss.detach().cpu()))
                    aux_losses.append(float(aux_loss.detach().cpu()))

                    if ppo_cfg.target_kl is not None and approx_kl > 1.5 * ppo_cfg.target_kl:
                        stop_early = True
                        break
                if stop_early:
                    break

            # Measure post-update KL across rollout batch (captures policy drift from aux loss too).
            post_kl_sum = 0.0
            with torch.no_grad():
                for start in range(0, batch_size, minibatch_size):
                    mb_inds = slice(start, min(start + minibatch_size, batch_size))
                    mb_obs, mb_act, mb_logp, _, _, _, mb_w_shift, mb_t_shift = _batch_slice(
                        flat, mb_inds, device
                    )
                    _, post_logp, _, _, _ = agent.get_action_and_value(
                        mb_obs,
                        wrist_shifts=mb_w_shift,
                        tripod_shifts=mb_t_shift,
                        action=mb_act,
                    )
                    lr_diff = post_logp - mb_logp
                    post_kl_sum += float(((lr_diff.exp() - 1.0) - lr_diff).sum().cpu())
            post_update_kl = post_kl_sum / max(1, batch_size)

            elapsed = max(1e-6, perf_counter() - t_start)
            tps = global_step / elapsed
            recent = ep_outcomes[-50:]
            log_row: dict[str, Any] = {
                "update": update,
                "global_step": global_step,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "policy_loss": float(np.mean(pg_losses)),
                "value_loss": float(np.mean(v_losses)),
                "aux_loss": float(np.mean(aux_losses)),
                "entropy": float(np.mean(ent_losses)),
                "approx_kl": float(np.mean(pre_kls)),
                "post_update_kl": post_update_kl,
                "clipfrac": float(np.mean(clipfracs)),
                "explained_variance": _explained_variance(flat["values"], flat["returns"]),
                "transitions_per_sec": tps,
                "rollout_episodes": len(ep_outcomes),
                "train_legal_strike_rate": (
                    float(np.mean([int(bool(o.get("legal_strike", False))) for o in recent]))
                    if recent
                    else 0.0
                ),
                "train_mean_credited_pins": (
                    float(np.mean([int(o.get("credited_pins", 0)) for o in recent]))
                    if recent
                    else 0.0
                ),
                "train_illegal_rate": (
                    float(np.mean([int(not bool(o.get("legal_delivery", False))) for o in recent]))
                    if recent
                    else 0.0
                ),
            }

            if (
                update % ppo_cfg.checkpoint_interval_updates == 0
                or update == num_updates
            ):
                torch.save(
                    {
                        "agent": agent.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "update": update,
                        "global_step": global_step,
                        "ppo_cfg": asdict(ppo_cfg),
                    },
                    out_dir / "latest.pt",
                )

            if update % ppo_cfg.eval_interval_updates == 0 or update == num_updates:
                vid_path = (
                    out_dir / "videos" / f"eval_step_{global_step:07d}.mp4"
                    if ppo_cfg.save_video
                    else None
                )
                ev = evaluate_actor(
                    actor=agent.actor,
                    cfg=wcfg,
                    dr_enabled=ppo_cfg.dr_enabled,
                    frame_stack=ppo_cfg.frame_stack,
                    gamma=ppo_cfg.gamma,
                    episodes=ppo_cfg.eval_episodes,
                    seed=ppo_cfg.seed + 10000,
                    device=device,
                    video_path=vid_path,
                )
                log_row.update({f"eval_{k}": v for k, v in ev.items() if k != "episodes"})
                score = (ev["legal_strike_rate"], ev["mean_credited_pins"], ev["mean_return"])
                if score > best_score or not (out_dir / "best_actor.pt").exists():
                    best_score = score
                    best_eval = ev
                    agent.actor.save_actor(out_dir / "best_actor.pt")

            for key, val in log_row.items():
                if isinstance(val, (int, float)) and key not in ("update", "global_step"):
                    writer.add_scalar(f"ppo/{key}", val, global_step)
            with metrics_path.open("a") as f:
                f.write(json.dumps(log_row) + "\n")
            last_log = log_row

        agent.actor.save_actor(out_dir / "final_actor.pt")
        if not (out_dir / "best_actor.pt").exists():
            agent.actor.save_actor(out_dir / "best_actor.pt")

    return {
        "run_dir": str(out_dir),
        "global_step": global_step,
        "best_eval": best_eval,
        "last_metrics": last_log,
    }
