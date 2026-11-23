"""Asymmetric dual-camera PPO training, GAE, DrQ shifts, and evaluation."""

from __future__ import annotations

import json
import math
import random
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping

import cv2
import numpy as np
import torch
import torch.nn.functional as F
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

VALIDATION_LOSS_NOTE = (
    "RL policy and critic losses optimize non-stationary expected returns rather than a fixed "
    "supervised validation loss; eval_aux_loss measures held-out supervised SmoothL1 loss on "
    "visible privileged_aux_targets during deterministic evaluation rollouts."
)


def _assert_finite_tensor(name: str, t: Tensor) -> None:
    if not torch.isfinite(t).all():
        nan_count = int(torch.isnan(t).sum().item())
        inf_count = int(torch.isinf(t).sum().item())
        fin = t[torch.isfinite(t)]
        fmin = float(fin.min().item()) if fin.numel() > 0 else float("nan")
        fmax = float(fin.max().item()) if fin.numel() > 0 else float("nan")
        raise FloatingPointError(
            f"Non-finite tensor {name}: shape={tuple(t.shape)} nan={nan_count} "
            f"inf={inf_count} finite_min={fmin} finite_max={fmax}"
        )


def _assert_finite_mapping(prefix: str, items: Mapping[str, float | Tensor]) -> None:
    for key, val in items.items():
        full_name = f"{prefix}.{key}"
        if isinstance(val, Tensor):
            _assert_finite_tensor(full_name, val)
        elif isinstance(val, (int, float)) and not isinstance(val, bool):
            if not math.isfinite(float(val)):
                raise FloatingPointError(f"Non-finite scalar {full_name}: {val}")


def _module_grad_and_param_stats(params: Iterable[nn.Parameter]) -> tuple[float, float]:
    grad_sq_sum = 0.0
    max_abs_param = 0.0
    for idx, p in enumerate(params):
        _assert_finite_tensor(f"param[{idx}]", p.data)
        if p.data.numel() > 0:
            max_abs_param = max(max_abs_param, float(p.data.detach().abs().max().item()))
        if p.grad is not None:
            _assert_finite_tensor(f"grad[{idx}]", p.grad)
            grad_sq_sum += float(p.grad.detach().pow(2).sum().item())
    return math.sqrt(grad_sq_sum), max_abs_param


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

    def __post_init__(self) -> None:
        for name in (
            "seed",
            "frame_stack",
            "num_envs",
            "rollout_steps",
            "total_timesteps",
            "num_epochs",
            "minibatch_size",
            "drq_pad_px",
            "eval_interval_updates",
            "eval_episodes",
            "checkpoint_interval_updates",
        ):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, int):
                raise ValueError(f"{name} must be an int, got {type(val).__name__}")
        for name in ("dr_enabled", "anneal_lr", "save_video"):
            val = getattr(self, name)
            if not isinstance(val, bool):
                raise ValueError(f"{name} must be a bool, got {type(val).__name__}")
        for name in (
            "num_envs",
            "rollout_steps",
            "total_timesteps",
            "num_epochs",
            "minibatch_size",
            "eval_interval_updates",
            "eval_episodes",
            "checkpoint_interval_updates",
            "frame_stack",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.drq_pad_px < 0:
            raise ValueError("drq_pad_px must be nonnegative")
        if self.encoder_type not in ("spatial_grid", "spatial_softmax"):
            raise ValueError(f"Unsupported encoder_type: {self.encoder_type!r}")
        for name in (
            "lr",
            "gamma",
            "gae_lambda",
            "clip_coef",
            "vf_coef",
            "ent_coef",
            "aux_coef",
            "max_grad_norm",
        ):
            raw_val = getattr(self, name)
            if isinstance(raw_val, bool) or not isinstance(raw_val, (int, float)):
                raise ValueError(f"{name} must be a real number, got {type(raw_val).__name__}")
            if not math.isfinite(float(raw_val)):
                raise ValueError(f"{name} must be finite, got {raw_val}")
        if self.lr <= 0.0:
            raise ValueError("lr must be positive")
        if not (0.0 < self.gamma <= 1.0):
            raise ValueError("gamma must be in (0, 1]")
        if not (0.0 < self.gae_lambda <= 1.0):
            raise ValueError("gae_lambda must be in (0, 1]")
        if self.clip_coef <= 0.0:
            raise ValueError("clip_coef must be positive")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        for name in ("vf_coef", "ent_coef", "aux_coef"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if self.target_kl is not None:
            if isinstance(self.target_kl, bool) or not isinstance(self.target_kl, (int, float)):
                raise ValueError("target_kl must be a positive finite number or None")
            if not math.isfinite(float(self.target_kl)) or self.target_kl <= 0.0:
                raise ValueError("target_kl must be positive and finite")


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
    aux_head: nn.Module | None = None,
) -> dict[str, Any]:
    """Evaluate standalone DualCameraActor deterministically without privileged inputs."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")

    was_training = actor.training
    orig_device = actor.log_std.device
    actor.eval()
    target_device = torch.device(device)
    actor.to(target_device)

    aux_was_training: bool | None = None
    aux_orig_device: torch.device | None = None
    if aux_head is not None:
        aux_was_training = aux_head.training
        aux_params = list(aux_head.parameters())
        if aux_params:
            aux_orig_device = aux_params[0].device
        aux_head.eval()
        aux_head.to(target_device)

    strikes = 0
    illegals = 0
    credited_list: list[int] = []
    full_knock_list: list[int] = []
    wobble_only_list: list[int] = []
    returns: list[float] = []
    steps_list: list[int] = []
    aux_err_sum = 0.0
    aux_weight_sum = 0.0

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
                ep_steps = 0
                done = False
                info: dict[str, Any] = {}
                while not done:
                    ep_steps += 1
                    if writer is not None and ep == 0:
                        w_img = env.render_camera("wrist_cam", vid_h, vid_w)
                        t_img = env.render_camera("tripod_cam", vid_h, vid_w)
                        panel = np.concatenate((w_img, t_img), axis=1)
                        writer.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

                    actor_obs = {k: obs[k] for k in ACTOR_OBS_KEYS}
                    actor_obs = _apply_occlusion(actor_obs, occlude_camera)
                    if aux_head is not None:
                        batched_actor_obs = {
                            k: torch.as_tensor(actor_obs[k], device=target_device).unsqueeze(0)
                            for k in ACTOR_OBS_KEYS
                        }
                        dist, z_vis = actor.forward_actor(batched_actor_obs)
                        action_t = (
                            torch.tanh(dist.mean)
                            if actor.squashed
                            else torch.clamp(dist.mean, -1.0, 1.0)
                        )
                        action = action_t.squeeze(0).cpu().numpy()

                        aux_pred = aux_head(z_vis)
                        aux_targets = torch.as_tensor(
                            obs["privileged_aux_targets"],
                            device=target_device,
                            dtype=aux_pred.dtype,
                        ).unsqueeze(0)
                        aux_mask = torch.as_tensor(
                            obs["privileged_aux_mask"],
                            device=target_device,
                            dtype=aux_pred.dtype,
                        ).unsqueeze(0)
                        weights = aux_mask.repeat_interleave(2, dim=-1)
                        per_elem = (
                            F.smooth_l1_loss(aux_pred, aux_targets, reduction="none") * weights
                        )
                        aux_err_sum += float(per_elem.sum().item())
                        aux_weight_sum += float(weights.sum().item())
                    else:
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
                steps_list.append(ep_steps)
    finally:
        if writer is not None:
            writer.release()
        actor.to(orig_device)
        if was_training:
            actor.train()
        if aux_head is not None:
            if aux_orig_device is not None:
                aux_head.to(aux_orig_device)
            if aux_was_training:
                aux_head.train()

    n = len(returns)
    result: dict[str, Any] = {
        "episodes": len(returns),
        "legal_strike_rate": strikes / n,
        "mean_credited_pins": float(np.mean(credited_list)),
        "mean_full_knock_pins": float(np.mean(full_knock_list)),
        "mean_wobble_only_pins": float(np.mean(wobble_only_list)),
        "illegal_rate": illegals / n,
        "mean_return": float(np.mean(returns)),
        "mean_episode_steps": float(np.mean(steps_list)),
    }
    if aux_head is not None:
        result["eval_aux_loss"] = (
            float(aux_err_sum / aux_weight_sum) if aux_weight_sum > 0.0 else None
        )
    return result


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


def _extract_ppo_final_losses(last_log: Mapping[str, Any]) -> dict[str, float]:
    keys = ("policy_loss", "value_loss", "aux_loss", "entropy", "approx_kl", "post_update_kl")
    return {
        k: float(last_log[k])
        for k in keys
        if k in last_log and isinstance(last_log[k], (int, float)) and not isinstance(last_log[k], bool)
    }


def _write_ppo_summary(
    out_dir: Path,
    *,
    global_step: int,
    total_timesteps: int,
    best_eval: dict[str, Any],
    last_eval: dict[str, Any],
    last_log: dict[str, Any],
    nan_inf_detected: bool,
    nan_inf_error: str | None = None,
    interrupted: bool = False,
) -> dict[str, Any]:
    eval_aux_loss = last_eval.get("eval_aux_loss")
    if eval_aux_loss is None and best_eval:
        eval_aux_loss = best_eval.get("eval_aux_loss")
    summary: dict[str, Any] = {
        "run_dir": str(out_dir),
        "global_step": int(global_step),
        "total_timesteps": int(total_timesteps),
        "best_eval": best_eval,
        "final_eval": last_eval,
        "last_metrics": last_log,
        "final_losses": _extract_ppo_final_losses(last_log),
        "eval_aux_loss": eval_aux_loss,
        "nan_inf_detected": bool(nan_inf_detected),
        "validation_loss_note": VALIDATION_LOSS_NOTE,
    }
    if nan_inf_error is not None:
        summary["nan_inf_error"] = nan_inf_error
    if interrupted:
        summary["interrupted"] = True
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def preflight_ppo(
    ppo_cfg: PPOConfig,
    workspace_cfg: WorkspaceConfig | None = None,
    *,
    steps: int = 150,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Run deterministic single-batch overfit and numerical stability verification for Asymmetric PPO."""
    _validate_ppo_config(ppo_cfg)
    if steps <= 0:
        raise ValueError("steps must be positive")
    wcfg = workspace_cfg if workspace_cfg is not None else load_config()
    rng = _set_seed(ppo_cfg.seed)
    device = resolve_device(ppo_cfg.device)
    img_hw = (wcfg.camera.render_height, wcfg.camera.render_width)

    target_batch = min(batch_size or ppo_cfg.minibatch_size, 32)
    if target_batch < 1:
        raise ValueError("batch_size must be positive")
    max_collect = max(target_batch, 32)

    agent = AsymmetricPPOAgent(
        frame_stack=ppo_cfg.frame_stack,
        action_dim=ACTION_DIM,
        encoder_type=ppo_cfg.encoder_type,
        img_hw=img_hw,
    ).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=ppo_cfg.lr, eps=1e-5)

    obs_seq: list[dict[str, np.ndarray]] = []
    act_seq: list[Tensor] = []
    logp_seq: list[Tensor] = []
    val_seq: list[Tensor] = []
    rew_seq: list[float] = []
    term_seq: list[bool] = []
    trunc_seq: list[bool] = []
    final_val_seq: list[float] = []

    agent.eval()
    with SO101BowlingEnv(
        cfg=wcfg,
        dr_enabled=ppo_cfg.dr_enabled,
        render_rgb=True,
        frame_stack=ppo_cfg.frame_stack,
        gamma=ppo_cfg.gamma,
    ) as env:
        obs, _ = env.reset(seed=ppo_cfg.seed)
        active_mask_count = 0.0
        for step_idx in range(max_collect):
            with torch.no_grad():
                obs_t = {
                    k: torch.as_tensor(v, device=device).unsqueeze(0)
                    for k, v in obs.items()
                }
                action, logprob, _, value, _ = agent.get_action_and_value(obs_t)
            clipped_action = np.clip(action.squeeze(0).cpu().numpy(), -1.0, 1.0)
            next_obs, reward, terminated, truncated, _ = env.step(clipped_action)
            final_val = 0.0
            if truncated and not terminated:
                with torch.no_grad():
                    n_obs_t = {
                        k: torch.as_tensor(v, device=device).unsqueeze(0)
                        for k, v in next_obs.items()
                    }
                    final_val = float(agent.get_value(n_obs_t).squeeze(0).cpu().item())

            obs_seq.append(obs)
            act_seq.append(action.squeeze(0).detach().cpu())
            logp_seq.append(logprob.squeeze(0).detach().cpu())
            val_seq.append(value.squeeze(0).detach().cpu())
            rew_seq.append(float(reward))
            term_seq.append(bool(terminated))
            trunc_seq.append(bool(truncated))
            final_val_seq.append(final_val)
            active_mask_count += float(np.sum(obs["privileged_aux_mask"]))

            if terminated or truncated:
                obs, _ = env.reset(seed=ppo_cfg.seed + step_idx + 1)
            else:
                obs = next_obs

            if len(obs_seq) >= target_batch and active_mask_count > 0.0:
                break

        with torch.no_grad():
            tail_obs_t = {
                k: torch.as_tensor(v, device=device).unsqueeze(0)
                for k, v in obs.items()
            }
            next_value = agent.get_value(tail_obs_t).cpu()

    rewards_t = torch.tensor(rew_seq, dtype=torch.float32).unsqueeze(1)
    values_t = torch.stack(val_seq, dim=0).unsqueeze(1)
    term_t = torch.tensor(term_seq, dtype=torch.bool).unsqueeze(1)
    trunc_t = torch.tensor(trunc_seq, dtype=torch.bool).unsqueeze(1)
    final_vals_t = torch.tensor(final_val_seq, dtype=torch.float32).unsqueeze(1)
    adv_t, ret_t = compute_gae(
        rewards=rewards_t,
        values=values_t,
        terminated=term_t,
        truncated=trunc_t,
        next_value=next_value,
        final_values=final_vals_t,
        gamma=ppo_cfg.gamma,
        gae_lambda=ppo_cfg.gae_lambda,
    )
    adv_flat = adv_t.squeeze(1)
    ret_flat = ret_t.squeeze(1)
    if adv_flat.numel() > 1:
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

    # Prefer transitions with active aux mask if available, up to target_batch.
    n_collected = len(obs_seq)
    mask_sums = [float(np.sum(o["privileged_aux_mask"])) for o in obs_seq]
    active_indices = [i for i, m in enumerate(mask_sums) if m > 0.0]
    inactive_indices = [i for i, m in enumerate(mask_sums) if m <= 0.0]
    chosen_indices = (active_indices + inactive_indices)[: min(target_batch, n_collected)]
    _ = rng  # keep deterministic seed setup consistent

    frozen_obs: dict[str, Tensor] = {
        k: torch.stack([torch.as_tensor(obs_seq[i][k]) for i in chosen_indices], dim=0).to(device)
        for k in (
            *ACTOR_OBS_KEYS,
            "privileged_state",
            "privileged_aux_targets",
            "privileged_aux_mask",
        )
    }
    frozen_act = torch.stack([act_seq[i] for i in chosen_indices], dim=0).to(device)
    frozen_logp = torch.stack([logp_seq[i] for i in chosen_indices], dim=0).to(device)
    frozen_adv = adv_flat[chosen_indices].to(device)
    frozen_ret = ret_flat[chosen_indices].to(device)

    aux_mask_active = float(frozen_obs["privileged_aux_mask"].sum().cpu().item())
    effective_steps = max(steps, 300 if ppo_cfg.lr < 1e-4 else steps)

    nan_inf_detected = False
    failure_reasons: list[str] = []
    initial_losses: dict[str, float] = {}
    final_losses: dict[str, float] = {}
    grad_norms: list[float] = []
    max_abs_param = 0.0
    actor_std_min = float("inf")
    actor_std_max = 0.0
    max_ratio = 0.0

    if aux_mask_active <= 0.0:
        failure_reasons.append("No active privileged_aux_mask elements in preflight batch")

    agent.train()
    try:
        for step_idx in range(effective_steps):
            dist, z_vis = agent.actor.forward_actor(frozen_obs)
            newlogp = dist.log_prob(frozen_act).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            newvalue = agent.get_value(frozen_obs)
            aux_pred = agent.aux_head(z_vis)
            ratio = (newlogp - frozen_logp).exp()

            pg_loss1 = -frozen_adv * ratio
            pg_loss2 = -frozen_adv * torch.clamp(
                ratio, 1.0 - ppo_cfg.clip_coef, 1.0 + ppo_cfg.clip_coef
            )
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            v_loss = 0.5 * ((newvalue - frozen_ret) ** 2).mean()
            ent_loss = entropy.mean()
            aux_loss = masked_aux_loss(
                aux_pred,
                frozen_obs["privileged_aux_targets"].to(dtype=torch.float32),
                frozen_obs["privileged_aux_mask"],
            )
            loss = (
                pg_loss
                + ppo_cfg.vf_coef * v_loss
                - ppo_cfg.ent_coef * ent_loss
                + ppo_cfg.aux_coef * aux_loss
            )

            _assert_finite_mapping(
                f"preflight_ppo.step_{step_idx}",
                {
                    "newlogp": newlogp,
                    "entropy": entropy,
                    "newvalue": newvalue,
                    "aux_pred": aux_pred,
                    "ratio": ratio,
                    "dist_mean": dist.mean,
                    "dist_stddev": dist.stddev,
                    "pg_loss": pg_loss,
                    "value_loss": v_loss,
                    "ent_loss": ent_loss,
                    "aux_loss": aux_loss,
                    "total_loss": loss,
                },
            )

            loss_snapshot = {
                "policy_loss": float(pg_loss.detach().cpu().item()),
                "value_loss": float(v_loss.detach().cpu().item()),
                "aux_loss": float(aux_loss.detach().cpu().item()),
                "entropy": float(ent_loss.detach().cpu().item()),
                "total_loss": float(loss.detach().cpu().item()),
            }
            if step_idx == 0:
                initial_losses = loss_snapshot
            final_losses = loss_snapshot

            actor_std_min = min(actor_std_min, float(dist.stddev.detach().min().cpu().item()))
            actor_std_max = max(actor_std_max, float(dist.stddev.detach().max().cpu().item()))
            max_ratio = max(max_ratio, float(ratio.detach().abs().max().cpu().item()))

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), ppo_cfg.max_grad_norm)
            _assert_finite_tensor(f"preflight_ppo.step_{step_idx}.grad_norm", grad_norm)
            raw_grad_norm, _ = _module_grad_and_param_stats(agent.parameters())
            grad_norms.append(float(raw_grad_norm))
            optimizer.step()
            _, step_max_param = _module_grad_and_param_stats(agent.parameters())
            max_abs_param = max(max_abs_param, step_max_param)
    except FloatingPointError as exc:
        nan_inf_detected = True
        failure_reasons.append(str(exc))

    max_grad_norm = max(grad_norms) if grad_norms else 0.0
    min_grad_norm = min(grad_norms) if grad_norms else 0.0

    aux_overfit_ok = (
        bool(final_losses)
        and bool(initial_losses)
        and final_losses["aux_loss"] < 0.08
        and final_losses["aux_loss"] < initial_losses["aux_loss"]
    )
    value_overfit_ok = (
        bool(final_losses)
        and bool(initial_losses)
        and final_losses["value_loss"] < initial_losses["value_loss"]
        and (
            final_losses["value_loss"] < 0.08
            or final_losses["value_loss"] <= 0.1 * initial_losses["value_loss"]
        )
    )
    total_loss_ok = (
        bool(final_losses)
        and bool(initial_losses)
        and final_losses["total_loss"] < initial_losses["total_loss"]
    )
    checks = {
        "finite_ok": not nan_inf_detected,
        "aux_mask_active_ok": aux_mask_active > 0.0,
        "non_zero_grads_ok": max_grad_norm > 0.0,
        "aux_overfit_ok": aux_overfit_ok,
        "value_overfit_ok": value_overfit_ok,
        "total_loss_ok": total_loss_ok,
    }
    for cname, ok in checks.items():
        if not ok and not any(cname in r for r in failure_reasons):
            if cname != "finite_ok" and cname != "aux_mask_active_ok":
                failure_reasons.append(f"Preflight check failed: {cname}")

    return {
        "algo": "ppo",
        "passed": all(checks.values()),
        "device": str(device),
        "batch_size": len(chosen_indices),
        "steps": effective_steps,
        "nan_inf_detected": nan_inf_detected,
        "aux_mask_active_elements": aux_mask_active,
        "initial_losses": initial_losses,
        "final_losses": final_losses,
        "numerical_stats": {
            "min_grad_norm": min_grad_norm,
            "max_grad_norm": max_grad_norm,
            "max_abs_param": max_abs_param,
            "actor_std_min": actor_std_min if math.isfinite(actor_std_min) else None,
            "actor_std_max": actor_std_max,
            "max_ratio": max_ratio,
        },
        "checks": checks,
        "failure_reasons": failure_reasons,
    }


def train_ppo(
    ppo_cfg: PPOConfig,
    workspace_cfg: WorkspaceConfig | None = None,
    run_dir: str | Path = "runs/ppo",
    resume_from: str | Path | None = None,
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
    last_eval: dict[str, Any] = {}
    last_log: dict[str, Any] = {}
    global_step = 0
    start_update = 1
    last_completed_update = 0

    if resume_from is not None:
        ckpt_path = Path(resume_from)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"PPO resume checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        agent.load_state_dict(ckpt["agent"])
        optimizer.load_state_dict(ckpt["optimizer"])
        last_completed_update = int(ckpt.get("update", 0))
        start_update = last_completed_update + 1
        global_step = int(ckpt.get("global_step", 0))
        if "best_score" in ckpt and ckpt["best_score"] is not None:
            best_score = tuple(float(x) for x in ckpt["best_score"])
            best_eval = dict(ckpt.get("best_eval", {}))
        elif (ckpt_path.parent / "summary.json").exists():
            prev_summary = json.loads((ckpt_path.parent / "summary.json").read_text())
            prev_be = prev_summary.get("best_eval") or {}
            if prev_be:
                best_eval = dict(prev_be)
                best_score = (
                    float(prev_be.get("legal_strike_rate", -1.0)),
                    float(prev_be.get("mean_credited_pins", -1.0)),
                    float(prev_be.get("mean_return", -float("inf"))),
                )
        if "last_log" in ckpt and ckpt["last_log"]:
            last_log = dict(ckpt["last_log"])
    elif metrics_path.exists():
        metrics_path.unlink()

    def _save_latest_ckpt(update_idx: int) -> None:
        torch.save(
            {
                "agent": agent.state_dict(),
                "optimizer": optimizer.state_dict(),
                "update": update_idx,
                "global_step": global_step,
                "best_score": list(best_score),
                "best_eval": best_eval,
                "last_log": last_log,
                "ppo_cfg": asdict(ppo_cfg),
            },
            out_dir / "latest.pt",
        )

    try:
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

            for update in range(start_update, num_updates + 1):
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

                    _assert_finite_mapping(
                        f"ppo.rollout.step_{global_step}",
                        {"action": action, "logprob": logprob, "value": value},
                    )
                    clipped_action = np.clip(action.cpu().numpy(), -1.0, 1.0)
                    new_obs, rewards, terminated, truncated, infos = vec_env.step(clipped_action)
                    _assert_finite_tensor(
                        f"ppo.rollout.step_{global_step}.rewards", torch.as_tensor(rewards)
                    )

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
                    _assert_finite_tensor(
                        f"ppo.rollout.step_{global_step}.final_values", final_values
                    )

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
                _assert_finite_tensor("ppo.rollout.next_value", next_value)
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
                            clipfracs.append(
                                float(((ratio - 1.0).abs() > ppo_cfg.clip_coef).float().mean().cpu())
                            )

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
                        _assert_finite_mapping(
                            f"ppo.update_{update}",
                            {
                                "newlogp": newlogp,
                                "entropy": entropy,
                                "newvalue": newvalue,
                                "aux_pred": aux_pred,
                                "ratio": ratio,
                                "pg_loss": pg_loss,
                                "value_loss": v_loss,
                                "ent_loss": ent_loss,
                                "aux_loss": aux_loss,
                                "loss": loss,
                            },
                        )

                        optimizer.zero_grad()
                        loss.backward()
                        grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), ppo_cfg.max_grad_norm)
                        _assert_finite_tensor(f"ppo.update_{update}.grad_norm", grad_norm)
                        _module_grad_and_param_stats(agent.parameters())
                        optimizer.step()
                        _module_grad_and_param_stats(agent.parameters())

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
                        aux_head=agent.aux_head,
                    )
                    last_eval = ev
                    log_row.update(
                        {
                            (k if k.startswith("eval_") else f"eval_{k}"): v
                            for k, v in ev.items()
                            if k != "episodes"
                        }
                    )
                    score = (ev["legal_strike_rate"], ev["mean_credited_pins"], ev["mean_return"])
                    if score > best_score or not (out_dir / "best_actor.pt").exists():
                        best_score = score
                        best_eval = ev
                        agent.actor.save_actor(out_dir / "best_actor.pt")
                    print(
                        f"[ppo] eval step={global_step}/{ppo_cfg.total_timesteps} "
                        f"strike={ev['legal_strike_rate']:.3f} pins={ev['mean_credited_pins']:.3f} "
                        f"steps={ev['mean_episode_steps']:.1f} return={ev['mean_return']:.3f} "
                        f"eval_aux={ev.get('eval_aux_loss')}",
                        flush=True,
                    )

                for key, val in log_row.items():
                    if (
                        isinstance(val, (int, float))
                        and not isinstance(val, bool)
                        and key not in ("update", "global_step")
                    ):
                        writer.add_scalar(f"ppo/{key}", val, global_step)
                with metrics_path.open("a") as f:
                    f.write(json.dumps(log_row) + "\n")
                last_log = log_row
                last_completed_update = update

                if (
                    update % ppo_cfg.checkpoint_interval_updates == 0
                    or update == num_updates
                ):
                    _save_latest_ckpt(update)

                print(
                    f"[ppo] upd={update}/{num_updates} step={global_step} "
                    f"pg={log_row['policy_loss']:.4f} vf={log_row['value_loss']:.4f} "
                    f"aux={log_row['aux_loss']:.4f} kl={log_row['post_update_kl']:.4f} "
                    f"tps={tps:.1f}",
                    flush=True,
                )

            agent.actor.save_actor(out_dir / "final_actor.pt")
            if not (out_dir / "best_actor.pt").exists():
                agent.actor.save_actor(out_dir / "best_actor.pt")
    except FloatingPointError as exc:
        _write_ppo_summary(
            out_dir,
            global_step=global_step,
            total_timesteps=ppo_cfg.total_timesteps,
            best_eval=best_eval,
            last_eval=last_eval,
            last_log=last_log,
            nan_inf_detected=True,
            nan_inf_error=str(exc),
        )
        raise
    except KeyboardInterrupt:
        _save_latest_ckpt(last_completed_update)
        agent.actor.save_actor(out_dir / "final_actor.pt")
        if not (out_dir / "best_actor.pt").exists():
            agent.actor.save_actor(out_dir / "best_actor.pt")
        _write_ppo_summary(
            out_dir,
            global_step=global_step,
            total_timesteps=ppo_cfg.total_timesteps,
            best_eval=best_eval,
            last_eval=last_eval,
            last_log=last_log,
            nan_inf_detected=False,
            interrupted=True,
        )
        raise

    return _write_ppo_summary(
        out_dir,
        global_step=global_step,
        total_timesteps=ppo_cfg.total_timesteps,
        best_eval=best_eval,
        last_eval=last_eval,
        last_log=last_log,
        nan_inf_detected=False,
    )
