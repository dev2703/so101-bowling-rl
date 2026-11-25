"""Stage 2: Online DAgger distillation from PrivilegedTeacherAgent into DualCameraActor."""

from __future__ import annotations

import json
import math
import random
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from so101_bowling.config import WorkspaceConfig, load_config
from so101_bowling.ppo import (
    VALIDATION_LOSS_NOTE,
    _assert_finite_params,
    _assert_finite_tensor,
    evaluate_actor,
    resolve_device,
)
from so101_bowling.ppo_models import (
    ACTION_DIM,
    AUX_MASK_DIM,
    AUX_TARGET_DIM,
    CAMERA_META_DIM,
    PROPRIO_DIM,
    DualCameraActor,
    _ortho_init,
    masked_aux_loss,
)
from so101_bowling.teacher import load_teacher
from so101_bowling.tracking import WandbConfig, WandbTracker
from so101_bowling.vec_env import make_vec_env

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DAGGER_YAML = _REPO_ROOT / "configs" / "dagger_dr.yaml"


@dataclass(frozen=True)
class DAggerConfig:
    seed: int = 0
    dr_enabled: bool = True
    frame_stack: int = 2
    num_envs: int = 4
    rounds: int = 12
    steps_per_round: int = 1500
    capacity: int = 24000
    epochs_per_round: int = 8
    batch_size: int = 128
    lr: float = 3.0e-4
    aux_coef: float = 0.5
    max_grad_norm: float = 1.0
    drq_pad_px: int = 4
    beta_start: float = 1.0
    beta_end: float = 0.0
    squashed: bool = True
    encoder_type: str = "spatial_grid"
    device: str = "auto"
    eval_episodes: int = 10
    save_video: bool = True

    def __post_init__(self) -> None:
        for name in (
            "seed",
            "frame_stack",
            "num_envs",
            "rounds",
            "steps_per_round",
            "capacity",
            "epochs_per_round",
            "batch_size",
            "drq_pad_px",
            "eval_episodes",
        ):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, int):
                raise ValueError(f"{name} must be an int, got {type(val).__name__}")
        for name in ("dr_enabled", "squashed", "save_video"):
            val = getattr(self, name)
            if not isinstance(val, bool):
                raise ValueError(f"{name} must be a bool, got {type(val).__name__}")
        for name in (
            "frame_stack",
            "num_envs",
            "rounds",
            "steps_per_round",
            "capacity",
            "epochs_per_round",
            "batch_size",
            "eval_episodes",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.drq_pad_px < 0:
            raise ValueError("drq_pad_px must be nonnegative")
        if self.encoder_type not in ("spatial_grid", "spatial_softmax"):
            raise ValueError(f"Unsupported encoder_type: {self.encoder_type!r}")
        for name in ("lr", "aux_coef", "max_grad_norm", "beta_start", "beta_end"):
            raw_val = getattr(self, name)
            if isinstance(raw_val, bool) or not isinstance(raw_val, (int, float)):
                raise ValueError(f"{name} must be a real number, got {type(raw_val).__name__}")
            if not math.isfinite(float(raw_val)):
                raise ValueError(f"{name} must be finite, got {raw_val}")
        if self.lr <= 0.0:
            raise ValueError("lr must be positive")
        if self.aux_coef < 0.0:
            raise ValueError("aux_coef must be nonnegative")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        for name in ("beta_start", "beta_end"):
            if not (0.0 <= float(getattr(self, name)) <= 1.0):
                raise ValueError(f"{name} must be in [0, 1]")


def load_dagger_config(path: str | Path | None = None) -> DAggerConfig:
    cfg_path = Path(path) if path is not None else _DEFAULT_DAGGER_YAML
    if not cfg_path.exists():
        raise FileNotFoundError(f"DAgger config not found: {cfg_path}")
    raw = yaml.safe_load(cfg_path.read_text()) or {}
    return DAggerConfig(**raw)


class _DAggerBuffer:
    """Compact CPU uint8 ring buffer storing dual-camera observations and teacher actions."""

    def __init__(self, capacity: int, frame_stack: int, img_hw: tuple[int, int]) -> None:
        h, w = img_hw
        self.capacity = int(capacity)
        self.size = 0
        self.ptr = 0
        self.wrist_rgb = torch.empty((capacity, frame_stack, h, w, 3), dtype=torch.uint8)
        self.tripod_rgb = torch.empty((capacity, frame_stack, h, w, 3), dtype=torch.uint8)
        self.camera_meta = torch.empty((capacity, CAMERA_META_DIM), dtype=torch.float32)
        self.proprio = torch.empty((capacity, PROPRIO_DIM), dtype=torch.float32)
        self.aux_targets = torch.empty((capacity, AUX_TARGET_DIM), dtype=torch.float32)
        self.aux_mask = torch.empty((capacity, AUX_MASK_DIM), dtype=torch.float32)
        self.teacher_actions = torch.empty((capacity, ACTION_DIM), dtype=torch.float32)

    def add_batch(self, obs: dict[str, np.ndarray], teacher_actions: np.ndarray) -> None:
        n = int(teacher_actions.shape[0])
        for i in range(n):
            idx = self.ptr
            self.wrist_rgb[idx].copy_(torch.from_numpy(obs["wrist_rgb"][i]))
            self.tripod_rgb[idx].copy_(torch.from_numpy(obs["tripod_rgb"][i]))
            self.camera_meta[idx].copy_(torch.from_numpy(obs["camera_meta"][i]))
            self.proprio[idx].copy_(torch.from_numpy(obs["proprio"][i]))
            self.aux_targets[idx].copy_(torch.from_numpy(obs["privileged_aux_targets"][i]))
            self.aux_mask[idx].copy_(torch.from_numpy(obs["privileged_aux_mask"][i]))
            self.teacher_actions[idx].copy_(torch.from_numpy(teacher_actions[i]))
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def slice_batch(self, indices: np.ndarray, device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        obs = {
            "wrist_rgb": self.wrist_rgb[indices].to(device),
            "tripod_rgb": self.tripod_rgb[indices].to(device),
            "camera_meta": self.camera_meta[indices].to(device),
            "proprio": self.proprio[indices].to(device),
            "privileged_aux_targets": self.aux_targets[indices].to(device),
            "privileged_aux_mask": self.aux_mask[indices].to(device),
        }
        return obs, self.teacher_actions[indices].to(device)


def train_dagger(
    cfg: DAggerConfig,
    teacher_path: str | Path,
    workspace_cfg: WorkspaceConfig | None = None,
    run_dir: str | Path = "runs/dagger",
    wandb_cfg: WandbConfig | None = None,
) -> dict[str, Any]:
    """Distill a PrivilegedTeacherAgent into a standalone deployable DualCameraActor."""
    wcfg = workspace_cfg if workspace_cfg is not None else load_config()
    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dagger_config.json").write_text(json.dumps(asdict(cfg), indent=2))

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = resolve_device(cfg.device)
    img_hw = (wcfg.camera.render_height, wcfg.camera.render_width)

    teacher = load_teacher(teacher_path, map_location=device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    actor = DualCameraActor(
        frame_stack=cfg.frame_stack,
        action_dim=ACTION_DIM,
        encoder_type=cfg.encoder_type,
        img_hw=img_hw,
        squashed=cfg.squashed,
    ).to(device)
    aux_head = nn.Sequential(
        _ortho_init(nn.Linear(256, 64)),
        nn.SiLU(),
        _ortho_init(nn.Linear(64, AUX_TARGET_DIM), std=0.01),
    ).to(device)
    params = list(actor.parameters()) + list(aux_head.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.lr)

    buf = _DAggerBuffer(capacity=cfg.capacity, frame_stack=cfg.frame_stack, img_hw=img_hw)
    metrics_path = out_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    tracker = WandbTracker(
        wandb_cfg,
        run_dir=out_dir,
        algo="dagger",
        train_config={**asdict(cfg), "teacher_path": str(teacher_path)},
    )
    if tracker.run_url:
        print(f"[dagger] wandb_url={tracker.run_url}", flush=True)

    best_score: tuple[float, float, float] = (-1.0, -1.0, -float("inf"))
    best_eval: dict[str, Any] = {}
    last_eval: dict[str, Any] = {}
    last_log: dict[str, Any] = {}
    global_step = 0
    nan_inf_detected = False
    nan_inf_error: str | None = None

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
                    gamma=wcfg.reward.gamma,
                    base_seed=cfg.seed,
                )
            )
            obs = vec_env.reset(seed=cfg.seed)
            t_start = perf_counter()

            for rnd in range(1, cfg.rounds + 1):
                if cfg.rounds > 1:
                    frac = (rnd - 1) / (cfg.rounds - 1)
                    beta = cfg.beta_start + frac * (cfg.beta_end - cfg.beta_start)
                else:
                    beta = cfg.beta_start

                actor.eval()
                round_outcomes: list[dict[str, Any]] = []
                # Sample policy choice per episode (resampling only when an env finishes).
                use_teacher = rng.random(cfg.num_envs) < beta
                collected = 0
                while collected < cfg.steps_per_round:
                    with torch.no_grad():
                        t_act = teacher.act(obs, deterministic=True).cpu().numpy().astype(np.float32)
                        s_act = actor.act(obs, deterministic=True).cpu().numpy().astype(np.float32)
                    buf.add_batch(obs, t_act)

                    exec_act = np.where(use_teacher[:, None], t_act, s_act)
                    obs, _, terminated, truncated, infos = vec_env.step(exec_act)
                    done_mask = terminated | truncated
                    if np.any(done_mask):
                        for env_idx in np.flatnonzero(done_mask):
                            use_teacher[env_idx] = bool(rng.random() < beta)
                    for info in infos:
                        if "outcome" in info:
                            round_outcomes.append(info["outcome"])
                    collected += cfg.num_envs
                    global_step += cfg.num_envs

                actor.train()
                aux_head.train()
                bc_losses: list[float] = []
                aux_losses: list[float] = []
                bs = min(cfg.batch_size, buf.size)
                for _ in range(cfg.epochs_per_round):
                    perm = rng.permutation(buf.size)
                    for start in range(0, buf.size, bs):
                        mb = perm[start : start + bs]
                        mb_obs, mb_t_act = buf.slice_batch(mb, device)
                        if cfg.drq_pad_px > 0:
                            w_sh = torch.randint(
                                -cfg.drq_pad_px, cfg.drq_pad_px + 1, (len(mb), 2), device=device
                            )
                            t_sh = torch.randint(
                                -cfg.drq_pad_px, cfg.drq_pad_px + 1, (len(mb), 2), device=device
                            )
                        else:
                            w_sh = t_sh = None
                        dist, z_vis = actor.forward_actor(mb_obs, wrist_shifts=w_sh, tripod_shifts=t_sh)
                        if actor.squashed:
                            pred_act = torch.tanh(dist.mean)
                            target_act = mb_t_act.clamp(-1.0 + 1e-3, 1.0 - 1e-3)
                        else:
                            pred_act = torch.clamp(dist.mean, -1.0, 1.0)
                            target_act = mb_t_act
                        bc_loss = F.mse_loss(pred_act, target_act)
                        aux_loss = masked_aux_loss(
                            aux_head(z_vis),
                            mb_obs["privileged_aux_targets"].to(dtype=torch.float32),
                            mb_obs["privileged_aux_mask"],
                        )
                        loss = bc_loss + cfg.aux_coef * aux_loss
                        _assert_finite_tensor(f"dagger.rnd_{rnd}.loss", loss)
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        grad_norm = nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                        _assert_finite_tensor(f"dagger.rnd_{rnd}.grad_norm", grad_norm)
                        optimizer.step()
                        _assert_finite_params(f"dagger.rnd_{rnd}", params)
                        bc_losses.append(float(bc_loss.detach().cpu()))
                        aux_losses.append(float(aux_loss.detach().cpu()))

                vid_path = (
                    out_dir / "videos" / f"eval_round_{rnd:03d}.mp4" if cfg.save_video else None
                )
                ev = evaluate_actor(
                    actor=actor,
                    cfg=wcfg,
                    dr_enabled=cfg.dr_enabled,
                    frame_stack=cfg.frame_stack,
                    gamma=wcfg.reward.gamma,
                    episodes=cfg.eval_episodes,
                    seed=cfg.seed + 10000,
                    device=device,
                    video_path=vid_path,
                    aux_head=aux_head,
                )
                last_eval = ev
                score = (ev["legal_strike_rate"], ev["mean_credited_pins"], ev["mean_return"])
                if score > best_score or not (out_dir / "actor_best.pt").exists():
                    best_score = score
                    best_eval = ev
                    actor.save_actor(out_dir / "actor_best.pt")
                actor.save_actor(out_dir / "actor_latest.pt")

                if vid_path is not None:
                    tracker.log_video(
                        "dagger/eval_video",
                        vid_path,
                        step=global_step,
                        fps=int(round(wcfg.robot.control_hz)),
                    )

                elapsed = max(1e-6, perf_counter() - t_start)
                log_row: dict[str, Any] = {
                    "round": rnd,
                    "global_step": global_step,
                    "beta": float(beta),
                    "buffer_size": buf.size,
                    "bc_loss": float(np.mean(bc_losses)),
                    "aux_loss": float(np.mean(aux_losses)),
                    "transitions_per_sec": global_step / elapsed,
                    "rollout_legal_strike_rate": (
                        float(np.mean([int(bool(o.get("legal_strike", False))) for o in round_outcomes]))
                        if round_outcomes
                        else 0.0
                    ),
                    "rollout_mean_credited_pins": (
                        float(np.mean([int(o.get("credited_pins", 0)) for o in round_outcomes]))
                        if round_outcomes
                        else 0.0
                    ),
                    **{
                        (k if k.startswith("eval_") else f"eval_{k}"): v
                        for k, v in ev.items()
                        if k != "episodes"
                    },
                }
                for key, val in log_row.items():
                    if isinstance(val, (int, float)) and not isinstance(val, bool) and key != "global_step":
                        writer.add_scalar(f"dagger/{key}", val, global_step)
                tracker.log_metrics("dagger", log_row, step=global_step)
                with metrics_path.open("a") as f:
                    f.write(json.dumps(log_row) + "\n")
                last_log = log_row
                print(
                    f"[dagger] rnd={rnd}/{cfg.rounds} step={global_step} beta={beta:.2f} "
                    f"bc={log_row['bc_loss']:.5f} aux={log_row['aux_loss']:.5f} "
                    f"eval_strike={ev['legal_strike_rate']:.3f} eval_pins={ev['mean_credited_pins']:.2f} "
                    f"eval_ret={ev['mean_return']:.2f}",
                    flush=True,
                )
    except FloatingPointError as exc:
        nan_inf_detected = True
        nan_inf_error = str(exc)
        eval_aux_loss = last_eval.get("eval_aux_loss") or (best_eval.get("eval_aux_loss") if best_eval else None)
        summary = {
            "run_dir": str(out_dir),
            "global_step": int(global_step),
            "total_timesteps": int(cfg.rounds * cfg.steps_per_round),
            "best_eval": best_eval,
            "final_eval": last_eval,
            "last_metrics": last_log,
            "final_losses": {
                k: float(last_log[k])
                for k in ("bc_loss", "aux_loss")
                if k in last_log and isinstance(last_log[k], (int, float))
            },
            "eval_aux_loss": eval_aux_loss,
            "nan_inf_detected": nan_inf_detected,
            "nan_inf_error": nan_inf_error,
            "validation_loss_note": VALIDATION_LOSS_NOTE,
            "wandb_url": tracker.run_url,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        tracker.update_summary(summary)
        tracker.finish(exit_code=1)
        raise
    except Exception:
        tracker.finish(exit_code=1)
        raise

    eval_aux_loss = last_eval.get("eval_aux_loss")
    if eval_aux_loss is None and best_eval:
        eval_aux_loss = best_eval.get("eval_aux_loss")
    summary = {
        "run_dir": str(out_dir),
        "global_step": int(global_step),
        "total_timesteps": int(cfg.rounds * cfg.steps_per_round),
        "best_eval": best_eval,
        "final_eval": last_eval,
        "last_metrics": last_log,
        "final_losses": {
            k: float(last_log[k])
            for k in ("bc_loss", "aux_loss")
            if k in last_log and isinstance(last_log[k], (int, float))
        },
        "eval_aux_loss": eval_aux_loss,
        "nan_inf_detected": nan_inf_detected,
        "validation_loss_note": VALIDATION_LOSS_NOTE,
        "wandb_url": tracker.run_url,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    tracker.update_summary(summary)
    tracker.finish(exit_code=0)
    return summary
