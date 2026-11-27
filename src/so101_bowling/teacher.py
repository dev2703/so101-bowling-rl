"""Stage 1: Fast privileged-state PPO teacher trained with render_rgb=False."""

from __future__ import annotations

import json
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
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from so101_bowling.config import WorkspaceConfig, load_config
from so101_bowling.env import SO101BowlingEnv
from so101_bowling.kinematics import ScriptedPushController
from so101_bowling.ppo import (
    _assert_finite_mapping,
    _assert_finite_params,
    _assert_finite_tensor,
    _explained_variance,
    compute_gae,
    resolve_device,
)
from so101_bowling.ppo_models import (
    ACTION_DIM,
    PRIVILEGED_STATE_DIM,
    PROPRIO_DIM,
    _ortho_init,
)
from so101_bowling.tracking import WandbConfig, WandbTracker
from so101_bowling.vec_env import make_vec_env

TEACHER_OBS_DIM = PRIVILEGED_STATE_DIM + PROPRIO_DIM
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_TEACHER_YAML = _REPO_ROOT / "configs" / "teacher_dr.yaml"


class PrivilegedTeacherAgent(nn.Module):
    """Fast MLP actor-critic consuming only privileged_state + proprio."""

    def __init__(
        self,
        obs_dim: int = TEACHER_OBS_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)

        self.actor_net = nn.Sequential(
            _ortho_init(nn.Linear(obs_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            _ortho_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.mean_head = _ortho_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

        self.critic_net = nn.Sequential(
            _ortho_init(nn.Linear(obs_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            _ortho_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            _ortho_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    @staticmethod
    def concat_obs(obs: Mapping[str, Any], device: torch.device) -> Tensor:
        priv = torch.as_tensor(obs["privileged_state"], device=device, dtype=torch.float32)
        prop = torch.as_tensor(obs["proprio"], device=device, dtype=torch.float32)
        if priv.ndim == 1:
            priv = priv.unsqueeze(0)
            prop = prop.unsqueeze(0)
        return torch.cat((priv, prop), dim=-1)

    def forward_actor(self, x: Tensor) -> Normal:
        h = self.actor_net(x)
        mean = self.mean_head(h)
        std = torch.exp(self.log_std.clamp(-5.0, 2.0)).expand_as(mean)
        if not (torch.isfinite(mean).all() and torch.isfinite(std).all()):
            raise FloatingPointError("Non-finite teacher actor output")
        return Normal(mean, std)

    def get_value(self, x: Tensor) -> Tensor:
        return self.critic_net(x).squeeze(-1)

    def get_action_and_value(
        self, x: Tensor, action: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        dist = self.forward_actor(x)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.get_value(x)
        return action, logprob, entropy, value

    @torch.no_grad()
    def act(self, obs: Mapping[str, Any], deterministic: bool = True) -> Tensor:
        device = self.log_std.device
        priv = obs["privileged_state"]
        single = np.asarray(priv).ndim == 1 if not isinstance(priv, Tensor) else priv.ndim == 1
        x = self.concat_obs(obs, device)
        dist = self.forward_actor(x)
        act = dist.mean if deterministic else dist.sample()
        act = torch.clamp(act, -1.0, 1.0)
        return act.squeeze(0) if single else act

    def save_teacher(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "hidden_dim": self.hidden_dim,
                "state_dict": self.state_dict(),
            },
            out,
        )
        return out


def load_teacher(path: str | Path, map_location: str | torch.device = "cpu") -> PrivilegedTeacherAgent:
    ckpt = torch.load(Path(path), map_location=map_location, weights_only=False)
    agent = PrivilegedTeacherAgent(
        obs_dim=int(ckpt.get("obs_dim", TEACHER_OBS_DIM)),
        action_dim=int(ckpt.get("action_dim", ACTION_DIM)),
        hidden_dim=int(ckpt.get("hidden_dim", 256)),
    )
    agent.load_state_dict(ckpt["state_dict"])
    agent.to(torch.device(map_location))
    agent.eval()
    return agent


@dataclass(frozen=True)
class TeacherPPOConfig:
    seed: int = 0
    dr_enabled: bool = True
    num_envs: int = 8
    rollout_steps: int = 128
    total_timesteps: int = 400000
    hidden_dim: int = 256
    device: str = "auto"
    lr: float = 3.0e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_epochs: int = 6
    minibatch_size: int = 128
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.005
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03
    bc_warmup_episodes: int = 60
    bc_warmup_epochs: int = 15
    bc_lr: float = 1.0e-3
    eval_interval_updates: int = 15
    eval_episodes: int = 20
    checkpoint_interval_updates: int = 30


def load_teacher_config(path: str | Path | None = None) -> TeacherPPOConfig:
    cfg_path = Path(path) if path is not None else _DEFAULT_TEACHER_YAML
    if not cfg_path.exists():
        raise FileNotFoundError(f"Teacher config not found: {cfg_path}")
    raw = yaml.safe_load(cfg_path.read_text()) or {}
    return TeacherPPOConfig(**raw)


@torch.no_grad()
def evaluate_teacher(
    agent: PrivilegedTeacherAgent,
    cfg: WorkspaceConfig,
    *,
    dr_enabled: bool = True,
    gamma: float = 0.99,
    episodes: int = 20,
    seed: int = 10000,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    was_training = agent.training
    orig_device = agent.log_std.device
    target_device = torch.device(device)
    agent.eval()
    agent.to(target_device)

    strikes = 0
    illegals = 0
    credited_list: list[int] = []
    full_list: list[int] = []
    returns: list[float] = []
    steps_list: list[int] = []
    try:
        with SO101BowlingEnv(
            cfg=cfg,
            dr_enabled=dr_enabled,
            render_rgb=False,
            frame_stack=2,
            gamma=gamma,
        ) as env:
            for ep in range(episodes):
                obs, _ = env.reset(seed=seed + ep)
                done = False
                ep_ret = 0.0
                ep_steps = 0
                info: dict[str, Any] = {}
                while not done:
                    ep_steps += 1
                    action = agent.act(obs, deterministic=True).cpu().numpy()
                    obs, reward, terminated, truncated, info = env.step(action)
                    ep_ret += float(reward)
                    done = terminated or truncated
                outcome = info.get("outcome", {})
                legal = bool(outcome.get("legal_delivery", False))
                strikes += int(bool(outcome.get("legal_strike", False)))
                illegals += int(not legal)
                credited_list.append(int(outcome.get("credited_pins", 0)))
                full_list.append(int(outcome.get("full_knock_pins", 0)))
                returns.append(ep_ret)
                steps_list.append(ep_steps)
    finally:
        agent.to(orig_device)
        if was_training:
            agent.train()

    n = max(1, len(returns))
    return {
        "episodes": len(returns),
        "legal_strike_rate": strikes / n,
        "mean_credited_pins": float(np.mean(credited_list)),
        "mean_full_knock_pins": float(np.mean(full_list)),
        "illegal_rate": illegals / n,
        "mean_return": float(np.mean(returns)),
        "mean_episode_steps": float(np.mean(steps_list)),
    }


def _bc_warmup_teacher(
    agent: PrivilegedTeacherAgent,
    wcfg: WorkspaceConfig,
    cfg: TeacherPPOConfig,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Fast render_rgb=False scripted BC warmup so teacher starts pushing on update 1."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    legal_count = 0
    with SO101BowlingEnv(
        cfg=wcfg,
        dr_enabled=cfg.dr_enabled,
        render_rgb=False,
        frame_stack=2,
        gamma=cfg.gamma,
    ) as env:
        controller = ScriptedPushController(env.model, env.ids, wcfg, env.q_min, env.q_max)
        for ep in range(cfg.bc_warmup_episodes):
            obs, _ = env.reset(seed=cfg.seed + 50000 + ep)
            controller.reset(env._last_snapshot.ball_pos, env._last_snapshot.pin_positions)
            done = False
            ep_x: list[np.ndarray] = []
            ep_y: list[np.ndarray] = []
            info: dict[str, Any] = {}
            while not done:
                act = controller.step(env._measured_q())
                ep_x.append(
                    np.concatenate(
                        (
                            np.asarray(obs["privileged_state"], dtype=np.float32),
                            np.asarray(obs["proprio"], dtype=np.float32),
                        )
                    )
                )
                ep_y.append(np.asarray(act, dtype=np.float32))
                obs, _, terminated, truncated, info = env.step(act)
                done = terminated or truncated
            if bool(info.get("outcome", {}).get("legal_delivery", False)):
                legal_count += 1
                xs.extend(ep_x)
                ys.extend(ep_y)

    if not xs:
        return {"bc_transitions": 0.0, "bc_loss": 0.0, "bc_legal_episodes": 0.0}

    x_t = torch.from_numpy(np.stack(xs, axis=0)).to(device)
    y_t = torch.from_numpy(np.stack(ys, axis=0)).to(device)
    opt = torch.optim.Adam(
        list(agent.actor_net.parameters()) + list(agent.mean_head.parameters()),
        lr=cfg.bc_lr,
    )
    agent.train()
    n = x_t.shape[0]
    bs = min(cfg.minibatch_size, n)
    final_loss = 0.0
    for _ in range(cfg.bc_warmup_epochs):
        perm = rng.permutation(n)
        for start in range(0, n, bs):
            idx = perm[start : start + bs]
            pred = agent.mean_head(agent.actor_net(x_t[idx]))
            loss = F.mse_loss(pred, y_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
            opt.step()
            final_loss = float(loss.detach().cpu().item())

    return {
        "bc_transitions": float(n),
        "bc_loss": final_loss,
        "bc_legal_episodes": float(legal_count),
    }


def train_teacher_ppo(
    cfg: TeacherPPOConfig,
    workspace_cfg: WorkspaceConfig | None = None,
    run_dir: str | Path = "runs/teacher",
    resume_from: str | Path | None = None,
    wandb_cfg: WandbConfig | None = None,
) -> dict[str, Any]:
    """Train PrivilegedTeacherAgent with fast render_rgb=False PPO."""
    wcfg = workspace_cfg if workspace_cfg is not None else load_config()
    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "teacher_config.json").write_text(json.dumps(asdict(cfg), indent=2))

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = resolve_device(cfg.device)

    agent = PrivilegedTeacherAgent(
        obs_dim=TEACHER_OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=cfg.hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)

    batch_size = cfg.num_envs * cfg.rollout_steps
    num_updates = max(1, cfg.total_timesteps // batch_size)
    minibatch_size = min(cfg.minibatch_size, batch_size)

    best_score: tuple[float, float, float] = (-1.0, -1.0, -float("inf"))
    best_eval: dict[str, Any] = {}
    last_eval: dict[str, Any] = {}
    last_log: dict[str, Any] = {}
    global_step = 0
    start_update = 1

    if resume_from is not None:
        ckpt = torch.load(Path(resume_from), map_location=device, weights_only=False)
        agent.load_state_dict(ckpt["agent"])
        optimizer.load_state_dict(ckpt["optimizer"])
        global_step = int(ckpt.get("global_step", 0))
        start_update = int(ckpt.get("update", 0)) + 1
        if ckpt.get("best_score") is not None:
            best_score = tuple(float(v) for v in ckpt["best_score"])
            best_eval = dict(ckpt.get("best_eval", {}))
        last_eval = dict(ckpt.get("last_eval", {}))
        last_log = dict(ckpt.get("last_log", {}))
    else:
        bc_stats = (
            _bc_warmup_teacher(agent, wcfg, cfg, device, rng)
            if cfg.bc_warmup_episodes > 0 and cfg.bc_warmup_epochs > 0
            else {"bc_transitions": 0.0, "bc_loss": 0.0, "bc_legal_episodes": 0.0}
        )
        print(
            f"[teacher] bc_warmup transitions={int(bc_stats['bc_transitions'])} "
            f"legal_eps={int(bc_stats['bc_legal_episodes'])} bc_loss={bc_stats['bc_loss']:.5f}",
            flush=True,
        )

    metrics_path = out_dir / "metrics.jsonl"
    if resume_from is None and metrics_path.exists():
        metrics_path.unlink()

    tracker = WandbTracker(
        wandb_cfg,
        run_dir=out_dir,
        algo="teacher_ppo",
        train_config=asdict(cfg),
    )
    if tracker.run_url:
        print(f"[teacher] wandb_url={tracker.run_url}", flush=True)

    obs_buf = torch.empty((cfg.rollout_steps, cfg.num_envs, TEACHER_OBS_DIM), dtype=torch.float32)
    act_buf = torch.empty((cfg.rollout_steps, cfg.num_envs, ACTION_DIM), dtype=torch.float32)
    logp_buf = torch.empty((cfg.rollout_steps, cfg.num_envs), dtype=torch.float32)
    val_buf = torch.empty((cfg.rollout_steps, cfg.num_envs), dtype=torch.float32)
    rew_buf = torch.empty((cfg.rollout_steps, cfg.num_envs), dtype=torch.float32)
    term_buf = torch.empty((cfg.rollout_steps, cfg.num_envs), dtype=torch.bool)
    trunc_buf = torch.empty((cfg.rollout_steps, cfg.num_envs), dtype=torch.bool)
    fval_buf = torch.zeros((cfg.rollout_steps, cfg.num_envs), dtype=torch.float32)

    try:
        with ExitStack() as stack:
            writer = SummaryWriter(log_dir=str(out_dir / "tb"))
            stack.callback(writer.close)
            vec_env = stack.enter_context(
                make_vec_env(
                    num_envs=cfg.num_envs,
                    cfg=wcfg,
                    dr_enabled=cfg.dr_enabled,
                    render_rgb=False,
                    frame_stack=2,
                    gamma=cfg.gamma,
                    base_seed=cfg.seed,
                )
            )
            if abs(vec_env.scorer_gamma - cfg.gamma) > 1e-6:
                raise ValueError(
                    f"Scorer gamma ({vec_env.scorer_gamma}) must match "
                    f"TeacherPPOConfig.gamma ({cfg.gamma})"
                )
            next_obs = vec_env.reset(seed=cfg.seed)
            ep_outcomes: list[dict[str, Any]] = []
            t_start = perf_counter()

            for update in range(start_update, num_updates + 1):
                if cfg.anneal_lr:
                    frac = 1.0 - (update - 1.0) / num_updates
                    optimizer.param_groups[0]["lr"] = frac * cfg.lr

                agent.eval()
                for step in range(cfg.rollout_steps):
                    global_step += cfg.num_envs
                    x_t = PrivilegedTeacherAgent.concat_obs(next_obs, device)
                    with torch.no_grad():
                        action, logprob, _, value = agent.get_action_and_value(x_t)
                    _assert_finite_mapping(
                        f"teacher.rollout.step_{global_step}",
                        {"action": action, "logprob": logprob, "value": value},
                    )
                    clipped = np.clip(action.cpu().numpy(), -1.0, 1.0)
                    new_obs, rewards, terminated, truncated, infos = vec_env.step(clipped)

                    final_values = torch.zeros(cfg.num_envs, dtype=torch.float32)
                    for idx, info in enumerate(infos):
                        if "outcome" in info:
                            ep_outcomes.append(info["outcome"])
                        if truncated[idx] and not terminated[idx] and "final_obs" in info:
                            with torch.no_grad():
                                fx = PrivilegedTeacherAgent.concat_obs(info["final_obs"], device)
                                final_values[idx] = agent.get_value(fx).squeeze(0).cpu()

                    obs_buf[step].copy_(x_t.cpu())
                    act_buf[step].copy_(action.cpu())
                    logp_buf[step].copy_(logprob.cpu())
                    val_buf[step].copy_(value.cpu())
                    rew_buf[step].copy_(torch.from_numpy(rewards))
                    term_buf[step].copy_(torch.from_numpy(terminated))
                    trunc_buf[step].copy_(torch.from_numpy(truncated))
                    fval_buf[step].copy_(final_values)
                    next_obs = new_obs

                with torch.no_grad():
                    next_val = agent.get_value(
                        PrivilegedTeacherAgent.concat_obs(next_obs, device)
                    ).cpu()
                advantages, returns = compute_gae(
                    rewards=rew_buf,
                    values=val_buf,
                    terminated=term_buf,
                    truncated=trunc_buf,
                    next_value=next_val,
                    final_values=fval_buf,
                    gamma=cfg.gamma,
                    gae_lambda=cfg.gae_lambda,
                )

                b_obs = obs_buf.reshape(batch_size, TEACHER_OBS_DIM).to(device)
                b_act = act_buf.reshape(batch_size, ACTION_DIM).to(device)
                b_logp = logp_buf.reshape(batch_size).to(device)
                b_adv = advantages.reshape(batch_size).to(device)
                b_ret = returns.reshape(batch_size).to(device)
                if b_adv.numel() > 1:
                    b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

                agent.train()
                mb_metrics: list[Tensor] = []
                stop_early = False
                for _ in range(cfg.num_epochs):
                    perm = rng.permutation(batch_size)
                    for start in range(0, batch_size, minibatch_size):
                        mb = perm[start : start + minibatch_size]
                        _, newlogp, entropy, newval = agent.get_action_and_value(
                            b_obs[mb], action=b_act[mb]
                        )
                        logratio = newlogp - b_logp[mb]
                        ratio = logratio.exp()
                        with torch.no_grad():
                            approx_kl_t = ((ratio - 1.0) - logratio).mean()

                        pg1 = -b_adv[mb] * ratio
                        pg2 = -b_adv[mb] * torch.clamp(
                            ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef
                        )
                        pg_loss = torch.max(pg1, pg2).mean()
                        v_loss = 0.5 * ((newval - b_ret[mb]) ** 2).mean()
                        ent_loss = entropy.mean()
                        loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent_loss
                        _assert_finite_tensor(f"teacher.upd_{update}.loss", loss)

                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                        _assert_finite_tensor(f"teacher.upd_{update}.grad_norm", grad_norm)
                        optimizer.step()
                        _assert_finite_params(f"teacher.upd_{update}", agent.parameters())

                        metric_row = (approx_kl_t, pg_loss, v_loss, ent_loss)
                        mb_metrics.append(torch.stack([t.detach() for t in metric_row]))
                        if (
                            cfg.target_kl is not None
                            and float(approx_kl_t.cpu()) > 1.5 * cfg.target_kl
                        ):
                            stop_early = True
                            break
                    if stop_early:
                        break

                # Single host transfer for every minibatch metric of this update.
                kls, pg_losses, v_losses, ent_losses = torch.stack(mb_metrics, dim=1).cpu().tolist()

                elapsed = max(1e-6, perf_counter() - t_start)
                tps = global_step / elapsed
                recent = ep_outcomes[-50:]
                log_row: dict[str, Any] = {
                    "update": update,
                    "global_step": global_step,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "policy_loss": float(np.mean(pg_losses)),
                    "value_loss": float(np.mean(v_losses)),
                    "entropy": float(np.mean(ent_losses)),
                    "approx_kl": float(np.mean(kls)),
                    "explained_variance": _explained_variance(
                        val_buf.reshape(batch_size), returns.reshape(batch_size)
                    ),
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

                if update % cfg.eval_interval_updates == 0 or update == num_updates:
                    ev = evaluate_teacher(
                        agent,
                        wcfg,
                        dr_enabled=cfg.dr_enabled,
                        gamma=cfg.gamma,
                        episodes=cfg.eval_episodes,
                        seed=cfg.seed + 10000,
                        device=device,
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
                    if score > best_score or not (out_dir / "best_teacher.pt").exists():
                        best_score = score
                        best_eval = ev
                        agent.save_teacher(out_dir / "best_teacher.pt")
                    print(
                        f"[teacher] eval step={global_step}/{cfg.total_timesteps} "
                        f"strike={ev['legal_strike_rate']:.3f} pins={ev['mean_credited_pins']:.3f} "
                        f"illegal={ev['illegal_rate']:.3f} return={ev['mean_return']:.3f}",
                        flush=True,
                    )

                for key, val in log_row.items():
                    if isinstance(val, (int, float)) and not isinstance(val, bool) and key not in ("update", "global_step"):
                        writer.add_scalar(f"teacher/{key}", val, global_step)
                tracker.log_metrics("teacher", log_row, step=global_step)
                with metrics_path.open("a") as f:
                    f.write(json.dumps(log_row) + "\n")
                last_log = log_row

                if update % cfg.checkpoint_interval_updates == 0 or update == num_updates:
                    torch.save(
                        {
                            "agent": agent.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "update": update,
                            "global_step": global_step,
                            "best_score": list(best_score),
                            "best_eval": best_eval,
                            "last_eval": last_eval,
                            "last_log": last_log,
                            "teacher_cfg": asdict(cfg),
                        },
                        out_dir / "latest.pt",
                    )

                if update % 5 == 0 or update == 1 or update == num_updates:
                    print(
                        f"[teacher] upd={update}/{num_updates} step={global_step} "
                        f"pg={log_row['policy_loss']:.4f} vf={log_row['value_loss']:.4f} "
                        f"pins={log_row['train_mean_credited_pins']:.2f} "
                        f"strike={log_row['train_legal_strike_rate']:.2f} tps={tps:.1f}",
                        flush=True,
                    )

            agent.save_teacher(out_dir / "final_teacher.pt")
            if not (out_dir / "best_teacher.pt").exists():
                agent.save_teacher(out_dir / "best_teacher.pt")
    except Exception:
        tracker.finish(exit_code=1)
        raise

    summary = {
        "run_dir": str(out_dir),
        "global_step": int(global_step),
        "total_timesteps": int(cfg.total_timesteps),
        "best_eval": best_eval,
        "final_eval": last_eval,
        "last_metrics": last_log,
        "nan_inf_detected": False,
        "wandb_url": tracker.run_url,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    tracker.update_summary(summary)
    tracker.finish(exit_code=0)
    return summary
