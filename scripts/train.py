"""Minimal self-contained PyTorch PPO trainer for SO101BowlingEnv."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from so101_bowling import SO101BowlingEnv, parse_policy_cli


def _ortho(layer: nn.Linear, std: float = np.sqrt(2.0)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class RunningObsNorm:
    """Welford running mean/std normalizer for 1D observations."""

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = eps

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        batch = x[None, :] if x.ndim == 1 else x
        b_mean = batch.mean(axis=0)
        b_var = batch.var(axis=0)
        b_count = batch.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_count
        self.mean = self.mean + delta * (b_count / tot)
        m_a = self.var * self.count
        m_b = b_var * b_count
        m2 = m_a + m_b + (delta**2) * (self.count * b_count / tot)
        self.var = m2 / tot
        self.count = tot

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.var, 1e-8))

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / self.std, -10.0, 10.0).astype(np.float32)


class Actor(nn.Module):
    """2-layer MLP Gaussian policy."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _ortho(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            _ortho(nn.Linear(hidden, hidden)),
            nn.Tanh(),
        )
        self.mean = _ortho(nn.Linear(hidden, act_dim), std=0.01)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def dist(self, x: torch.Tensor) -> Normal:
        mean = self.mean(self.net(x))
        std = torch.exp(self.log_std.clamp(-5.0, 2.0)).expand_as(mean)
        return Normal(mean, std)


class ActorCritic(nn.Module):
    """Actor-Critic pairing deployable Actor with a 2-layer MLP value head."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.actor = Actor(obs_dim, act_dim, hidden)
        self.critic = nn.Sequential(
            _ortho(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            _ortho(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            _ortho(nn.Linear(hidden, 1), std=1.0),
        )

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x).squeeze(-1)

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.actor.dist(x)
        if action is None:
            action = dist.sample()
        logp = dist.log_prob(action).sum(dim=-1)
        ent = dist.entropy().sum(dim=-1)
        return action, logp, ent, self.value(x)


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


@dataclass(frozen=True)
class PPOConfig:
    total_steps: int = 100_000
    rollout_steps: int = 1024
    epochs: int = 4
    batch_size: int = 128
    lr: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    hidden: int = 128
    seed: int = 0
    out: str = "checkpoints/ppo.pt"
    device: str = "auto"
    save_every: int = 20_000
    resume: str = ""


def parse_args(argv: list[str] | None = None) -> PPOConfig:
    return parse_policy_cli(PPOConfig, argv, description="Train PPO on SO101BowlingEnv.")


def _ckpt_siblings(out_path: Path) -> tuple[Path, Path]:
    suffix = out_path.suffix or ".pt"
    latest_path = out_path.parent / f"{out_path.stem}_latest{suffix}"
    best_path = out_path.parent / f"{out_path.stem}_best{suffix}"
    return latest_path, best_path


def _save_checkpoint(
    path: Path,
    model: ActorCritic,
    opt: torch.optim.Optimizer,
    obs_norm: RunningObsNorm,
    obs_dim: int,
    act_dim: int,
    cfg: PPOConfig,
    global_step: int,
    best_ret: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor_state_dict": {k: v.detach().cpu() for k, v in model.actor.state_dict().items()},
            "critic_state_dict": {k: v.detach().cpu() for k, v in model.critic.state_dict().items()},
            "optimizer_state_dict": opt.state_dict(),
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "obs_mean": torch.from_numpy(obs_norm.mean.astype(np.float32)),
            "obs_std": torch.from_numpy(obs_norm.std.astype(np.float32)),
            "obs_var": torch.from_numpy(obs_norm.var.astype(np.float32)),
            "obs_count": float(obs_norm.count),
            "global_step": int(global_step),
            "best_return": float(best_ret),
            "args": asdict(cfg),
        },
        path,
    )


def train(cfg: PPOConfig) -> Path:
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = resolve_device(cfg.device)

    env = SO101BowlingEnv(seed=cfg.seed)
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    obs_norm = RunningObsNorm(obs_dim)

    model = ActorCritic(obs_dim, act_dim, hidden=cfg.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    out_path = Path(cfg.out)
    suffix = out_path.suffix or ".pt"
    latest_path, best_path = _ckpt_siblings(out_path)

    global_step = 0
    best_ret = -float("inf")
    if cfg.resume:
        ckpt = torch.load(Path(cfg.resume), map_location=device, weights_only=True)
        model.actor.load_state_dict(ckpt["actor_state_dict"], strict=True)
        if "critic_state_dict" in ckpt:
            model.critic.load_state_dict(ckpt["critic_state_dict"], strict=True)
        if "optimizer_state_dict" in ckpt:
            opt.load_state_dict(ckpt["optimizer_state_dict"])
        obs_norm.mean = ckpt["obs_mean"].cpu().numpy().astype(np.float64)
        if "obs_var" in ckpt:
            obs_norm.var = ckpt["obs_var"].cpu().numpy().astype(np.float64)
        else:
            obs_norm.var = ckpt["obs_std"].cpu().numpy().astype(np.float64) ** 2
        obs_norm.count = float(ckpt.get("obs_count", 1.0))
        global_step = int(ckpt.get("global_step", 0))
        best_ret = float(ckpt.get("best_return", -float("inf")))

    rollout_n = cfg.rollout_steps
    batch_size = min(cfg.batch_size, rollout_n)
    remaining_steps = max(0, cfg.total_steps - global_step)
    num_updates = (remaining_steps + rollout_n - 1) // rollout_n
    total_updates = max(1, (cfg.total_steps + rollout_n - 1) // rollout_n)
    start_update = global_step // rollout_n
    next_save_step = ((global_step // cfg.save_every) + 1) * cfg.save_every if cfg.save_every > 0 else 0

    obs_buf = torch.empty((rollout_n, obs_dim), dtype=torch.float32)
    act_buf = torch.empty((rollout_n, act_dim), dtype=torch.float32)
    logp_buf = torch.empty(rollout_n, dtype=torch.float32)
    rew_buf = torch.empty(rollout_n, dtype=torch.float32)
    done_buf = torch.empty(rollout_n, dtype=torch.bool)
    val_buf = torch.empty(rollout_n, dtype=torch.float32)

    raw_obs, _ = env.reset(seed=cfg.seed)
    obs_norm.update(raw_obs)
    ep_ret = 0.0
    ep_count = 0
    recent: deque[tuple[float, bool, int, bool, bool]] = deque(maxlen=50)
    t0 = perf_counter()

    for update in range(1, num_updates + 1):
        if cfg.anneal_lr:
            frac = max(0.0, 1.0 - (start_update + update - 1.0) / total_updates)
            opt.param_groups[0]["lr"] = frac * cfg.lr

        model.eval()
        for t in range(rollout_n):
            norm_obs = obs_norm.normalize(raw_obs)
            x = torch.from_numpy(norm_obs).unsqueeze(0).to(device)
            with torch.no_grad():
                act, logp, _, val = model.get_action_and_value(x)
            act_np = np.clip(act.squeeze(0).cpu().numpy(), -1.0, 1.0)
            next_raw, reward, term, trunc, info = env.step(act_np)
            done = bool(term or trunc)

            obs_buf[t].copy_(torch.from_numpy(norm_obs))
            act_buf[t].copy_(act.squeeze(0).cpu())
            logp_buf[t] = logp.squeeze(0).cpu()
            rew_buf[t] = float(reward)
            done_buf[t] = done
            val_buf[t] = val.squeeze(0).cpu()

            ep_ret += float(reward)
            global_step += 1

            if done:
                ep_count += 1
                recent.append((
                    ep_ret,
                    bool(info.get("legal_delivery", False)),
                    int(info.get("credited_pins", 0)),
                    bool(info.get("legal_strike", False)),
                    bool(info.get("touched_ball", False)),
                ))
                ep_ret = 0.0
                raw_obs, _ = env.reset()
            else:
                raw_obs = next_raw
            obs_norm.update(raw_obs)

        with torch.no_grad():
            next_x = torch.from_numpy(obs_norm.normalize(raw_obs)).unsqueeze(0).to(device)
            next_val = model.value(next_x).squeeze(0).cpu()

        adv = torch.zeros(rollout_n, dtype=torch.float32)
        last_gae = 0.0
        for t in reversed(range(rollout_n)):
            non_term = 0.0 if done_buf[t] else 1.0
            nv = next_val if t == rollout_n - 1 else val_buf[t + 1]
            delta = rew_buf[t] + cfg.gamma * nv * non_term - val_buf[t]
            last_gae = float(delta) + cfg.gamma * cfg.gae_lambda * non_term * last_gae
            adv[t] = last_gae
        returns = adv + val_buf

        b_obs = obs_buf.to(device)
        b_act = act_buf.to(device)
        b_logp = logp_buf.to(device)
        b_adv = ((adv - adv.mean()) / (adv.std() + 1e-8)).to(device)
        b_ret = returns.to(device)

        model.train()
        for _ in range(cfg.epochs):
            perm = rng.permutation(rollout_n)
            for start in range(0, rollout_n, batch_size):
                mb = perm[start : start + batch_size]
                _, new_logp, ent, new_val = model.get_action_and_value(b_obs[mb], b_act[mb])
                ratio = (new_logp - b_logp[mb]).exp()
                pg1 = -b_adv[mb] * ratio
                pg2 = -b_adv[mb] * ratio.clamp(1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()
                v_loss = 0.5 * ((new_val - b_ret[mb]) ** 2).mean()
                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent.mean()

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                opt.step()

        if recent:
            m_ret = float(np.mean([r[0] for r in recent]))
            m_leg = float(np.mean([r[1] for r in recent]))
            m_pin = float(np.mean([r[2] for r in recent]))
            m_str = float(np.mean([r[3] for r in recent]))
            m_touch = float(np.mean([r[4] for r in recent]))
        else:
            m_ret = m_leg = m_pin = m_str = m_touch = 0.0

        if len(recent) >= 10 and m_ret > best_ret:
            best_ret = m_ret
            _save_checkpoint(best_path, model, opt, obs_norm, obs_dim, act_dim, cfg, global_step, best_ret)

        if cfg.save_every > 0 and global_step >= next_save_step:
            step_path = out_path.parent / f"{out_path.stem}_step_{global_step}{suffix}"
            _save_checkpoint(step_path, model, opt, obs_norm, obs_dim, act_dim, cfg, global_step, best_ret)
            _save_checkpoint(latest_path, model, opt, obs_norm, obs_dim, act_dim, cfg, global_step, best_ret)
            while global_step >= next_save_step:
                next_save_step += cfg.save_every

        if update == 1 or update % max(1, num_updates // 20) == 0 or update == num_updates:
            sps = global_step / max(1e-6, perf_counter() - t0)
            print(
                f"step={global_step:7d}/{cfg.total_steps} eps={ep_count:4d} "
                f"ret={m_ret:6.2f} touch={m_touch:.2f} legal={m_leg:.2f} pins={m_pin:.2f} "
                f"strike={m_str:.2f} sps={sps:.0f}",
                flush=True,
            )

    _save_checkpoint(out_path, model, opt, obs_norm, obs_dim, act_dim, cfg, global_step, best_ret)
    _save_checkpoint(latest_path, model, opt, obs_norm, obs_dim, act_dim, cfg, global_step, best_ret)
    print(f"Saved checkpoint to {out_path}", flush=True)
    return out_path


def main(argv: list[str] | None = None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
