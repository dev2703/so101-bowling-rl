"""Minimal self-contained PyTorch PPO trainer for SO101BowlingEnv."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from time import perf_counter

import gymnasium as gym
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
    num_envs: int = 1
    eval_every: int = 10_000
    eval_episodes: int = 10
    dr: bool = True


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
            "best_return_kind": "eval",
            "args": asdict(cfg),
        },
        path,
    )


def _sync_rollout_model(rollout_model: ActorCritic, model: ActorCritic) -> None:
    if rollout_model is not model:
        rollout_model.load_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items()})
    rollout_model.eval()


def _make_env(dr: bool) -> SO101BowlingEnv:
    """Top-level so AsyncVectorEnv workers can unpickle it under the spawn start method."""
    return SO101BowlingEnv(dr=dr)


def _make_vector_env(cfg: PPOConfig, num_envs: int) -> gym.vector.VectorEnv:
    env_fns = [partial(_make_env, dr=cfg.dr) for _ in range(num_envs)]
    vector_cls = gym.vector.AsyncVectorEnv if num_envs > 1 else gym.vector.SyncVectorEnv
    return vector_cls(env_fns, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)


def _evaluate_greedy(
    env: SO101BowlingEnv, model: ActorCritic, obs_norm: RunningObsNorm, cfg: PPOConfig
) -> tuple[float, float, float, float, float]:
    """Mean (return, touch, legal, pins, strike) of the mean-action policy on fixed seeds."""
    results = []
    for ep in range(cfg.eval_episodes):
        obs, _ = env.reset(seed=10_000 + cfg.seed * 100 + ep)
        ep_ret, done, info = 0.0, False, {}
        while not done:
            x = torch.from_numpy(obs_norm.normalize(obs)).unsqueeze(0)
            with torch.no_grad():
                act = model.actor.dist(x).mean.squeeze(0).numpy()
            obs, reward, term, trunc, info = env.step(np.clip(act, -1.0, 1.0))
            ep_ret += float(reward)
            done = bool(term or trunc)
        results.append((
            ep_ret,
            info["touched_ball"],
            info["legal_delivery"],
            info["credited_pins"],
            info["legal_strike"],
        ))
    return tuple(float(v) for v in np.mean(np.asarray(results, dtype=np.float64), axis=0))


def train(cfg: PPOConfig) -> Path:
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = resolve_device(cfg.device)

    num_envs = max(1, cfg.num_envs)
    envs = eval_env = None
    try:
        envs = _make_vector_env(cfg, num_envs)
        obs_dim = int(envs.single_observation_space.shape[0])
        act_dim = int(envs.single_action_space.shape[0])
        obs_norm = RunningObsNorm(obs_dim)

        model = ActorCritic(obs_dim, act_dim, hidden=cfg.hidden).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)
        # Small-batch inference is faster on CPU, so accelerators get a CPU mirror for rollouts.
        rollout_model = model if device.type == "cpu" else ActorCritic(obs_dim, act_dim, hidden=cfg.hidden)

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
            best_ret = float(ckpt.get("best_return", -float("inf"))) if ckpt.get("best_return_kind") == "eval" else -float("inf")

        steps_per_env = max(1, cfg.rollout_steps // num_envs)
        batch_steps = steps_per_env * num_envs
        batch_size = min(cfg.batch_size, batch_steps)
        remaining_steps = max(0, cfg.total_steps - global_step)
        num_updates = (remaining_steps + batch_steps - 1) // batch_steps
        total_updates = max(1, (cfg.total_steps + batch_steps - 1) // batch_steps)
        start_update = global_step // batch_steps
        next_save_step = ((global_step // cfg.save_every) + 1) * cfg.save_every if cfg.save_every > 0 else 0
        next_eval_step = ((global_step // cfg.eval_every) + 1) * cfg.eval_every if cfg.eval_every > 0 else 0
        if cfg.eval_every > 0:
            eval_env = SO101BowlingEnv(dr=cfg.dr)

        obs_buf = torch.empty((steps_per_env, num_envs, obs_dim), dtype=torch.float32)
        act_buf = torch.empty((steps_per_env, num_envs, act_dim), dtype=torch.float32)
        logp_buf = torch.empty((steps_per_env, num_envs), dtype=torch.float32)
        rew_buf = torch.empty((steps_per_env, num_envs), dtype=torch.float32)
        done_buf = torch.empty((steps_per_env, num_envs), dtype=torch.bool)
        val_buf = torch.empty((steps_per_env, num_envs), dtype=torch.float32)

        raw_obs, _ = envs.reset(seed=[cfg.seed + i for i in range(num_envs)])
        obs_norm.update(raw_obs)
        ep_rets = np.zeros(num_envs, dtype=np.float64)
        ep_count = 0
        # (ep_ret, legal, pins, strike, touched, end_reason,
        #  shaping, contact, cost, launch, approach, pin_rew, penalty)
        recent: deque[tuple] = deque(maxlen=50)
        t0 = perf_counter()

        for update in range(1, num_updates + 1):
            if cfg.anneal_lr:
                frac = max(0.0, 1.0 - (start_update + update - 1.0) / total_updates)
                opt.param_groups[0]["lr"] = frac * cfg.lr

            _sync_rollout_model(rollout_model, model)
            for t in range(steps_per_env):
                norm_obs = obs_norm.normalize(raw_obs)
                x = torch.from_numpy(norm_obs)
                with torch.no_grad():
                    act, logp, _, val = rollout_model.get_action_and_value(x)
                act_np = np.clip(act.numpy(), -1.0, 1.0)
                # With SAME_STEP autoreset, finished envs already return their reset observation.
                raw_obs, reward, term, trunc, infos = envs.step(act_np)
                done = np.logical_or(term, trunc)

                obs_buf[t].copy_(x)
                act_buf[t].copy_(act)
                logp_buf[t].copy_(logp)
                rew_buf[t].copy_(torch.from_numpy(np.asarray(reward, dtype=np.float32)))
                done_buf[t].copy_(torch.from_numpy(done))
                val_buf[t].copy_(val)

                ep_rets += reward
                global_step += num_envs

                for i in np.flatnonzero(done):
                    if not infos["_final_info"][i]:
                        continue
                    # final_info is columnar: one array per key, indexed by env.
                    fi = infos["final_info"]
                    outcome = fi["outcome"]
                    ep_count += 1
                    recent.append((
                        float(ep_rets[i]),
                        bool(fi["legal_delivery"][i]),
                        int(fi["credited_pins"][i]),
                        bool(fi["legal_strike"][i]),
                        bool(fi["touched_ball"][i]),
                        str(fi["end_reason"][i]),
                        float(outcome["shaping_reward"][i]),
                        float(outcome["contact_reward"][i]),
                        float(outcome["cost_reward"][i]),
                        float(outcome["launch_reward"][i]),
                        float(outcome["approach_reward"][i]),
                        float(outcome["pin_reward"][i]),
                        float(outcome["penalty_reward"][i]),
                    ))
                    ep_rets[i] = 0.0
                obs_norm.update(raw_obs)

            with torch.no_grad():
                next_val = rollout_model.value(torch.from_numpy(obs_norm.normalize(raw_obs)))

            adv = torch.zeros((steps_per_env, num_envs), dtype=torch.float32)
            last_gae = torch.zeros(num_envs, dtype=torch.float32)
            for t in reversed(range(steps_per_env)):
                non_term = (~done_buf[t]).float()
                nv = next_val if t == steps_per_env - 1 else val_buf[t + 1]
                delta = rew_buf[t] + cfg.gamma * nv * non_term - val_buf[t]
                last_gae = delta + cfg.gamma * cfg.gae_lambda * non_term * last_gae
                adv[t] = last_gae
            returns = adv + val_buf

            adv = adv.flatten()
            b_obs = obs_buf.flatten(0, 1).to(device)
            b_act = act_buf.flatten(0, 1).to(device)
            b_logp = logp_buf.flatten().to(device)
            b_adv = ((adv - adv.mean()) / (adv.std() + 1e-8)).to(device)
            b_ret = returns.flatten().to(device)

            model.train()
            for _ in range(cfg.epochs):
                perm = rng.permutation(batch_steps)
                for start in range(0, batch_steps, batch_size):
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
                m_parts = [float(np.mean([r[k] for r in recent])) for k in range(6, 13)]
                top = ",".join(
                    f"{reason}:{count / len(recent):.0%}"
                    for reason, count in Counter(r[5] for r in recent).most_common(2)
                )
            else:
                m_ret = m_leg = m_pin = m_str = m_touch = 0.0
                m_parts = [0.0] * 7
                top = "none"

            if cfg.eval_every > 0 and global_step >= next_eval_step:
                _sync_rollout_model(rollout_model, model)
                det_ret, e_touch, e_leg, e_pin, e_str = _evaluate_greedy(eval_env, rollout_model, obs_norm, cfg)
                print(
                    f"EVAL step={global_step:7d} det_ret={det_ret:6.2f} touch={e_touch:.2f} "
                    f"legal={e_leg:.2f} pins={e_pin:.2f} strike={e_str:.2f}",
                    flush=True,
                )
                if det_ret > best_ret:
                    best_ret = det_ret
                    _save_checkpoint(best_path, model, opt, obs_norm, obs_dim, act_dim, cfg, global_step, best_ret)
                while global_step >= next_eval_step:
                    next_eval_step += cfg.eval_every

            if cfg.save_every > 0 and global_step >= next_save_step:
                step_path = out_path.parent / f"{out_path.stem}_step_{global_step}{suffix}"
                _save_checkpoint(step_path, model, opt, obs_norm, obs_dim, act_dim, cfg, global_step, best_ret)
                _save_checkpoint(latest_path, model, opt, obs_norm, obs_dim, act_dim, cfg, global_step, best_ret)
                while global_step >= next_save_step:
                    next_save_step += cfg.save_every

            if update == 1 or update % max(1, num_updates // 20) == 0 or update == num_updates:
                sps = global_step / max(1e-6, perf_counter() - t0)
                shp, cnt, cst, lch, app, pin, pen = m_parts
                print(
                    f"step={global_step:7d}/{cfg.total_steps} eps={ep_count:4d} "
                    f"ret={m_ret:6.2f} touch={m_touch:.2f} legal={m_leg:.2f} pins={m_pin:.2f} "
                    f"strike={m_str:.2f} sps={sps:.0f} "
                    f"| R[shp={shp:.2f} cnt={cnt:.2f} cst={cst:.2f} lch={lch:.2f} "
                    f"app={app:.2f} pin={pin:.2f} pen={pen:.2f}] top={top}",
                    flush=True,
                )

        _save_checkpoint(out_path, model, opt, obs_norm, obs_dim, act_dim, cfg, global_step, best_ret)
        _save_checkpoint(latest_path, model, opt, obs_norm, obs_dim, act_dim, cfg, global_step, best_ret)
        print(f"Saved checkpoint to {out_path}", flush=True)
        return out_path
    finally:
        if envs is not None:
            envs.close()
        if eval_env is not None:
            eval_env.close()


def main(argv: list[str] | None = None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
