"""Eval"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
import time

import mujoco.viewer
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import Actor  # noqa: E402

from so101_bowling import SO101BowlingEnv  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a PPO checkpoint or random policy.")
    p.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint from scripts/train.py.")
    p.add_argument("--random", action="store_true", help="Run a uniform random policy.")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gui", action="store_true", help="Launch passive MuJoCo viewer (use mjpython on macOS).")
    p.add_argument("--stochastic", action="store_true", help="Sample from policy instead of using mean action.")
    p.add_argument(
        "--reset-delay",
        type=float,
        default=0.0,
        help="Seconds to pause between episodes before resetting (e.g. for GUI inspection or real-world pin setup).",
    )
    return p.parse_args(argv)


def _load_policy(ckpt_path: str | Path) -> tuple[Actor, np.ndarray, np.ndarray]:
    ckpt = torch.load(Path(ckpt_path), map_location="cpu", weights_only=True)
    hidden = int(ckpt.get("args", {}).get("hidden", 128))
    actor = Actor(int(ckpt["obs_dim"]), int(ckpt["act_dim"]), hidden=hidden)
    actor.load_state_dict(ckpt["actor_state_dict"], strict=True)
    actor.eval()
    mean = ckpt["obs_mean"].numpy().astype(np.float32)
    std = ckpt["obs_std"].numpy().astype(np.float32)
    return actor, mean, std


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    use_random = args.random or args.checkpoint is None
    actor = obs_mean = obs_std = None
    if not use_random:
        actor, obs_mean, obs_std = _load_policy(args.checkpoint)

    rng = np.random.default_rng(args.seed)
    env = SO101BowlingEnv(seed=args.seed)
    viewer = mujoco.viewer.launch_passive(env.model, env.data) if args.gui else None

    returns: list[float] = []
    legals: list[bool] = []
    pins: list[int] = []
    strikes: list[bool] = []
    reasons: Counter[str] = Counter()

    try:
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + ep)
            ep_ret = 0.0
            steps = 0
            done = False
            info: dict = {}
            while not done:
                t_step = time.perf_counter()
                if use_random:
                    act = rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)
                else:
                    assert actor is not None and obs_mean is not None and obs_std is not None
                    norm_obs = np.clip((obs - obs_mean) / obs_std, -10.0, 10.0).astype(np.float32)
                    with torch.no_grad():
                        dist = actor.dist(torch.from_numpy(norm_obs).unsqueeze(0))
                        raw = dist.sample() if args.stochastic else dist.mean
                    act = np.clip(raw.squeeze(0).numpy(), -1.0, 1.0)

                obs, rew, term, trunc, info = env.step(act)
                ep_ret += float(rew)
                steps += 1
                done = bool(term or trunc)

                if viewer is not None:
                    if not viewer.is_running():
                        return {}
                    viewer.sync()
                    rem = env._control_dt - (time.perf_counter() - t_step)
                    if rem > 0.0:
                        time.sleep(rem)

            legal = bool(info.get("legal_delivery", False))
            credited = int(info.get("credited_pins", 0))
            strike = bool(info.get("legal_strike", False))
            reason = str(info.get("end_reason", "unknown"))
            returns.append(ep_ret)
            legals.append(legal)
            pins.append(credited)
            strikes.append(strike)
            reasons[reason] += 1
            print(
                f"ep={ep:02d} steps={steps:2d} ret={ep_ret:6.2f} "
                f"legal={int(legal)} pins={credited} strike={int(strike)} reason={reason}",
                flush=True,
            )
            if args.reset_delay > 0.0 and ep < args.episodes - 1:
                delay_end = time.perf_counter() + args.reset_delay
                while time.perf_counter() < delay_end:
                    if viewer is not None:
                        if not viewer.is_running():
                            return {}
                        viewer.sync()
                    time.sleep(min(0.05, max(0.0, delay_end - time.perf_counter())))
    finally:
        if viewer is not None:
            viewer.close()

    n = max(1, len(returns))
    summary = {
        "episodes": len(returns),
        "legal_rate": float(np.mean(legals)),
        "hit_rate": float(np.mean([p > 0 for p in pins])),
        "strike_rate": float(np.mean(strikes)),
        "mean_credited_pins": float(np.mean(pins)),
        "mean_return": float(np.mean(returns)),
        "end_reasons": dict(reasons),
    }
    print(
        f"SUMMARY ({'random' if use_random else args.checkpoint}): "
        f"legal={summary['legal_rate']:.2f} hit={summary['hit_rate']:.2f} "
        f"strike={summary['strike_rate']:.2f} pins={summary['mean_credited_pins']:.2f} "
        f"ret={summary['mean_return']:.2f} reasons={summary['end_reasons']}",
        flush=True,
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    evaluate(parse_args(argv))


if __name__ == "__main__":
    main()
