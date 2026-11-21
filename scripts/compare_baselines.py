"""Paired-seed baseline comparison CLI: scripted 4-DOF vs PPO vs SAC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from so101_bowling.config import WorkspaceConfig, load_config
from so101_bowling.env import SO101BowlingEnv
from so101_bowling.kinematics import ScriptedPushController
from so101_bowling.ppo import evaluate_actor, evaluate_camera_ablation, resolve_device
from so101_bowling.ppo_models import load_actor


def evaluate_scripted_4dof(
    cfg: WorkspaceConfig,
    dr_enabled: bool = False,
    episodes: int = 10,
    seed: int = 1000,
) -> dict[str, Any]:
    """Evaluate privileged ScriptedPushController across paired seeds."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")

    strikes = 0
    illegals = 0
    credited_list: list[int] = []
    full_knock_list: list[int] = []
    wobble_only_list: list[int] = []
    returns: list[float] = []

    with SO101BowlingEnv(
        cfg=cfg,
        dr_enabled=dr_enabled,
        render_rgb=False,
        frame_stack=2,
        gamma=cfg.reward.gamma,
    ) as env:
        controller = ScriptedPushController(env.model, env.ids, cfg, env.q_min, env.q_max)
        for ep in range(episodes):
            _, _ = env.reset(seed=seed + ep)
            controller.reset(env._last_snapshot.ball_pos, env._last_snapshot.pin_positions)
            ep_ret = 0.0
            done = False
            info: dict[str, Any] = {}
            while not done:
                action = controller.step(env._measured_q())
                _, reward, terminated, truncated, info = env.step(action)
                ep_ret += float(reward)
                done = terminated or truncated

            outcome = info.get("outcome", {})
            legal = bool(outcome.get("legal_delivery", False))
            strikes += int(bool(outcome.get("legal_strike", False)))
            illegals += int(not legal)
            credited_list.append(int(outcome.get("credited_pins", 0)))
            full_knock_list.append(int(outcome.get("full_knock_pins", 0)))
            wobble_only_list.append(int(outcome.get("wobble_only_pins", 0)))
            returns.append(ep_ret)

    n = len(returns)
    return {
        "privileged_state": True,
        "episodes": n,
        "legal_strike_rate": strikes / n,
        "mean_credited_pins": float(np.mean(credited_list)),
        "mean_full_knock_pins": float(np.mean(full_knock_list)),
        "mean_wobble_only_pins": float(np.mean(wobble_only_list)),
        "illegal_rate": illegals / n,
        "mean_return": float(np.mean(returns)),
    }


def _evaluate_actor_baseline(
    checkpoint_path: str | Path,
    cfg: WorkspaceConfig,
    dr_enabled: bool,
    episodes: int,
    seed: int,
    device: Any,
    ablate_cameras: bool,
) -> dict[str, Any]:
    actor = load_actor(checkpoint_path, map_location=device)
    metrics = evaluate_actor(
        actor=actor,
        cfg=cfg,
        dr_enabled=dr_enabled,
        frame_stack=actor.frame_stack,
        gamma=cfg.reward.gamma,
        episodes=episodes,
        seed=seed,
        device=device,
    )
    metrics["privileged_state"] = False
    if ablate_cameras:
        metrics["camera_ablation"] = evaluate_camera_ablation(
            actor=actor,
            cfg=cfg,
            dr_enabled=dr_enabled,
            frame_stack=actor.frame_stack,
            gamma=cfg.reward.gamma,
            episodes=episodes,
            seed=seed,
            device=device,
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare scripted 4-DOF, PPO, and SAC baselines on paired seeds.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--workspace-config", type=str, default=None)
    parser.add_argument("--ppo-checkpoint", type=str, default=None)
    parser.add_argument("--sac-checkpoint", type=str, default=None)
    parser.add_argument("--dr", action="store_true", help="Evaluate with domain randomization enabled.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--ablate-cameras", action="store_true", help="Also run camera ablation on actor checkpoints.")
    parser.add_argument("--out-json", type=str, default=None, help="Optional path to write JSON comparison report.")
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    ws_cfg = load_config(args.workspace_config)
    device = resolve_device(args.device)

    report: dict[str, Any] = {
        "seed": args.seed,
        "episodes": args.episodes,
        "dr_enabled": bool(args.dr),
        "scripted_4dof": evaluate_scripted_4dof(
            cfg=ws_cfg,
            dr_enabled=args.dr,
            episodes=args.episodes,
            seed=args.seed,
        ),
    }

    if args.ppo_checkpoint is not None:
        report["ppo"] = _evaluate_actor_baseline(
            checkpoint_path=args.ppo_checkpoint,
            cfg=ws_cfg,
            dr_enabled=args.dr,
            episodes=args.episodes,
            seed=args.seed,
            device=device,
            ablate_cameras=args.ablate_cameras,
        )

    if args.sac_checkpoint is not None:
        report["sac"] = _evaluate_actor_baseline(
            checkpoint_path=args.sac_checkpoint,
            cfg=ws_cfg,
            dr_enabled=args.dr,
            episodes=args.episodes,
            seed=args.seed,
            device=device,
            ablate_cameras=args.ablate_cameras,
        )

    payload = json.dumps(report, indent=2)
    if args.out_json is not None:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload)
    print(payload)


if __name__ == "__main__":
    main()
