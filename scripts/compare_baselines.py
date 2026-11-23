"""Paired-seed baseline comparison CLI: scripted 4-DOF vs PPO vs SAC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from so101_bowling.config import WorkspaceConfig, load_config
from so101_bowling.env import SO101BowlingEnv
from so101_bowling.hub import _ACTOR_CANDIDATES
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
    steps_list: list[int] = []

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
            ep_steps = 0
            done = False
            info: dict[str, Any] = {}
            while not done:
                ep_steps += 1
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
            steps_list.append(ep_steps)

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
        "mean_episode_steps": float(np.mean(steps_list)),
    }


def _resolve_baseline_paths(
    checkpoint_arg: str | None,
    run_dir_arg: str | None,
) -> tuple[Path, Path]:
    if checkpoint_arg is not None:
        ckpt_path = Path(checkpoint_arg)
        rdir = Path(run_dir_arg) if run_dir_arg is not None else ckpt_path.parent
        return ckpt_path, rdir
    assert run_dir_arg is not None
    rdir = Path(run_dir_arg)
    for cand in _ACTOR_CANDIDATES:
        candidate = rdir / cand
        if candidate.exists():
            return candidate, rdir
    raise FileNotFoundError(
        f"No actor checkpoint from {_ACTOR_CANDIDATES} found in run directory {rdir}"
    )


def _load_training_metrics(run_dir: Path) -> dict[str, Any] | None:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    keys = (
        "global_step",
        "total_timesteps",
        "final_losses",
        "eval_aux_loss",
        "nan_inf_detected",
        "validation_loss_note",
        "best_eval",
    )
    return {k: summary[k] for k in keys if k in summary}


def _evaluate_actor_baseline(
    checkpoint_path: str | Path,
    cfg: WorkspaceConfig,
    dr_enabled: bool,
    episodes: int,
    seed: int,
    device: Any,
    ablate_cameras: bool,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    ckpt = Path(checkpoint_path)
    actor = load_actor(ckpt, map_location=device)
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
    training_metrics = _load_training_metrics(run_dir if run_dir is not None else ckpt.parent)
    if training_metrics is not None:
        metrics["training_metrics"] = training_metrics
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare scripted 4-DOF, PPO, and SAC baselines on paired seeds.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--workspace-config", type=str, default=None)
    parser.add_argument("--ppo-checkpoint", type=str, default=None)
    parser.add_argument("--ppo-run-dir", type=str, default=None)
    parser.add_argument("--sac-checkpoint", type=str, default=None)
    parser.add_argument("--sac-run-dir", type=str, default=None)
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

    if args.ppo_checkpoint is not None or args.ppo_run_dir is not None:
        ppo_ckpt, ppo_rdir = _resolve_baseline_paths(args.ppo_checkpoint, args.ppo_run_dir)
        report["ppo"] = _evaluate_actor_baseline(
            checkpoint_path=ppo_ckpt,
            cfg=ws_cfg,
            dr_enabled=args.dr,
            episodes=args.episodes,
            seed=args.seed,
            device=device,
            ablate_cameras=args.ablate_cameras,
            run_dir=ppo_rdir,
        )

    if args.sac_checkpoint is not None or args.sac_run_dir is not None:
        sac_ckpt, sac_rdir = _resolve_baseline_paths(args.sac_checkpoint, args.sac_run_dir)
        report["sac"] = _evaluate_actor_baseline(
            checkpoint_path=sac_ckpt,
            cfg=ws_cfg,
            dr_enabled=args.dr,
            episodes=args.episodes,
            seed=args.seed,
            device=device,
            ablate_cameras=args.ablate_cameras,
            run_dir=sac_rdir,
        )

    payload = json.dumps(report, indent=2)
    if args.out_json is not None:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload)
    print(payload)


if __name__ == "__main__":
    main()
