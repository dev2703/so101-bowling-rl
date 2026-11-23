"""CLI to evaluate a standalone exported DualCameraActor checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from so101_bowling.config import load_config
from so101_bowling.hub import _ACTOR_CANDIDATES
from so101_bowling.ppo import evaluate_actor, evaluate_camera_ablation, resolve_device
from so101_bowling.ppo_models import load_actor


def _resolve_checkpoint_path(checkpoint: str | None, run_dir: str | None) -> tuple[Path, Path]:
    if checkpoint is not None:
        ckpt_path = Path(checkpoint)
        rdir = Path(run_dir) if run_dir is not None else ckpt_path.parent
        return ckpt_path, rdir
    if run_dir is None:
        raise ValueError("Either --checkpoint or --run-dir must be provided")
    rdir = Path(run_dir)
    for cand in _ACTOR_CANDIDATES:
        candidate_path = rdir / cand
        if candidate_path.exists():
            return candidate_path, rdir
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate standalone exported DualCameraActor.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to exported actor .pt.")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Training run directory containing actor checkpoint and summary.json.",
    )
    parser.add_argument("--workspace-config", type=str, default=None)
    parser.add_argument("--dr", action="store_true", help="Evaluate with domain randomization enabled.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--ablate-cameras", action="store_true", help="Also eval with wrist/tripod occluded.")
    parser.add_argument("--video-out", type=str, default=None, help="Optional side-by-side MP4 path.")
    args = parser.parse_args(argv)

    if args.checkpoint is None and args.run_dir is None:
        parser.error("Provide --checkpoint or --run-dir")

    ckpt_path, run_dir = _resolve_checkpoint_path(args.checkpoint, args.run_dir)
    wcfg = load_config(args.workspace_config)
    device = resolve_device(args.device)
    actor = load_actor(ckpt_path, map_location=device)

    if args.ablate_cameras:
        results: dict[str, Any] = evaluate_camera_ablation(
            actor=actor,
            cfg=wcfg,
            dr_enabled=args.dr,
            frame_stack=actor.frame_stack,
            gamma=wcfg.reward.gamma,
            episodes=args.episodes,
            seed=args.seed,
            device=device,
        )
        if args.video_out is not None:
            evaluate_actor(
                actor=actor,
                cfg=wcfg,
                dr_enabled=args.dr,
                frame_stack=actor.frame_stack,
                gamma=wcfg.reward.gamma,
                episodes=1,
                seed=args.seed,
                device=device,
                video_path=args.video_out,
            )
    else:
        results = evaluate_actor(
            actor=actor,
            cfg=wcfg,
            dr_enabled=args.dr,
            frame_stack=actor.frame_stack,
            gamma=wcfg.reward.gamma,
            episodes=args.episodes,
            seed=args.seed,
            device=device,
            video_path=args.video_out,
        )

    training_metrics = _load_training_metrics(run_dir)
    if training_metrics is not None:
        results["training_metrics"] = training_metrics

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
