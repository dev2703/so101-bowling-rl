"""CLI to train dual-camera Asymmetric SAC on SO-101 bowling."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace

from so101_bowling.config import load_config
from so101_bowling.sac import load_sac_config, preflight_sac, train_sac
from so101_bowling.tracking import WandbConfig, parse_wandb_tags


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train dual-camera Asymmetric SAC on SO-101 bowling.")
    parser.add_argument("--config", type=str, default=None, help="Path to SAC YAML config.")
    parser.add_argument("--workspace-config", type=str, default=None, help="Path to workspace YAML.")
    parser.add_argument("--out-dir", "--run-dir", dest="out_dir", type=str, default="runs/sac", help="Output directory for logs/checkpoints.")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--capacity", type=int, default=None)
    parser.add_argument("--learning-starts", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--utd-ratio", type=float, default=None)
    parser.add_argument("--eval-interval-steps", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--encoder-type", choices=("spatial_grid", "spatial_softmax"), default=None)
    parser.add_argument("--dr", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable DR.")
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--no-video", action="store_true", help="Disable evaluation video recording.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint_latest.pt to resume.")
    parser.add_argument("--demo-dataset", type=str, default=None, help="Path to collected dataset dir to prefill replay buffer.")
    parser.add_argument("--pretrain-steps", type=int, default=None, help="Offline SAC updates on demo buffer before online loop.")
    parser.add_argument("--preflight", action="store_true", help="Run single-batch overfit & NaN/Inf preflight.")
    parser.add_argument("--preflight-steps", type=int, default=150)
    parser.add_argument("--preflight-batch-size", type=int, default=None)
    parser.add_argument("--wandb-project", type=str, default=None, help="W&B project name (enables W&B logging).")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity/team name.")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run display name.")
    parser.add_argument("--wandb-group", type=str, default=None, help="W&B run group.")
    parser.add_argument("--wandb-tags", type=str, default=None, help="Comma-separated W&B tags.")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args(argv)

    sac_cfg = load_sac_config(args.config)
    overrides: dict[str, object] = {}
    for field_name in (
        "total_timesteps",
        "num_envs",
        "capacity",
        "learning_starts",
        "batch_size",
        "utd_ratio",
        "eval_interval_steps",
        "eval_episodes",
        "seed",
        "device",
        "encoder_type",
        "save_video",
    ):
        val = getattr(args, field_name)
        if val is not None:
            overrides[field_name] = val
    if args.pretrain_steps is not None:
        overrides["demo_pretrain_steps"] = args.pretrain_steps
    if args.dr is not None:
        overrides["dr_enabled"] = args.dr
    if args.no_video:
        overrides["save_video"] = False
    if overrides:
        sac_cfg = replace(sac_cfg, **overrides)

    wcfg = load_config(args.workspace_config)
    if args.preflight:
        report = preflight_sac(
            cfg=sac_cfg,
            workspace_cfg=wcfg,
            steps=args.preflight_steps,
            batch_size=args.preflight_batch_size,
        )
        print(json.dumps(report, indent=2), flush=True)
        if not report["passed"] or report["nan_inf_detected"]:
            sys.exit(1)
        return

    wandb_cfg = (
        WandbConfig(
            project=args.wandb_project,
            entity=args.wandb_entity,
            run_name=args.wandb_run_name,
            group=args.wandb_group,
            tags=parse_wandb_tags(args.wandb_tags),
            mode=args.wandb_mode,
        )
        if args.wandb_project
        else None
    )
    summary = train_sac(
        cfg=sac_cfg,
        workspace_cfg=wcfg,
        run_dir=args.out_dir,
        resume_from=args.resume,
        demo_dataset=args.demo_dataset,
        wandb_cfg=wandb_cfg,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
