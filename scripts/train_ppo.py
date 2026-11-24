"""CLI to train the dual-camera Asymmetric PPO baseline."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace

from so101_bowling.config import load_config
from so101_bowling.ppo import load_ppo_config, preflight_ppo, train_ppo
from so101_bowling.tracking import WandbConfig, parse_wandb_tags


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train dual-camera Asymmetric PPO on SO-101 bowling.")
    parser.add_argument("--config", type=str, default=None, help="Path to PPO YAML config.")
    parser.add_argument("--workspace-config", type=str, default=None, help="Path to workspace YAML.")
    parser.add_argument("--dr", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable DR.")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--encoder-type", choices=("spatial_grid", "spatial_softmax"), default=None)
    parser.add_argument("--run-dir", type=str, default="runs/ppo")
    parser.add_argument("--resume", type=str, default=None, help="Path to latest.pt checkpoint to resume.")
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=None)
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

    ppo_cfg = load_ppo_config(args.config)
    overrides = {}
    for field_name in (
        "total_timesteps",
        "num_envs",
        "rollout_steps",
        "seed",
        "device",
        "encoder_type",
        "save_video",
    ):
        val = getattr(args, field_name)
        if val is not None:
            overrides[field_name] = val
    if args.dr is not None:
        overrides["dr_enabled"] = args.dr
    if overrides:
        ppo_cfg = replace(ppo_cfg, **overrides)

    wcfg = load_config(args.workspace_config)
    if args.preflight:
        report = preflight_ppo(
            ppo_cfg=ppo_cfg,
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
    summary = train_ppo(
        ppo_cfg=ppo_cfg,
        workspace_cfg=wcfg,
        run_dir=args.run_dir,
        resume_from=args.resume,
        wandb_cfg=wandb_cfg,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
