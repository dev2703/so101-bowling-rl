"""CLI to train dual-camera Asymmetric SAC on SO-101 bowling."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

from so101_bowling.config import load_config
from so101_bowling.sac import load_sac_config, train_sac


def main() -> None:
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
    args = parser.parse_args()

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
    if args.dr is not None:
        overrides["dr_enabled"] = args.dr
    if args.no_video:
        overrides["save_video"] = False
    if overrides:
        sac_cfg = replace(sac_cfg, **overrides)

    wcfg = load_config(args.workspace_config)
    summary = train_sac(
        cfg=sac_cfg,
        workspace_cfg=wcfg,
        run_dir=args.out_dir,
        resume_from=args.resume,
        demo_dataset=args.demo_dataset,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
