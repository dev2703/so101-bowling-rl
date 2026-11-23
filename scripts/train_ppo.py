"""CLI to train the dual-camera Asymmetric PPO baseline."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

from so101_bowling.config import load_config
from so101_bowling.ppo import load_ppo_config, train_ppo


def main() -> None:
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
    args = parser.parse_args()

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
    summary = train_ppo(
        ppo_cfg=ppo_cfg,
        workspace_cfg=wcfg,
        run_dir=args.run_dir,
        resume_from=args.resume,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
