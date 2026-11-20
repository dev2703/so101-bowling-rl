"""CLI to evaluate a standalone exported DualCameraActor checkpoint."""

from __future__ import annotations

import argparse
import json

from so101_bowling.config import load_config
from so101_bowling.ppo import evaluate_actor, evaluate_camera_ablation, resolve_device
from so101_bowling.ppo_models import load_actor


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate standalone exported DualCameraActor.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to exported actor .pt.")
    parser.add_argument("--workspace-config", type=str, default=None)
    parser.add_argument("--dr", action="store_true", help="Evaluate with domain randomization enabled.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--ablate-cameras", action="store_true", help="Also eval with wrist/tripod occluded.")
    parser.add_argument("--video-out", type=str, default=None, help="Optional side-by-side MP4 path.")
    args = parser.parse_args()

    wcfg = load_config(args.workspace_config)
    device = resolve_device(args.device)
    actor = load_actor(args.checkpoint, map_location=device)

    if args.ablate_cameras:
        results = evaluate_camera_ablation(
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

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
