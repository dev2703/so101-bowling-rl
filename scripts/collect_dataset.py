"""CLI to collect dual-camera SO-101 bowling episode datasets."""

from __future__ import annotations

import argparse
import json

from so101_bowling.config import load_config
from so101_bowling.dataset import collect_dataset
from so101_bowling.ppo import resolve_device
from so101_bowling.ppo_models import load_actor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect dual-camera bowling episode dataset (scripted 4-DOF or trained actor)."
    )
    parser.add_argument("--out-dir", type=str, required=True, help="Output dataset directory.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to save.")
    parser.add_argument("--workspace-config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional exported actor .pt.")
    parser.add_argument("--policy-name", type=str, default=None, help="Label for dataset_info.json.")
    parser.add_argument("--dr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render-rgb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--frame-stack", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--only-legal", action="store_true", help="Keep only legal deliveries.")
    parser.add_argument("--no-jitter", action="store_true", help="Disable scripted push param jitter.")
    args = parser.parse_args()

    wcfg = load_config(args.workspace_config)
    actor = None
    frame_stack = args.frame_stack
    if args.checkpoint is not None:
        device = resolve_device(args.device)
        actor = load_actor(args.checkpoint, map_location=device)
        frame_stack = actor.frame_stack

    info = collect_dataset(
        out_dir=args.out_dir,
        num_episodes=args.episodes,
        cfg=wcfg,
        dr_enabled=args.dr,
        render_rgb=args.render_rgb,
        frame_stack=frame_stack,
        seed=args.seed,
        actor=actor,
        policy_name=args.policy_name,
        jitter_scripted=not args.no_jitter,
        only_legal=args.only_legal,
    )
    summary = {k: v for k, v in info.items() if k != "workspace_config"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
