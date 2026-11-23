"""End-to-end headless GPU training runner for Colab Pro (with status file & HF Hub push)."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _write_status(path: Path, state: str, extra: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {
        "state": state,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", choices=("sac", "ppo"), default="sac")
    parser.add_argument("--config", type=str, default=None, help="Path to SAC or PPO YAML config.")
    parser.add_argument("--workspace-config", type=str, default=None)
    parser.add_argument("--run-dir", type=str, default="runs/colab_run")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from.")
    parser.add_argument(
        "--collect-demos",
        type=int,
        default=0,
        help="If > 0, collect N scripted 4-DOF demo episodes before training (and warm-start SAC).",
    )
    parser.add_argument("--demo-dir", type=str, default=None, help="Directory for demo dataset.")
    parser.add_argument("--hf-model-repo", type=str, default=None, help="HF model repo to push on completion.")
    parser.add_argument("--hf-dataset-repo", type=str, default=None, help="HF dataset repo to push demos.")
    parser.add_argument("--hf-private", action="store_true")
    parser.add_argument("--mujoco-gl", type=str, default="egl", help="MUJOCO_GL backend (default: egl).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mujoco_gl and "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = args.mujoco_gl

    from so101_bowling.config import load_config
    from so101_bowling.dataset import collect_dataset
    from so101_bowling.hub import push_dataset_to_hub, push_model_to_hub
    from so101_bowling.ppo import load_ppo_config, train_ppo
    from so101_bowling.sac import load_sac_config, train_sac

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    _write_status(status_path, "running", {"algo": args.algo, "run_dir": str(run_dir)})

    try:
        wcfg = load_config(args.workspace_config)
        demo_dir = Path(args.demo_dir) if args.demo_dir else (run_dir / "demos")
        demo_info = None

        if args.collect_demos > 0:
            _write_status(status_path, "collecting_demos", {"num_episodes": args.collect_demos})
            demo_info = collect_dataset(
                out_dir=demo_dir,
                num_episodes=args.collect_demos,
                cfg=wcfg,
                dr_enabled=args.dr,
                render_rgb=True,
                seed=args.seed or 0,
                only_legal=True,
            )
            if args.hf_dataset_repo:
                push_dataset_to_hub(
                    dataset_dir=demo_dir,
                    repo_id=args.hf_dataset_repo,
                    private=args.hf_private,
                )

        _write_status(status_path, "training", {"algo": args.algo})
        if args.algo == "sac":
            sac_cfg = load_sac_config(args.config)
            overrides: dict[str, Any] = {"dr_enabled": args.dr}
            if args.total_timesteps is not None:
                overrides["total_timesteps"] = args.total_timesteps
            if args.num_envs is not None:
                overrides["num_envs"] = args.num_envs
            if args.seed is not None:
                overrides["seed"] = args.seed
            sac_cfg = replace(sac_cfg, **overrides)
            summary = train_sac(
                cfg=sac_cfg,
                workspace_cfg=wcfg,
                run_dir=run_dir,
                resume_from=args.resume,
                demo_dataset=demo_dir if demo_dir.exists() and (demo_dir / "episodes").is_dir() else None,
            )
        else:
            ppo_cfg = load_ppo_config(args.config)
            overrides = {"dr_enabled": args.dr}
            if args.total_timesteps is not None:
                overrides["total_timesteps"] = args.total_timesteps
            if args.num_envs is not None:
                overrides["num_envs"] = args.num_envs
            if args.seed is not None:
                overrides["seed"] = args.seed
            ppo_cfg = replace(ppo_cfg, **overrides)
            summary = train_ppo(
                ppo_cfg=ppo_cfg,
                workspace_cfg=wcfg,
                run_dir=run_dir,
                resume_from=args.resume,
            )

        model_url = None
        if args.hf_model_repo:
            _write_status(status_path, "pushing_model", {"repo_id": args.hf_model_repo})
            model_url = push_model_to_hub(
                run_dir=run_dir,
                repo_id=args.hf_model_repo,
                private=args.hf_private,
            )

        _write_status(
            status_path,
            "completed",
            {
                "algo": args.algo,
                "summary": summary,
                "demo_episodes": demo_info["num_episodes"] if demo_info else 0,
                "hf_model_url": model_url,
            },
        )
        print(json.dumps({"state": "completed", "summary": summary, "hf_model_url": model_url}, indent=2))
    except Exception as exc:
        _write_status(
            status_path,
            "failed",
            {"error": str(exc), "traceback": traceback.format_exc()},
        )
        raise


if __name__ == "__main__":
    main()
