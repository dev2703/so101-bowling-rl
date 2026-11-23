"""End-to-end headless GPU training runner for Colab Pro (with status file & HF Hub push)."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
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
    parser.add_argument(
        "--hf-private",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create private HF repos (default: True).",
    )
    parser.add_argument("--mujoco-gl", type=str, default="egl", help="MUJOCO_GL backend (default: egl).")

    # Common overrides
    parser.add_argument("--encoder-type", choices=("spatial_grid", "spatial_softmax"), default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--frame-stack", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--aux-coef", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--drq-pad-px", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=None,
        help="Generic eval interval (updates for PPO, steps for SAC unless algo-specific flag is set).",
    )

    # Preflight options
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run single-batch overfit & NaN/Inf preflight verification and exit.",
    )
    parser.add_argument("--preflight-steps", type=int, default=150)
    parser.add_argument("--preflight-batch-size", type=int, default=None)

    # PPO overrides
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--anneal-lr", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--clip-coef", type=float, default=None)
    parser.add_argument("--vf-coef", type=float, default=None)
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument("--gae-lambda", type=float, default=None)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--eval-interval-updates", type=int, default=None)
    parser.add_argument("--checkpoint-interval-updates", type=int, default=None)

    # SAC overrides
    parser.add_argument("--actor-lr", type=float, default=None)
    parser.add_argument("--critic-lr", type=float, default=None)
    parser.add_argument("--alpha-lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--utd-ratio", type=float, default=None)
    parser.add_argument("--learning-starts", type=int, default=None)
    parser.add_argument("--capacity", type=int, default=None)
    parser.add_argument("--policy-delay", type=int, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--init-alpha", type=float, default=None)
    parser.add_argument("--autotune-alpha", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--eval-interval-steps", type=int, default=None)
    parser.add_argument("--checkpoint-interval-steps", type=int, default=None)
    parser.add_argument("--pretrain-steps", type=int, default=None)

    return parser.parse_args(argv)


def _build_sac_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {"dr_enabled": args.dr}
    mapping = {
        "total_timesteps": args.total_timesteps,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "encoder_type": args.encoder_type,
        "device": args.device,
        "frame_stack": args.frame_stack,
        "gamma": args.gamma,
        "aux_coef": args.aux_coef,
        "max_grad_norm": args.max_grad_norm,
        "drq_pad_px": args.drq_pad_px,
        "eval_episodes": args.eval_episodes,
        "save_video": args.save_video,
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,
        "alpha_lr": args.alpha_lr,
        "batch_size": args.batch_size,
        "utd_ratio": args.utd_ratio,
        "learning_starts": args.learning_starts,
        "capacity": args.capacity,
        "policy_delay": args.policy_delay,
        "tau": args.tau,
        "init_alpha": args.init_alpha,
        "autotune_alpha": args.autotune_alpha,
        "eval_interval_steps": (
            args.eval_interval_steps
            if args.eval_interval_steps is not None
            else args.eval_interval
        ),
        "checkpoint_interval_steps": args.checkpoint_interval_steps,
        "demo_pretrain_steps": args.pretrain_steps,
    }
    for k, v in mapping.items():
        if v is not None:
            overrides[k] = v
    return overrides


def _build_ppo_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {"dr_enabled": args.dr}
    mapping = {
        "total_timesteps": args.total_timesteps,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "encoder_type": args.encoder_type,
        "device": args.device,
        "frame_stack": args.frame_stack,
        "gamma": args.gamma,
        "aux_coef": args.aux_coef,
        "max_grad_norm": args.max_grad_norm,
        "drq_pad_px": args.drq_pad_px,
        "eval_episodes": args.eval_episodes,
        "save_video": args.save_video,
        "lr": args.lr,
        "anneal_lr": args.anneal_lr,
        "rollout_steps": args.rollout_steps,
        "num_epochs": args.num_epochs,
        "minibatch_size": args.minibatch_size,
        "clip_coef": args.clip_coef,
        "vf_coef": args.vf_coef,
        "ent_coef": args.ent_coef,
        "gae_lambda": args.gae_lambda,
        "target_kl": args.target_kl,
        "eval_interval_updates": (
            args.eval_interval_updates
            if args.eval_interval_updates is not None
            else args.eval_interval
        ),
        "checkpoint_interval_updates": args.checkpoint_interval_updates,
    }
    for k, v in mapping.items():
        if v is not None:
            overrides[k] = v
    return overrides


def _salvage_push_model(
    run_dir: Path,
    args: argparse.Namespace,
    actor_candidates: tuple[str, ...],
    push_fn: Any,
    reason: str,
) -> str | None:
    if not args.hf_model_repo:
        return None
    if not any((run_dir / cand).exists() for cand in actor_candidates):
        return None
    try:
        print(
            f"[colab_train] salvaging checkpoints to {args.hf_model_repo} ({reason})",
            flush=True,
        )
        return push_fn(
            run_dir=run_dir,
            repo_id=args.hf_model_repo,
            private=args.hf_private,
            commit_message=f"Salvage checkpoints on {reason}",
        )
    except Exception as push_exc:
        print(f"[colab_train] salvage Hub push failed: {push_exc}", flush=True)
        return None


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mujoco_gl and "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = args.mujoco_gl

    from so101_bowling.config import load_config
    from so101_bowling.dataset import collect_dataset
    from so101_bowling.hub import _ACTOR_CANDIDATES, push_dataset_to_hub, push_model_to_hub
    from so101_bowling.ppo import load_ppo_config, preflight_ppo, train_ppo
    from so101_bowling.sac import load_sac_config, preflight_sac, train_sac

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    _write_status(status_path, "running", {"algo": args.algo, "run_dir": str(run_dir)})
    print(f"[colab_train] stage=running algo={args.algo} run_dir={run_dir}", flush=True)

    def _handle_sigterm(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"Received signal {signum}")

    prev_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _handle_sigterm)

    try:
        wcfg = load_config(args.workspace_config)

        if args.preflight:
            _write_status(status_path, "preflight", {"algo": args.algo})
            print(f"[colab_train] stage=preflight algo={args.algo}", flush=True)
            try:
                if args.algo == "sac":
                    sac_cfg = replace(load_sac_config(args.config), **_build_sac_overrides(args))
                    report = preflight_sac(
                        cfg=sac_cfg,
                        workspace_cfg=wcfg,
                        steps=args.preflight_steps,
                        batch_size=args.preflight_batch_size,
                    )
                else:
                    ppo_cfg = replace(load_ppo_config(args.config), **_build_ppo_overrides(args))
                    report = preflight_ppo(
                        ppo_cfg=ppo_cfg,
                        workspace_cfg=wcfg,
                        steps=args.preflight_steps,
                        batch_size=args.preflight_batch_size,
                    )
            except FloatingPointError as exc:
                report = {
                    "algo": args.algo,
                    "passed": False,
                    "nan_inf_detected": True,
                    "checks": {"finite_ok": False},
                    "failure_reasons": [str(exc)],
                }
            (run_dir / "preflight_report.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(report, indent=2), flush=True)
            if not report["passed"] or report["nan_inf_detected"]:
                _write_status(
                    status_path,
                    "failed",
                    {
                        "algo": args.algo,
                        "preflight": report,
                        "nan_inf_detected": bool(report["nan_inf_detected"]),
                    },
                )
                sys.exit(1)
            _write_status(status_path, "preflight_passed", {"algo": args.algo, "preflight": report})
            return

        demo_dir = Path(args.demo_dir) if args.demo_dir else (run_dir / "demos")
        demo_info = None

        if args.collect_demos > 0:
            _write_status(status_path, "collecting_demos", {"num_episodes": args.collect_demos})
            print(
                f"[colab_train] stage=collecting_demos episodes={args.collect_demos} dir={demo_dir}",
                flush=True,
            )
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
        print(f"[colab_train] stage=training algo={args.algo}", flush=True)
        if args.algo == "sac":
            sac_cfg = replace(load_sac_config(args.config), **_build_sac_overrides(args))
            summary = train_sac(
                cfg=sac_cfg,
                workspace_cfg=wcfg,
                run_dir=run_dir,
                resume_from=args.resume,
                demo_dataset=demo_dir if demo_dir.exists() and (demo_dir / "episodes").is_dir() else None,
            )
        else:
            ppo_cfg = replace(load_ppo_config(args.config), **_build_ppo_overrides(args))
            summary = train_ppo(
                ppo_cfg=ppo_cfg,
                workspace_cfg=wcfg,
                run_dir=run_dir,
                resume_from=args.resume,
            )

        model_url = None
        if args.hf_model_repo:
            _write_status(status_path, "pushing_model", {"repo_id": args.hf_model_repo})
            print(f"[colab_train] stage=pushing_model repo={args.hf_model_repo}", flush=True)
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
        print(f"[colab_train] stage=completed hf_model_url={model_url}", flush=True)
        print(
            json.dumps({"state": "completed", "summary": summary, "hf_model_url": model_url}, indent=2),
            flush=True,
        )
    except SystemExit:
        raise
    except KeyboardInterrupt as exc:
        salvage_url = _salvage_push_model(
            run_dir, args, _ACTOR_CANDIDATES, push_model_to_hub, "interrupt"
        )
        _write_status(
            status_path,
            "interrupted",
            {"algo": args.algo, "error": str(exc) or "KeyboardInterrupt", "salvage_url": salvage_url},
        )
        print("[colab_train] stage=interrupted", flush=True)
        sys.exit(130)
    except Exception as exc:
        salvage_url = _salvage_push_model(
            run_dir, args, _ACTOR_CANDIDATES, push_model_to_hub, "failure"
        )
        _write_status(
            status_path,
            "failed",
            {
                "algo": args.algo,
                "nan_inf_detected": isinstance(exc, FloatingPointError),
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "salvage_url": salvage_url,
            },
        )
        print(f"[colab_train] stage=failed error={exc}", flush=True)
        raise
    finally:
        signal.signal(signal.SIGTERM, prev_sigterm)


if __name__ == "__main__":
    main()
