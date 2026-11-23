"""Hugging Face Hub push/pull helpers for SO-101 bowling actors and datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from so101_bowling.ppo_models import DualCameraActor, load_actor

_ACTOR_CANDIDATES = ("actor_best.pt", "best_actor.pt", "actor_latest.pt", "final_actor.pt")


def _get_hf_api(token: str | None = None) -> Any:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError(
            "huggingface-hub is required for Hub operations; install with `uv sync --extra hub`"
        ) from exc
    return HfApi(token=token)


def build_model_card(
    repo_id: str,
    run_dir: Path,
    algo: str,
    summary: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> str:
    """Generate a Hugging Face Model Card (README.md) with YAML frontmatter."""
    best_eval = (summary or {}).get("best_eval") or {}
    lines = [
        "---",
        "library_name: pytorch",
        "tags:",
        "  - reinforcement-learning",
        "  - robotics",
        "  - mujoco",
        "  - so101",
        "  - bowling",
        f"  - {algo}",
        "---",
        "",
        f"# {repo_id}",
        "",
        f"Dual-camera `{algo.upper()}` policy (`DualCameraActor`) trained on SO-101 3-pin tabletop push-bowling in MuJoCo.",
        "",
        "## Evaluation Summary",
        "",
    ]
    if best_eval:
        lines.extend(
            [
                "| Metric | Value |",
                "| --- | --- |",
                f"| Legal Strike Rate | `{best_eval.get('legal_strike_rate', 0.0):.3f}` |",
                f"| Mean Credited Pins | `{best_eval.get('mean_credited_pins', 0.0):.3f}` |",
                f"| Illegal Rate | `{best_eval.get('illegal_rate', 0.0):.3f}` |",
                f"| Mean Return | `{best_eval.get('mean_return', 0.0):.3f}` |",
                "",
            ]
        )
    else:
        lines.append("No evaluation summary found in run directory.\n")

    if config:
        lines.extend(
            [
                "## Training Config",
                "",
                "```json",
                json.dumps(config, indent=2),
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## Usage",
            "",
            "```python",
            "from so101_bowling.hub import load_actor_from_hub",
            f'actor = load_actor_from_hub("{repo_id}")',
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def build_dataset_card(repo_id: str, info: dict[str, Any]) -> str:
    """Generate a Hugging Face Dataset Card (README.md) with YAML frontmatter."""
    lines = [
        "---",
        "tags:",
        "  - robotics",
        "  - mujoco",
        "  - so101",
        "  - bowling",
        "  - reinforcement-learning",
        "---",
        "",
        f"# {repo_id}",
        "",
        "Dual-camera (`wrist_cam` + `tripod_cam`) episode dataset for SO-101 3-pin tabletop push-bowling.",
        "",
        "## Dataset Summary",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Source Policy | `{info.get('source_policy', 'unknown')}` |",
        f"| Episodes | `{info.get('num_episodes', 0)}` |",
        f"| Total Transitions | `{info.get('total_transitions', 0)}` |",
        f"| Domain Randomization | `{info.get('dr_enabled', False)}` |",
        f"| Image Resolution (H×W) | `{info.get('img_hw', [64, 112])}` |",
        f"| Frame Stack | `{info.get('frame_stack', 2)}` |",
        f"| Legal Delivery Rate | `{info.get('legal_delivery_rate', 0.0):.3f}` |",
        f"| Legal Strike Rate | `{info.get('legal_strike_rate', 0.0):.3f}` |",
        f"| Mean Credited Pins | `{info.get('mean_credited_pins', 0.0):.3f}` |",
        "",
        "## Structure",
        "",
        "- `dataset_info.json`: aggregate statistics and workspace configuration",
        "- `episodes.jsonl`: per-episode metadata, seeds, and `DeliveryScorer` outcomes",
        "- `episodes/ep_XXXXXX.npz`: compressed numpy archives containing dual-camera RGB stacks, proprio, camera meta, privileged state, actions, rewards, and done flags",
        "",
    ]
    return "\n".join(lines)


def push_model_to_hub(
    run_dir: str | Path,
    repo_id: str,
    *,
    private: bool = False,
    token: str | None = None,
    commit_message: str | None = None,
    api: Any | None = None,
) -> str:
    """Upload exported actor checkpoints, configs, metrics, videos, and Model Card to HF Hub."""
    root = Path(run_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Run directory not found: {root}")

    found_actors = [name for name in _ACTOR_CANDIDATES if (root / name).exists()]
    if not found_actors:
        raise FileNotFoundError(
            f"No exported actor checkpoint ({_ACTOR_CANDIDATES}) found in {root}"
        )

    algo = "sac" if (root / "sac_config.json").exists() else "ppo"
    cfg_path = root / f"{algo}_config.json"
    config = json.loads(cfg_path.read_text()) if cfg_path.exists() else None
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None

    readme_path = root / "README.md"
    readme_path.write_text(
        build_model_card(repo_id=repo_id, run_dir=root, algo=algo, summary=summary, config=config),
        encoding="utf-8",
    )

    hf_api = api if api is not None else _get_hf_api(token=token)
    hf_api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    hf_api.upload_folder(
        folder_path=str(root),
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=[
            "README.md",
            "*actor*.pt",
            "*_config.json",
            "summary.json",
            "metrics.jsonl",
            "status.json",
            "videos/*.mp4",
        ],
        commit_message=commit_message or f"Upload SO-101 bowling {algo.upper()} actor",
    )
    return f"https://huggingface.co/{repo_id}"


def push_dataset_to_hub(
    dataset_dir: str | Path,
    repo_id: str,
    *,
    private: bool = False,
    token: str | None = None,
    commit_message: str | None = None,
    api: Any | None = None,
) -> str:
    """Upload a collected bowling episode dataset and Dataset Card to HF Hub."""
    root = Path(dataset_dir)
    info_path = root / "dataset_info.json"
    if not info_path.exists() or not (root / "episodes").is_dir():
        raise FileNotFoundError(
            f"Dataset directory {root} must contain dataset_info.json and episodes/"
        )

    info = json.loads(info_path.read_text())
    (root / "README.md").write_text(
        build_dataset_card(repo_id=repo_id, info=info),
        encoding="utf-8",
    )

    hf_api = api if api is not None else _get_hf_api(token=token)
    hf_api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    hf_api.upload_folder(
        folder_path=str(root),
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["README.md", "dataset_info.json", "episodes.jsonl", "episodes/*.npz"],
        commit_message=commit_message or "Upload SO-101 bowling dual-camera dataset",
    )
    return f"https://huggingface.co/datasets/{repo_id}"


def load_actor_from_hub(
    repo_id: str,
    filename: str = "actor_best.pt",
    map_location: str = "cpu",
    revision: str | None = None,
    token: str | None = None,
    downloader: Any | None = None,
) -> DualCameraActor:
    """Download and load a standalone DualCameraActor checkpoint from Hugging Face Hub."""
    if downloader is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "huggingface-hub is required for Hub operations; install with `uv sync --extra hub`"
            ) from exc
        downloader = hf_hub_download

    local_path = downloader(
        repo_id=repo_id,
        filename=filename,
        repo_type="model",
        revision=revision,
        token=token,
    )
    return load_actor(local_path, map_location=map_location)
