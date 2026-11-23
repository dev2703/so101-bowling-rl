"""Offline unit tests for Hugging Face Hub integration and Colab training CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from so101_bowling.hub import (
    build_dataset_card,
    build_model_card,
    load_actor_from_hub,
    push_dataset_to_hub,
    push_model_to_hub,
)
from so101_bowling.ppo_models import DualCameraActor


class _FakeHfApi:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.uploaded: list[dict[str, Any]] = []

    def create_repo(self, **kwargs: Any) -> None:
        self.created.append(kwargs)

    def upload_folder(self, **kwargs: Any) -> None:
        self.uploaded.append(kwargs)


def test_push_model_and_load_from_hub_offline(tmp_path: Path) -> None:
    run_dir = tmp_path / "sac_run"
    run_dir.mkdir()
    actor = DualCameraActor(frame_stack=2, squashed=True)
    actor_path = actor.save_actor(run_dir / "actor_best.pt")
    (run_dir / "sac_config.json").write_text(json.dumps({"seed": 0, "dr_enabled": True}))
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "best_eval": {
                    "legal_strike_rate": 0.75,
                    "mean_credited_pins": 2.6,
                    "illegal_rate": 0.05,
                    "mean_return": 18.2,
                }
            }
        )
    )

    api = _FakeHfApi()
    url = push_model_to_hub(run_dir, "dev2703/so101-bowling-sac", private=True, api=api)
    assert url == "https://huggingface.co/dev2703/so101-bowling-sac"
    assert api.created[0]["repo_type"] == "model"
    assert api.created[0]["private"] is True
    assert api.uploaded[0]["repo_id"] == "dev2703/so101-bowling-sac"
    readme = (run_dir / "README.md").read_text()
    assert "0.750" in readme
    assert "dev2703/so101-bowling-sac" in readme

    loaded = load_actor_from_hub(
        "dev2703/so101-bowling-sac",
        filename="actor_best.pt",
        downloader=lambda **kw: str(actor_path),
    )
    assert loaded.squashed is True
    for k, v in actor.state_dict().items():
        assert torch.equal(v, loaded.state_dict()[k])


def test_push_model_raises_without_actor_checkpoint(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="actor checkpoint"):
        push_model_to_hub(empty_dir, "user/repo", api=_FakeHfApi())


def test_push_dataset_to_hub_offline(tmp_path: Path) -> None:
    ds_dir = tmp_path / "ds"
    (ds_dir / "episodes").mkdir(parents=True)
    (ds_dir / "dataset_info.json").write_text(
        json.dumps(
            {
                "source_policy": "scripted_4dof",
                "num_episodes": 10,
                "total_transitions": 250,
                "dr_enabled": True,
                "img_hw": [64, 112],
                "frame_stack": 2,
                "legal_delivery_rate": 1.0,
                "legal_strike_rate": 0.9,
                "mean_credited_pins": 2.9,
            }
        )
    )
    api = _FakeHfApi()
    url = push_dataset_to_hub(ds_dir, "dev2703/so101-bowling-demos", api=api)
    assert url == "https://huggingface.co/datasets/dev2703/so101-bowling-demos"
    assert api.created[0]["repo_type"] == "dataset"
    readme = (ds_dir / "README.md").read_text()
    assert "scripted_4dof" in readme
    assert "250" in readme


def test_colab_train_cli_arg_parsing() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "colab_train",
        Path(__file__).resolve().parents[1] / "scripts" / "colab_train.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    args = mod.parse_args(
        [
            "--algo",
            "sac",
            "--total-timesteps",
            "100000",
            "--collect-demos",
            "50",
            "--hf-model-repo",
            "dev2703/so101-bowling-sac",
            "--hf-dataset-repo",
            "dev2703/so101-bowling-demos",
        ]
    )
    assert args.algo == "sac"
    assert args.total_timesteps == 100000
    assert args.collect_demos == 50
    assert args.dr is True
    assert args.mujoco_gl == "egl"
