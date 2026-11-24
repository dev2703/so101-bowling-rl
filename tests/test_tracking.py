"""Offline unit tests for optional Weights & Biases tracking and Colab W&B wiring."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from typing import Any

import pytest

from so101_bowling.tracking import WandbConfig, WandbTracker, parse_wandb_tags


class _FakeWandbRun:
    def __init__(self, url: str = "https://wandb.ai/test/proj/runs/abc123") -> None:
        self.url = url
        self.logged: list[tuple[dict[str, Any], int]] = []
        self.summary: dict[str, Any] = {}
        self.finish_codes: list[int] = []

    def log(self, data: dict[str, Any], step: int = 0) -> None:
        self.logged.append((dict(data), int(step)))

    def finish(self, exit_code: int = 0) -> None:
        self.finish_codes.append(int(exit_code))


class _FakeWandbModule:
    def __init__(self) -> None:
        self.init_calls: list[dict[str, Any]] = []
        self.run = _FakeWandbRun()

    class Video:
        def __init__(self, path: str, fps: int = 20, format: str = "mp4") -> None:
            self.path = path
            self.fps = fps
            self.format = format

    def init(self, **kwargs: Any) -> _FakeWandbRun:
        self.init_calls.append(kwargs)
        return self.run


def test_wandb_config_validation_and_tags() -> None:
    assert WandbConfig().enabled is False
    assert WandbConfig(project="bowling", mode="disabled").enabled is False
    assert WandbConfig(project="bowling", mode="online").enabled is True
    assert WandbConfig(project="bowling", mode="offline").enabled is True
    with pytest.raises(ValueError, match="wandb mode"):
        WandbConfig(project="bowling", mode="invalid")
    assert parse_wandb_tags("sac, dr, sac, , bowling") == ("sac", "dr", "bowling")
    assert parse_wandb_tags(None) == ()


def test_wandb_tracker_noop_when_disabled(tmp_path: Path) -> None:
    tracker = WandbTracker(None, run_dir=tmp_path, algo="sac", train_config={"seed": 0})
    assert tracker.active is False
    assert tracker.run_url is None
    tracker.log_metrics("sac", {"critic_loss": 0.5}, step=10)
    tracker.log_video("sac/eval_video", tmp_path / "missing.mp4", step=10)
    tracker.update_summary({"global_step": 10})
    tracker.finish(exit_code=0)


def test_wandb_tracker_logs_metrics_video_and_summary(tmp_path: Path) -> None:
    fake_wb = _FakeWandbModule()
    cfg = WandbConfig(
        project="so101-bowling",
        entity="dev2703",
        run_name="sac-dr-test",
        group="colab-dr",
        tags=("sac", "dr"),
        mode="online",
    )
    tracker = WandbTracker(
        cfg,
        run_dir=tmp_path,
        algo="sac",
        train_config={"seed": 0, "dr_enabled": True},
        wandb_module=fake_wb,
    )
    assert tracker.active is True
    assert tracker.run_url == "https://wandb.ai/test/proj/runs/abc123"
    assert fake_wb.init_calls[0]["project"] == "so101-bowling"
    assert fake_wb.init_calls[0]["entity"] == "dev2703"
    assert fake_wb.init_calls[0]["name"] == "sac-dr-test"
    assert fake_wb.init_calls[0]["tags"] == ["sac", "dr"]

    tracker.log_metrics(
        "sac",
        {
            "global_step": 100,
            "critic_loss": 0.25,
            "nan_val": float("nan"),
            "flag": True,
            "text": "ignore",
        },
        step=100,
    )
    assert fake_wb.run.logged == [({"sac/critic_loss": 0.25}, 100)]

    vid = tmp_path / "eval.mp4"
    vid.write_bytes(b"fake-mp4")
    tracker.log_video("sac/eval_video", vid, step=100, fps=20)
    assert len(fake_wb.run.logged) == 2
    vid_obj = fake_wb.run.logged[1][0]["sac/eval_video"]
    assert isinstance(vid_obj, _FakeWandbModule.Video)
    assert vid_obj.path == str(vid)

    tracker.update_summary(
        {
            "global_step": 100,
            "nan_inf_detected": False,
            "best_eval": {"legal_strike_rate": 0.8, "mean_credited_pins": 2.7},
        }
    )
    assert fake_wb.run.summary["global_step"] == 100
    assert fake_wb.run.summary["best_eval/legal_strike_rate"] == 0.8

    tracker.finish(exit_code=0)
    tracker.finish(exit_code=1)  # idempotent
    assert fake_wb.run.finish_codes == [0]
    assert tracker.active is False


def test_colab_train_and_launch_wandb_wiring(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    spec_ct = importlib.util.spec_from_file_location(
        "colab_train_wb", repo_root / "scripts" / "colab_train.py"
    )
    assert spec_ct is not None and spec_ct.loader is not None
    ct_mod = importlib.util.module_from_spec(spec_ct)
    spec_ct.loader.exec_module(ct_mod)

    args = ct_mod.parse_args(
        [
            "--algo",
            "ppo",
            "--wandb-project",
            "so101-bowling",
            "--wandb-entity",
            "team",
            "--wandb-run-name",
            "ppo-dr",
            "--wandb-group",
            "gpu-pair",
            "--wandb-tags",
            "ppo,dr",
            "--wandb-mode",
            "offline",
        ]
    )
    assert args.wandb_project == "so101-bowling"
    assert args.wandb_entity == "team"
    assert args.wandb_run_name == "ppo-dr"
    assert args.wandb_group == "gpu-pair"
    assert args.wandb_tags == "ppo,dr"
    assert args.wandb_mode == "offline"

    spec_lc = importlib.util.spec_from_file_location(
        "launch_colab_wb", repo_root / "scripts" / "launch_colab.py"
    )
    assert spec_lc is not None and spec_lc.loader is not None
    lc_mod = importlib.util.module_from_spec(spec_lc)
    spec_lc.loader.exec_module(lc_mod)

    secret_wb = "wandb_secret_key_999999999999"
    monkeypatch.setenv("WANDB_API_KEY", secret_wb)
    lc_mod.main(
        [
            "--dry-run",
            "launch",
            "-s",
            "bowling-wb",
            "--gpu",
            "T4",
            "--",
            "--algo",
            "sac",
            "--wandb-project",
            "so101-bowling",
        ]
    )
    out = capsys.readouterr().out
    assert secret_wb not in out
    assert "upload '<redacted_wandb_key>' /content/.wandb_api_key_tmp -s bowling-wb" in out

    boot = lc_mod.build_bootstrap_script(
        repo_url=lc_mod.DEFAULT_REPO_URL,
        branch="main",
        remote_repo_dir="/content/so101-bowling-rl",
    )
    spawn = lc_mod.build_spawn_script(
        remote_repo_dir="/content/so101-bowling-rl",
        remote_run_dir="/content/runs/sac",
        train_args=["--algo", "sac", "--wandb-project", "so101-bowling"],
        session="bowling-wb",
    )
    ast.parse(boot)
    ast.parse(spawn)
    assert '"--extra", "tracking"' in boot
    assert "/content/.wandb_api_key_tmp" in boot
    assert "/content/.wandb_api_key_tmp" in spawn
    assert 'env["WANDB_API_KEY"]' in spawn
