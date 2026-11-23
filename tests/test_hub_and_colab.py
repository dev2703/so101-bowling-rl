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
    assert args.hf_private is True
    assert args.mujoco_gl == "egl"


def test_load_actor_from_hub_ppo_fallback_and_explicit_fail(tmp_path: Path) -> None:
    ppo_actor = DualCameraActor(frame_stack=2, squashed=False)
    ppo_path = ppo_actor.save_actor(tmp_path / "best_actor.pt")
    calls: list[str] = []

    def fake_downloader(**kwargs: Any) -> str:
        fname = kwargs["filename"]
        calls.append(fname)
        if fname == "best_actor.pt":
            return str(ppo_path)
        raise FileNotFoundError(fname)

    loaded = load_actor_from_hub("user/so101-ppo", downloader=fake_downloader)
    assert loaded.squashed is False
    assert calls == ["actor_best.pt", "best_actor.pt"]

    # Explicit missing filename must NOT fall back silently.
    explicit_calls: list[str] = []

    def fail_downloader(**kwargs: Any) -> str:
        explicit_calls.append(kwargs["filename"])
        raise FileNotFoundError(kwargs["filename"])

    with pytest.raises(FileNotFoundError):
        load_actor_from_hub("user/so101-ppo", filename="missing.pt", downloader=fail_downloader)
    assert explicit_calls == ["missing.pt"]

    # All candidates missing when filename=None raises FileNotFoundError.
    all_calls: list[str] = []

    def all_missing_downloader(**kwargs: Any) -> str:
        all_calls.append(kwargs["filename"])
        raise FileNotFoundError(kwargs["filename"])

    with pytest.raises(FileNotFoundError, match="No actor checkpoint"):
        load_actor_from_hub("user/so101-empty", downloader=all_missing_downloader)
    assert all_calls == ["actor_best.pt", "best_actor.pt", "actor_latest.pt", "final_actor.pt"]


def test_build_model_card_includes_steps_and_loss_metadata(tmp_path: Path) -> None:
    summary = {
        "global_step": 1000,
        "total_timesteps": 2000,
        "best_eval": {
            "legal_strike_rate": 0.9,
            "mean_credited_pins": 2.8,
            "illegal_rate": 0.0,
            "mean_return": 19.5,
            "mean_episode_steps": 41.25,
        },
        "eval_aux_loss": 0.0123,
        "final_losses": {"policy_loss": -0.04, "value_loss": 0.03, "aux_loss": 0.01},
        "nan_inf_detected": False,
        "validation_loss_note": "Eval aux loss tracks visible targets.",
    }
    for algo, expected_hint in (("ppo", "best_actor.pt"), ("sac", "actor_best.pt")):
        card = build_model_card(
            repo_id=f"user/so101-{algo}",
            run_dir=tmp_path,
            algo=algo,
            summary=summary,
            config={"seed": 0},
        )
        assert "Mean Episode Steps" in card
        assert "41.25" in card
        assert "Training & Validation Metrics" in card
        assert "0.012300" in card
        assert "Eval aux loss tracks visible targets." in card
        assert expected_hint in card


def test_hf_private_defaults_true_and_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.util
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    spec_ct = importlib.util.spec_from_file_location(
        "colab_train_priv", repo_root / "scripts" / "colab_train.py"
    )
    assert spec_ct is not None and spec_ct.loader is not None
    ct_mod = importlib.util.module_from_spec(spec_ct)
    spec_ct.loader.exec_module(ct_mod)

    assert ct_mod.parse_args([]).hf_private is True
    assert ct_mod.parse_args(["--no-hf-private"]).hf_private is False

    spec_push = importlib.util.spec_from_file_location(
        "push_to_hub_priv", repo_root / "scripts" / "push_to_hub.py"
    )
    assert spec_push is not None and spec_push.loader is not None
    push_mod = importlib.util.module_from_spec(spec_push)
    spec_push.loader.exec_module(push_mod)

    seen_private: list[bool] = []
    monkeypatch.setattr(
        push_mod,
        "push_model_to_hub",
        lambda **kw: seen_private.append(bool(kw["private"])) or "https://hf/m",
    )
    monkeypatch.setattr(sys, "argv", ["push_to_hub.py", "model", "--run-dir", str(tmp_path), "--repo-id", "u/r"])
    push_mod.main()
    monkeypatch.setattr(sys, "argv", ["push_to_hub.py", "model", "--run-dir", str(tmp_path), "--repo-id", "u/r", "--no-private"])
    push_mod.main()
    assert seen_private == [True, False]


def test_colab_train_overrides_preflight_and_signal_salvage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "colab_train_full", repo_root / "scripts" / "colab_train.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    args = mod.parse_args(
        [
            "--algo",
            "sac",
            "--actor-lr",
            "1e-4",
            "--critic-lr",
            "2e-4",
            "--alpha-lr",
            "5e-5",
            "--batch-size",
            "64",
            "--utd-ratio",
            "1.0",
            "--learning-starts",
            "500",
            "--capacity",
            "10000",
            "--pretrain-steps",
            "250",
            "--eval-interval",
            "2000",
            "--eval-episodes",
            "5",
            "--no-save-video",
            "--preflight",
            "--preflight-steps",
            "20",
            "--preflight-batch-size",
            "4",
        ]
    )
    sac_ov = mod._build_sac_overrides(args)
    assert sac_ov["actor_lr"] == 1e-4
    assert sac_ov["demo_pretrain_steps"] == 250
    assert sac_ov["eval_interval_steps"] == 2000
    assert sac_ov["save_video"] is False

    # Failed preflight exits with code 1 and writes preflight_report.json + status.json.
    import so101_bowling.sac as sac_mod

    monkeypatch.setattr(
        sac_mod,
        "preflight_sac",
        lambda **kw: {
            "algo": "sac",
            "passed": False,
            "nan_inf_detected": True,
            "failure_reasons": ["forced_fail"],
        },
    )
    preflight_dir = tmp_path / "pf_fail"
    with pytest.raises(SystemExit) as exc_info:
        mod.main(["--algo", "sac", "--run-dir", str(preflight_dir), "--mujoco-gl", "", "--preflight"])
    assert exc_info.value.code == 1
    assert (preflight_dir / "preflight_report.json").exists()
    assert json.loads((preflight_dir / "status.json").read_text())["state"] == "failed"

    # KeyboardInterrupt salvages existing actor checkpoint to HF Hub and exits 130.
    import so101_bowling.hub as hub_mod

    salvage_dir = tmp_path / "salvage_run"
    salvage_dir.mkdir()
    DualCameraActor(frame_stack=2, squashed=True).save_actor(salvage_dir / "actor_latest.pt")
    pushed: list[str] = []
    monkeypatch.setattr(
        hub_mod,
        "push_model_to_hub",
        lambda **kw: pushed.append(kw["repo_id"]) or "https://hf/salvaged",
    )
    monkeypatch.setattr(
        sac_mod,
        "train_sac",
        lambda **kw: (_ for _ in ()).throw(KeyboardInterrupt("preempted")),
    )
    with pytest.raises(SystemExit) as kb_exc:
        mod.main(["--algo", "sac", "--run-dir", str(salvage_dir), "--mujoco-gl", "", "--hf-model-repo", "user/salvage"])
    assert kb_exc.value.code == 130
    assert pushed == ["user/salvage"]
    assert json.loads((salvage_dir / "status.json").read_text())["state"] == "interrupted"


def test_launch_colab_command_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import importlib.util
    import subprocess

    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "launch_colab", repo_root / "scripts" / "launch_colab.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    secret_token = "hf_super_secret_test_token_12345"
    monkeypatch.setenv("HF_TOKEN", secret_token)

    mod.main(
        [
            "--dry-run",
            "--colab-auth",
            "oauth2",
            "launch",
            "-s",
            "bowling-test",
            "--gpu",
            "T4",
            "--branch",
            "feat/m1-preflight-dr-hub-fixes",
            "--",
            "--algo",
            "sac",
            "--preflight",
        ]
    )
    out = capsys.readouterr().out
    assert secret_token not in out
    assert "colab --auth oauth2 sessions" in out
    assert "colab --auth oauth2 new -s bowling-test --gpu T4" in out
    assert "colab --auth oauth2 upload '<redacted_hf_token>' /content/.hf_token_tmp -s bowling-test" in out
    assert "--timeout 300 -f" in out
    assert "--timeout 30 -f" in out

    # Also verify non-dry-run with mocked subprocess.run builds bootstrap & spawn scripts accurately.
    recorded_cmds: list[list[str]] = []
    captured_scripts: list[str] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        recorded_cmds.append(list(cmd))
        if "-f" in cmd:
            f_idx = cmd.index("-f") + 1
            captured_scripts.append(Path(cmd[f_idx]).read_text(encoding="utf-8"))
        if "sessions" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if "exec" in cmd and "--timeout" in cmd and "30" in cmd:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=json.dumps({"session": "bowling-test", "pid": 4321}),
                stderr="",
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    mod.main(
        [
            "launch",
            "-s",
            "bowling-test",
            "--gpu",
            "L4",
            "--branch",
            "feat/m1-preflight-dr-hub-fixes",
            "--",
            "--algo",
            "ppo",
        ]
    )
    assert any("new" in c and "L4" in c for c in recorded_cmds)
    assert len(captured_scripts) == 2
    assert '"sync", "--extra", "hub", "--extra", "dev"' in captured_scripts[0]
    assert "start_new_session=True" in captured_scripts[1]
    assert "MUJOCO_GL" in captured_scripts[1]
    assert secret_token not in captured_scripts[0]
    assert secret_token not in captured_scripts[1]
