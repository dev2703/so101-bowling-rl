import builtins
import json
import runpy
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from so101_bowling.config import WorkspaceConfig
from so101_bowling.ppo_models import DualCameraActor
from so101_bowling.real_eval import (
    TrialRecord,
    run_passive_camera_gate,
    run_real_trials,
    run_trial_episode,
    summarize_trials,
    wilson_ci,
)


@pytest.mark.parametrize(
    ("successes", "total", "expected"),
    [
        (0, 0, (0, 0)),
        (0, 10, (0, 0.2775328)),
        (10, 10, (0.7224672, 1)),
        (5, 10, (0.2365931, 0.7634069)),
    ],
)
def test_wilson_ci(successes, total, expected):
    assert wilson_ci(successes, total) == pytest.approx(expected)


@pytest.mark.parametrize(("successes", "total"), [(11, 10), (-1, 10), (0, -1)])
def test_wilson_ci_rejects_invalid_counts(successes, total):
    with pytest.raises(ValueError):
        wilson_ci(successes, total)


class FakeCamera:
    def __init__(self, captures, rgb):
        self.frames = iter(
            SimpleNamespace(rgb=rgb, capture_t=t, frame_seq=seq)
            for seq, t in enumerate(captures)
        )

    def read_latest(self):
        return next(self.frames)

    def __getattr__(self, key):
        raise AssertionError(f"Unexpected camera access {key}")


def test_passive_camera_gate_static_scene_and_stalled_timestamps(tmp_path, monkeypatch):
    # Any import of the bridge module would fail on the evaluator's camera-only path.
    monkeypatch.setitem(sys.modules, "so101_bowling.hardware", None)
    rgb = np.zeros((64, 112, 3), dtype=np.uint8)
    rgb[:, :30, 0] = 255
    tripod_rgb = np.flip(rgb, axis=1)
    wrist = FakeCamera([1, 1.125, 1.125, 1.375], rgb)
    tripod = FakeCamera([0.96875, 1.09375, 1.21875, 1.34375], tripod_rgb)
    clock = iter([1.03125, 1.15625, 1.28125, 1.5625]).__next__

    report = run_passive_camera_gate(
        wrist, tripod, WorkspaceConfig(), tmp_path, num_samples=4,
        actor=DualCameraActor(frame_stack=3).eval(), clock=clock,
    )

    assert report["cameras"]["wrist"] == {"effective_fps": 2 / 0.375, "stale_rate": 0.25}
    assert report["cameras"]["tripod"] == {"effective_fps": 8, "stale_rate": 0.25}
    assert report["inter_camera_skew_s"] == pytest.approx(
        {"median": 0.03125, "p95": 0.084375, "max": 0.09375}
    )
    latency = report["actor_inference"]
    assert latency["num_samples"] == 4
    assert latency["actions_valid"]
    assert 0 < latency["median_ms"] <= latency["p95_ms"] <= latency["max_ms"]
    assert "Omitted" in latency["auxiliary_head"]
    np.testing.assert_array_equal(cv2.imread(str(tmp_path / "wrist_crop.png")), rgb[:, :, ::-1])
    np.testing.assert_array_equal(
        cv2.imread(str(tmp_path / "tripod_crop.png")), tripod_rgb[:, :, ::-1]
    )
    np.testing.assert_array_equal(
        cv2.imread(str(tmp_path / "crops_side_by_side.png")),
        np.concatenate((rgb, tripod_rgb), axis=1)[:, :, ::-1],
    )
    assert json.loads((tmp_path / "passive_gate_report.json").read_text()) == report


def test_summarize_trials_keeps_aborts_and_separates_modes():
    strike = TrialRecord(1, "alpha", "live", True, True, 3, 0, False, None,
                         "human_video_review", 4, None, None)
    aborted = replace(strike, trial_id=2, legal_strike=False, legal_delivery=False,
                      credited_pins=0, aborted=True, abort_reason="stale_camera")
    partial = replace(strike, trial_id=3, legal_strike=False, credited_pins=2)
    dry = replace(strike, trial_id=4, mode="dry_run")
    beta = replace(strike, trial_id=5, policy_name="beta", legal_strike=False,
                   credited_pins=1)
    summary = summarize_trials([strike, aborted, partial, dry, beta])
    assert summary["dry_run"] == {"n_total": 1, "n_aborted": 0}
    assert summary["live"]["alpha"] == {
        "n_total": 3,
        "n_aborted": 1,
        "n_labeled": 3,
        "n_unlabeled": 0,
        "legal_strike_rate": 1 / 3,
        "legal_strike_ci95": wilson_ci(1, 3),
        "illegal_rate": 1 / 3,
        "illegal_ci95": wilson_ci(1, 3),
        "abort_rate": 1 / 3,
        "abort_ci95": wilson_ci(1, 3),
        "mean_credited_pins": 5 / 3,
        "credited_pins_distribution": {0: 1, 1: 0, 2: 1, 3: 1},
    }
    assert summary["live"]["beta"]["credited_pins_distribution"] == {0: 0, 1: 1, 2: 0, 3: 0}


class FakeBridge:
    def __init__(self, name, resets, abort_at=None):
        self.name = name
        self.resets = resets
        self.abort_at = abort_at
        self.estopped = False

    def reset_episode(self):
        self.resets.append(self.name)
        self.steps = 0

    def emergency_stop(self, reason):
        self.estopped = True

    def step(self):
        self.steps += 1
        aborted = self.steps == self.abort_at
        wrist = np.full((12, 16, 3), [255, 0, 0], dtype=np.uint8)
        tripod = np.full((18, 24, 3), [0, 0, 255], dtype=np.uint8)
        return SimpleNamespace(
            t=self.steps / 20, q_rad=np.zeros(4), action=np.ones(4),
            applied_delta_rad=np.full(4, 0.01), guard_reason=None,
            wrist_age_s=float("inf") if aborted else 0.01, tripod_age_s=0.02,
            aborted=aborted, abort_reason="stale_camera" if aborted else None,
            wrist_rgb=wrist, tripod_rgb=tripod,
        )


def test_run_trial_episode_writes_synchronized_video_and_trace_on_abort(tmp_path):
    resets = []
    bridge = FakeBridge("alpha", resets, abort_at=2)
    record = run_trial_episode(
        bridge, 5, tmp_path, 7, "alpha", mode="live",
        outcome_label={"legal_delivery": True, "legal_strike": True, "credited_pins": 3},
        outcome_labeler="operator_cli",
    )
    assert resets == ["alpha"]
    assert bridge.steps == record.steps == 2
    assert record.aborted and record.abort_reason == "stale_camera"
    assert record.labeled and record.wobble_only_pins == 0
    assert not record.legal_delivery and not record.legal_strike and record.credited_pins == 0
    assert record.outcome_labeler == "operator_cli"
    trace_path = tmp_path / "trial_007_trace.jsonl"
    assert record.trace_path == str(trace_path)
    first, last = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert first == {
        "t": 0.05, "q_rad": [0] * 4, "action": [1] * 4,
        "applied_delta_rad": [0.01] * 4, "guard_reason": None,
        "wrist_age_s": 0.01, "tripod_age_s": 0.02, "aborted": False, "abort_reason": None,
        "torque_disabled_confirmed": None,
    }
    assert last["aborted"] and last["wrist_age_s"] is None
    assert record.video_path == str(tmp_path / "trial_007.mp4")
    cap = cv2.VideoCapture(record.video_path)
    try:
        for _ in range(2):
            ok, frame = cap.read()
            assert ok and frame.shape == (24, 64, 3)
            assert frame[0, 0, 2] > 240
            assert frame[0, -1, 0] > 240
        assert not cap.read()[0]
    finally:
        cap.release()


def test_run_real_trials_interleaves_policies_and_saves_summary(tmp_path):
    resets = []
    policies = [(name, FakeBridge(name, resets)) for name in ("alpha", "beta")]
    summary = run_real_trials(policies, 4, 1, tmp_path)
    assert resets == ["alpha", "beta", "alpha", "beta"]
    assert summary == {
        "live": {}, "dry_run": {"n_total": 4, "n_aborted": 0},
        "stopped_due_to_estop": False, "torque_disable_unconfirmed_trial_ids": [],
    }
    assert json.loads((tmp_path / "summary.json").read_text()) == summary
    trials = [json.loads(line) for line in (tmp_path / "trials.jsonl").read_text().splitlines()]
    assert [trial["trial_id"] for trial in trials] == [1, 2, 3, 4]
    assert [trial["policy_name"] for trial in trials] == resets
    assert {trial["outcome_labeler"] for trial in trials} == {"dry_run_unlabeled"}
    assert all(trial["labeled"] is False for trial in trials)


def test_run_real_trials_records_live_outcome_labels(tmp_path):
    bridge = FakeBridge("alpha", [])
    label = {"legal_delivery": True, "legal_strike": True, "credited_pins": 3}

    def label_trial(record):
        assert record.trial_id == 2
        assert record.video_path is not None
        return {"legal_delivery": True, "legal_strike": False, "credited_pins": 1,
                "wobble_only_pins": 2}

    summary = run_real_trials(
        [("alpha", bridge)], 2, 1, tmp_path, mode="live",
        outcome_labeler="operator_cli", outcome_labels={1: label}, label_trial=label_trial,
    )
    assert summary["live"]["alpha"]["legal_strike_rate"] == 0.5
    assert summary["live"]["alpha"]["mean_credited_pins"] == 2
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["live"]["alpha"]["credited_pins_distribution"] == {
        "0": 0, "1": 1, "2": 0, "3": 1,
    }
    _, second_trial = (tmp_path / "trials.jsonl").read_text().splitlines()
    assert json.loads(second_trial)["wobble_only_pins"] == 2


def test_run_trial_episode_preserves_latched_estop(tmp_path):
    resets = []
    bridge = FakeBridge("alpha", resets)
    bridge.estopped = True
    bridge.estop_reason = "stale_camera_frame"
    record = run_trial_episode(bridge, 5, tmp_path, 1, "alpha")
    assert not resets
    assert record.aborted and record.abort_reason == "stale_camera_frame"
    assert record.steps == 0
    assert record.video_path is None and record.trace_path is None



def test_pending_live_outcomes_are_excluded_from_scoring(tmp_path):
    bridge = FakeBridge("alpha", [])
    pending = run_trial_episode(bridge, 1, tmp_path, 1, "alpha", mode="live")
    assert not pending.labeled
    assert pending.outcome_labeler == "human_video_review"
    assert not pending.legal_delivery and not pending.legal_strike and pending.credited_pins == 0
    assert summarize_trials([pending])["live"]["alpha"] == {
        "n_total": 1,
        "n_aborted": 0,
        "n_unlabeled": 1,
        "n_labeled": 0,
        "legal_strike_rate": None,
        "legal_strike_ci95": (0.0, 0.0),
        "illegal_rate": None,
        "illegal_ci95": (0.0, 0.0),
        "abort_rate": 0.0,
        "abort_ci95": wilson_ci(0, 1),
        "mean_credited_pins": None,
        "credited_pins_distribution": dict.fromkeys(range(4), 0),
    }

    strike = run_trial_episode(
        bridge, 1, tmp_path, 2, "alpha", mode="live",
        outcome_label={"legal_delivery": True, "legal_strike": True, "credited_pins": 3},
    )
    bridge.abort_at = 1
    aborted = run_trial_episode(bridge, 1, tmp_path, 3, "alpha", mode="live")
    assert aborted.labeled and aborted.aborted
    assert summarize_trials([pending, strike, aborted])["live"]["alpha"] == {
        "n_total": 3,
        "n_aborted": 1,
        "n_unlabeled": 1,
        "n_labeled": 2,
        "legal_strike_rate": 0.5,
        "legal_strike_ci95": wilson_ci(1, 2),
        "illegal_rate": 0.5,
        "illegal_ci95": wilson_ci(1, 2),
        "abort_rate": 1 / 3,
        "abort_ci95": wilson_ci(1, 3),
        "mean_credited_pins": 1.5,
        "credited_pins_distribution": {0: 1, 1: 0, 2: 0, 3: 1},
    }


@pytest.mark.parametrize(("delivery", "strike", "pins"), [
    (True, True, 2), (False, True, 3), (True, False, -1), (True, False, 4),
])
def test_invalid_outcome_label_is_rejected_before_reset(tmp_path, delivery, strike, pins):
    resets = []
    with pytest.raises(ValueError):
        run_trial_episode(
            FakeBridge("alpha", resets), 1, tmp_path, 1, "alpha", mode="live",
            outcome_label={"legal_delivery": delivery, "legal_strike": strike, "credited_pins": pins},
        )
    assert resets == []


class UnconfirmedStopBridge(FakeBridge):
    def emergency_stop(self, reason):
        super().emergency_stop(reason)
        self.last_estop_result = {"torque_disabled_confirmed": False}
        return self.last_estop_result

    def step(self):
        trace = super().step()
        if trace.aborted:
            self.emergency_stop(trace.abort_reason)
            trace.torque_disabled_confirmed = False
        return trace


def test_torque_disable_failure_is_recorded_and_warned(tmp_path, capsys):
    bridge = UnconfirmedStopBridge("alpha", [], abort_at=1)
    record = run_trial_episode(bridge, 1, tmp_path, 1, "alpha", mode="live")
    assert record.torque_disabled_confirmed is False
    assert json.loads(Path(record.trace_path).read_text())["torque_disabled_confirmed"] is False
    assert "CRITICAL WARNING" in capsys.readouterr().err


def test_recording_exception_warns_on_unconfirmed_torque_disable(tmp_path, monkeypatch, capsys):
    bridge = UnconfirmedStopBridge("alpha", [])

    def fail_step():
        raise RuntimeError("step failed")

    monkeypatch.setattr(bridge, "step", fail_step)
    with pytest.raises(RuntimeError, match="step failed"):
        run_trial_episode(bridge, 1, tmp_path, 1, "alpha", mode="live")
    assert bridge.estopped
    assert "CRITICAL WARNING" in capsys.readouterr().err


@pytest.mark.parametrize(("failed_operation", "error"), [
    pytest.param("open", "Unable to open video writer", id="open"),
    pytest.param("write", "video write failed", id="write"),
    pytest.param("release", "video release failed", id="release"),
])
def test_real_trials_save_recording_abort(tmp_path, monkeypatch, capsys, failed_operation, error):
    bridge = UnconfirmedStopBridge("alpha", [])
    frames_written = []

    class FailingWriter:
        def isOpened(self):
            return failed_operation != "open"

        def write(self, frame):
            if failed_operation == "write":
                raise OSError("video write failed")
            frames_written.append(frame)

        def release(self):
            if failed_operation in {"write", "release"}:
                raise OSError("video release failed")

    monkeypatch.setattr(cv2, "VideoWriter", lambda *args: FailingWriter())

    with pytest.raises(OSError, match=error):
        run_real_trials([("alpha", bridge)], 2, 1, tmp_path, mode="live")

    assert bridge.estopped is True
    if failed_operation == "release":
        assert len(frames_written) == 1
    (line,) = (tmp_path / "trials.jsonl").read_text().splitlines()
    trial = json.loads(line)
    assert trial["trial_id"] == 1
    assert trial["aborted"] is True
    assert trial["abort_reason"] == "trial_recording_failed"
    assert trial["labeled"] is True
    assert trial["legal_delivery"] is False
    assert trial["legal_strike"] is False
    assert trial["credited_pins"] == trial["wobble_only_pins"] == 0
    assert trial["torque_disabled_confirmed"] is False
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["live"]["alpha"]["n_total"] == 1
    assert summary["live"]["alpha"]["n_aborted"] == 1
    assert summary["stopped_due_to_estop"] is True
    assert summary["torque_disable_unconfirmed_trial_ids"] == [1]
    assert "CRITICAL WARNING" in capsys.readouterr().err


def test_real_trials_save_unlabeled_attempt_when_labeler_raises(tmp_path):
    bridge = FakeBridge("alpha", [])

    def fail_label(record):
        raise OSError("labeling failed")

    with pytest.raises(OSError, match="labeling failed"):
        run_real_trials([("alpha", bridge)], 1, 1, tmp_path, mode="live", label_trial=fail_label)

    (line,) = (tmp_path / "trials.jsonl").read_text().splitlines()
    trial = json.loads(line)
    assert trial["labeled"] is False
    assert trial["aborted"] is False
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["live"]["alpha"]["n_total"] == 1
    assert summary["live"]["alpha"]["n_unlabeled"] == 1


def test_real_trials_stop_launching_on_estopped_bridge(tmp_path):
    resets = []
    stopped = UnconfirmedStopBridge("alpha", resets, abort_at=1)
    running = FakeBridge("beta", resets)
    summary = run_real_trials(
        [("alpha", stopped), ("beta", running)], 4, 1, tmp_path, mode="live",
    )
    assert resets == ["alpha", "beta", "beta"]
    assert summary["stopped_due_to_estop"] is True
    assert summary["torque_disable_unconfirmed_trial_ids"] == [1]
    assert summary["live"]["alpha"]["n_total"] == summary["live"]["alpha"]["n_aborted"] == 1
    trials = [json.loads(line) for line in (tmp_path / "trials.jsonl").read_text().splitlines()]
    assert [trial["trial_id"] for trial in trials] == [1, 2, 4]
    assert trials[0]["torque_disabled_confirmed"] is False


def _run_cli(monkeypatch, *args):
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_real_transfer.py"
    monkeypatch.setattr(sys, "argv", [str(script), *map(str, args)])
    runpy.run_path(str(script), run_name="__main__")


def test_passive_gate_cli_replays_mp4s_without_lerobot(tmp_path, monkeypatch, capsys):
    import so101_bowling

    import_module = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name.split(".")[0] != "lerobot"
        return import_module(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "so101_bowling.hardware", raising=False)
    monkeypatch.delattr(so101_bowling, "hardware", raising=False)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    videos = [tmp_path / f"{name}.mp4" for name in ("wrist", "tripod")]
    for video in videos:
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 20, (16, 12))
        try:
            assert writer.isOpened()
            for _ in range(2):
                writer.write(np.zeros((12, 16, 3), dtype=np.uint8))
        finally:
            writer.release()

    _run_cli(
        monkeypatch, "passive-gate", "--wrist-video", videos[0], "--tripod-video", videos[1],
        "--samples", 2, "--out-dir", tmp_path,
    )
    report = json.loads(capsys.readouterr().out)
    assert report["num_samples"] == 2
    assert report["cameras"]["wrist"]["effective_fps"] == pytest.approx(20)
    assert (tmp_path / "passive_gate_report.json").exists()


def test_run_trials_cli_live_requires_calibration(tmp_path, monkeypatch):
    hardware_config = tmp_path / "hardware.yaml"
    hardware_config.write_text("calibration_path: null\n")
    with pytest.raises(SystemExit, match="--live requires --calibration or calibration_path"):
        _run_cli(
            monkeypatch, "run-trials", "--live", "--hardware-config", hardware_config,
            "--checkpoint", "missing.pt", "--policy-name", "alpha", "--out-dir", tmp_path,
        )
    assert not (tmp_path / "trials.jsonl").exists()


@pytest.mark.parametrize("calibration_source", [None, "cli", "config"])
def test_run_trials_cli_uses_dry_run_default_and_loads_live_calibration(
    tmp_path, monkeypatch, capsys, calibration_source,
):
    import so101_bowling
    from so101_bowling import config, hardware, ppo_models

    assert so101_bowling.hardware is sys.modules["so101_bowling.hardware"]
    assert so101_bowling.config is sys.modules["so101_bowling.config"]
    assert so101_bowling.ppo_models is sys.modules["so101_bowling.ppo_models"]

    live = calibration_source is not None
    calibration_path = tmp_path / "calibration.json"
    hardware_config = tmp_path / "hardware.yaml"
    configured_path = str(calibration_path) if calibration_source == "config" else None
    hardware_config.write_text(json.dumps({"dry_run": False, "calibration_path": configured_path}))
    calibration = object()

    def load_calibration(*, live_json_path):
        assert live_json_path == (str(calibration_path) if live else None)
        return calibration

    class Camera:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def wait_ready(self):
            return True

    class Bridge(UnconfirmedStopBridge):
        def __init__(self, actor, workspace, cal, hw_cfg, wrist, tripod):
            assert hw_cfg.dry_run is (not live)
            assert cal is calibration
            super().__init__("alpha", [], abort_at=1 if live else None)

        def close(self):
            pass

        def connect_robot(self):
            pass

        reset = FakeBridge.reset_episode

    monkeypatch.setattr(config, "load_calibration", load_calibration)
    monkeypatch.setattr(hardware, "OpenCVTimestampedCamera", Camera)
    monkeypatch.setattr(hardware, "SO101BowlingHardwareBridge", Bridge)
    monkeypatch.setattr(ppo_models, "load_actor", lambda path: object())
    args = []
    if live:
        args.append("--live")
    if calibration_source == "cli":
        args.extend(("--calibration", calibration_path))
    _run_cli(
        monkeypatch, "run-trials", "--hardware-config", hardware_config,
        "--checkpoint", "fake.pt", "--policy-name", "alpha", "--out-dir", tmp_path,
        "--max-steps", 1, *args,
    )
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["dry_run"]["n_total"] == (0 if live else 1)
    if live:
        assert "CRITICAL WARNING: Torque disable unconfirmed" in captured.err
        assert "CRITICAL WARNING: Live trials stopped" in captured.err
    else:
        assert captured.err == ""
