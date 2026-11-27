"""Camera diagnostics and trial evidence collection for real transfer."""

from __future__ import annotations

import json
import math
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from functools import partial
from itertools import cycle, islice
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from so101_bowling.config import WorkspaceConfig
from so101_bowling.perception_utils import TimestampedFrameStack, preprocess_camera_frame


class CameraFrame(Protocol):
    rgb: np.ndarray
    capture_t: float
    frame_seq: int


class Camera(Protocol):
    def read_latest(self) -> CameraFrame | None: ...


class TrialBridge(Protocol):
    @property
    def estopped(self) -> bool: ...

    def reset_episode(self) -> Any: ...
    def step(self) -> Any: ...
    def emergency_stop(self, reason: str) -> Any: ...


def wilson_ci(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if not 0 <= successes <= total:
        raise ValueError("Expected 0 <= successes <= total")
    if total == 0:
        return 0.0, 0.0
    proportion = successes / total
    z2 = z * z
    denominator = 1 + z2 / total
    center = (proportion + z2 / (2 * total)) / denominator
    margin = abs(z) * math.sqrt(
        proportion * (1 - proportion) / total + z2 / (4 * total**2)
    ) / denominator
    return (
        0.0 if successes == 0 else max(0.0, center - margin),
        1.0 if successes == total else min(1.0, center + margin),
    )


@dataclass
class TrialRecord:
    trial_id: int
    policy_name: str
    mode: str
    legal_delivery: bool
    legal_strike: bool
    credited_pins: int
    wobble_only_pins: int
    aborted: bool
    abort_reason: str | None
    outcome_labeler: str
    steps: int
    video_path: str | None
    trace_path: str | None
    torque_disabled_confirmed: bool | None = None
    labeled: bool = True

    def __post_init__(self) -> None:
        if self.mode not in ("live", "dry_run"):
            raise ValueError("mode must be live or dry_run")
        for count in (self.credited_pins, self.wobble_only_pins):
            if type(count) is not int or not 0 <= count <= 3:
                raise ValueError("Pin counts must be integers from 0 to 3")


def summarize_trials(records: list[TrialRecord]) -> dict[str, Any]:
    live: dict[str, list[TrialRecord]] = {}
    dry_run: list[TrialRecord] = []
    for record in records:
        if record.mode == "dry_run":
            dry_run.append(record)
        else:
            live.setdefault(record.policy_name, []).append(record)

    policies = {}
    for name, attempts in live.items():
        total = len(attempts)
        labeled = [attempt for attempt in attempts if attempt.labeled]
        n_labeled = len(labeled)
        distribution = dict.fromkeys(range(4), 0)
        for attempt in labeled:
            distribution[attempt.credited_pins] += 1
        strikes = sum(attempt.legal_strike for attempt in labeled)
        illegal = sum(not attempt.legal_delivery for attempt in labeled)
        aborts = sum(attempt.aborted for attempt in attempts)
        policies[name] = {
            "n_total": total,
            "n_aborted": aborts,
            "n_labeled": n_labeled,
            "n_unlabeled": total - n_labeled,
            "legal_strike_rate": strikes / n_labeled if n_labeled else None,
            "legal_strike_ci95": wilson_ci(strikes, n_labeled),
            "illegal_rate": illegal / n_labeled if n_labeled else None,
            "illegal_ci95": wilson_ci(illegal, n_labeled),
            "abort_rate": aborts / total,
            "abort_ci95": wilson_ci(aborts, total),
            "mean_credited_pins": (
                sum(attempt.credited_pins for attempt in labeled) / n_labeled
                if n_labeled else None
            ),
            "credited_pins_distribution": distribution,
        }
    return {
        "live": policies,
        "dry_run": {"n_total": len(dry_run), "n_aborted": sum(r.aborted for r in dry_run)},
    }


def _percentiles(samples: list[float], suffix: str = "") -> dict[str, float | None]:
    values = np.percentile(samples, [50, 95, 100]).tolist() if samples else [None] * 3
    return {label + suffix: value for label, value in zip(("median", "p95", "max"), values)}


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_passive_camera_gate(
    wrist_cam: Camera,
    tripod_cam: Camera,
    cfg: WorkspaceConfig,
    out_dir: str | Path,
    num_samples: int = 40,
    max_camera_age_s: float = 0.20,
    actor=None,
    clock=time.monotonic,
    sample_interval_s: float = 0.0,
) -> dict[str, Any]:
    """Sample producer timing and save the final crops in actor input order.

    Pass clock=None for offline replay to measure age relative to the newest
    producer timestamp in the current sample.
    """
    if num_samples < 1 or max_camera_age_s < 0 or sample_interval_s < 0:
        raise ValueError("Expected positive sample count and nonnegative timing settings")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cameras = {"wrist": wrist_cam, "tripod": tripod_cam}
    stack_k = getattr(actor, "frame_stack", 2)
    stacks = {name: TimestampedFrameStack(k=stack_k) for name in cameras}
    captures: dict[str, list[float]] = {name: [] for name in cameras}
    stale = dict.fromkeys(cameras, 0)
    crops: dict[str, np.ndarray] = {}
    skew: list[float] = []
    latencies: list[float] = []
    actions_valid = True

    for sample in range(num_samples):
        frames = {name: camera.read_latest() for name, camera in cameras.items()}
        now = (
            clock() if clock is not None
            else max((frame.capture_t for frame in frames.values() if frame is not None), default=0.0)
        )
        actor_obs = {"proprio": np.zeros(12, dtype=np.float32)}
        meta = []
        for name, frame in frames.items():
            stack = stacks[name]
            fresh = False
            if frame is not None:
                crop = preprocess_camera_frame(
                    frame.rgb, cfg.camera.render_height, cfg.camera.render_width
                )
                crops[name] = crop
                if stack.push(crop, frame.capture_t):
                    captures[name].append(frame.capture_t)
                    fresh = stack.is_fresh(now, max_camera_age_s)
            stale[name] += not fresh
            meta.extend((min(10.0, stack.age_s(now)), float(fresh)))
            history = stack.frames
            if history:
                actor_obs[f"{name}_rgb"] = np.stack((history[0],) * (stack_k - len(history)) + history)

        wrist, tripod = frames["wrist"], frames["tripod"]
        if wrist is not None and tripod is not None:
            skew.append(abs(wrist.capture_t - tripod.capture_t))
        if actor is not None and "wrist_rgb" in actor_obs and "tripod_rgb" in actor_obs:
            actor_obs["camera_meta"] = np.asarray(meta, dtype=np.float32)
            started = time.perf_counter()
            action = actor.act(actor_obs, deterministic=True)
            if hasattr(action, "detach"):
                action = action.detach().cpu().numpy()
            latencies.append((time.perf_counter() - started) * 1000)
            action = np.asarray(action)
            actions_valid &= bool(
                action.shape == (4,) and np.isfinite(action).all() and (np.abs(action) <= 1).all()
            )
        if sample_interval_s and sample + 1 < num_samples:
            time.sleep(sample_interval_s)

    metrics = {}
    for name, times in captures.items():
        metrics[name] = {
            "effective_fps": (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0,
            "stale_rate": stale[name] / num_samples,
        }

    images = {f"{name}_crop.png": crop for name, crop in crops.items()}
    if len(crops) == 2:
        images["crops_side_by_side.png"] = np.concatenate((crops["wrist"], crops["tripod"]), axis=1)
    for filename, rgb in images.items():
        if not cv2.imwrite(str(out_dir / filename), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
            raise OSError(f"Unable to write {out_dir / filename}")

    report: dict[str, Any] = {
        "num_samples": num_samples,
        "max_camera_age_s": max_camera_age_s,
        "cameras": metrics,
        "inter_camera_skew_s": _percentiles(skew),
    }
    if actor is not None:
        report["actor_inference"] = {
            **_percentiles(latencies, "_ms"),
            "num_samples": len(latencies),
            "actions_valid": bool(latencies) and actions_valid,
            "auxiliary_head": "Omitted on standalone exports.",
        }
    _write_json(out_dir / "passive_gate_report.json", report)
    return report


_TRACE_FIELDS = (
    "t", "q_rad", "action", "applied_delta_rad", "guard_reason", "wrist_age_s",
    "tripod_age_s", "aborted", "abort_reason",
)


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.ndarray, np.generic)):
        return _json_value(value.tolist())
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _record_estop(
    record: TrialRecord, bridge: TrialBridge, result: dict[str, Any] | None = None
) -> None:
    result = result or getattr(bridge, "last_estop_result", None)
    if result is not None and result.get("torque_disabled_confirmed") is not None:
        record.torque_disabled_confirmed = result["torque_disabled_confirmed"]
    if record.torque_disabled_confirmed is False:
        print(
            f"CRITICAL WARNING: Trial {record.trial_id} E-stop did not confirm torque disable.",
            file=sys.stderr,
        )


def run_trial_episode(
    bridge: TrialBridge,
    max_steps: int,
    out_dir: str | Path,
    trial_id: int,
    policy_name: str,
    mode: str = "dry_run",
    outcome_label: dict[str, Any] | None = None,
    outcome_labeler: str = "human_video_review",
    fps: float = 20.0,
    on_abort: Callable[[TrialRecord], None] | None = None,
) -> TrialRecord:
    """Record one episode. Each video pair comes from the corresponding step."""
    if max_steps < 1 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Expected positive max_steps and fps")
    record = TrialRecord(
        trial_id, policy_name, mode, False, False, 0, 0, False, None,
        outcome_labeler if mode == "live" or outcome_label is not None else "dry_run_unlabeled",
        0, None, None, labeled=False,
    )
    if outcome_label is not None:
        record = _label_trial(record, outcome_label)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"trial_{trial_id:03d}"
    trace_path = stem.with_name(stem.name + "_trace.jsonl")
    video_path = stem.with_suffix(".mp4")
    writer = None
    video_size = None
    estop_result = None
    if not getattr(bridge, "estopped", False):
        try:
            bridge.reset_episode()
            with trace_path.open("w", encoding="utf-8") as trace_file:
                record.trace_path = str(trace_path)
                for _ in range(max_steps):
                    if getattr(bridge, "estopped", False):
                        break
                    trace = bridge.step()
                    row = {key: _json_value(getattr(trace, key)) for key in _TRACE_FIELDS}
                    torque_confirmed = getattr(trace, "torque_disabled_confirmed", None)
                    row["torque_disabled_confirmed"] = _json_value(torque_confirmed)
                    if torque_confirmed is not None:
                        record.torque_disabled_confirmed = torque_confirmed
                    record.steps += 1
                    record.aborted = bool(trace.aborted)
                    record.abort_reason = trace.abort_reason
                    if record.aborted and not getattr(bridge, "estopped", False):
                        estop_result = bridge.emergency_stop(record.abort_reason or "trial_aborted")
                    trace_file.write(json.dumps(row, allow_nan=False) + "\n")
                    raw_wrist = getattr(trace, "raw_wrist_rgb", None)
                    raw_tripod = getattr(trace, "raw_tripod_rgb", None)
                    if raw_wrist is not None and raw_tripod is not None:
                        vid_wrist, vid_tripod = raw_wrist, raw_tripod
                        high_res_video = True
                    else:
                        vid_wrist, vid_tripod = trace.wrist_rgb, trace.tripod_rgb
                        high_res_video = False
                    if vid_wrist is not None and vid_tripod is not None:
                        if video_size is None:
                            height, width = vid_wrist.shape[:2]
                            video_size = (2 * width, height) if high_res_video else (4 * width, 2 * height)
                            writer = cv2.VideoWriter(
                                str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, video_size
                            )
                            if not writer.isOpened():
                                raise OSError(f"Unable to open video writer for {video_path}")
                        panel_size = (video_size[0] // 2, video_size[1])
                        w_panel = vid_wrist if vid_wrist.shape[:2] == (panel_size[1], panel_size[0]) else cv2.resize(vid_wrist, panel_size)
                        t_panel = vid_tripod if vid_tripod.shape[:2] == (panel_size[1], panel_size[0]) else cv2.resize(vid_tripod, panel_size)
                        pair = np.concatenate((w_panel, t_panel), axis=1)
                        writer.write(cv2.cvtColor(pair, cv2.COLOR_RGB2BGR))
                        record.video_path = str(video_path)
                    if record.aborted:
                        break
            if writer is not None:
                active_writer, writer = writer, None
                active_writer.release()
        except BaseException:
            if not getattr(bridge, "estopped", False) and not record.aborted:
                estop_result = bridge.emergency_stop("trial_recording_failed")
            record = replace(
                record,
                aborted=True,
                abort_reason=(
                    record.abort_reason
                    or getattr(bridge, "estop_reason", None)
                    or "trial_recording_failed"
                ),
                labeled=True,
                legal_delivery=False,
                legal_strike=False,
                credited_pins=0,
                wobble_only_pins=0,
            )
            _record_estop(record, bridge, estop_result)
            if on_abort is not None:
                on_abort(record)
            raise
        finally:
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass

    if getattr(bridge, "estopped", False) and not record.aborted:
        record.aborted = True
        record.abort_reason = getattr(bridge, "estop_reason", None) or "bridge_estopped"
    if record.aborted:
        record = replace(
            record, labeled=True, legal_delivery=False, legal_strike=False,
            credited_pins=0, wobble_only_pins=0,
        )
        _record_estop(record, bridge, estop_result)
    elif record.torque_disabled_confirmed is False:
        _record_estop(record, bridge)
    return record


def _label_trial(record: TrialRecord, label: dict[str, Any]) -> TrialRecord:
    if label["legal_strike"] and (not label["legal_delivery"] or label["credited_pins"] != 3):
        raise ValueError("A legal strike requires a legal delivery and 3 credited pins")
    return replace(
        record,
        labeled=True,
        legal_delivery=label["legal_delivery"],
        legal_strike=label["legal_strike"],
        credited_pins=label["credited_pins"],
        wobble_only_pins=label.get("wobble_only_pins", 0),
    )


def run_real_trials(
    policies: list[tuple[str, TrialBridge]],
    num_trials: int,
    max_steps: int,
    out_dir: str | Path,
    mode: str = "dry_run",
    outcome_labeler: str = "human_video_review",
    fps: float = 20.0,
    outcome_labels: dict[int, dict[str, Any]] | None = None,
    label_trial: Callable[[TrialRecord], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Schedule num_trials in round-robin order and skip E-stopped live bridges."""
    if not policies or num_trials < 1:
        raise ValueError("Expected at least one policy and one trial")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[TrialRecord] = []
    labels = outcome_labels or {}
    stopped_due_to_estop = False
    with (out_dir / "trials.jsonl").open("w", encoding="utf-8") as trials_file:
        def save_trial(bridge: TrialBridge, record: TrialRecord) -> None:
            nonlocal stopped_due_to_estop
            if mode == "live" and getattr(bridge, "estopped", False):
                stopped_due_to_estop = True
            records.append(record)
            trials_file.write(json.dumps(asdict(record), allow_nan=False) + "\n")
            trials_file.flush()

        try:
            for trial_id, (name, bridge) in enumerate(islice(cycle(policies), num_trials), start=1):
                if mode == "live" and getattr(bridge, "estopped", False):
                    stopped_due_to_estop = True
                    continue
                record = run_trial_episode(
                    bridge, max_steps, out_dir, trial_id, name, mode,
                    outcome_label=labels.get(trial_id), outcome_labeler=outcome_labeler, fps=fps,
                    on_abort=partial(save_trial, bridge),
                )
                try:
                    if mode == "live" and not record.aborted and trial_id not in labels and label_trial:
                        record = _label_trial(record, label_trial(record))
                finally:
                    save_trial(bridge, record)
        finally:
            summary = summarize_trials(records)
            summary["stopped_due_to_estop"] = stopped_due_to_estop
            summary["torque_disable_unconfirmed_trial_ids"] = [
                record.trial_id for record in records if record.torque_disabled_confirmed is False
            ]
            _write_json(out_dir / "summary.json", summary)
    return summary
