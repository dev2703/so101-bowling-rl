"""Run passive dual-camera checks or record real-transfer trials."""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

from so101_bowling.config import load_calibration, load_config
from so101_bowling.real_eval import TrialRecord, run_passive_camera_gate, run_real_trials


def _open_live_cameras(stack, camera_type, hardware, wrist_index=None, tripod_index=None):
    settings = {"width": hardware.cam_width, "height": hardware.cam_height, "fps": hardware.cam_fps}
    wrist = stack.enter_context(camera_type(
        hardware.wrist_cam_index if wrist_index is None else wrist_index,
        rotation_deg=hardware.wrist_rotation_deg, **settings,
    ))
    tripod = stack.enter_context(camera_type(
        hardware.tripod_cam_index if tripod_index is None else tripod_index,
        rotation_deg=hardware.tripod_rotation_deg, **settings,
    ))
    for name, camera in (("wrist", wrist), ("tripod", tripod)):
        if not camera.wait_ready():
            raise RuntimeError(f"No initial frame from the {name} camera")
    return wrist, tripod


def _passive_gate(args: argparse.Namespace) -> dict:
    from so101_bowling.hardware import (
        OpenCVTimestampedCamera,
        VideoFileTimestampedCamera,
        load_hardware_config,
    )

    workspace = load_config(args.workspace_config)
    hardware = load_hardware_config(args.hardware_config)
    actor = None
    if args.checkpoint is not None:
        from so101_bowling.ppo_models import load_actor

        actor = load_actor(args.checkpoint)
    with ExitStack() as stack:
        if args.wrist_video is not None:
            wrist = stack.enter_context(VideoFileTimestampedCamera(
                args.wrist_video, rotation_deg=hardware.wrist_rotation_deg,
            ))
            tripod = stack.enter_context(VideoFileTimestampedCamera(
                args.tripod_video, rotation_deg=hardware.tripod_rotation_deg,
            ))
            interval, clock = 0.0, None
        else:
            wrist, tripod = _open_live_cameras(
                stack, OpenCVTimestampedCamera, hardware, args.wrist_index, args.tripod_index,
            )
            interval, clock = 1 / hardware.control_hz, time.monotonic
        return run_passive_camera_gate(
            wrist, tripod, workspace, args.out_dir, num_samples=args.samples,
            max_camera_age_s=hardware.max_camera_age_s, actor=actor,
            clock=clock, sample_interval_s=interval,
        )


def _operator_label(record: TrialRecord) -> dict:
    print(f"Trial {record.trial_id} video {record.video_path}")
    while True:
        answer = input("Enter legal delivery (y/n), credited pins, wobble-only pins: ").split()
        try:
            delivery, credit_text, wobble_text = answer
            credited, wobble = int(credit_text), int(wobble_text)
            if delivery not in ("y", "n") or min(credited, wobble) < 0 or credited + wobble > 3:
                raise ValueError
        except ValueError:
            print("Enter y or n followed by two pin counts totaling at most 3.")
            continue
        return {
            "legal_delivery": delivery == "y",
            "legal_strike": delivery == "y" and credited == 3,
            "credited_pins": credited,
            "wobble_only_pins": wobble,
        }


def _run_trials(args: argparse.Namespace) -> dict:
    from so101_bowling.hardware import (
        OpenCVTimestampedCamera,
        SO101BowlingHardwareBridge,
        load_hardware_config,
    )
    hardware = load_hardware_config(args.hardware_config)
    calibration_path = str(args.calibration) if args.calibration is not None else hardware.calibration_path
    if args.live and not calibration_path:
        raise SystemExit("run-trials --live requires --calibration or calibration_path in the hardware config")
    hardware = replace(hardware, dry_run=not args.live, calibration_path=calibration_path)
    workspace = load_config(args.workspace_config)
    calibration = load_calibration(live_json_path=hardware.calibration_path)

    from so101_bowling.ppo_models import load_actor

    actor = load_actor(args.checkpoint)

    class PacedBridge(SO101BowlingHardwareBridge):
        def reset_episode(self) -> None:
            self.reset()
            self._next_step_t = time.monotonic()

        def step(self):
            delay = self._next_step_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._next_step_t = time.monotonic() + 1 / hardware.control_hz
            return super().step()

    with ExitStack() as stack:
        wrist, tripod = _open_live_cameras(stack, OpenCVTimestampedCamera, hardware)
        bridge = PacedBridge(actor, workspace, calibration, hardware, wrist, tripod)
        stack.callback(bridge.close)
        bridge.connect_robot()
        try:
            return run_real_trials(
                [(args.policy_name, bridge)], args.trials, args.max_steps, args.out_dir,
                mode="live" if args.live else "dry_run", outcome_labeler=args.labeler,
                fps=hardware.control_hz,
                label_trial=_operator_label if args.labeler == "operator_cli" else None,
            )
        except BaseException:
            if args.live and not bridge.estopped:
                bridge.emergency_stop("trial_run_interrupted")
            raise


def _positive_int(value: str) -> int:
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--hardware-config", type=Path)
    common.add_argument("--workspace-config", type=Path)
    common.add_argument("--out-dir", type=Path, required=True)
    subcommands = parser.add_subparsers(dest="command", required=True)

    passive = subcommands.add_parser("passive-gate", parents=[common], help="Sample camera timing.")
    passive.add_argument("--wrist-video", type=Path)
    passive.add_argument("--tripod-video", type=Path)
    passive.add_argument("--wrist-index", type=int)
    passive.add_argument("--tripod-index", type=int)
    passive.add_argument("--samples", type=_positive_int, default=40)
    passive.add_argument("--checkpoint", type=Path)

    trials = subcommands.add_parser("run-trials", parents=[common], help="Record policy trials.")
    trials.add_argument("--checkpoint", type=Path, required=True)
    trials.add_argument("--policy-name", required=True)
    trials.add_argument("--trials", type=_positive_int, default=1)
    trials.add_argument("--max-steps", type=_positive_int, default=80)
    trials.add_argument("--calibration", type=Path, help="Path to live LeRobot calibration JSON.")
    trials.add_argument("--live", action="store_true", help="Enable motor commands.")
    trials.add_argument("--labeler", default="human_video_review",
                        help="Use operator_cli to enter live outcomes after each recording.")

    args = parser.parse_args()
    if args.command == "passive-gate":
        if (args.wrist_video is None) != (args.tripod_video is None):
            passive.error("--wrist-video and --tripod-video must be supplied together")
        if args.wrist_video is not None and (args.wrist_index is not None or args.tripod_index is not None):
            passive.error("video replay and camera indices are mutually exclusive")
        report = _passive_gate(args)
    else:
        report = _run_trials(args)
        unconfirmed = report.get("torque_disable_unconfirmed_trial_ids", [])
        if unconfirmed:
            trial_ids = ", ".join(map(str, unconfirmed))
            print(f"CRITICAL WARNING: Torque disable unconfirmed for trial(s) {trial_ids}.", file=sys.stderr)
        if report.get("stopped_due_to_estop"):
            print("CRITICAL WARNING: Live trials stopped on an E-stopped bridge.", file=sys.stderr)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
