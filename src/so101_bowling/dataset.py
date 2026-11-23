"""Dual-camera bowling episode dataset collection and SAC replay-buffer loader."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from so101_bowling.config import WorkspaceConfig, load_config
from so101_bowling.env import SO101BowlingEnv
from so101_bowling.kinematics import PushParams, ScriptedPushController
from so101_bowling.ppo_models import ACTOR_OBS_KEYS, DualCameraActor

_OBS_KEYS = (
    "wrist_rgb",
    "tripod_rgb",
    "camera_meta",
    "proprio",
    "privileged_state",
    "privileged_aux_targets",
    "privileged_aux_mask",
)
_NEXT_KEYS = (
    "wrist_rgb",
    "tripod_rgb",
    "camera_meta",
    "proprio",
    "privileged_state",
)


def collect_dataset(
    out_dir: str | Path,
    num_episodes: int = 50,
    cfg: WorkspaceConfig | None = None,
    dr_enabled: bool = True,
    render_rgb: bool = True,
    frame_stack: int = 2,
    seed: int = 0,
    actor: DualCameraActor | None = None,
    policy_name: str | None = None,
    jitter_scripted: bool = True,
    only_legal: bool = False,
    max_attempts_multiplier: int = 5,
) -> dict[str, Any]:
    """Collect dual-camera bowling episodes from ScriptedPushController or a DualCameraActor.

    Saves compressed per-episode ``.npz`` files under ``out_dir/episodes/``,
    per-episode summaries in ``out_dir/episodes.jsonl``, and aggregate metadata
    in ``out_dir/dataset_info.json``.
    """
    if num_episodes < 1:
        raise ValueError("num_episodes must be positive")
    if max_attempts_multiplier < 1:
        raise ValueError("max_attempts_multiplier must be positive")

    wcfg = cfg if cfg is not None else load_config()
    root = Path(out_dir)
    ep_dir = root / "episodes"
    ep_dir.mkdir(parents=True, exist_ok=True)
    for stale_ep in ep_dir.glob("ep_*.npz"):
        stale_ep.unlink()
    jsonl_path = root / "episodes.jsonl"

    rng = np.random.default_rng(seed)
    source = policy_name or ("actor" if actor is not None else "scripted_4dof")
    if actor is not None:
        actor.eval()

    saved_episodes = 0
    total_transitions = 0
    attempts = 0
    max_attempts = num_episodes * max_attempts_multiplier
    outcomes: list[dict[str, Any]] = []

    with (
        SO101BowlingEnv(
            cfg=wcfg,
            dr_enabled=dr_enabled,
            render_rgb=render_rgb,
            frame_stack=frame_stack,
            gamma=wcfg.reward.gamma,
        ) as env,
        jsonl_path.open("w", encoding="utf-8") as jf,
    ):
        while saved_episodes < num_episodes and attempts < max_attempts:
            ep_seed = seed + attempts
            attempts += 1
            obs, reset_info = env.reset(seed=ep_seed)

            controller = None
            if actor is None:
                if jitter_scripted:
                    params = PushParams(
                        aim_yaw_offset_rad=float(rng.uniform(-0.04, 0.04)),
                        push_speed_mps=float(rng.uniform(0.75, 0.95)),
                        follow_through_m=float(rng.uniform(0.12, 0.16)),
                    )
                else:
                    params = PushParams()
                controller = ScriptedPushController(
                    env.model, env.ids, wcfg, env.q_min, env.q_max, params=params
                )
                controller.reset(env._last_snapshot.ball_pos, env._last_snapshot.pin_positions)

            obs_buf: dict[str, list[np.ndarray]] = {k: [] for k in _OBS_KEYS}
            next_buf: dict[str, list[np.ndarray]] = {k: [] for k in _NEXT_KEYS}
            act_buf: list[np.ndarray] = []
            rew_buf: list[float] = []
            term_buf: list[bool] = []
            trunc_buf: list[bool] = []

            done = False
            info: dict[str, Any] = {}
            while not done:
                if actor is not None:
                    actor_obs = {k: obs[k] for k in ACTOR_OBS_KEYS}
                    action = actor.act(actor_obs, deterministic=True).cpu().numpy().astype(np.float32)
                else:
                    assert controller is not None
                    action = controller.step(env._measured_q())

                next_obs, reward, terminated, truncated, info = env.step(action)
                for k in _OBS_KEYS:
                    obs_buf[k].append(obs[k])
                for k in _NEXT_KEYS:
                    next_buf[k].append(next_obs[k])
                act_buf.append(np.asarray(action, dtype=np.float32))
                rew_buf.append(float(reward))
                term_buf.append(bool(terminated))
                trunc_buf.append(bool(truncated))
                obs = next_obs
                done = terminated or truncated

            outcome = info.get("outcome", {})
            if only_legal and not outcome.get("legal_delivery", False):
                if attempts % 10 == 0:
                    print(
                        f"[collect_dataset] attempt {attempts}/{max_attempts}: "
                        f"saved {saved_episodes}/{num_episodes} "
                        f"(skipped: {outcome.get('invalid_reason')})",
                        flush=True,
                    )
                continue

            ep_path = ep_dir / f"ep_{saved_episodes:06d}.npz"
            arrays: dict[str, np.ndarray] = {
                **{f"obs_{k}": np.stack(obs_buf[k], axis=0) for k in _OBS_KEYS},
                **{f"next_{k}": np.stack(next_buf[k], axis=0) for k in _NEXT_KEYS},
                "actions": np.stack(act_buf, axis=0),
                "rewards": np.asarray(rew_buf, dtype=np.float32),
                "terminated": np.asarray(term_buf, dtype=bool),
                "truncated": np.asarray(trunc_buf, dtype=bool),
            }
            np.savez_compressed(ep_path, **arrays)

            ep_meta = {
                "episode_id": saved_episodes,
                "seed": ep_seed,
                "steps": len(rew_buf),
                "episode_return": float(sum(rew_buf)),
                "file": f"episodes/{ep_path.name}",
                "outcome": outcome,
                "dr_enabled": bool(dr_enabled),
                "randomization": _to_jsonable(reset_info.get("randomization", {})),
            }
            jf.write(json.dumps(ep_meta) + "\n")
            outcomes.append(outcome)
            total_transitions += len(rew_buf)
            saved_episodes += 1
            if saved_episodes % 10 == 0 or saved_episodes == num_episodes:
                legal_so_far = sum(int(bool(o.get("legal_delivery", False))) for o in outcomes)
                print(
                    f"[collect_dataset] saved {saved_episodes}/{num_episodes} "
                    f"legal={legal_so_far}/{saved_episodes} "
                    f"transitions={total_transitions} (attempts={attempts})",
                    flush=True,
                )

    if only_legal and saved_episodes < num_episodes:
        raise RuntimeError(
            f"collect_dataset(only_legal=True) collected only {saved_episodes}/{num_episodes} "
            f"legal episodes within max_attempts={max_attempts}."
        )

    n = max(1, saved_episodes)
    dataset_info: dict[str, Any] = {
        "source_policy": source,
        "num_episodes": saved_episodes,
        "attempts": attempts,
        "total_transitions": total_transitions,
        "dr_enabled": bool(dr_enabled),
        "render_rgb": bool(render_rgb),
        "frame_stack": frame_stack,
        "img_hw": [wcfg.camera.render_height, wcfg.camera.render_width],
        "seed": seed,
        "only_legal": bool(only_legal),
        "legal_delivery_rate": (
            sum(int(bool(o.get("legal_delivery", False))) for o in outcomes) / n
            if outcomes
            else 0.0
        ),
        "legal_strike_rate": (
            sum(int(bool(o.get("legal_strike", False))) for o in outcomes) / n
            if outcomes
            else 0.0
        ),
        "mean_credited_pins": (
            float(np.mean([int(o.get("credited_pins", 0)) for o in outcomes]))
            if outcomes
            else 0.0
        ),
        "workspace_config": asdict(wcfg),
    }
    (root / "dataset_info.json").write_text(json.dumps(dataset_info, indent=2) + "\n")
    return dataset_info


def _to_jsonable(val: Any) -> Any:
    if isinstance(val, (np.ndarray, np.generic)):
        return _to_jsonable(val.tolist())
    if isinstance(val, dict):
        return {str(k): _to_jsonable(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_to_jsonable(v) for v in val]
    return val


def load_dataset_into_replay_buffer(
    dataset_dir: str | Path,
    buffer: Any,
    max_transitions: int | None = None,
) -> int:
    """Load saved ``episodes/ep_*.npz`` transitions into a ``RawFrameReplayBuffer``.

    Packs transitions across ``buffer.num_envs`` parallel lanes to preserve
    per-env frame-pool deduplication. Returns the number of transitions inserted.
    """
    root = Path(dataset_dir)
    ep_files = sorted((root / "episodes").glob("ep_*.npz"))
    if not ep_files:
        raise FileNotFoundError(f"No episode .npz files found in {root / 'episodes'}")

    num_envs = int(buffer.num_envs)
    # Concatenate all transitions in episode order, then slice into num_envs contiguous lanes.
    obs_lists: dict[str, list[np.ndarray]] = {k: [] for k in _OBS_KEYS}
    next_lists: dict[str, list[np.ndarray]] = {k: [] for k in _NEXT_KEYS}
    act_list: list[np.ndarray] = []
    rew_list: list[np.ndarray] = []
    term_list: list[np.ndarray] = []
    trunc_list: list[np.ndarray] = []

    total = 0
    for ep_path in ep_files:
        with np.load(ep_path) as data:
            n_steps = int(data["rewards"].shape[0])
            if n_steps == 0:
                continue
            for k in _OBS_KEYS:
                obs_lists[k].append(data[f"obs_{k}"])
            for k in _NEXT_KEYS:
                next_lists[k].append(data[f"next_{k}"])
            act_list.append(data["actions"])
            rew_list.append(data["rewards"])
            term_list.append(data["terminated"])
            trunc_list.append(data["truncated"])
            total += n_steps
            if max_transitions is not None and total >= max_transitions:
                break

    if total < num_envs:
        raise ValueError(
            f"Dataset has only {total} transitions; need at least num_envs={num_envs}"
        )

    limit = min(total, max_transitions) if max_transitions is not None else total
    steps_per_lane = limit // num_envs
    usable = steps_per_lane * num_envs

    obs_all = {k: np.concatenate(obs_lists[k], axis=0)[:usable] for k in _OBS_KEYS}
    next_all = {k: np.concatenate(next_lists[k], axis=0)[:usable] for k in _NEXT_KEYS}
    actions_all = np.concatenate(act_list, axis=0)[:usable]
    rewards_all = np.concatenate(rew_list, axis=0)[:usable]
    term_all = np.concatenate(term_list, axis=0)[:usable]
    trunc_all = np.concatenate(trunc_list, axis=0)[:usable]

    # Reshape into (num_envs, steps_per_lane, ...) so each env lane gets contiguous steps.
    obs_lanes = {k: v.reshape(num_envs, steps_per_lane, *v.shape[1:]) for k, v in obs_all.items()}
    next_lanes = {k: v.reshape(num_envs, steps_per_lane, *v.shape[1:]) for k, v in next_all.items()}
    act_lanes = actions_all.reshape(num_envs, steps_per_lane, *actions_all.shape[1:])
    rew_lanes = rewards_all.reshape(num_envs, steps_per_lane)
    term_lanes = term_all.reshape(num_envs, steps_per_lane)
    trunc_lanes = trunc_all.reshape(num_envs, steps_per_lane)

    for step in range(steps_per_lane):
        obs_step = {k: obs_lanes[k][:, step] for k in _OBS_KEYS}
        next_step = {k: next_lanes[k][:, step] for k in _NEXT_KEYS}
        trunc_step = trunc_lanes[:, step]
        infos: list[dict[str, Any]] = []
        for env_idx in range(num_envs):
            if trunc_step[env_idx] and not term_lanes[env_idx, step]:
                infos.append({"final_obs": {k: next_step[k][env_idx] for k in _NEXT_KEYS}})
            else:
                infos.append({})
        buffer.add(
            obs=obs_step,
            actions=act_lanes[:, step],
            rewards=rew_lanes[:, step],
            next_obs=next_step,
            terminated=term_lanes[:, step],
            truncated=trunc_step,
            infos=infos,
        )

    return usable
