"""Synchronous and spawn-based vectorized SO-101 bowling environments."""

from __future__ import annotations

import multiprocessing as mp
import os
from typing import Any

import numpy as np

from so101_bowling.config import WorkspaceConfig, load_config
from so101_bowling.env import SO101BowlingEnv


def _stack_obs(obs_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.stack([obs[k] for obs in obs_list], axis=0) for k in obs_list[0]}


def _worker_loop(
    conn: Any,
    cfg: WorkspaceConfig,
    dr_enabled: bool,
    render_rgb: bool,
    frame_stack: int,
    gamma: float,
    worker_seed: int,
) -> None:
    import cv2

    cv2.setNumThreads(1)

    with conn, SO101BowlingEnv(
        cfg=cfg,
        dr_enabled=dr_enabled,
        render_rgb=render_rgb,
        frame_stack=frame_stack,
        gamma=gamma,
        seed=worker_seed,
    ) as env:
        conn.send((env.observation_space, env.action_space, float(env.scorer.config.gamma)))
        try:
            while True:
                cmd, payload = conn.recv()
                if cmd == "close":
                    return
                if cmd == "reset":
                    obs, _ = env.reset(seed=payload)
                    conn.send(obs)
                elif cmd == "step":
                    obs, reward, terminated, truncated, info = env.step(payload)
                    if terminated or truncated:
                        info = dict(info)
                        info["final_obs"] = obs
                        obs, _ = env.reset()
                    conn.send((obs, float(reward), bool(terminated), bool(truncated), info))
                else:
                    raise ValueError(f"Unknown worker command: {cmd}")
        except EOFError:
            return


class SyncVecBowlingEnv:
    """Single-process vectorized wrapper matching SubprocVecBowlingEnv."""

    def __init__(
        self,
        cfg: WorkspaceConfig,
        dr_enabled: bool = False,
        render_rgb: bool = True,
        frame_stack: int = 2,
        gamma: float = 0.99,
        base_seed: int = 0,
    ) -> None:
        self.num_envs = 1
        self._env = SO101BowlingEnv(
            cfg=cfg,
            dr_enabled=dr_enabled,
            render_rgb=render_rgb,
            frame_stack=frame_stack,
            gamma=gamma,
            seed=base_seed,
        )
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.scorer_gamma = float(self._env.scorer.config.gamma)
        self._closed = False

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        obs, _ = self._env.reset(seed=seed)
        return _stack_obs([obs])

    def step(
        self, actions: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        act = np.asarray(actions[0], dtype=np.float32)
        obs, reward, terminated, truncated, info = self._env.step(act)
        if terminated or truncated:
            info = dict(info)
            info["final_obs"] = obs
            obs, _ = self._env.reset()
        return (
            _stack_obs([obs]),
            np.asarray([reward], dtype=np.float32),
            np.asarray([terminated], dtype=bool),
            np.asarray([truncated], dtype=bool),
            [info],
        )

    def close(self) -> None:
        if not self._closed:
            self._env.close()
            self._closed = True

    def __enter__(self) -> SyncVecBowlingEnv:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class SubprocVecBowlingEnv:
    """Spawn-based multiprocessing vectorized wrapper keeping worker GL contexts isolated."""

    def __init__(
        self,
        num_envs: int,
        cfg: WorkspaceConfig,
        dr_enabled: bool = False,
        render_rgb: bool = True,
        frame_stack: int = 2,
        gamma: float = 0.99,
        base_seed: int = 0,
    ) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        self.num_envs = num_envs
        self._closed = False
        ctx = mp.get_context("spawn")
        self._parents: list[Any] = []
        self._procs: list[Any] = []

        thread_vars = (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
        try:
            previous = {name: os.environ.get(name) for name in thread_vars}
            try:
                os.environ.update(dict.fromkeys(thread_vars, "1"))
                for idx in range(num_envs):
                    parent, child = ctx.Pipe()
                    self._parents.append(parent)
                    with child:
                        proc = ctx.Process(
                            target=_worker_loop,
                            args=(
                                child,
                                cfg,
                                dr_enabled,
                                render_rgb,
                                frame_stack,
                                gamma,
                                base_seed + idx,
                            ),
                            daemon=True,
                        )
                        self._procs.append(proc)
                        proc.start()
            finally:
                for name, value in previous.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value

            spaces_list = [parent.recv() for parent in self._parents]
            self.observation_space, self.action_space, self.scorer_gamma = spaces_list[0]
        except BaseException:
            self.close()
            raise

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        for idx, parent in enumerate(self._parents):
            worker_seed = None if seed is None else seed + idx
            parent.send(("reset", worker_seed))
        return _stack_obs([parent.recv() for parent in self._parents])

    def step(
        self, actions: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        for idx, parent in enumerate(self._parents):
            parent.send(("step", np.asarray(actions[idx], dtype=np.float32)))
        results = [parent.recv() for parent in self._parents]
        obs = _stack_obs([r[0] for r in results])
        rewards = np.asarray([r[1] for r in results], dtype=np.float32)
        terminated = np.asarray([r[2] for r in results], dtype=bool)
        truncated = np.asarray([r[3] for r in results], dtype=bool)
        infos = [r[4] for r in results]
        return obs, rewards, terminated, truncated, infos

    def close(self) -> None:
        if self._closed:
            return
        for parent in self._parents:
            try:
                parent.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
            try:
                parent.close()
            except OSError:
                pass
        self._parents.clear()

        remaining = []
        for proc in self._procs:
            if proc.pid is not None:
                proc.join(timeout=5.0)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=2.0)
                    if proc.is_alive():
                        proc.kill()
                        proc.join(timeout=2.0)
                if proc.is_alive():
                    remaining.append(proc)
                    continue
            proc.close()
        self._procs = remaining
        self._closed = not remaining

    def __enter__(self) -> SubprocVecBowlingEnv:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def make_vec_env(
    num_envs: int,
    cfg: WorkspaceConfig | None = None,
    dr_enabled: bool = False,
    render_rgb: bool = True,
    frame_stack: int = 2,
    gamma: float = 0.99,
    base_seed: int = 0,
) -> SyncVecBowlingEnv | SubprocVecBowlingEnv:
    """Create SyncVecBowlingEnv for 1 env or spawn-based SubprocVecBowlingEnv for >1."""
    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs}")
    workspace_cfg = cfg if cfg is not None else load_config()
    if num_envs == 1:
        return SyncVecBowlingEnv(
            cfg=workspace_cfg,
            dr_enabled=dr_enabled,
            render_rgb=render_rgb,
            frame_stack=frame_stack,
            gamma=gamma,
            base_seed=base_seed,
        )
    return SubprocVecBowlingEnv(
        num_envs=num_envs,
        cfg=workspace_cfg,
        dr_enabled=dr_enabled,
        render_rgb=render_rgb,
        frame_stack=frame_stack,
        gamma=gamma,
        base_seed=base_seed,
    )
