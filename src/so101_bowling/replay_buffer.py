"""Vectorized replay with generation-checked references to raw camera frames."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch


_CAMERAS = ("wrist", "tripod")
_SUCCESSOR_FIELDS = ("proprio", "camera_meta", "privileged_state")


class RawFrameReplayBuffer:
    """Store vector-env transitions and sample rows whose camera frames still exist."""

    def __init__(
        self,
        capacity: int,
        num_envs: int,
        frame_stack: int = 2,
        img_hw: tuple[int, int] = (64, 112),
        action_dim: int = 4,
        state_dim: int = 46,
        proprio_dim: int = 12,
        camera_meta_dim: int = 4,
        aux_target_dim: int = 6,
        aux_mask_dim: int = 3,
        frame_pool_size_per_env: int | None = None,
    ) -> None:
        if num_envs < 1 or frame_stack < 1 or min(img_hw) < 1:
            raise ValueError("num_envs, frame_stack, and image dimensions must be positive")
        per_env_cap = capacity // num_envs
        if per_env_cap < frame_stack + 2:
            raise ValueError("Per-env capacity must be at least frame_stack + 2")
        pool_len = (
            per_env_cap + frame_stack + 4
            if frame_pool_size_per_env is None
            else frame_pool_size_per_env
        )
        if pool_len < 1:
            raise ValueError("Frame pool size must be positive")

        self.num_envs = num_envs
        self.per_env_cap = per_env_cap
        self.capacity = per_env_cap * num_envs
        self._stack_shape = (frame_stack, *img_hw, 3)
        self._pool_len = pool_len
        self._search_len = min(pool_len, frame_stack + 4)
        self._write_ptr = 0
        self._truncated_final_images: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._obs_dims = {
            "proprio": proprio_dim,
            "camera_meta": camera_meta_dim,
            "privileged_state": state_dim,
            "privileged_aux_targets": aux_target_dim,
            "privileged_aux_mask": aux_mask_dim,
        }
        self._next_dims = {key: self._obs_dims[key] for key in _SUCCESSOR_FIELDS}
        self._action_dim = action_dim

        ring_shape = (num_envs, per_env_cap)
        refs_shape = (*ring_shape, frame_stack)
        for camera in _CAMERAS:
            setattr(self, f"{camera}_pool", np.zeros((num_envs, pool_len, *img_hw, 3), np.uint8))
            setattr(self, f"{camera}_pool_gen", np.zeros((num_envs, pool_len), np.uint64))
            setattr(self, f"_{camera}_pool_ptr", np.zeros(num_envs, np.intp))
            for prefix in ("obs", "next"):
                setattr(self, f"{prefix}_{camera}_idx", np.zeros(refs_shape, np.intp))
                setattr(self, f"{prefix}_{camera}_gen", np.zeros(refs_shape, np.uint64))
        for prefix, dims in (("obs", self._obs_dims), ("next", self._next_dims)):
            for key, dim in dims.items():
                setattr(self, f"{prefix}_{key}", np.zeros((*ring_shape, dim), np.float32))

        self.actions = np.zeros((*ring_shape, action_dim), np.float32)
        self.rewards = np.zeros(ring_shape, np.float32)
        self.terminated = np.zeros(ring_shape, bool)
        self.truncated = np.zeros(ring_shape, bool)
        self._is_ordinary = np.zeros(ring_shape, bool)
        self.occupied = np.zeros(ring_shape, bool)

    @staticmethod
    def _checked_array(
        value: Any, shape: tuple[int, ...], dtype: Any, name: str, *, exact_dtype: bool = False
    ) -> np.ndarray:
        value = np.asarray(value) if exact_dtype else np.asarray(value, dtype=dtype)
        if value.shape != shape or value.dtype != np.dtype(dtype):
            raise ValueError(f"{name} must have shape {shape} and dtype {np.dtype(dtype)}")
        return value

    def _check_obs(
        self, obs: Mapping[str, np.ndarray], dims: Mapping[str, int], leading: tuple[int, ...]
    ) -> dict[str, np.ndarray]:
        result = {
            key: self._checked_array(obs[key], (*leading, dim), np.float32, key)
            for key, dim in dims.items()
        }
        for camera in _CAMERAS:
            key = f"{camera}_rgb"
            result[key] = self._checked_array(
                obs[key], (*leading, *self._stack_shape), np.uint8, key, exact_dtype=True
            )
        return result

    def _store_stacks(
        self, prefix: str, env: int, transition: int, obs: Mapping[str, np.ndarray]
    ) -> None:
        for camera in _CAMERAS:
            pool = getattr(self, f"{camera}_pool")[env]
            generations = getattr(self, f"{camera}_pool_gen")[env]
            pointers = getattr(self, f"_{camera}_pool_ptr")
            indices = getattr(self, f"{prefix}_{camera}_idx")[env, transition]
            refs = getattr(self, f"{prefix}_{camera}_gen")[env, transition]
            ptr = int(pointers[env])
            for k, rgb in enumerate(obs[f"{camera}_rgb"]):
                for offset in range(1, self._search_len + 1):
                    slot = (ptr - offset) % self._pool_len
                    if generations[slot] and np.array_equal(pool[slot], rgb):
                        break
                else:
                    slot = ptr
                    # The slot before ptr holds the most recent write, including after wrap.
                    generation = int(generations[ptr - 1]) + 1
                    pool[slot] = rgb
                    generations[slot] = generation
                    ptr = (ptr + 1) % self._pool_len
                indices[k] = slot
                refs[k] = generations[slot]
            pointers[env] = ptr

    def add(
        self,
        obs: Mapping[str, np.ndarray],
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: Mapping[str, np.ndarray],
        terminated: np.ndarray,
        truncated: np.ndarray,
        infos: Sequence[Mapping[str, Any]],
    ) -> None:
        leading = (self.num_envs,)
        obs = self._check_obs(obs, self._obs_dims, leading)
        next_obs = self._check_obs(next_obs, self._next_dims, leading)
        actions = self._checked_array(actions, (*leading, self._action_dim), np.float32, "actions")
        rewards = self._checked_array(rewards, leading, np.float32, "rewards")
        terminated = self._checked_array(terminated, leading, bool, "terminated")
        truncated = self._checked_array(truncated, leading, bool, "truncated") & ~terminated
        ordinary = ~(terminated | truncated)
        if len(infos) != self.num_envs:
            raise ValueError("infos must have one entry per environment")

        finals = {}
        for env in np.flatnonzero(truncated):
            if "final_obs" not in infos[env]:
                raise ValueError(f"Truncated transition for env {env} missing info['final_obs']")
            finals[env] = self._check_obs(infos[env]["final_obs"], self._next_dims, ())

        t = self._write_ptr
        for env in range(self.num_envs):
            self._truncated_final_images.pop((env, t), None)
            self._store_stacks("obs", env, t, {f"{c}_rgb": obs[f"{c}_rgb"][env] for c in _CAMERAS})
            if ordinary[env]:
                successor = {key: value[env] for key, value in next_obs.items()}
                self._store_stacks("next", env, t, successor)
            else:
                for camera in _CAMERAS:
                    getattr(self, f"next_{camera}_gen")[env, t] = 0
                successor = finals.get(env, {})
                if truncated[env]:
                    self._truncated_final_images[env, t] = (
                        successor["wrist_rgb"].copy(),
                        successor["tripod_rgb"].copy(),
                    )

            for key in self._next_dims:
                getattr(self, f"next_{key}")[env, t] = successor.get(key, 0.0)

        for key in self._obs_dims:
            getattr(self, f"obs_{key}")[:, t] = obs[key]
        self.actions[:, t] = actions
        self.rewards[:, t] = rewards
        self.terminated[:, t] = terminated
        self.truncated[:, t] = truncated
        self._is_ordinary[:, t] = ordinary
        self.occupied[:, t] = True
        self._write_ptr = (t + 1) % self.per_env_cap

    def _valid_mask(self) -> np.ndarray:
        env_idx = np.arange(self.num_envs)[:, None, None]
        valid = self.occupied.copy()
        for camera in _CAMERAS:
            generations = getattr(self, f"{camera}_pool_gen")
            obs_ok = np.all(
                generations[env_idx, getattr(self, f"obs_{camera}_idx")]
                == getattr(self, f"obs_{camera}_gen"),
                axis=-1,
            )
            next_ok = np.all(
                generations[env_idx, getattr(self, f"next_{camera}_idx")]
                == getattr(self, f"next_{camera}_gen"),
                axis=-1,
            )
            valid &= obs_ok & (~self._is_ordinary | next_ok)
        return valid

    def valid_indices(self) -> tuple[np.ndarray, np.ndarray]:
        return np.nonzero(self._valid_mask())

    @property
    def valid_size(self) -> int:
        return int(self._valid_mask().sum())

    def __len__(self) -> int:
        return self.valid_size

    def memory_bytes(self) -> int:
        return sum(value.nbytes for value in vars(self).values() if isinstance(value, np.ndarray)) + sum(
            wrist.nbytes + tripod.nbytes for wrist, tripod in self._truncated_final_images.values()
        )

    def sample(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
        rng: np.random.Generator | None = None,
    ) -> dict[str, Any]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        envs, transitions = self.valid_indices()
        if envs.size < batch_size:
            raise ValueError(f"Requested {batch_size} samples with only {envs.size} valid transitions")
        if rng is None:
            rng = np.random.default_rng()
        selected = rng.choice(envs.size, size=batch_size, replace=False)
        envs, transitions = envs[selected], transitions[selected]
        rows = (envs, transitions)

        def tensor(value: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(value, device=device)

        obs = {key: tensor(getattr(self, f"obs_{key}")[rows]) for key in self._obs_dims}
        next_obs = {key: tensor(getattr(self, f"next_{key}")[rows]) for key in self._next_dims}
        ordinary = self._is_ordinary[rows]
        ordinary_rows = (envs[ordinary], transitions[ordinary])
        truncated_rows = np.flatnonzero(self.truncated[rows])
        for camera_idx, camera in enumerate(_CAMERAS):
            key = f"{camera}_rgb"
            pool = getattr(self, f"{camera}_pool")
            obs[key] = tensor(pool[envs[:, None], getattr(self, f"obs_{camera}_idx")[rows]])
            successor = np.zeros((batch_size, *self._stack_shape), dtype=np.uint8)
            successor[ordinary] = pool[
                ordinary_rows[0][:, None], getattr(self, f"next_{camera}_idx")[ordinary_rows]
            ]
            for row in truncated_rows:
                successor[row] = self._truncated_final_images[envs[row], transitions[row]][camera_idx]
            next_obs[key] = tensor(successor)
        return {
            "obs": obs,
            "next_obs": next_obs,
            "actions": tensor(self.actions[rows]),
            "rewards": tensor(self.rewards[rows]),
            "terminated": tensor(self.terminated[rows]),
            "truncated": tensor(self.truncated[rows]),
        }
