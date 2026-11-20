"""Camera timing and fixed physical scaling for policy observations."""

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np


def preprocess_camera_frame(rgb: np.ndarray, out_h: int = 64, out_w: int = 112) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(*rgb.shape[:2], out_h, out_w) < 1:
        raise ValueError("Expected a nonempty H×W×3 RGB frame and positive output dimensions")
    height, width = rgb.shape[:2]
    crop_height = min(height, max(1, width * out_h // out_w))
    crop_width = min(width, max(1, height * out_w // out_h))
    top, left = (height - crop_height) // 2, (width - crop_width) // 2
    crop = rgb[top:top + crop_height, left:left + crop_width]
    if crop.dtype != np.uint8:
        crop = np.clip(crop, 0, 255).astype(np.uint8)
    resized = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(resized)


@dataclass
class CameraDelayBuffer:
    """One buffer per camera. Each push advances its frame clock by one tick.

    Delay jitter is uniform over integer ticks in ±jitter_frames. When several
    captures are due on one tick, the newest is delivered. Between deliveries,
    push returns the previous frame with its original capture timestamp.
    """

    delay_frames: int = 0
    jitter_frames: int = 0
    drop_probability: float = 0.0
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def __post_init__(self) -> None:
        if self.delay_frames < 0 or self.jitter_frames < 0:
            raise ValueError("Frame delay and jitter must be nonnegative")
        if not 0.0 <= self.drop_probability <= 1.0:
            raise ValueError("Drop probability must be in [0, 1]")
        self.reset()

    def reset(self) -> None:
        self._tick = -1
        self._delivered_tick = -1
        self._pending: list[tuple[int, int, float, np.ndarray]] = []
        self._latest: tuple[np.ndarray, float] | None = None

    def push(self, frame: np.ndarray, capture_t: float) -> tuple[np.ndarray, float] | None:
        self._tick += 1
        if self.drop_probability == 0.0 or self.rng.random() >= self.drop_probability:
            jitter = 0
            if self.jitter_frames:
                jitter = int(self.rng.integers(-self.jitter_frames, self.jitter_frames + 1))
            due = self._tick + max(0, self.delay_frames + jitter)
            self._pending.append((due, self._tick, capture_t, frame.copy()))

        pending = []
        for due, tick, timestamp, captured in self._pending:
            if due <= self._tick:
                if tick > self._delivered_tick:
                    self._delivered_tick = tick
                    self._latest = captured, timestamp
            elif tick > self._delivered_tick:
                pending.append((due, tick, timestamp, captured))
        self._pending = pending
        return self._latest


@dataclass
class TimestampedFrameStack:
    """One history per camera, ordered oldest to newest."""

    k: int = 2

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("Frame stack size must be positive")
        self.reset()

    def reset(self) -> None:
        self._samples: deque[tuple[np.ndarray, float]] = deque(maxlen=self.k)
        self._fresh = False
        self._last_seen_t = -np.inf

    def push(self, frame: np.ndarray, capture_t: float) -> bool:
        """Keep distinct captures and report whether the stack advanced."""
        in_order = capture_t > self._last_seen_t
        self._last_seen_t = max(self._last_seen_t, capture_t)
        self._fresh = in_order and (
            not self._samples or not np.array_equal(self._samples[-1][0], frame)
        )
        if self._fresh:
            self._samples.append((frame.copy(), capture_t))
        return self._fresh

    @property
    def frames(self) -> tuple[np.ndarray, ...]:
        return tuple(frame for frame, _ in self._samples)

    @property
    def timestamps(self) -> tuple[float, ...]:
        return tuple(timestamp for _, timestamp in self._samples)

    def age_s(self, now_t: float) -> float:
        return max(0.0, now_t - self._samples[-1][1]) if self._samples else float("inf")

    def is_fresh(self, now_t: float, max_age_s: float) -> bool:
        return self._fresh and self.age_s(now_t) <= max_age_s


class CausalJointVelocityEstimator:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._t: float | None = None
        self._q: np.ndarray | None = None
        self._dq: np.ndarray | None = None

    def update(self, t: float, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if self._t is None:
            self._dq = np.zeros_like(q)
        elif t <= self._t:
            return self._dq.copy()
        else:
            self._dq = (q - self._q) / (t - self._t)
        self._t = t
        self._q = q.copy()
        return self._dq.copy()

    def is_stale(self, now_t: float, max_stale_s: float) -> bool:
        return self._t is None or now_t - self._t > max_stale_s


def scale_proprio(q_rad, dq_rad_s, prev_applied_delta, q_min, q_max, max_delta, dq_scale=4.0) -> np.ndarray:
    """Concatenate clipped joint position, velocity, and previous action features."""
    lower, upper = np.asarray(q_min), np.asarray(q_max)
    joint_span = upper - lower
    if np.any(joint_span <= 0) or np.any(np.asarray(max_delta) <= 0) or dq_scale <= 0:
        raise ValueError("Joint limits and scales must define positive ranges")
    position = 2.0 * (np.asarray(q_rad) - lower) / joint_span - 1.0
    velocity = np.asarray(dq_rad_s) / dq_scale
    delta = np.asarray(prev_applied_delta) / max_delta
    return np.clip(np.concatenate((position, velocity, delta)), -1.0, 1.0).astype(np.float32)
