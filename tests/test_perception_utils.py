import cv2
import numpy as np
import pytest

from so101_bowling.perception_utils import (
    CameraDelayBuffer,
    CausalJointVelocityEstimator,
    TimestampedFrameStack,
    preprocess_camera_frame,
    scale_proprio,
)


@pytest.mark.parametrize("height,width", [(720, 1280), (720, 960)])
def test_camera_resizes_center_crop_with_target_aspect(height, width):
    rgb = np.random.default_rng(0).integers(0, 256, (height, width, 3), dtype=np.uint8)
    if width == 1280:
        cropped = rgb[:, 10:-10]
    else:
        cropped = rgb[86:634]
    expected = cv2.resize(cropped, (112, 64), interpolation=cv2.INTER_AREA)
    actual = preprocess_camera_frame(rgb)
    assert actual.shape == (64, 112, 3)
    assert actual.dtype == np.uint8 and actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, expected)


def test_timestamped_stack_accepts_stationary_live_frames_and_rejects_stale_timestamps():
    stack = TimestampedFrameStack()
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    assert np.isinf(stack.age_s(1.0))
    assert not stack.is_fresh(1.0, 0.1)
    assert stack.push(frame, 1.0)
    assert stack.is_fresh(1.05, 0.1)
    assert not stack.is_fresh(1.2, 0.1)
    # Repeated capture timestamp stays stale and preserves original age.
    assert not stack.push(frame.copy(), 1.0)
    assert not stack.is_fresh(1.05, 0.2)
    assert stack.has_recent_frame(1.05, 0.2)
    assert not stack.has_recent_frame(1.25, 0.2)
    assert not stack.has_recent_frame(float("nan"), 0.2)
    assert stack.age_s(1.2) == pytest.approx(0.2)
    # Pixel-identical stationary frame with a newer capture timestamp is fresh.
    assert stack.push(frame.copy(), 1.1)
    assert stack.is_fresh(1.12, 0.05)
    assert stack.timestamps == (1.0, 1.1)
    # Out-of-order older timestamp is rejected even with different pixels.
    frame[:] = 1
    assert not stack.push(frame, 1.05)
    assert stack.push(frame, 1.2)
    assert stack.timestamps == (1.1, 1.2)
    assert stack.frames[0][0, 0, 0] == 0
    assert stack.push(frame + 1, 1.3)
    assert stack.timestamps == (1.2, 1.3)
    assert stack.is_fresh(1.34, 0.05)


def test_camera_delay_seed_initializes_without_duplicate_pending_delivery():
    buf = CameraDelayBuffer(delay_frames=2)
    f0 = np.array([10], dtype=np.uint8)
    buf.seed(f0, 0.0)
    # First live tick is delayed by 2 frames, so latest stays the seeded frame at t=0.0.
    delivered = buf.push(np.array([11], dtype=np.uint8), 0.05)
    assert delivered is not None and delivered[1] == 0.0
    delivered = buf.push(np.array([12], dtype=np.uint8), 0.10)
    assert delivered is not None and delivered[1] == 0.0
    delivered = buf.push(np.array([13], dtype=np.uint8), 0.15)
    assert delivered is not None and delivered[1] == 0.05
    np.testing.assert_array_equal(delivered[0], [11])


def test_camera_delay_is_per_camera_and_retains_capture_timestamps():
    frame = np.array([1])
    slow = CameraDelayBuffer(delay_frames=2)
    fast = CameraDelayBuffer()
    assert slow.push(frame, 0.0) is None
    assert fast.push(frame, 0.0)[1] == 0.0
    frame[0] = 2
    assert slow.push(frame, 0.1) is None
    delivered, timestamp = slow.push(frame, 0.2)
    assert timestamp == 0.0
    np.testing.assert_array_equal(delivered, [1])
    slow.drop_probability = 1.0
    assert slow.push(frame, 0.3)[1] == 0.1
    assert slow.push(frame, 0.4)[1] == 0.2
    assert slow.push(frame, 0.5)[1] == 0.2
    slow.reset()
    assert slow.push(frame, 0.6) is None


def test_jitter_preserves_causality_and_never_delivers_out_of_order():
    buffer = CameraDelayBuffer(delay_frames=2, jitter_frames=2, rng=np.random.default_rng(42))
    received = []
    for tick in range(30):
        sample = buffer.push(np.array([tick]), float(tick))
        if sample is not None:
            frame, capture_t = sample
            assert 0 <= tick - capture_t <= 4
            assert frame.item() == capture_t
            received.append(capture_t)
    assert received == sorted(received)
    assert len(set(received)) < len(received)
    assert len(set(received)) > 1


def test_causal_finite_difference_ignores_duplicate_and_old_feedback():
    estimator = CausalJointVelocityEstimator()
    assert estimator.is_stale(0.0, 0.1)
    assert estimator.update(1.0, [0.0, 1.0]) == pytest.approx([0.0, 0.0])
    assert estimator.update(1.2, [0.4, 0.8]) == pytest.approx([2.0, -1.0])
    assert estimator.update(1.2, [10.0, 10.0]) == pytest.approx([2.0, -1.0])
    assert estimator.update(1.1, [10.0, 10.0]) == pytest.approx([2.0, -1.0])
    assert not estimator.is_stale(1.24, 0.1)
    assert estimator.is_stale(1.31, 0.1)
    assert estimator.update(1.4, [0.6, 1.0]) == pytest.approx([1.0, 1.0])


def test_scale_proprio_fixed_ranges_and_clipping():
    actual = scale_proprio(
        [-2.0, 2.0, 6.0], [-8.0, 2.0, 4.0], [-0.2, 0.05, 0.5],
        [-1.0, 0.0, 0.0], [1.0, 4.0, 5.0], [0.1, 0.1, 0.2],
    )
    assert actual.dtype == np.float32
    assert actual == pytest.approx([-1, 0, 1, -1, 0.5, 1, -1, 0.5, 1])
