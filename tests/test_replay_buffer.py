import numpy as np
import pytest
import torch

from so101_bowling import load_config
from so101_bowling.replay_buffer import RawFrameReplayBuffer
from so101_bowling.vec_env import SyncVecBowlingEnv


_HW = (2, 3)
_CAMERAS = ("wrist_rgb", "tripod_rgb")
_NEXT_FIELDS = (*_CAMERAS, "proprio", "camera_meta", "privileged_state")


def _obs(wrist, tripod=None, *, marker=0):
    if tripod is None:
        tripod = wrist
    batch = len(wrist)
    result = {}
    for camera, frames in zip(_CAMERAS, (wrist, tripod)):
        pixels = np.asarray(frames, dtype=np.uint8)[..., None, None, None]
        result[camera] = np.broadcast_to(pixels, (batch, 2, *_HW, 3)).copy()
    for key, width in (
        ("proprio", 12), ("camera_meta", 4), ("privileged_state", 46),
        ("privileged_aux_targets", 6), ("privileged_aux_mask", 3),
    ):
        result[key] = np.full((batch, width), marker, dtype=np.float32)
    return result


def _add(buf, obs, next_obs, step, *, terminated=None, truncated=None, infos=None):
    batch = buf.num_envs
    rewards = np.arange(batch, dtype=np.float32) + step * batch
    buf.add(
        obs, np.repeat(rewards[:, None], 4, axis=1), rewards, next_obs,
        np.zeros(batch, dtype=bool) if terminated is None else np.array(terminated),
        np.zeros(batch, dtype=bool) if truncated is None else np.array(truncated),
        [{} for _ in range(batch)] if infos is None else infos,
    )


def _assert_sample_parity(buf, expected):
    assert len(buf) > 0
    batch = buf.sample(len(buf), rng=np.random.default_rng(72))
    indices = set(zip(*buf.valid_indices()))
    expected_rewards = {int(buf.rewards[row]) for row in indices}
    assert set(batch["rewards"].tolist()) == expected_rewards
    for row, reward in enumerate(batch["rewards"].tolist()):
        env, obs, successor = expected[reward]
        for key, value in obs.items():
            np.testing.assert_array_equal(batch["obs"][key][row].numpy(), value[env])
        for key in _NEXT_FIELDS:
            np.testing.assert_array_equal(batch["next_obs"][key][row].numpy(), successor[key])


def test_byte_exact_stack_parity_under_drops_delays_and_wraparound():
    buf = RawFrameReplayBuffer(13, 2, img_hw=_HW)
    assert buf.capacity == 12
    stacks = [[(1, 1), (101, 101)], [(51, 51), (151, 151)]]
    obs = _obs(*stacks)
    expected = {}
    writes = np.ones((2, 2), dtype=int)  # One initial frame per camera and environment.
    for step in range(36):
        for camera, camera_stacks in enumerate(stacks):
            for env, (_, newest) in enumerate(camera_stacks):
                frame = 2 + step + camera * 100 + env * 50
                if step % 9 == 8:
                    camera_stacks[env] = (frame, frame)
                    writes[camera, env] += 1
                elif (step + 2 * camera + env) % 4 in (0, 1):
                    camera_stacks[env] = (newest, frame)
                    writes[camera, env] += 1
        successor = _obs(*stacks, marker=step + 1)
        _add(buf, obs, successor, step)
        for env in range(2):
            expected[step * 2 + env] = (
                env, obs, {key: successor[key][env] for key in _NEXT_FIELDS},
            )
        assert len(buf) == min((step + 1) * buf.num_envs, buf.capacity)
        np.testing.assert_array_equal(buf.wrist_pool_gen.max(axis=1), writes[0])
        np.testing.assert_array_equal(buf.tripod_pool_gen.max(axis=1), writes[1])
        _assert_sample_parity(buf, expected)
        obs = successor
    assert buf.wrist_pool_gen.max() > buf.wrist_pool.shape[1]
    assert buf.tripod_pool_gen.max() > buf.tripod_pool.shape[1]


def test_oldest_edge_overwritten_frame_dependencies_excluded():
    buf = RawFrameReplayBuffer(8, 2, img_hw=_HW, frame_pool_size_per_env=3)
    _add(buf, _obs([(1, 1)] * 2), _obs([(0, 0)] * 2), 0, terminated=[True, True])
    _add(buf, _obs([(1, 2), (2, 3)]), _obs([(2, 3), (1, 1)]), 1)
    assert buf.valid_size == 4
    _add(buf, _obs([(2, 3)] * 2), _obs([(3, 4)] * 2), 2)

    assert buf.occupied.sum() == 6
    for env, obs_ok, next_ok in [(0, False, True), (1, True, False)]:
        for prefix, ok in [("obs", obs_ok), ("next", next_ok)]:
            indices = getattr(buf, f"{prefix}_wrist_idx")[env, 1]
            generations = getattr(buf, f"{prefix}_wrist_gen")[env, 1]
            assert bool(np.all(buf.wrist_pool_gen[env, indices] == generations)) is ok
    np.testing.assert_array_equal(buf.valid_indices(), ([0, 1], [2, 2]))
    assert buf.valid_size == len(buf) == 2
    assert sorted(buf.sample(2)["rewards"].tolist()) == [4, 5]
    for size in (0, -1, 3):
        with pytest.raises(ValueError):
            buf.sample(size)


def test_successor_cases_ordinary_truncated_and_terminated():
    buf = RawFrameReplayBuffer(16, 4, img_hw=_HW)
    obs = _obs([(1, 2)] * 4, marker=1)
    next_obs = _obs([(2, 3), (4, 4), (5, 5), (6, 6)], marker=2)
    final_obs = {key: value[0] for key, value in _obs([(2, 9)], [(7, 8)], marker=9).items()}
    terminated = [False, False, True, True]
    truncated = [False, True, False, True]
    infos = [{}, {"final_obs": final_obs}, {"final_obs": {}}, {}]
    _add(buf, obs, next_obs, 0, terminated=terminated, truncated=truncated, infos=infos)
    assert set(buf._truncated_final_images) == {(1, 0)}
    expected_final = {key: value.copy() for key, value in final_obs.items()}
    final_obs["wrist_rgb"].fill(255)
    final_obs["tripod_rgb"].fill(255)
    expected = {
        env: (
            env,
            obs,
            {
                key: (
                    next_obs[key][env] if env == 0 else expected_final[key] if env == 1
                    else np.zeros_like(next_obs[key][env])
                )
                for key in _NEXT_FIELDS
            },
        )
        for env in range(4)
    }
    _assert_sample_parity(buf, expected)
    batch = buf.sample(4)
    order = batch["rewards"].argsort()
    assert batch["actions"][order].tolist() == [[float(i)] * 4 for i in range(4)]
    assert batch["terminated"][order].tolist() == terminated
    assert batch["truncated"][order].tolist() == [False, True, False, False]
    for _ in range(3):
        _add(buf, obs, next_obs, 1)

    arrays = {key: value.copy() for key, value in vars(buf).items() if isinstance(value, np.ndarray)}
    saved_images = {
        row: (wrist.copy(), tripod.copy())
        for row, (wrist, tripod) in buf._truncated_final_images.items()
    }
    for bad_next, bad_infos, error in [
        (next_obs, [{}, {}, {}, {}], r"env 1 missing info\['final_obs'\]"),
        (next_obs, [{}, {"final_obs": {**final_obs, "tripod_rgb": np.zeros(1)}}, {}, {}],
         "tripod_rgb"),
        ({**next_obs, "wrist_rgb": next_obs["wrist_rgb"].astype(np.float32)}, infos, "wrist_rgb"),
        ({**next_obs, "proprio": next_obs["proprio"][:1]}, infos, "proprio"),
    ]:
        with pytest.raises(ValueError, match=error):
            _add(buf, next_obs, bad_next, 2, truncated=truncated, terminated=terminated, infos=bad_infos)
        assert buf._write_ptr == 0
        for key, before in arrays.items():
            np.testing.assert_array_equal(getattr(buf, key), before)
        assert buf._truncated_final_images.keys() == saved_images.keys()
        for row, before in saved_images.items():
            np.testing.assert_array_equal(buf._truncated_final_images[row], before)


def test_all_truncated_workload_evicts_boundary_images_bounded_memory():
    buf = RawFrameReplayBuffer(8, 2, img_hw=_HW)
    obs = _obs([(10, 10), (20, 20)])
    initial_bytes = buf.memory_bytes()
    boundary_bytes = sum(obs[key][0].nbytes for key in _CAMERAS)
    bound = initial_bytes + buf.capacity * boundary_bytes
    for step in range(5 * buf.capacity):
        final = _obs([(10, step), (20, step)])
        infos = [
            {"final_obs": {key: value[env] for key, value in final.items()}}
            for env in range(buf.num_envs)
        ]
        _add(buf, obs, obs, step, truncated=[True, True], infos=infos)
        count = min((step + 1) * buf.num_envs, buf.capacity)
        assert len(buf._truncated_final_images) == count
        assert buf.memory_bytes() == initial_bytes + count * boundary_bytes <= bound
    assert buf.memory_bytes() == bound


def test_memory_bytes_and_uint8_storage():
    buf = RawFrameReplayBuffer(4, 1, img_hw=_HW)
    obs = _obs([(12, 12)])
    final = {key: value[0] for key, value in obs.items()}
    _add(buf, obs, obs, 0, truncated=[True], infos=[{"final_obs": final}])
    for pool, gen in (
        (buf.wrist_pool, buf.wrist_pool_gen), (buf.tripod_pool, buf.tripod_pool_gen),
    ):
        assert pool.dtype == np.uint8
        assert pool.shape == (1, 10, *_HW, 3)
        assert gen.dtype == np.uint64
        assert np.count_nonzero(gen) == 1
    expected_bytes = sum(
        value.nbytes for value in vars(buf).values() if isinstance(value, np.ndarray)
    )
    expected_bytes += sum(final[key].nbytes for key in _CAMERAS)
    assert buf.memory_bytes() == expected_bytes

    batch = buf.sample(1, device=torch.device("cpu"))
    for field in ("obs", "next_obs"):
        for key, tensor in batch[field].items():
            assert tensor.device.type == "cpu"
            assert tensor.dtype == (torch.uint8 if key in _CAMERAS else torch.float32)
    assert batch["actions"].dtype == batch["rewards"].dtype == torch.float32
    assert batch["terminated"].dtype == batch["truncated"].dtype == torch.bool


def test_real_env_integration_with_dr_camera_drops():
    history = []
    repeated_after_delivery = np.zeros(len(_CAMERAS), dtype=bool)
    delivered_before = np.zeros(len(_CAMERAS), dtype=bool)
    with SyncVecBowlingEnv(load_config(), dr_enabled=True, frame_stack=2) as env:
        obs = env.reset(seed=42)
        buf = RawFrameReplayBuffer(16, env.num_envs, img_hw=obs["wrist_rgb"].shape[-3:-1])
        action = np.zeros((env.num_envs, *env.action_space.shape), dtype=np.float32)
        for step in range(80):
            next_obs, rewards, terminated, truncated, infos = env.step(action)
            if terminated[0]:
                successor = {key: np.zeros_like(next_obs[key][0]) for key in _NEXT_FIELDS}
            elif truncated[0]:
                successor = infos[0]["final_obs"]
            else:
                successor = {key: next_obs[key][0] for key in _NEXT_FIELDS}
                for i, key in enumerate(_CAMERAS):
                    unchanged = np.array_equal(obs[key], next_obs[key])
                    repeated_after_delivery[i] |= unchanged and delivered_before[i]
                    delivered_before[i] |= not unchanged

            buf.add(obs, action, rewards, next_obs, terminated, truncated, infos)
            history.append((obs, successor, rewards[0]))
            obs = next_obs

            # The environment can return identical rewards, so align samples using
            # the seeded selection order and map each ring slot to its latest step.
            _, slots = buf.valid_indices()
            rng = np.random.default_rng(step)
            batch = buf.sample(len(slots), rng=np.random.default_rng(step))
            order = rng.choice(len(slots), size=len(slots), replace=False)
            for row, slot in enumerate(slots[order]):
                original_step = step - (step - slot) % buf.per_env_cap
                original_obs, original_successor, original_reward = history[original_step]
                assert batch["rewards"][row].item() == original_reward
                for key in _CAMERAS:
                    np.testing.assert_array_equal(batch["obs"][key][row], original_obs[key][0])
                    np.testing.assert_array_equal(batch["next_obs"][key][row], original_successor[key])
    assert repeated_after_delivery.all()


def test_cached_valid_indices_invalidates_only_on_add(monkeypatch: pytest.MonkeyPatch):
    buf = RawFrameReplayBuffer(8, 2, img_hw=_HW)
    obs0 = _obs([(1, 1), (2, 2)])
    obs1 = _obs([(3, 3), (4, 4)])
    _add(buf, obs0, obs1, 0)

    calls = 0
    orig_valid_mask = buf._valid_mask

    def counting_valid_mask():
        nonlocal calls
        calls += 1
        return orig_valid_mask()

    monkeypatch.setattr(buf, "_valid_mask", counting_valid_mask)
    assert len(buf) == 2
    assert buf.valid_size == 2
    buf.sample(2, rng=np.random.default_rng(0))
    buf.sample(2, rng=np.random.default_rng(1))
    assert calls == 1

    # Failed add() must leave cached indices intact.
    with pytest.raises(ValueError, match="proprio"):
        _add(buf, obs1, {**obs0, "proprio": obs0["proprio"][:1]}, 1)
    assert len(buf) == 2
    assert calls == 1

    # Valid add() invalidates cache; next access recomputes once.
    _add(buf, obs1, _obs([(5, 5), (6, 6)]), 1)
    assert len(buf) == 4
    buf.sample(4, rng=np.random.default_rng(2))
    assert calls == 2
