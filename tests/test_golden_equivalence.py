"""Cross-platform equivalence invariants for MJCF actuators and seeded rollouts."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

from so101_bowling.assets import build_scene_xml
from so101_bowling.config import load_config
from so101_bowling.env import SO101BowlingEnv
from so101_bowling.kinematics import ScriptedPushController

_REPO_ROOT = Path(__file__).resolve().parents[1]

_BASELINES = {
    False: {
        "q_lift": np.array(
            [-7.200885667626235e-05, -0.5515272203521672, 0.15244546468965828, 1.6380627293335335]
        ),
        "q_stage": np.array(
            [-7.200074082437685e-05, -0.5140403901858117, 0.6910011931580063, 1.1991930976321175]
        ),
        "q_push_end": np.array(
            [-2.958186197031868e-05, 0.5506015489227235, -0.17529657794008033, 0.06707611576036218]
        ),
        "final_proprio": np.array(
            [
                -1.2509188309195451e-05, -0.10177411884069443, 0.2750278413295746, 0.5298193097114563,
                -2.2313590974931685e-09, 0.00013194355415180326, 0.0001272904482902959, 1.672376492933836e-05,
                0.0, 0.0, 0.0, 0.0,
            ],
            dtype=np.float32,
        ),
        "total_reward": 20.765086297121282,
    },
    True: {
        "q_lift": np.array(
            [-0.27540426037242643, -0.535683507876529, 0.14177090300411294, 1.6380627293335335]
        ),
        "q_stage": np.array(
            [-0.2751805207890703, -0.5033947016374208, 0.6844393629541056, 1.1855291119334734]
        ),
        "q_push_end": np.array(
            [-0.06330249034734499, 0.5210860857104715, -0.1318255867727599, 0.057585761506078546]
        ),
        "final_proprio": np.array(
            [
                -0.14700818061828613, -0.29123204946517944, 0.4219188988208771, 0.7106606364250183,
                -1.2493826943682507e-05, -0.005218390841037035, -0.10270030796527863, 0.06725332140922546,
                -2.9370061383815482e-05, -0.01395358331501484, -0.27574387192726135, 0.18413905799388885,
            ],
            dtype=np.float32,
        ),
        "total_reward": 0.0702139355606693,
    },
}


def _run_rollout(dr_enabled: bool):
    trace: list[tuple[dict[str, np.ndarray], float, np.ndarray]] = []
    with SO101BowlingEnv(render_rgb=False, dr_enabled=dr_enabled, seed=42) as env:
        obs, _ = env.reset(seed=42)
        for arr in obs.values():
            assert np.isfinite(arr).all()
        ctrl = ScriptedPushController(env.model, env.ids, env.cfg, env.q_min, env.q_max)
        ctrl.reset(
            env.data.xpos[env.ids.ball_body_id].copy(),
            env.data.xpos[env.ids.pin_body_ids].copy(),
        )
        waypoints = (ctrl.q_lift.copy(), ctrl.q_stage.copy(), ctrl.q_push_end.copy())
        total_reward = 0.0
        for _ in range(25):
            action = ctrl.step(env.data.qpos[env.ids.active_qpos_addrs].copy())
            assert np.isfinite(action).all()
            assert np.all((action >= -1.0) & (action <= 1.0))
            obs, reward, done, _, _ = env.step(action)
            assert np.isfinite(reward)
            for arr in obs.values():
                assert np.isfinite(arr).all()
            total_reward += float(reward)
            trace.append(({k: v.copy() for k, v in obs.items()}, float(reward), action.copy()))
            if done:
                break
    return waypoints, trace, total_reward


def test_compiled_actuators_and_mjcf_defaults() -> None:
    xml_path = _REPO_ROOT / "assets" / "so101" / "so101_new_calib_camera.xml"
    root = ET.parse(xml_path).getroot()
    so101_default = root.find(".//default[@class='so101_new_calib']")
    assert so101_default is not None
    assert so101_default.find("position") is None

    model = mujoco.MjModel.from_xml_string(build_scene_xml(load_config()))
    assert model.nu == 6
    np.testing.assert_allclose(model.actuator_gainprm[:, 0], 998.22, atol=1e-9, rtol=0.0)
    np.testing.assert_allclose(model.actuator_biasprm[:, 1], -998.22, atol=1e-9, rtol=0.0)
    np.testing.assert_allclose(model.actuator_biasprm[:, 2], -2.731, atol=1e-9, rtol=0.0)
    np.testing.assert_allclose(
        model.actuator_forcerange, np.tile([-3.35, 3.35], (6, 1)), atol=1e-9, rtol=0.0
    )


@pytest.mark.parametrize("dr_enabled", [False, True])
def test_seeded_rollout_determinism_and_cross_platform_baseline(dr_enabled: bool) -> None:
    wp1, trace1, total1 = _run_rollout(dr_enabled)
    wp2, trace2, total2 = _run_rollout(dr_enabled)

    assert len(trace1) == len(trace2)
    for w1, w2 in zip(wp1, wp2, strict=True):
        np.testing.assert_array_equal(w1, w2)
    assert total1 == total2
    for (obs1, r1, a1), (obs2, r2, a2) in zip(trace1, trace2, strict=True):
        assert r1 == r2
        np.testing.assert_array_equal(a1, a2)
        assert obs1.keys() == obs2.keys()
        for k in obs1:
            np.testing.assert_array_equal(obs1[k], obs2[k])

    baseline = _BASELINES[dr_enabled]
    np.testing.assert_allclose(wp1[0], baseline["q_lift"], atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(wp1[1], baseline["q_stage"], atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(wp1[2], baseline["q_push_end"], atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(trace1[-1][0]["proprio"], baseline["final_proprio"], atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(total1, baseline["total_reward"], atol=1e-4, rtol=1e-4)
