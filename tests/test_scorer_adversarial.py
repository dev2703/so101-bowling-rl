from dataclasses import replace

import numpy as np
import pytest

from so101_bowling.scorer import DeliveryPhase, DeliveryScorer, SubstepSnapshot


class Delivery:
    def __init__(self):
        self.scorer = DeliveryScorer()
        self.snap = SubstepSnapshot(
            t=0.0,
            paddle_pos=np.array([0.15, 0.0, 0.05]),
            ball_pos=np.array([0.25, 0.0, 0.05]),
            ball_vel=np.zeros(3),
            pin_positions=np.array([[0.60, 0.0, 0.08], [0.63, 0.03, 0.08], [0.63, -0.03, 0.08]]),
            pin_up_vectors=np.tile([0.0, 0.0, 1.0], (3, 1)),
            pin_speeds=np.zeros(3),
            max_arm_x=0.40,
            pusher_ball_contact=False,
            other_arm_ball_contact=False,
            arm_pin_contact=False,
            table_slam=False,
            ball_pin_contacts=(False, False, False),
            pin_pin_contacts=((False, False, False),) * 3,
            paddle_ball_surface_dist=0.1,
        )
        self.reset()

    def reset(self):
        self.scorer.reset(self.snap.ball_pos, self.snap.pin_positions, self.snap.pin_up_vectors)

    def step(self, **changes):
        self.snap = replace(self.snap, t=self.snap.t + self.scorer.config.sim_dt, **changes)
        self.scorer.update_substep(self.snap)

    def start_push(self, displacement=0.035, velocity=(0.95, 0.0, 0.0)):
        self.step(pusher_ball_contact=True, paddle_ball_surface_dist=0.0)
        self.step(ball_pos=self.snap.ball_pos + [displacement, 0.0, 0.0], ball_vel=np.array(velocity))

    def separate(self, steps=5):
        for _ in range(steps):
            self.step(pusher_ball_contact=False, paddle_ball_surface_dist=0.03)

    def launch(self, velocity=(0.95, 0.0, 0.0)):
        self.start_push(velocity=velocity)
        self.separate()
        assert self.scorer.phase == DeliveryPhase.RELEASED

    def outcome(self, unresolved=False):
        return self.scorer.finalize_episode(self.snap, unresolved)


def tilted(degrees):
    angle = np.radians(degrees)
    return [np.sin(angle), 0.0, np.cos(angle)]


def assert_illegal(delivery, reason):
    outcome = delivery.outcome()
    assert not outcome.legal_delivery
    assert outcome.invalid_reason == reason
    assert outcome.credited_pins == 0
    assert outcome.launch_reward == outcome.pin_reward == 0
    assert outcome.penalty_reward == outcome.total_terminal_reward == -2.0


def test_side_nudge_consumes_attempt_without_moving_stage():
    delivery = Delivery()
    stage = delivery.scorer.p_stage.copy()
    assert stage == pytest.approx([0.1465, 0.0, 0.05])
    delivery.step(ball_pos=np.array([0.25, 0.017, 0.05]))
    delivery.step(pusher_ball_contact=True)
    assert delivery.scorer.p_stage == pytest.approx(stage)
    assert_illegal(delivery, "ball_moved_before_pusher_contact")


def test_slap_cannot_use_free_roll_to_meet_push_displacement():
    delivery = Delivery()
    delivery.start_push(displacement=0.005)
    delivery.step(ball_pos=np.array([0.43, 0.0, 0.05]), pusher_ball_contact=False, paddle_ball_surface_dist=0.03)
    delivery.separate(4)
    assert_illegal(delivery, "insufficient_push_displacement_slap")


def test_continuous_push_and_transitive_strike_keep_original_launch_velocity():
    delivery = Delivery()
    delivery.launch()
    delivery.step(
        ball_pos=np.array([0.60, 0.0, 0.05]),
        ball_vel=np.array([-0.3, 0.5, 0.0]),
        ball_pin_contacts=(True, False, False),
        pin_pin_contacts=[(1, 2), (0, 1)],
        pin_up_vectors=np.array([tilted(60)] * 3),
    )
    assert delivery.scorer.should_end_delivery(0.1, 1.0) == (True, "released")
    outcome = delivery.outcome()
    assert outcome.legal_delivery and outcome.legal_strike
    assert outcome.credited_pins == outcome.full_knock_pins == 3
    assert outcome.contact_forward_disp_m == pytest.approx(0.035)
    assert outcome.launch_angle_err_rad == 0
    assert outcome.launch_speed == pytest.approx(0.95)
    assert outcome.launch_reward == pytest.approx(1.20)
    assert outcome.pin_reward == 20.0
    assert outcome.total_terminal_reward == pytest.approx(21.20)
    state = delivery.scorer.critic_state(0.4)
    assert state.dtype == np.float32
    assert state == pytest.approx([0, 0, 1, 0, 0.4, 0.035, *([np.pi / 3] * 3), 1, 1, 1])


def test_short_contact_chatter_and_small_gap_reset_separation():
    delivery = Delivery()
    delivery.start_push()
    delivery.separate(4)
    delivery.step(paddle_ball_surface_dist=0.01)
    delivery.separate(4)
    assert delivery.scorer.phase == DeliveryPhase.PUSHING
    delivery.step(pusher_ball_contact=True)
    delivery.separate()
    assert delivery.outcome().legal_delivery


def test_restrike_within_one_20hz_control_tick_revokes_rewards():
    delivery = Delivery()
    delivery.start_push()
    start = delivery.snap.t
    delivery.separate()
    delivery.step(ball_pin_contacts=(True, False, False), pin_up_vectors=np.array([tilted(60)] * 3))
    delivery.step(pusher_ball_contact=True)
    assert delivery.snap.t - start < 0.05
    assert_illegal(delivery, "double_tap_restrike")


def test_pin_impact_during_release_confirmation_is_illegal():
    delivery = Delivery()
    delivery.start_push()
    delivery.separate(4)
    delivery.step(ball_pin_contacts=(True, False, False))
    assert_illegal(delivery, "pre_release_pin_impact")


@pytest.mark.parametrize(
    ("flags", "reason", "coast"),
    [
        ({"arm_pin_contact": True}, "arm_pin_contact", False),
        ({"max_arm_x": 0.441}, "foul_line_crossing", False),
        ({"max_arm_x": 0.441}, "foul_line_crossing", True),
        ({"table_slam": True}, "table_slam", True),
        ({"other_arm_ball_contact": True}, "non_pusher_ball_contact", False),
    ],
)
def test_hard_illegalities(flags, reason, coast):
    delivery = Delivery()
    if coast:
        delivery.launch()
    else:
        delivery.start_push()
    delivery.step(**flags)
    assert delivery.scorer.should_end_delivery(0.1, 1.0) == (True, reason)
    assert_illegal(delivery, reason)


def test_late_release_past_rack_is_illegal():
    delivery = Delivery()
    delivery.start_push()
    delivery.step(ball_pos=np.array([0.575, 0.0, 0.05]), pusher_ball_contact=False, paddle_ball_surface_dist=0.03)
    delivery.separate(4)
    assert_illegal(delivery, "late_release_past_rack")


@pytest.mark.parametrize("mode", ["no_action", "near_miss", "table_vibration"])
def test_no_impact_means_no_credit_or_peak_tilt(mode):
    delivery = Delivery()
    if mode != "no_action":
        delivery.launch()
    if mode == "table_vibration":
        for _ in range(10):
            delivery.step(pin_up_vectors=np.array([tilted(60)] * 3))
    outcome = delivery.outcome()
    assert outcome.credited_pins == outcome.pin_reward == 0
    assert not delivery.scorer.pin_peak_tilt_deg.any()
    if mode == "no_action":
        assert outcome.invalid_reason == "no_legal_release"
        assert outcome.penalty_reward == 0


def test_impact_wobble_uses_reset_up_and_requires_consecutive_substeps():
    delivery = Delivery()
    baseline = np.array([tilted(8), tilted(0), tilted(0)])
    delivery.snap = replace(delivery.snap, pin_up_vectors=baseline)
    delivery.reset()
    delivery.launch()
    wobble = np.array([tilted(20), tilted(50), tilted(0)])
    delivery.step(ball_pin_contacts=(True, False, False), pin_up_vectors=wobble)
    delivery.step(ball_pin_contacts=(False, False, False), pin_up_vectors=baseline)
    for _ in range(3):
        delivery.step(pin_up_vectors=wobble)
    assert not delivery.scorer.pin_wobbled.any()
    delivery.step()
    delivery.step(pin_up_vectors=baseline)
    outcome = delivery.outcome()
    assert outcome.wobble_only_pins == outcome.credited_pins == 1
    assert outcome.full_knock_pins == 0
    assert outcome.pin_reward == 5.0
    assert delivery.scorer.pin_peak_tilt_deg == pytest.approx([12, 0, 0])


def test_impacted_off_table_pin_is_full_credit_and_full_knock():
    delivery = Delivery()
    delivery.launch()
    positions = delivery.snap.pin_positions.copy()
    positions[0, 2] = -0.051
    positions[1, 2] = -0.06
    delivery.step(pin_positions=positions, ball_pin_contacts=(True, False, False))
    outcome = delivery.outcome()
    assert outcome.off_table_pins == outcome.full_knock_pins == outcome.credited_pins == 1
    assert outcome.wobble_only_pins == 0
    assert outcome.pin_reward == 5.0


def test_coast_waits_for_rolling_ball_and_scores_late_impact_on_timeout():
    delivery = Delivery()
    delivery.launch()
    for _ in range(80):
        delivery.step()
    assert delivery.scorer.is_coast_settled(0.4) == (False, False)
    delivery.step(
        ball_pos=np.array([0.61, 0.0, 0.05]),
        ball_pin_contacts=(True, False, False),
        pin_up_vectors=np.array([tilted(50), tilted(0), tilted(0)]),
    )
    done, unresolved = delivery.scorer.is_coast_settled(2.5)
    assert done and unresolved
    outcome = delivery.outcome(unresolved)
    assert outcome.unresolved_motion and outcome.credited_pins == 1


def test_ball_stopped_short_can_settle_after_uninterrupted_quiet_dwell():
    delivery = Delivery()
    delivery.launch()
    for _ in range(49):
        delivery.step(ball_vel=np.zeros(3))
    assert delivery.scorer.is_coast_settled(0.245) == (False, False)
    delivery.step(pin_speeds=np.array([0.0, 0.03, 0.0]))
    for _ in range(50):
        delivery.step(pin_speeds=np.zeros(3))
    assert delivery.scorer.is_coast_settled(0.5) == (True, False)


@pytest.mark.parametrize("count", [0, 1, 2])
def test_worst_next_pin_outcome_beats_best_adjacent_terminal_reward(count):
    def score(pins, velocity):
        delivery = Delivery()
        delivery.launch(velocity=velocity)
        delivery.step(
            ball_pin_contacts=tuple(i < pins for i in range(3)),
            pin_up_vectors=np.array([tilted(60)] * 3),
        )
        return delivery.outcome()

    best = score(count, (5.0, 0.0, 0.0))
    worst = score(count + 1, (0.1, 0.0, 0.0))
    assert best.launch_reward == pytest.approx(1.2)
    assert worst.launch_reward == 0
    assert worst.total_terminal_reward > best.total_terminal_reward + 0.4


def test_discounted_potential_cancels_and_terminal_step_charges_jerk():
    scorer = Delivery().scorer
    stage = scorer.p_stage
    assert scorer.compute_potential(stage) == pytest.approx(0.4)
    assert scorer.compute_potential(scorer.ball_init_pos) == 0.0
    assert scorer.compute_potential(2 * scorer.ball_init_pos - stage) < 0.01
    first = scorer.step_control_reward(stage, [0], [0], False)
    second = scorer.step_control_reward(stage, [1], [0], True)
    assert first == pytest.approx(0.99 * 0.4)
    assert second == pytest.approx(-0.405)
    assert first + 0.99 * second == pytest.approx(-0.99 * 0.005)
    assert scorer.compute_potential(stage) == 0.0


def test_timeout_during_push_without_release_is_unpenalized():
    delivery = Delivery()
    delivery.start_push()
    assert delivery.scorer.should_end_delivery(1.0, 1.0) == (True, "delivery_timeout")
    outcome = delivery.outcome()
    assert not outcome.legal_delivery
    assert outcome.invalid_reason == "no_legal_release"
    assert outcome.total_terminal_reward == 0


def test_off_table_falling_ball_and_pin_settle_cleanly_without_timeout():
    delivery = Delivery()
    delivery.launch()
    delivery.step(
        ball_pos=np.array([0.62, 0.0, 0.05]),
        ball_pin_contacts=(True, False, False),
    )
    # Ball and impacted pin 0 fall off the back desk edge while still carrying horizontal speed.
    pin_pos = np.array([[0.85, 0.0, -0.12], [0.695, -0.044, 0.0], [0.695, 0.044, 0.0]])
    pin_speeds = np.array([1.2, 0.0, 0.0])
    for _ in range(50):
        delivery.step(
            ball_pos=np.array([0.90, 0.0, -0.15]),
            ball_vel=np.array([0.5, 0.0, -1.5]),
            pin_positions=pin_pos,
            pin_speeds=pin_speeds,
        )
    assert delivery.scorer.is_coast_settled(0.5) == (True, False)
    outcome = delivery.outcome(False)
    assert outcome.legal_delivery and not outcome.unresolved_motion
    assert outcome.off_table_pins == 1 and outcome.credited_pins == 1


def _matrix_contacts(*edges: tuple[int, int]) -> np.ndarray:
    mat = np.zeros((3, 3), dtype=bool)
    for i, j in edges:
        mat[i, j] = True
    return mat


@pytest.mark.parametrize(
    ("pin_pin_contacts", "expected_impacted"),
    [
        (_matrix_contacts((2, 1), (1, 0)), [True, True, True]),
        (
            tuple(tuple(bool(v) for v in row) for row in _matrix_contacts((2, 1), (1, 0))),
            [True, True, True],
        ),
        ([(2, 1), (1, 0)], [True, True, True]),
        ([(0, 1)], [True, True, False]),
        ([], [True, False, False]),
    ],
)
def test_pin_pin_transitive_propagation_formats(pin_pin_contacts, expected_impacted):
    delivery = Delivery()
    delivery.launch()
    delivery.step(
        ball_pin_contacts=(True, False, False),
        pin_pin_contacts=pin_pin_contacts,
        pin_up_vectors=np.array([tilted(60)] * 3),
    )
    np.testing.assert_array_equal(delivery.scorer.pin_impacted, expected_impacted)
    outcome = delivery.outcome()
    expected_count = int(np.count_nonzero(expected_impacted))
    assert outcome.credited_pins == outcome.full_knock_pins == expected_count
    assert outcome.legal_strike is (expected_count == 3)
