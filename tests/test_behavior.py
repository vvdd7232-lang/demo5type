import unittest

from demo5type.behavior import (
    BehaviorState,
    BehaviorStateMachine,
    ControllerConfig,
    SceneObservation,
    TrackedTarget,
)


def frame(
    time_s,
    *,
    targets=(),
    sound_level=0.0,
    impact_magnitude_ns=0.0,
    threat_detected=False,
    threat_position_world=None,
    finger_contact=False,
    right_hand_position_world=(0.3, 0.0, 1.0),
):
    return SceneObservation(
        time_s=time_s,
        head_position_world=(0.0, 0.0, 1.6),
        right_hand_position_world=right_hand_position_world,
        targets=tuple(targets),
        sound_level=sound_level,
        impact_magnitude_ns=impact_magnitude_ns,
        threat_detected=threat_detected,
        threat_position_world=threat_position_world,
        finger_contact=finger_contact,
    )


class BehaviorStateMachineTests(unittest.TestCase):
    def test_empty_scene_explores_and_emits_scan_and_wander_intents(self):
        controller = BehaviorStateMachine()
        intent = controller.update(frame(0.0))
        self.assertEqual(intent.state, BehaviorState.EXPLORE)
        self.assertIsNone(intent.approach_target_world)
        self.assertFalse(intent.grasp_requested)
        self.assertIsNotNone(intent.gaze_direction_world)
        self.assertEqual(intent.wander_direction_world, (1.0, 0.0, 0.0))
        self.assertGreater(intent.wander_speed_m_s, 0.0)

        scan_time = 3.141592653589793 / (2 * controller.config.exploration_scan_frequency_rad_s)
        scanned = controller.update(frame(scan_time))
        self.assertGreater(scanned.gaze_direction_world[1], 0.0)

    def test_nearest_moving_person_is_gazed_at_but_not_grabbed(self):
        person = TrackedTarget(
            "person", (1.0, 0.0, 1.6), is_moving=True, is_person=True, is_interactable=True
        )
        intent = BehaviorStateMachine().update(frame(0.0, targets=(person,)))
        self.assertEqual(intent.state, BehaviorState.EXPLORE)
        self.assertEqual(intent.attention_target_id, "person")
        self.assertEqual(intent.look_at_world, person.position_world)
        self.assertIsNone(intent.interaction_target_id)
        self.assertIsNone(intent.right_hand_target_world)

    def test_interactable_target_drives_approach_and_reach_goals(self):
        controller = BehaviorStateMachine()
        target = TrackedTarget("box", (0.9, 0.0, 1.0), is_interactable=True)
        intent = controller.update(frame(0.0, targets=(target,)))
        self.assertEqual(intent.state, BehaviorState.INTEREST)
        self.assertEqual(intent.approach_target_world, target.position_world)
        self.assertEqual(intent.right_hand_target_world, target.position_world)
        self.assertFalse(intent.grasp_requested)

    def test_contact_only_requests_grasp_when_hand_is_close(self):
        controller = BehaviorStateMachine()
        close_target = TrackedTarget("can", (0.45, 0.0, 1.0), is_interactable=True)
        close = controller.update(frame(0.0, targets=(close_target,), finger_contact=True))
        self.assertTrue(close.grasp_requested)

        far_controller = BehaviorStateMachine()
        far_target = TrackedTarget("can", (0.9, 0.0, 1.0), is_interactable=True)
        far = far_controller.update(frame(0.0, targets=(far_target,), finger_contact=True))
        self.assertFalse(far.grasp_requested)

    def test_interest_uses_exit_radius_hysteresis(self):
        controller = BehaviorStateMachine(
            ControllerConfig(interest_enter_radius_m=2.0, interest_exit_radius_m=3.0)
        )
        initial = TrackedTarget("box", (1.8, 0.0, 1.6), is_interactable=True)
        self.assertEqual(controller.update(frame(0.0, targets=(initial,))).state, BehaviorState.INTEREST)

        still_in_exit_radius = TrackedTarget("box", (2.5, 0.0, 1.6), is_interactable=True)
        self.assertEqual(
            controller.update(frame(0.1, targets=(still_in_exit_radius,))).state,
            BehaviorState.INTEREST,
        )

        outside = TrackedTarget("box", (3.1, 0.0, 1.6), is_interactable=True)
        self.assertEqual(controller.update(frame(0.2, targets=(outside,))).state, BehaviorState.EXPLORE)

    def test_fright_overrides_interest_and_then_returns_to_target(self):
        controller = BehaviorStateMachine(
            ControllerConfig(fright_duration_s=1.0, fright_cooldown_s=0.5)
        )
        target = TrackedTarget("box", (1.0, 0.0, 1.0), is_interactable=True)
        controller.update(frame(0.0, targets=(target,)))

        frightened = controller.update(
            frame(
                0.1,
                targets=(target,),
                sound_level=0.95,
                threat_position_world=(2.0, 0.0, 1.0),
            )
        )
        self.assertEqual(frightened.state, BehaviorState.FRIGHT)
        self.assertTrue(frightened.protect_head)
        self.assertLess(frightened.retreat_direction_world[0], 0.0)
        self.assertGreater(frightened.retreat_speed_m_s, 0.0)
        self.assertIsNone(frightened.right_hand_target_world)

        still_frightened = controller.update(frame(0.8, targets=(target,)))
        self.assertEqual(still_frightened.state, BehaviorState.FRIGHT)
        recovered = controller.update(frame(1.2, targets=(target,)))
        self.assertEqual(recovered.state, BehaviorState.INTEREST)

    def test_startle_cooldown_suppresses_retrigger(self):
        controller = BehaviorStateMachine(
            ControllerConfig(fright_duration_s=1.0, fright_cooldown_s=1.0)
        )
        first = controller.update(frame(0.0, threat_detected=True))
        self.assertEqual(first.state, BehaviorState.FRIGHT)
        controller.update(frame(1.1))  # fear is over; cooldown continues to t=2
        during_cooldown = controller.update(frame(1.5, threat_detected=True))
        self.assertEqual(during_cooldown.state, BehaviorState.EXPLORE)
        after_cooldown = controller.update(frame(2.1, threat_detected=True))
        self.assertEqual(after_cooldown.state, BehaviorState.FRIGHT)

    def test_time_must_not_move_backwards(self):
        controller = BehaviorStateMachine()
        controller.update(frame(1.0))
        with self.assertRaises(ValueError):
            controller.update(frame(0.5))

    def test_duplicate_target_ids_are_rejected(self):
        duplicate = TrackedTarget("same", (1.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            frame(0.0, targets=(duplicate, duplicate))


if __name__ == "__main__":
    unittest.main()
