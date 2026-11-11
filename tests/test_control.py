import math
import unittest

from demo5type.behavior import BehaviorIntent, BehaviorState
from demo5type.control import (
    ControlMode,
    InteractionControlConfig,
    command_from_intent,
)


class ControlBridgeTests(unittest.TestCase):
    def test_exploration_velocity_is_normalized_and_clamped(self):
        intent = BehaviorIntent(
            state=BehaviorState.EXPLORE,
            reason="scan",
            gaze_direction_world=(1.0, 0.0, 0.0),
            wander_direction_world=(3.0, 4.0, 0.0),
            wander_speed_m_s=0.9,
        )
        command = command_from_intent(
            intent,
            base_position_world=(0.0, 0.0, 0.0),
            base_forward_world=(1.0, 0.0, 0.0),
            config=InteractionControlConfig(wander_speed_limit_m_s=0.2),
        )
        self.assertEqual(command.mode, ControlMode.WANDER)
        self.assertAlmostEqual(math.hypot(*command.base_velocity_world), 0.2)
        self.assertEqual(command.gaze_direction_world, (1.0, 0.0, 0.0))

    def test_approach_goal_stops_short_of_object_center(self):
        target = (3.0, 2.0, 1.0)
        intent = BehaviorIntent(
            state=BehaviorState.INTEREST,
            reason="near object",
            interaction_target_id="box",
            approach_target_world=target,
        )
        command = command_from_intent(
            intent,
            base_position_world=(0.0, 0.0, 0.0),
            base_forward_world=(1.0, 0.0, 0.0),
        )
        self.assertEqual(command.mode, ControlMode.APPROACH)
        self.assertEqual(command.navigation_target_world[2], 0.0)
        self.assertAlmostEqual(
            math.hypot(
                command.navigation_target_world[0] - target[0],
                command.navigation_target_world[1] - target[1],
            ),
            0.85,
        )
        self.assertNotEqual(command.navigation_target_world, target)

    def test_near_target_switches_to_reach_and_preserves_grasp_request(self):
        intent = BehaviorIntent(
            state=BehaviorState.INTEREST,
            reason="contact",
            interaction_target_id="can",
            approach_target_world=(1.0, 0.0, 0.8),
            right_hand_target_world=(1.0, 0.0, 0.8),
            grasp_requested=True,
        )
        command = command_from_intent(
            intent,
            base_position_world=(0.0, 0.0, 0.0),
            base_forward_world=(1.0, 0.0, 0.0),
        )
        self.assertEqual(command.mode, ControlMode.REACH)
        self.assertEqual(command.right_hand_target_world, (1.0, 0.0, 0.8))
        self.assertTrue(command.grasp_requested)
        self.assertIsNone(command.navigation_target_world)

    def test_far_target_does_not_start_reaching_early(self):
        intent = BehaviorIntent(
            state=BehaviorState.INTEREST,
            reason="approach",
            approach_target_world=(4.0, 0.0, 0.0),
            right_hand_target_world=(4.0, 0.0, 0.0),
            grasp_requested=True,
        )
        command = command_from_intent(
            intent,
            base_position_world=(0.0, 0.0, 0.0),
            base_forward_world=(1.0, 0.0, 0.0),
        )
        self.assertEqual(command.mode, ControlMode.APPROACH)
        self.assertFalse(command.grasp_requested)
        self.assertIsNone(command.right_hand_target_world)

    def test_fright_command_is_speed_limited_and_keeps_guard_signal(self):
        intent = BehaviorIntent(
            state=BehaviorState.FRIGHT,
            reason="loud sound",
            retreat_direction_world=(-2.0, 0.0, 0.0),
            retreat_speed_m_s=1.5,
            protect_head=True,
            grasp_requested=True,
        )
        command = command_from_intent(
            intent,
            base_position_world=(0.0, 0.0, 0.0),
            base_forward_world=(1.0, 0.0, 0.0),
            config=InteractionControlConfig(retreat_speed_limit_m_s=0.6),
        )
        self.assertEqual(command.mode, ControlMode.RETREAT)
        self.assertAlmostEqual(math.hypot(*command.base_velocity_world), 0.6)
        self.assertTrue(command.protect_head)
        self.assertFalse(command.grasp_requested)

    def test_config_requires_reach_switch_outside_standoff(self):
        with self.assertRaises(ValueError):
            InteractionControlConfig(approach_standoff_m=1.0, reach_switch_distance_m=1.0)


if __name__ == "__main__":
    unittest.main()
