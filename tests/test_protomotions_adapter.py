import math
import unittest

from demo5type.control import ControlCommand, ControlMode
from integrations.protomotions.runtime_adapter import (
    CommandGateConfig,
    ProtoMotionsBridgeConfig,
    ProtoMotionsCommandDispatcher,
    dispatch_to_protomotions,
)


class FakeCanvas:
    has_velocity_command = True


class FakeControlConfig:
    def __init__(self):
        self.canvas = FakeCanvas()


class FakeController:
    def __init__(self):
        self.config = FakeControlConfig()
        self.calls = []

    def hold(self, env_ids):
        self.calls.append(("hold", env_ids))

    def set_location(self, env_ids, position_xy, seconds):
        self.calls.append(("location", env_ids, position_xy, seconds))

    def set_reaching(self, env_ids, position_xyz, seconds):
        self.calls.append(("reaching", env_ids, position_xyz, seconds))

    def set_steering(self, env_ids, speed, heading, facing=None):
        self.calls.append(("steering", env_ids, speed, heading, facing))


class ProtoMotionsAdapterTests(unittest.TestCase):
    def test_approach_dispatches_batched_location_goal_with_bounded_horizon(self):
        controller = FakeController()
        env_ids = [1, 4]
        command = ControlCommand(
            mode=ControlMode.APPROACH,
            reason="approach",
            navigation_target_world=(2.0, 1.0, 0.0),
            navigation_speed_m_s=0.25,
        )
        dispatch_to_protomotions(
            controller,
            env_ids,
            command,
            base_position_world=(0.0, 0.0, 0.0),
            config=ProtoMotionsBridgeConfig(max_goal_duration_s=3.0),
        )
        name, called_ids, positions, seconds = controller.calls[0]
        self.assertEqual(name, "location")
        self.assertIs(called_ids, env_ids)
        self.assertEqual(positions, [[2.0, 1.0], [2.0, 1.0]])
        self.assertEqual(seconds, 3.0)

    def test_reach_dispatches_same_world_target_to_each_selected_environment(self):
        controller = FakeController()
        command = ControlCommand(
            mode=ControlMode.REACH,
            reason="reach",
            right_hand_target_world=(0.5, 0.2, 1.1),
        )
        dispatch_to_protomotions(
            controller,
            [0, 2, 3],
            command,
            base_position_world=(0.0, 0.0, 0.0),
        )
        self.assertEqual(
            controller.calls[0],
            ("reaching", [0, 2, 3], [[0.5, 0.2, 1.1]] * 3, 1.0),
        )

    def test_steering_dispatches_heading_and_gaze_facing(self):
        controller = FakeController()
        command = ControlCommand(
            mode=ControlMode.RETREAT,
            reason="startle",
            base_velocity_world=(0.0, -0.6, 0.0),
            gaze_direction_world=(1.0, 0.0, 0.0),
        )
        dispatch_to_protomotions(
            controller,
            [5],
            command,
            base_position_world=(0.0, 0.0, 0.0),
        )
        name, _, speed, heading, facing = controller.calls[0]
        self.assertEqual(name, "steering")
        self.assertAlmostEqual(speed, 0.6)
        self.assertAlmostEqual(heading, -math.pi / 2)
        self.assertEqual(facing, 0.0)

    def test_steering_refuses_checkpoint_without_velocity_goal_channel(self):
        controller = FakeController()
        controller.config.canvas.has_velocity_command = False
        command = ControlCommand(
            mode=ControlMode.WANDER,
            reason="walk",
            base_velocity_world=(0.1, 0.0, 0.0),
        )
        with self.assertRaisesRegex(ValueError, "velocity-command goals"):
            dispatch_to_protomotions(
                controller,
                [0],
                command,
                base_position_world=(0.0, 0.0, 0.0),
            )
        self.assertEqual(controller.calls, [])

    def test_auxiliary_behaviors_are_never_silently_ignored(self):
        controller = FakeController()
        command = ControlCommand(
            mode=ControlMode.RETREAT,
            reason="startle",
            base_velocity_world=(-0.5, 0.0, 0.0),
            protect_head=True,
        )
        with self.assertRaises(NotImplementedError):
            dispatch_to_protomotions(
                controller,
                [0],
                command,
                base_position_world=(0.0, 0.0, 0.0),
            )
        self.assertEqual(controller.calls, [])

        auxiliary = []
        dispatch_to_protomotions(
            controller,
            [0],
            command,
            base_position_world=(0.0, 0.0, 0.0),
            on_auxiliary_intent=lambda env_ids, value: auxiliary.append((env_ids, value)),
        )
        self.assertEqual(len(auxiliary), 1)
        self.assertIs(auxiliary[0][1], command)

    def test_dispatcher_gates_small_frame_updates_and_sends_mode_changes(self):
        controller = FakeController()
        dispatcher = ProtoMotionsCommandDispatcher(
            gate_config=CommandGateConfig(
                min_dispatch_interval_s=0.1,
                position_update_threshold_m=0.1,
            )
        )
        approach = ControlCommand(
            mode=ControlMode.APPROACH,
            reason="approach",
            navigation_target_world=(1.0, 0.0, 0.0),
            navigation_speed_m_s=0.5,
        )
        kwargs = {"base_position_world": (0.0, 0.0, 0.0)}
        self.assertTrue(dispatcher.update(controller, [0], approach, now_s=0.0, **kwargs))
        self.assertFalse(dispatcher.update(controller, [0], approach, now_s=0.05, **kwargs))

        tiny_move = ControlCommand(
            mode=ControlMode.APPROACH,
            reason="approach",
            navigation_target_world=(1.05, 0.0, 0.0),
            navigation_speed_m_s=0.5,
        )
        self.assertFalse(dispatcher.update(controller, [0], tiny_move, now_s=0.11, **kwargs))
        moved = ControlCommand(
            mode=ControlMode.APPROACH,
            reason="approach",
            navigation_target_world=(1.2, 0.0, 0.0),
            navigation_speed_m_s=0.5,
        )
        self.assertTrue(dispatcher.update(controller, [0], moved, now_s=0.12, **kwargs))

        reaching = ControlCommand(
            mode=ControlMode.REACH,
            reason="reach",
            right_hand_target_world=(0.5, 0.0, 1.0),
        )
        self.assertTrue(dispatcher.update(controller, [0], reaching, now_s=0.13, **kwargs))
        self.assertEqual([call[0] for call in controller.calls], ["location", "location", "reaching"])

    def test_dispatcher_rejects_time_reversal(self):
        controller = FakeController()
        dispatcher = ProtoMotionsCommandDispatcher()
        command = ControlCommand(mode=ControlMode.HOLD, reason="hold")
        kwargs = {"base_position_world": (0.0, 0.0, 0.0)}
        dispatcher.update(controller, [0], command, now_s=1.0, **kwargs)
        self.assertFalse(dispatcher.update(controller, [0], command, now_s=2.0, **kwargs))
        with self.assertRaises(ValueError):
            dispatcher.update(controller, [0], command, now_s=1.5, **kwargs)

    def test_empty_batch_is_a_no_op(self):
        controller = FakeController()
        command = ControlCommand(mode=ControlMode.HOLD, reason="empty")
        dispatch_to_protomotions(
            controller,
            [],
            command,
            base_position_world=(0.0, 0.0, 0.0),
        )
        self.assertEqual(controller.calls, [])


if __name__ == "__main__":
    unittest.main()
