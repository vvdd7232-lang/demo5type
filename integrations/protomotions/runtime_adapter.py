"""Dispatch backend-neutral commands to ProtoMotions' native-goal controller.

This file imports no simulator packages at module import time. Pass the live
``env.control_manager.components["masked_mimic"]`` object from a custom
inference loop and a batch of environment IDs. The caller still owns sensor
acquisition, collision checking, resets, and the simulation loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from demo5type.control import ControlCommand, ControlMode
from demo5type.geometry import Vec3, as_vec3, distance, dot, norm, normalize


@dataclass(frozen=True, slots=True)
class ProtoMotionsBridgeConfig:
    min_goal_duration_s: float = 0.5
    max_goal_duration_s: float = 2.0
    reach_goal_duration_s: float = 1.0
    steering_speed_limit_m_s: float = 0.8

    def __post_init__(self) -> None:
        for name in ("min_goal_duration_s", "max_goal_duration_s", "reach_goal_duration_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_goal_duration_s < self.min_goal_duration_s:
            raise ValueError("max_goal_duration_s must be at least min_goal_duration_s")
        if not math.isfinite(self.steering_speed_limit_m_s) or self.steering_speed_limit_m_s <= 0:
            raise ValueError("steering_speed_limit_m_s must be finite and positive")


def _replicate(values: Sequence[float], count: int) -> list[list[float]]:
    return [list(values) for _ in range(count)]


def _set_steering(
    controller: Any,
    env_ids: Any,
    command: ControlCommand,
    config: ProtoMotionsBridgeConfig,
) -> None:
    velocity = as_vec3(command.base_velocity_world, "base_velocity_world")
    speed = math.hypot(velocity[0], velocity[1])
    if speed <= 1e-9:
        controller.hold(env_ids)
        return

    steering_config = getattr(controller, "config", None)
    canvas = getattr(steering_config, "canvas", None)
    if canvas is not None and not canvas.has_velocity_command:
        raise ValueError(
            "Wander/retreat requires a MaskedMimic checkpoint with velocity-command goals"
        )
    if not hasattr(controller, "set_steering"):
        raise TypeError("controller does not provide set_steering()")

    heading = math.atan2(velocity[1], velocity[0])
    facing = None
    if command.gaze_direction_world is not None:
        gaze = as_vec3(command.gaze_direction_world, "gaze_direction_world")
        if math.hypot(gaze[0], gaze[1]) > 1e-9:
            facing = math.atan2(gaze[1], gaze[0])

    controller.set_steering(
        env_ids,
        min(speed, config.steering_speed_limit_m_s),
        heading,
        facing=facing,
    )


def dispatch_to_protomotions(
    controller: Any,
    env_ids: Any,
    command: ControlCommand,
    *,
    base_position_world: Vec3,
    config: ProtoMotionsBridgeConfig = ProtoMotionsBridgeConfig(),
    on_auxiliary_intent: Callable[[Any, ControlCommand], None] | None = None,
) -> None:
    """Apply one shared command to a batch using ProtoMotions native-goal APIs.

    ``env_ids`` should be a tensor on the simulator device, as expected by
    ProtoMotions. World-space position goals are replicated to each selected
    row. Call separately for groups whose targets differ. Protective-pose and
    physical-grasp requests require an explicit ``on_auxiliary_intent`` hook;
    they are never silently treated as implemented by a reaching goal. The
    hook may be called again when a command changes, so it should be idempotent.
    """
    count = len(env_ids)
    if count == 0:
        return

    needs_auxiliary = command.protect_head or command.grasp_requested
    if needs_auxiliary and on_auxiliary_intent is None:
        raise NotImplementedError(
            "protect_head/grasp_requested need a simulator-specific auxiliary handler"
        )

    if command.mode is ControlMode.HOLD:
        controller.hold(env_ids)
    elif command.mode is ControlMode.APPROACH:
        target = command.navigation_target_world
        if target is None:
            raise ValueError("APPROACH command requires navigation_target_world")
        position = as_vec3(target, "navigation_target_world")
        base = as_vec3(base_position_world, "base_position_world")
        speed = command.navigation_speed_m_s
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("approach navigation_speed_m_s must be finite and positive")
        distance = math.hypot(position[0] - base[0], position[1] - base[1])
        duration = distance / speed if distance > 1e-9 else config.min_goal_duration_s
        duration = min(config.max_goal_duration_s, max(config.min_goal_duration_s, duration))
        if not hasattr(controller, "set_location"):
            raise TypeError("controller does not provide set_location()")
        controller.set_location(env_ids, _replicate(position[:2], count), duration)
    elif command.mode is ControlMode.REACH:
        target = command.right_hand_target_world
        if target is None:
            raise ValueError("REACH command requires right_hand_target_world")
        position = as_vec3(target, "right_hand_target_world")
        if not hasattr(controller, "set_reaching"):
            raise TypeError("controller does not provide set_reaching()")
        controller.set_reaching(
            env_ids,
            _replicate(position, count),
            config.reach_goal_duration_s,
        )
    elif command.mode in (ControlMode.WANDER, ControlMode.RETREAT):
        _set_steering(controller, env_ids, command, config)
    else:
        raise ValueError(f"unsupported control mode: {command.mode!r}")

    if needs_auxiliary:
        on_auxiliary_intent(env_ids, command)


@dataclass(frozen=True, slots=True)
class CommandGateConfig:
    """Suppress per-frame goal resets while retaining meaningful updates."""

    min_dispatch_interval_s: float = 0.1
    position_update_threshold_m: float = 0.08
    direction_update_threshold_rad: float = 0.12
    speed_update_threshold_m_s: float = 0.05

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_dispatch_interval_s) or self.min_dispatch_interval_s < 0:
            raise ValueError("min_dispatch_interval_s must be finite and non-negative")
        for name in (
            "position_update_threshold_m",
            "direction_update_threshold_rad",
            "speed_update_threshold_m_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


def _point_changed(left: Vec3 | None, right: Vec3 | None, threshold: float) -> bool:
    if left is None or right is None:
        return left is not right
    return distance(as_vec3(left), as_vec3(right)) > threshold


def _direction_changed(
    left: Vec3 | None,
    right: Vec3 | None,
    threshold_rad: float,
) -> bool:
    if left is None or right is None:
        return left is not right
    left_vec = as_vec3(left)
    right_vec = as_vec3(right)
    if norm(left_vec) <= 1e-9 or norm(right_vec) <= 1e-9:
        return (norm(left_vec) <= 1e-9) != (norm(right_vec) <= 1e-9)
    cosine = max(-1.0, min(1.0, dot(normalize(left_vec), normalize(right_vec))))
    return math.acos(cosine) > threshold_rad


def _command_changed_enough(
    previous: ControlCommand,
    current: ControlCommand,
    config: CommandGateConfig,
) -> bool:
    if (
        previous.protect_head != current.protect_head
        or previous.grasp_requested != current.grasp_requested
        or previous.attention_target_id != current.attention_target_id
    ):
        return True
    if _point_changed(previous.navigation_target_world, current.navigation_target_world, config.position_update_threshold_m):
        return True
    if _point_changed(previous.right_hand_target_world, current.right_hand_target_world, config.position_update_threshold_m):
        return True
    if _point_changed(previous.look_at_world, current.look_at_world, config.position_update_threshold_m):
        return True
    if _direction_changed(previous.gaze_direction_world, current.gaze_direction_world, config.direction_update_threshold_rad):
        return True

    old_velocity = as_vec3(previous.base_velocity_world, "previous base velocity")
    new_velocity = as_vec3(current.base_velocity_world, "current base velocity")
    old_speed = math.hypot(old_velocity[0], old_velocity[1])
    new_speed = math.hypot(new_velocity[0], new_velocity[1])
    if abs(old_speed - new_speed) > config.speed_update_threshold_m_s:
        return True
    if old_speed > 1e-9 and new_speed > 1e-9:
        old_heading = math.atan2(old_velocity[1], old_velocity[0])
        new_heading = math.atan2(new_velocity[1], new_velocity[0])
        heading_delta = abs((new_heading - old_heading + math.pi) % (2 * math.pi) - math.pi)
        if heading_delta > config.direction_update_threshold_rad:
            return True
    return abs(previous.navigation_speed_m_s - current.navigation_speed_m_s) > config.speed_update_threshold_m_s


class ProtoMotionsCommandDispatcher:
    """Edge/threshold gate around the runtime adapter.

    Create one dispatcher per homogeneous environment batch. It avoids resetting
    ProtoMotions' timed goal on every physics step: task mode and grasp/guard
    transitions dispatch immediately, while drifting targets are rate-limited
    and sent only after crossing configured thresholds. Reset it when the batch
    membership or simulation episode changes.
    """

    def __init__(
        self,
        bridge_config: ProtoMotionsBridgeConfig = ProtoMotionsBridgeConfig(),
        gate_config: CommandGateConfig = CommandGateConfig(),
    ) -> None:
        self.bridge_config = bridge_config
        self.gate_config = gate_config
        self.reset()

    def reset(self) -> None:
        self._last_command: ControlCommand | None = None
        self._last_dispatch_time_s: float | None = None
        self._last_update_time_s: float | None = None

    def update(
        self,
        controller: Any,
        env_ids: Any,
        command: ControlCommand,
        *,
        now_s: float,
        base_position_world: Vec3,
        on_auxiliary_intent: Callable[[Any, ControlCommand], None] | None = None,
    ) -> bool:
        """Dispatch if the command changed enough; return whether it was sent."""
        if not math.isfinite(now_s) or now_s < 0:
            raise ValueError("now_s must be finite and non-negative")
        if self._last_update_time_s is not None and now_s < self._last_update_time_s:
            raise ValueError("now_s moved backwards; reset the dispatcher between episodes")
        self._last_update_time_s = now_s
        if len(env_ids) == 0:
            return False

        previous = self._last_command
        if previous is not None:
            if command.mode is not previous.mode:
                should_dispatch = True
            elif now_s - self._last_dispatch_time_s < self.gate_config.min_dispatch_interval_s:
                should_dispatch = (
                    command.protect_head != previous.protect_head
                    or command.grasp_requested != previous.grasp_requested
                    or command.attention_target_id != previous.attention_target_id
                )
            else:
                should_dispatch = _command_changed_enough(previous, command, self.gate_config)
        else:
            should_dispatch = True

        if not should_dispatch:
            return False

        dispatch_to_protomotions(
            controller,
            env_ids,
            command,
            base_position_world=base_position_world,
            config=self.bridge_config,
            on_auxiliary_intent=on_auxiliary_intent,
        )
        self._last_command = command
        self._last_dispatch_time_s = now_s
        return True
