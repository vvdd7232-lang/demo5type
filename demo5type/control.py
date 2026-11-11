"""Map behavior intents to bounded, backend-neutral navigation/hand goals.

The result is still a command suggestion: a simulator integration must route
navigation targets through collision checking and send hand/grasp goals to the
chosen controller. This module deliberately does not write actuator commands.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .behavior import BehaviorIntent, BehaviorState
from .geometry import Vec3, as_vec3, norm, normalize


class ControlMode(str, Enum):
    HOLD = "hold"
    WANDER = "wander"
    APPROACH = "approach"
    REACH = "reach"
    RETREAT = "retreat"


@dataclass(frozen=True, slots=True)
class InteractionControlConfig:
    approach_standoff_m: float = 0.85
    reach_switch_distance_m: float = 1.2
    wander_speed_limit_m_s: float = 0.2
    approach_speed_m_s: float = 0.35
    retreat_speed_limit_m_s: float = 0.8

    def __post_init__(self) -> None:
        positive = (
            "approach_standoff_m",
            "reach_switch_distance_m",
            "approach_speed_m_s",
        )
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.reach_switch_distance_m <= self.approach_standoff_m:
            raise ValueError("reach_switch_distance_m must exceed approach_standoff_m")
        for name in ("wander_speed_limit_m_s", "retreat_speed_limit_m_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class ControlCommand:
    mode: ControlMode
    reason: str
    base_velocity_world: Vec3 = (0.0, 0.0, 0.0)
    navigation_target_world: Vec3 | None = None
    navigation_speed_m_s: float = 0.0
    gaze_direction_world: Vec3 | None = None
    look_at_world: Vec3 | None = None
    right_hand_target_world: Vec3 | None = None
    attention_target_id: str | None = None
    protect_head: bool = False
    grasp_requested: bool = False


def _horizontal_direction(vector: Vec3 | None, fallback: Vec3) -> Vec3:
    if vector is None:
        vector = (0.0, 0.0, 0.0)
    value = as_vec3(vector, "direction")
    horizontal_fallback = (fallback[0], fallback[1], 0.0)
    return normalize((value[0], value[1], 0.0), fallback=horizontal_fallback)


def _bounded_speed(value: float, limit: float, name: str) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return min(value, limit)


def command_from_intent(
    intent: BehaviorIntent,
    *,
    base_position_world: Vec3,
    base_forward_world: Vec3,
    config: InteractionControlConfig = InteractionControlConfig(),
) -> ControlCommand:
    """Create a bounded high-level command from one state-machine intent.

    For interaction targets, the navigation goal is placed at a horizontal
    stand-off from the object's center, not on the object itself. ``base_z`` is
    retained. A real backend must still plan around walls and other colliders.
    """
    base_position = as_vec3(base_position_world, "base_position_world")
    base_forward = as_vec3(base_forward_world, "base_forward_world")
    heading_fallback = _horizontal_direction(base_forward, (1.0, 0.0, 0.0))

    gaze = None
    if intent.gaze_direction_world is not None:
        gaze = normalize(
            as_vec3(intent.gaze_direction_world, "gaze_direction_world"),
            fallback=heading_fallback,
        )
    look_at = (
        None
        if intent.look_at_world is None
        else as_vec3(intent.look_at_world, "look_at_world")
    )
    common = {
        "reason": intent.reason,
        "gaze_direction_world": gaze,
        "look_at_world": look_at,
        "attention_target_id": intent.attention_target_id,
        "protect_head": intent.protect_head,
    }

    if intent.state is BehaviorState.FRIGHT:
        direction = _horizontal_direction(intent.retreat_direction_world, (0.0, 0.0, 0.0))
        speed = _bounded_speed(
            intent.retreat_speed_m_s,
            config.retreat_speed_limit_m_s,
            "retreat_speed_m_s",
        )
        return ControlCommand(
            mode=ControlMode.RETREAT if norm(direction) > 0 else ControlMode.HOLD,
            base_velocity_world=(
                speed * direction[0],
                speed * direction[1],
                0.0,
            ),
            right_hand_target_world=None,
            grasp_requested=False,
            **common,
        )

    if intent.state is BehaviorState.INTEREST and intent.approach_target_world is not None:
        target = as_vec3(intent.approach_target_world, "approach_target_world")
        delta_xy = (base_position[0] - target[0], base_position[1] - target[1], 0.0)
        distance_xy = norm(delta_xy)
        away_from_target = normalize(
            delta_xy,
            fallback=(-heading_fallback[0], -heading_fallback[1], 0.0),
        )
        navigation_goal = (
            target[0] + away_from_target[0] * config.approach_standoff_m,
            target[1] + away_from_target[1] * config.approach_standoff_m,
            base_position[2],
        )
        can_reach = (
            intent.right_hand_target_world is not None
            and distance_xy <= config.reach_switch_distance_m
        )
        if can_reach:
            hand_goal = as_vec3(intent.right_hand_target_world, "right_hand_target_world")
            return ControlCommand(
                mode=ControlMode.REACH,
                right_hand_target_world=hand_goal,
                grasp_requested=intent.grasp_requested,
                **common,
            )

        return ControlCommand(
            mode=ControlMode.APPROACH,
            navigation_target_world=navigation_goal,
            navigation_speed_m_s=config.approach_speed_m_s,
            right_hand_target_world=None,
            grasp_requested=False,
            **common,
        )

    if intent.state is BehaviorState.EXPLORE:
        direction = _horizontal_direction(intent.wander_direction_world, heading_fallback)
        speed = _bounded_speed(
            intent.wander_speed_m_s,
            config.wander_speed_limit_m_s,
            "wander_speed_m_s",
        )
        return ControlCommand(
            mode=ControlMode.WANDER if speed > 0 else ControlMode.HOLD,
            base_velocity_world=(
                speed * direction[0],
                speed * direction[1],
                0.0,
            ),
            right_hand_target_world=None,
            grasp_requested=False,
            **common,
        )

    return ControlCommand(mode=ControlMode.HOLD, **common)
