"""Scalar reference rewards for gaze, reaching, contact, and lifting."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .geometry import Vec3, as_vec3, dot, norm, normalize


@dataclass(frozen=True, slots=True)
class InteractionRewardWeights:
    gaze: float = 0.25
    reach: float = 0.5
    contact: float = 1.0
    lift: float = 1.5
    reach_scale_m: float = 0.5
    lift_target_m: float = 0.35

    def __post_init__(self) -> None:
        for name in ("gaze", "reach", "contact", "lift"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} reward weight must be finite and non-negative")
        for name in ("reach_scale_m", "lift_target_m"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    gaze: float
    reach: float
    contact: float
    lift: float
    total: float


DEFAULT_REWARD_WEIGHTS = InteractionRewardWeights()


def compute_interaction_rewards(
    *,
    gaze_forward_world: Vec3,
    head_to_target_world: Vec3 | None,
    right_hand_to_target_world: Vec3 | None,
    finger_contact: bool,
    object_lifted_m: float = 0.0,
    weights: InteractionRewardWeights = DEFAULT_REWARD_WEIGHTS,
) -> RewardBreakdown:
    """Compute bounded task terms; this is a reference, not an RL tensor kernel.

    ``object_lifted_m`` is the object's upward displacement from its initial
    support height. Lift reward is gated by finger contact so a coincidental
    upward motion does not count as a successful grasp.
    """
    if not math.isfinite(object_lifted_m):
        raise ValueError("object_lifted_m must be finite")

    gaze = 0.0
    if head_to_target_world is not None:
        target_direction = as_vec3(head_to_target_world, "head_to_target_world")
        forward = as_vec3(gaze_forward_world, "gaze_forward_world")
        if norm(target_direction) > 1e-12 and norm(forward) > 1e-12:
            gaze = max(0.0, min(1.0, dot(normalize(forward), normalize(target_direction))))

    reach = 0.0
    if right_hand_to_target_world is not None:
        hand_delta = as_vec3(right_hand_to_target_world, "right_hand_to_target_world")
        reach = math.exp(-norm(hand_delta) / weights.reach_scale_m)

    contact = 1.0 if finger_contact else 0.0
    lift = 0.0
    if finger_contact:
        lift = max(0.0, min(1.0, object_lifted_m / weights.lift_target_m))

    total = (
        weights.gaze * gaze
        + weights.reach * reach
        + weights.contact * contact
        + weights.lift * lift
    )
    return RewardBreakdown(gaze=gaze, reach=reach, contact=contact, lift=lift, total=total)
