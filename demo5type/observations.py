"""Reference policy features for a visible object and the humanoid's hand."""

from __future__ import annotations

from .geometry import Vec3, as_vec3, subtract

OBJECT_OBSERVATION_SIZE = 7


def build_object_observation(
    *,
    head_position_world: Vec3,
    right_hand_position_world: Vec3,
    object_position_world: Vec3 | None,
) -> tuple[float, ...]:
    """Encode head-to-object and hand-to-object offsets plus a visibility bit.

    The six offsets are in world axes and meters. A simulator adapter should
    rotate them into the policy's chosen body-local frame, if required. The
    final scalar keeps the feature dimension fixed when no object is visible.
    """
    head = as_vec3(head_position_world, "head_position_world")
    hand = as_vec3(right_hand_position_world, "right_hand_position_world")
    if object_position_world is None:
        return (0.0,) * OBJECT_OBSERVATION_SIZE

    target = as_vec3(object_position_world, "object_position_world")
    return (*subtract(target, head), *subtract(target, hand), 1.0)
