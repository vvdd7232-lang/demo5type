"""Minimal 3-D vector helpers; kept dependency-free for local tests and tools."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TypeAlias

Vec3: TypeAlias = tuple[float, float, float]
_EPSILON = 1e-12


def as_vec3(value: Iterable[float], name: str = "vector") -> Vec3:
    """Convert a three-component iterable to finite floats."""
    try:
        components = tuple(float(component) for component in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain three finite numbers") from exc

    if len(components) != 3 or not all(math.isfinite(component) for component in components):
        raise ValueError(f"{name} must contain three finite numbers")
    return components  # type: ignore[return-value]


def add(left: Vec3, right: Vec3) -> Vec3:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def subtract(left: Vec3, right: Vec3) -> Vec3:
    """Return ``left - right``."""
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def dot(left: Vec3, right: Vec3) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def norm(vector: Vec3) -> float:
    return math.hypot(vector[0], vector[1], vector[2])


def normalize(vector: Vec3, fallback: Vec3 = (0.0, 0.0, 0.0)) -> Vec3:
    """Normalize a vector, using a normalized fallback for near-zero input."""
    magnitude = norm(vector)
    if magnitude > _EPSILON:
        return (vector[0] / magnitude, vector[1] / magnitude, vector[2] / magnitude)

    fallback_magnitude = norm(fallback)
    if fallback_magnitude > _EPSILON:
        return (
            fallback[0] / fallback_magnitude,
            fallback[1] / fallback_magnitude,
            fallback[2] / fallback_magnitude,
        )
    return (0.0, 0.0, 0.0)


def distance(left: Vec3, right: Vec3) -> float:
    return norm(subtract(left, right))
