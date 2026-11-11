"""Deterministic samplers for per-episode friction and occasional impulses.

These functions only sample values. A physics-backend adapter must apply them
through Isaac Lab or ProtoMotions APIs; this module does not modify a simulator.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .geometry import Vec3


@dataclass(frozen=True, slots=True)
class RandomizationConfig:
    friction_min: float = 0.55
    friction_max: float = 1.25
    push_probability_per_step: float = 0.001
    impulse_min_ns: float = 0.6
    impulse_max_ns: float = 1.8
    max_vertical_direction_fraction: float = 0.2

    def __post_init__(self) -> None:
        values = (
            self.friction_min,
            self.friction_max,
            self.push_probability_per_step,
            self.impulse_min_ns,
            self.impulse_max_ns,
            self.max_vertical_direction_fraction,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("randomization parameters must be finite")
        if self.friction_min <= 0 or self.friction_max < self.friction_min:
            raise ValueError("friction range must be positive and ordered")
        if not 0 <= self.push_probability_per_step <= 1:
            raise ValueError("push probability must be between zero and one")
        if self.impulse_min_ns < 0 or self.impulse_max_ns < self.impulse_min_ns:
            raise ValueError("impulse range must be non-negative and ordered")
        if not 0 <= self.max_vertical_direction_fraction < 1:
            raise ValueError("vertical direction fraction must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class EpisodeRandomization:
    floor_friction: float


@dataclass(frozen=True, slots=True)
class Disturbance:
    impulse_world_ns: Vec3
    magnitude_ns: float


DEFAULT_RANDOMIZATION = RandomizationConfig()


def sample_episode_randomization(
    rng: random.Random,
    config: RandomizationConfig = DEFAULT_RANDOMIZATION,
) -> EpisodeRandomization:
    """Sample one floor-friction coefficient for a new episode."""
    return EpisodeRandomization(
        floor_friction=rng.uniform(config.friction_min, config.friction_max)
    )


def sample_disturbance(
    rng: random.Random,
    config: RandomizationConfig = DEFAULT_RANDOMIZATION,
) -> Disturbance | None:
    """Maybe sample a horizontal-dominant impulse, in newton-seconds."""
    if rng.random() >= config.push_probability_per_step:
        return None

    magnitude = rng.uniform(config.impulse_min_ns, config.impulse_max_ns)
    z_fraction = rng.uniform(
        -config.max_vertical_direction_fraction,
        config.max_vertical_direction_fraction,
    )
    horizontal_fraction = math.sqrt(1.0 - z_fraction * z_fraction)
    angle = rng.uniform(-math.pi, math.pi)
    vector = (
        magnitude * horizontal_fraction * math.cos(angle),
        magnitude * horizontal_fraction * math.sin(angle),
        magnitude * z_fraction,
    )
    return Disturbance(impulse_world_ns=vector, magnitude_ns=magnitude)
