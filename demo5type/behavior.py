"""High-level behavior state machine for a humanoid interaction prototype.

This module emits goals and flags only. It does not produce joint actions,
simulate contact, or claim that a policy has learned the behavior.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from .geometry import Vec3, as_vec3, distance, normalize, subtract


class BehaviorState(str, Enum):
    EXPLORE = "explore"
    INTEREST = "interest"
    FRIGHT = "fright"


@dataclass(frozen=True, slots=True)
class TrackedTarget:
    target_id: str
    position_world: Vec3
    is_moving: bool = False
    is_person: bool = False
    is_interactable: bool = False

    def __post_init__(self) -> None:
        if not self.target_id.strip():
            raise ValueError("target_id must not be empty")
        object.__setattr__(self, "position_world", as_vec3(self.position_world, "target position"))


@dataclass(frozen=True, slots=True)
class SceneObservation:
    time_s: float
    head_position_world: Vec3
    right_hand_position_world: Vec3
    head_forward_world: Vec3 = (1.0, 0.0, 0.0)
    base_forward_world: Vec3 = (1.0, 0.0, 0.0)
    targets: tuple[TrackedTarget, ...] = ()
    sound_level: float = 0.0
    impact_magnitude_ns: float = 0.0
    threat_detected: bool = False
    threat_position_world: Vec3 | None = None
    finger_contact: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_s) or self.time_s < 0:
            raise ValueError("time_s must be finite and non-negative")
        if not math.isfinite(self.sound_level) or not 0 <= self.sound_level <= 1:
            raise ValueError("sound_level must be normalized to [0, 1]")
        if not math.isfinite(self.impact_magnitude_ns) or self.impact_magnitude_ns < 0:
            raise ValueError("impact_magnitude_ns must be finite and non-negative")

        object.__setattr__(self, "head_position_world", as_vec3(self.head_position_world, "head position"))
        object.__setattr__(
            self,
            "right_hand_position_world",
            as_vec3(self.right_hand_position_world, "right hand position"),
        )
        object.__setattr__(self, "head_forward_world", as_vec3(self.head_forward_world, "head forward"))
        object.__setattr__(self, "base_forward_world", as_vec3(self.base_forward_world, "base forward"))
        if self.threat_position_world is not None:
            object.__setattr__(
                self,
                "threat_position_world",
                as_vec3(self.threat_position_world, "threat position"),
            )

        targets = tuple(self.targets)
        if not all(isinstance(target, TrackedTarget) for target in targets):
            raise ValueError("targets must contain TrackedTarget values")
        identifiers = [target.target_id for target in targets]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("target_id values must be unique within one observation")
        object.__setattr__(self, "targets", targets)


@dataclass(frozen=True, slots=True)
class ControllerConfig:
    interest_enter_radius_m: float = 4.0
    interest_exit_radius_m: float = 5.5
    reach_activation_radius_m: float = 1.0
    grasp_contact_radius_m: float = 0.22
    startle_sound_threshold: float = 0.8
    startle_impulse_threshold_ns: float = 1.5
    fright_duration_s: float = 1.5
    fright_cooldown_s: float = 2.0
    exploration_walk_speed_m_s: float = 0.15
    exploration_scan_amplitude_rad: float = 0.55
    exploration_scan_frequency_rad_s: float = 0.7
    retreat_speed_m_s: float = 0.8
    retreat_fallback_world: Vec3 = (-1.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        positive_fields = (
            "interest_enter_radius_m",
            "reach_activation_radius_m",
            "grasp_contact_radius_m",
            "startle_sound_threshold",
            "startle_impulse_threshold_ns",
            "fright_duration_s",
        )
        for name in positive_fields:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

        if (
            not math.isfinite(self.interest_exit_radius_m)
            or self.interest_exit_radius_m < self.interest_enter_radius_m
        ):
            raise ValueError("interest_exit_radius_m must be finite and at least the enter radius")
        for name in ("fright_cooldown_s", "exploration_walk_speed_m_s", "exploration_scan_frequency_rad_s", "retreat_speed_m_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.startle_sound_threshold > 1:
            raise ValueError("startle_sound_threshold must be normalized to [0, 1]")
        if (
            not math.isfinite(self.exploration_scan_amplitude_rad)
            or not 0 <= self.exploration_scan_amplitude_rad <= math.pi / 2
        ):
            raise ValueError("exploration scan amplitude must be in [0, pi/2]")

        fallback = as_vec3(self.retreat_fallback_world, "retreat fallback")
        if math.hypot(fallback[0], fallback[1]) <= 1e-12:
            raise ValueError("retreat fallback must have a non-zero horizontal component")
        object.__setattr__(self, "retreat_fallback_world", fallback)


@dataclass(frozen=True, slots=True)
class BehaviorIntent:
    state: BehaviorState
    reason: str
    attention_target_id: str | None = None
    interaction_target_id: str | None = None
    look_at_world: Vec3 | None = None
    gaze_direction_world: Vec3 | None = None
    approach_target_world: Vec3 | None = None
    right_hand_target_world: Vec3 | None = None
    wander_direction_world: Vec3 | None = None
    wander_speed_m_s: float = 0.0
    retreat_direction_world: Vec3 | None = None
    retreat_speed_m_s: float = 0.0
    protect_head: bool = False
    grasp_requested: bool = False


class BehaviorStateMachine:
    """Translate per-frame sensor inputs into high-level goals.

    Use one instance per simulated character/environment. ``reset`` should be
    called when that environment is reset. Times must be monotonically
    increasing between resets.
    """

    def __init__(self, config: ControllerConfig = ControllerConfig()) -> None:
        self.config = config
        self.reset()

    @property
    def state(self) -> BehaviorState:
        return self._state

    def reset(self) -> None:
        self._state = BehaviorState.EXPLORE
        self._interaction_target_id: str | None = None
        self._fright_until_s = float("-inf")
        self._cooldown_until_s = float("-inf")
        self._last_time_s: float | None = None

    def update(self, observation: SceneObservation) -> BehaviorIntent:
        if self._last_time_s is not None and observation.time_s < self._last_time_s:
            raise ValueError("time_s moved backwards; reset the controller between episodes")
        self._last_time_s = observation.time_s

        if self._is_startled(observation) and observation.time_s >= self._cooldown_until_s:
            self._state = BehaviorState.FRIGHT
            self._interaction_target_id = None
            self._fright_until_s = observation.time_s + self.config.fright_duration_s
            self._cooldown_until_s = self._fright_until_s + self.config.fright_cooldown_s

        if self._state is BehaviorState.FRIGHT and observation.time_s < self._fright_until_s:
            return self._fright_intent(observation)

        if self._state is BehaviorState.FRIGHT:
            self._state = BehaviorState.EXPLORE
            self._interaction_target_id = None

        target = self._select_interaction_target(observation)
        if target is not None:
            self._state = BehaviorState.INTEREST
            self._interaction_target_id = target.target_id
            hand_distance = distance(observation.right_hand_position_world, target.position_world)
            can_reach = hand_distance <= self.config.reach_activation_radius_m
            return BehaviorIntent(
                state=BehaviorState.INTEREST,
                reason="interactable_target_in_range",
                attention_target_id=target.target_id,
                interaction_target_id=target.target_id,
                look_at_world=target.position_world,
                gaze_direction_world=normalize(
                    subtract(target.position_world, observation.head_position_world),
                    fallback=observation.head_forward_world,
                ),
                approach_target_world=target.position_world,
                right_hand_target_world=target.position_world if can_reach else None,
                grasp_requested=(
                    observation.finger_contact
                    and hand_distance <= self.config.grasp_contact_radius_m
                ),
            )

        self._state = BehaviorState.EXPLORE
        self._interaction_target_id = None
        salient = self._nearest_target(observation, lambda item: item.is_moving or item.is_person)
        base_forward = normalize(
            (observation.base_forward_world[0], observation.base_forward_world[1], 0.0),
            fallback=(1.0, 0.0, 0.0),
        )
        if salient is not None:
            gaze_direction = normalize(
                subtract(salient.position_world, observation.head_position_world),
                fallback=observation.head_forward_world,
            )
        else:
            scan_yaw = self.config.exploration_scan_amplitude_rad * math.sin(
                observation.time_s * self.config.exploration_scan_frequency_rad_s
            )
            cos_yaw = math.cos(scan_yaw)
            sin_yaw = math.sin(scan_yaw)
            gaze_direction = normalize(
                (
                    base_forward[0] * cos_yaw - base_forward[1] * sin_yaw,
                    base_forward[0] * sin_yaw + base_forward[1] * cos_yaw,
                    0.0,
                ),
                fallback=observation.head_forward_world,
            )

        return BehaviorIntent(
            state=BehaviorState.EXPLORE,
            reason="no_interactable_target_in_range",
            attention_target_id=None if salient is None else salient.target_id,
            look_at_world=None if salient is None else salient.position_world,
            gaze_direction_world=gaze_direction,
            wander_direction_world=base_forward,
            wander_speed_m_s=self.config.exploration_walk_speed_m_s,
        )

    def _is_startled(self, observation: SceneObservation) -> bool:
        return (
            observation.threat_detected
            or observation.sound_level >= self.config.startle_sound_threshold
            or observation.impact_magnitude_ns >= self.config.startle_impulse_threshold_ns
        )

    def _select_interaction_target(self, observation: SceneObservation) -> TrackedTarget | None:
        current = None
        if self._state is BehaviorState.INTEREST and self._interaction_target_id is not None:
            current = next(
                (
                    item
                    for item in observation.targets
                    if (
                        item.target_id == self._interaction_target_id
                        and item.is_interactable
                        and not item.is_person
                    )
                ),
                None,
            )
        if current is not None:
            if distance(observation.head_position_world, current.position_world) <= self.config.interest_exit_radius_m:
                return current

        candidate = self._nearest_target(
            observation,
            lambda item: item.is_interactable and not item.is_person,
        )
        if candidate is not None and distance(observation.head_position_world, candidate.position_world) <= self.config.interest_enter_radius_m:
            return candidate
        return None

    @staticmethod
    def _nearest_target(
        observation: SceneObservation,
        predicate: Callable[[TrackedTarget], bool],
    ) -> TrackedTarget | None:
        candidates = [item for item in observation.targets if predicate(item)]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda item: (
                distance(observation.head_position_world, item.position_world),
                item.target_id,
            ),
        )

    def _fright_intent(self, observation: SceneObservation) -> BehaviorIntent:
        fallback = (
            self.config.retreat_fallback_world[0],
            self.config.retreat_fallback_world[1],
            0.0,
        )
        threat = observation.threat_position_world
        if threat is not None:
            away = (
                observation.head_position_world[0] - threat[0],
                observation.head_position_world[1] - threat[1],
                0.0,
            )
            retreat = normalize(away, fallback=fallback)
            gaze = normalize(
                subtract(threat, observation.head_position_world),
                fallback=(-retreat[0], -retreat[1], 0.0),
            )
        else:
            retreat = normalize(fallback)
            gaze = normalize((-retreat[0], -retreat[1], 0.0))

        return BehaviorIntent(
            state=BehaviorState.FRIGHT,
            reason="startle_or_threat",
            look_at_world=threat,
            gaze_direction_world=gaze,
            retreat_direction_world=retreat,
            retreat_speed_m_s=self.config.retreat_speed_m_s,
            protect_head=True,
        )
