"""Run a tiny sensor-driven scenario without Isaac Sim or a GPU.

This exercises only the high-level rule logic; it is not a physical simulation.
"""

from __future__ import annotations

import json

from .behavior import (
    BehaviorStateMachine,
    SceneObservation,
    TrackedTarget,
)
from .control import command_from_intent
from .observations import build_object_observation
from .rewards import compute_interaction_rewards


def run_demo() -> list[dict[str, object]]:
    controller = BehaviorStateMachine()
    frames = [
        SceneObservation(
            time_s=0.0,
            head_position_world=(0.0, 0.0, 1.6),
            right_hand_position_world=(0.35, 0.0, 1.0),
        ),
        SceneObservation(
            time_s=1.0,
            head_position_world=(0.0, 0.0, 1.6),
            right_hand_position_world=(0.35, 0.0, 1.0),
            targets=(TrackedTarget("can", (2.8, 0.0, 0.7), is_interactable=True),),
        ),
        SceneObservation(
            time_s=2.0,
            head_position_world=(0.0, 0.0, 1.6),
            right_hand_position_world=(0.35, 0.0, 1.0),
            targets=(TrackedTarget("can", (0.9, 0.0, 1.0), is_interactable=True),),
        ),
        SceneObservation(
            time_s=2.2,
            head_position_world=(0.0, 0.0, 1.6),
            right_hand_position_world=(0.35, 0.0, 1.0),
            targets=(TrackedTarget("can", (0.52, 0.0, 1.0), is_interactable=True),),
            finger_contact=True,
        ),
        SceneObservation(
            time_s=3.0,
            head_position_world=(0.0, 0.0, 1.6),
            right_hand_position_world=(0.35, 0.0, 1.0),
            targets=(TrackedTarget("can", (0.52, 0.0, 1.0), is_interactable=True),),
            sound_level=0.95,
            threat_position_world=(2.0, 0.0, 1.0),
        ),
        SceneObservation(
            time_s=5.0,
            head_position_world=(0.0, 0.0, 1.6),
            right_hand_position_world=(0.35, 0.0, 1.0),
            targets=(TrackedTarget("can", (0.52, 0.0, 1.0), is_interactable=True),),
        ),
    ]

    records: list[dict[str, object]] = []
    for frame in frames:
        intent = controller.update(frame)
        command = command_from_intent(
            intent,
            base_position_world=(0.0, 0.0, 0.0),
            base_forward_world=frame.base_forward_world,
        )
        target = next(
            (item for item in frame.targets if item.target_id == intent.interaction_target_id),
            None,
        )
        relative = build_object_observation(
            head_position_world=frame.head_position_world,
            right_hand_position_world=frame.right_hand_position_world,
            object_position_world=None if target is None else target.position_world,
        )
        reward = compute_interaction_rewards(
            gaze_forward_world=frame.head_forward_world,
            head_to_target_world=None if target is None else relative[0:3],
            right_hand_to_target_world=None if target is None else relative[3:6],
            finger_contact=frame.finger_contact and intent.grasp_requested,
        )
        records.append(
            {
                "time_s": frame.time_s,
                "state": intent.state.value,
                "reason": intent.reason,
                "attention_target_id": intent.attention_target_id,
                "interaction_target_id": intent.interaction_target_id,
                "control_mode": command.mode.value,
                "base_velocity_world": command.base_velocity_world,
                "navigation_target_world": command.navigation_target_world,
                "navigation_speed_m_s": round(command.navigation_speed_m_s, 3),
                "right_hand_goal": command.right_hand_target_world,
                "gaze_direction": command.gaze_direction_world,
                "protect_head": command.protect_head,
                "grasp_requested": intent.grasp_requested,
                "reward_terms": {
                    "gaze": round(reward.gaze, 3),
                    "reach": round(reward.reach, 3),
                    "contact": round(reward.contact, 3),
                    "lift": round(reward.lift, 3),
                    "total": round(reward.total, 3),
                },
            }
        )
    return records


def main() -> None:
    for record in run_demo():
        print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
