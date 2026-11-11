"""ProtoMotions 3 inference hook for a native right-hand reach goal.

This config is intentionally small: it reuses ProtoMotions' trained
MaskedMimic controller and only changes its inference-time task. The selected
checkpoint must support native goals, and the body name must be conditionable
for that checkpoint/robot.
"""

from protomotions.envs.control.masked_mimic_tasks.controller import (
    configure_masked_mimic_task,
)
from protomotions.utils.experiment_configs import ExperimentConfigs


RIGHT_HAND_BODY_NAME = "R_Hand"  # Default SMPL body name; change for another robot.


def apply_inference_overrides(configs: ExperimentConfigs, args) -> None:
    """Select keyboard-controlled reaching for a compatible MaskedMimic policy."""
    if args.full_eval:
        raise ValueError("Interactive reaching cannot run motion-tracking full evaluation")

    configure_masked_mimic_task(configs.env, configs.robot, "reaching")
    task = configs.env.control_components["masked_mimic"]
    task.reach_body_name = RIGHT_HAND_BODY_NAME
    task.command_source = "keyboard"
