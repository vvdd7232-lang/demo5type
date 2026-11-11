"""ProtoMotions inference hook for external Demo 5 Type command dispatch.

Use this preset from a custom inference loop that calls
``runtime_adapter.dispatch_to_protomotions``. Unlike ``reach_inference.py``,
it does not register keyboard handlers.
"""

from protomotions.envs.control.masked_mimic_tasks.controller import (
    configure_masked_mimic_task,
)
from protomotions.utils.experiment_configs import ExperimentConfigs


RIGHT_HAND_BODY_NAME = "R_Hand"  # Standard SMPL body name; adapt to the chosen robot.


def apply_inference_overrides(configs: ExperimentConfigs, args) -> None:
    """Configure a script-driven native reach mode for a compatible checkpoint."""
    if args.full_eval:
        raise ValueError("Behavior-driven goals cannot run motion-tracking full evaluation")

    configure_masked_mimic_task(configs.env, configs.robot, "reaching")
    task = configs.env.control_components["masked_mimic"]
    task.reach_body_name = RIGHT_HAND_BODY_NAME
    task.command_source = "script"
