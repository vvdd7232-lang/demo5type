import math
import unittest

from demo5type.rewards import InteractionRewardWeights, compute_interaction_rewards


class InteractionRewardTests(unittest.TestCase):
    def test_aligned_gaze_reach_contact_and_lift(self):
        result = compute_interaction_rewards(
            gaze_forward_world=(1.0, 0.0, 0.0),
            head_to_target_world=(2.0, 0.0, 0.0),
            right_hand_to_target_world=(0.0, 0.0, 0.0),
            finger_contact=True,
            object_lifted_m=0.35,
        )
        self.assertEqual(result.gaze, 1.0)
        self.assertEqual(result.reach, 1.0)
        self.assertEqual(result.contact, 1.0)
        self.assertEqual(result.lift, 1.0)
        self.assertGreater(result.total, 0.0)

    def test_no_contact_means_no_lift_reward(self):
        result = compute_interaction_rewards(
            gaze_forward_world=(1.0, 0.0, 0.0),
            head_to_target_world=(1.0, 0.0, 0.0),
            right_hand_to_target_world=(1.0, 0.0, 0.0),
            finger_contact=False,
            object_lifted_m=1.0,
        )
        self.assertEqual(result.contact, 0.0)
        self.assertEqual(result.lift, 0.0)

    def test_opposite_gaze_is_clamped_to_zero(self):
        result = compute_interaction_rewards(
            gaze_forward_world=(1.0, 0.0, 0.0),
            head_to_target_world=(-1.0, 0.0, 0.0),
            right_hand_to_target_world=None,
            finger_contact=False,
        )
        self.assertEqual(result.gaze, 0.0)
        self.assertEqual(result.reach, 0.0)

    def test_zero_direction_has_finite_reward(self):
        result = compute_interaction_rewards(
            gaze_forward_world=(0.0, 0.0, 0.0),
            head_to_target_world=(0.0, 0.0, 0.0),
            right_hand_to_target_world=(0.0, 0.0, 0.0),
            finger_contact=False,
        )
        self.assertEqual(result.gaze, 0.0)
        self.assertTrue(math.isfinite(result.total))

    def test_invalid_weights_are_rejected(self):
        with self.assertRaises(ValueError):
            InteractionRewardWeights(reach_scale_m=0.0)


if __name__ == "__main__":
    unittest.main()
