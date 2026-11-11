import unittest

from demo5type.observations import OBJECT_OBSERVATION_SIZE, build_object_observation


class ObjectObservationTests(unittest.TestCase):
    def test_offsets_are_object_minus_head_and_hand(self):
        features = build_object_observation(
            head_position_world=(1.0, 2.0, 3.0),
            right_hand_position_world=(2.0, 2.0, 2.0),
            object_position_world=(4.0, 5.0, 6.0),
        )
        self.assertEqual(features, (3.0, 3.0, 3.0, 2.0, 3.0, 4.0, 1.0))
        self.assertEqual(len(features), OBJECT_OBSERVATION_SIZE)

    def test_missing_target_keeps_feature_dimension(self):
        features = build_object_observation(
            head_position_world=(0.0, 0.0, 1.0),
            right_hand_position_world=(0.5, 0.0, 0.8),
            object_position_world=None,
        )
        self.assertEqual(features, (0.0,) * OBJECT_OBSERVATION_SIZE)

    def test_rejects_non_finite_coordinates(self):
        with self.assertRaises(ValueError):
            build_object_observation(
                head_position_world=(0.0, 0.0, 1.0),
                right_hand_position_world=(0.5, 0.0, 0.8),
                object_position_world=(float("nan"), 0.0, 0.0),
            )


if __name__ == "__main__":
    unittest.main()
