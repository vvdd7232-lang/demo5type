import math
import random
import unittest

from demo5type.domain_randomization import (
    RandomizationConfig,
    sample_disturbance,
    sample_episode_randomization,
)


class DomainRandomizationTests(unittest.TestCase):
    def test_episode_samples_are_reproducible_and_bounded(self):
        config = RandomizationConfig(friction_min=0.7, friction_max=1.1)
        first = sample_episode_randomization(random.Random(42), config)
        second = sample_episode_randomization(random.Random(42), config)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first.floor_friction, 0.7)
        self.assertLessEqual(first.floor_friction, 1.1)

    def test_zero_probability_never_pushes(self):
        config = RandomizationConfig(push_probability_per_step=0.0)
        self.assertIsNone(sample_disturbance(random.Random(1), config))

    def test_certain_push_has_requested_impulse_magnitude(self):
        config = RandomizationConfig(
            push_probability_per_step=1.0,
            impulse_min_ns=1.0,
            impulse_max_ns=1.0,
        )
        disturbance = sample_disturbance(random.Random(7), config)
        self.assertIsNotNone(disturbance)
        self.assertAlmostEqual(disturbance.magnitude_ns, 1.0)
        self.assertAlmostEqual(math.hypot(*disturbance.impulse_world_ns), 1.0)

    def test_invalid_ranges_are_rejected(self):
        with self.assertRaises(ValueError):
            RandomizationConfig(friction_min=1.2, friction_max=0.8)


if __name__ == "__main__":
    unittest.main()
