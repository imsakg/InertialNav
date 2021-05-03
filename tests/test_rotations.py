"""Scalar-first Hamilton/body-to-NED attitude convention tests."""

import math
import unittest

from inertialnav import rotations


class QuaternionNormalizationTests(unittest.TestCase):
    def test_identity_general_and_negative_sign_are_normalized(self):
        self.assertEqual(rotations.normalize((2, 0, 0, 0)), (1, 0, 0, 0))
        self.assertEqual(rotations.normalize((-2, 0, 0, 0)), (-1, 0, 0, 0))
        actual = rotations.normalize((1, -2, 3, -4))
        self.assertIsInstance(actual, tuple)
        self.assertEqual(len(actual), 4)
        self.assertAlmostEqual(math.hypot(*actual), 1, places=15)
        for value, reference in zip(actual, (1, -2, 3, -4)):
            self.assertAlmostEqual(value, reference / math.sqrt(30), places=15)

    def test_extreme_finite_magnitudes_do_not_overflow_or_underflow(self):
        for magnitude in (1e308, 1e-300, 5e-324):
            self.assertEqual(rotations.normalize((magnitude,) * 4), (0.5,) * 4)

    def test_does_not_mutate_input(self):
        original = [2, 0, 0, 0]
        rotations.normalize(original)
        self.assertEqual(original, [2, 0, 0, 0])

    def test_rejects_zero_dimensions_and_all_nonfinite_components(self):
        for invalid in ((0, 0, 0, 0), (), (1, 2, 3), (1, 2, 3, 4, 5), None, "1234"):
            with self.assertRaises(ValueError):
                rotations.normalize(invalid)
        for index in range(4):
            for invalid in (math.nan, math.inf, -math.inf, None):
                quaternion = [1.0] * 4
                quaternion[index] = invalid
                with self.assertRaises(ValueError):
                    rotations.normalize(quaternion)


if __name__ == "__main__":
    unittest.main()
