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


class HamiltonAlgebraTests(unittest.TestCase):
    def test_ordered_basis_products_are_hamilton_not_commutative(self):
        i, j, k = (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)
        self.assertEqual(rotations.multiply(i, j), k)
        self.assertEqual(rotations.multiply(j, i), tuple(-value for value in k))
        self.assertEqual(rotations.multiply(i, i), (-1, 0, 0, 0))

    def test_identity_and_conjugate_product_preserve_raw_norm(self):
        quaternion = (1, -2, 3, -4)
        self.assertEqual(rotations.multiply((1, 0, 0, 0), quaternion), quaternion)
        self.assertEqual(rotations.multiply(quaternion, (1, 0, 0, 0)), quaternion)
        conjugated = rotations.conjugate(quaternion)
        self.assertEqual(conjugated, (1, 2, -3, 4))
        self.assertEqual(rotations.conjugate(conjugated), quaternion)
        expected = (sum(value * value for value in quaternion), 0, 0, 0)
        self.assertEqual(rotations.multiply(quaternion, conjugated), expected)

    def test_product_is_associative_and_unit_inverse_is_conjugate(self):
        q1 = rotations.normalize((1, 2, -3, 4))
        q2 = rotations.normalize((-2, 3, 1, 5))
        q3 = rotations.normalize((7, -1, 2, 0))
        left = rotations.multiply(rotations.multiply(q1, q2), q3)
        right = rotations.multiply(q1, rotations.multiply(q2, q3))
        inverse = rotations.multiply(q1, rotations.conjugate(q1))
        for actual, reference in zip(left, right):
            self.assertAlmostEqual(actual, reference, places=14)
        for actual, reference in zip(inverse, (1, 0, 0, 0)):
            self.assertAlmostEqual(actual, reference, places=14)

    def test_bad_operands_and_unrepresentable_raw_products_are_rejected(self):
        for invalid in ((0, 0, 0, 0), (), (1, 2, 3), (1, 2, 3, 4, 5), None, "1234",
                        (1, math.nan, 0, 0), (1, 0, math.inf, 0), (-math.inf, 0, 0, 0)):
            with self.assertRaises(ValueError):
                rotations.conjugate(invalid)
            for left, right in ((invalid, (1, 0, 0, 0)), ((1, 0, 0, 0), invalid)):
                with self.assertRaises(ValueError):
                    rotations.multiply(left, right)
        with self.assertRaises(ValueError):
            rotations.multiply((1e308, 0, 0, 0), (1e308, 0, 0, 0))


if __name__ == "__main__":
    unittest.main()
