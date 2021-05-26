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


class ActiveVectorRotationTests(unittest.TestCase):
    def assertVectorClose(self, actual, expected):
        self.assertIsInstance(actual, tuple)
        self.assertEqual(len(actual), 3)
        for value, reference in zip(actual, expected):
            self.assertAlmostEqual(value, reference, places=13)

    def test_positive_yaw_maps_body_forward_to_ned_east(self):
        half = math.sqrt(0.5)
        self.assertVectorClose(rotations.rotate((half, 0, 0, half), (1, 0, 0)), (0, 1, 0))
        self.assertVectorClose(rotations.rotate((half, half, 0, 0), (0, 1, 0)), (0, 0, 1))
        self.assertVectorClose(rotations.rotate((half, 0, half, 0), (1, 0, 0)), (0, 0, -1))

    def test_identity_zero_and_scaled_sign_equivalent_attitudes(self):
        vector = (2, -3, 5)
        self.assertVectorClose(rotations.rotate((7, 0, 0, 0), vector), vector)
        self.assertEqual(rotations.rotate((1, 2, 3, 4), (0, 0, 0)), (0, 0, 0))
        baseline = rotations.rotate((1, 2, -3, 4), vector)
        for scale in (-7, 1e100, 1e-100):
            actual = rotations.rotate(tuple(scale * value for value in (1, 2, -3, 4)), vector)
            self.assertVectorClose(actual, baseline)

    def test_rotation_agrees_with_sandwich_and_composition(self):
        q1, q2 = rotations.normalize((1, 2, 3, 4)), rotations.normalize((2, -3, 4, 1))
        vector = (0.3, -2, 5)
        sandwich = rotations.multiply(rotations.multiply(q1, (0,) + vector), rotations.conjugate(q1))
        actual = rotations.rotate(q1, vector)
        self.assertVectorClose(actual, sandwich[1:])
        self.assertAlmostEqual(math.hypot(*actual), math.hypot(*vector), places=13)
        self.assertVectorClose(rotations.rotate(rotations.conjugate(q1), actual), vector)
        self.assertVectorClose(rotations.rotate(rotations.multiply(q1, q2), vector),
                               rotations.rotate(q1, rotations.rotate(q2, vector)))

    def test_vector_and_attitude_validation(self):
        for invalid in ((), (1, 2), (1, 2, 3, 4), None, "123",
                        (math.nan, 0, 0), (0, math.inf, 0), (0, 0, -math.inf)):
            with self.assertRaises(ValueError):
                rotations.rotate((1, 0, 0, 0), invalid)
        for invalid in ((0, 0, 0, 0), (1, 2, 3), (1, math.nan, 0, 0)):
            with self.assertRaises(ValueError):
                rotations.rotate(invalid, (1, 2, 3))


class EulerToQuaternionTests(unittest.TestCase):
    def test_zero_and_single_axis_half_angles(self):
        self.assertEqual(rotations.from_euler(0, 0, 0), (1, 0, 0, 0))
        half = math.sqrt(0.5)
        for angles, expected in (((math.pi / 2, 0, 0), (half, half, 0, 0)),
                                 ((0, math.pi / 2, 0), (half, 0, half, 0)),
                                 ((0, 0, math.pi / 2), (half, 0, 0, half))):
            for actual, reference in zip(rotations.from_euler(*angles), expected):
                self.assertAlmostEqual(actual, reference, places=14)

    def test_zyx_sequence_is_yaw_times_pitch_times_roll(self):
        roll, pitch, yaw = 0.3, -0.6, 1.2
        qx = (math.cos(roll / 2), math.sin(roll / 2), 0, 0)
        qy = (math.cos(pitch / 2), 0, math.sin(pitch / 2), 0)
        qz = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
        expected = rotations.multiply(rotations.multiply(qz, qy), qx)
        actual = rotations.from_euler(roll, pitch, yaw)
        for value, reference in zip(actual, expected):
            self.assertAlmostEqual(value, reference, places=14)
        self.assertAlmostEqual(math.hypot(*actual), 1, places=14)

    def test_radians_and_body_to_ned_heading(self):
        actual = rotations.rotate(rotations.from_euler(0, 0, math.pi / 2), (1, 0, 0))
        for value, reference in zip(actual, (0, 1, 0)):
            self.assertAlmostEqual(value, reference, places=14)
        self.assertAlmostEqual(math.hypot(*rotations.from_euler(1e100, -1e100, 1e100)), 1, places=14)

    def test_rejects_nonfinite_or_nonnumeric_euler_angles(self):
        for index in range(3):
            for invalid in (math.nan, math.inf, -math.inf, None, "bad"):
                angles = [0.0] * 3
                angles[index] = invalid
                with self.assertRaises(ValueError):
                    rotations.from_euler(*angles)


class QuaternionToEulerTests(unittest.TestCase):
    def test_identity_and_regular_radian_round_trips(self):
        self.assertEqual(rotations.to_euler((1, 0, 0, 0)), (0, 0, 0))
        for angles in ((0.3, -0.6, 1.2), (-2.4, 1.4, -2.7), (2.8, -1.4, 3.0)):
            actual = rotations.to_euler(rotations.from_euler(*angles))
            self.assertIsInstance(actual, tuple)
            for value, reference in zip(actual, angles):
                self.assertAlmostEqual(value, reference, places=13)

    def test_canonical_ranges_reconstruct_noncanonical_inputs(self):
        for angles in ((4.0, 2.4, -5.0), (-6.0, -2.0, 7.0), (0.3, math.pi, -0.7)):
            original = rotations.from_euler(*angles)
            roll, pitch, yaw = rotations.to_euler(original)
            self.assertLessEqual(abs(roll), math.pi)
            self.assertLessEqual(abs(pitch), math.pi / 2)
            self.assertLessEqual(abs(yaw), math.pi)
            recovered = rotations.from_euler(roll, pitch, yaw)
            self.assertAlmostEqual(abs(sum(a * b for a, b in zip(original, recovered))), 1, places=14)

    def test_both_gimbal_locks_choose_zero_roll_without_changing_rotation(self):
        for pitch in (-math.pi / 2, math.pi / 2):
            for roll, yaw in ((0.3, 1.2), (-2.0, -0.7), (0, 0)):
                original = rotations.from_euler(roll, pitch, yaw)
                actual = rotations.to_euler(original)
                self.assertEqual(actual[0], 0.0)
                self.assertAlmostEqual(actual[1], pitch, places=14)
                recovered = rotations.from_euler(*actual)
                for vector in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    for value, reference in zip(rotations.rotate(recovered, vector),
                                                rotations.rotate(original, vector)):
                        self.assertAlmostEqual(value, reference, places=13)

    def test_normalizes_and_is_sign_invariant(self):
        original = rotations.from_euler(0.3, 0.7, -1.2)
        expected = rotations.to_euler(original)
        for scale in (-7.0, 1e300, 1e-300):
            actual = rotations.to_euler(tuple(scale * value for value in original))
            for value, reference in zip(actual, expected):
                self.assertAlmostEqual(value, reference, places=13)

    def test_rejects_bad_quaternions(self):
        for invalid in ((0, 0, 0, 0), (), (1, 2, 3), (1, 2, 3, 4, 5), None, "1234",
                        (math.nan, 0, 0, 0), (1, math.inf, 0, 0), (1, 0, -math.inf, 0)):
            with self.assertRaises(ValueError):
                rotations.to_euler(invalid)


class RotationVectorTests(unittest.TestCase):
    def test_zero_and_quarter_turn_axes(self):
        self.assertEqual(rotations.from_rotation_vector((0, 0, 0)), (1, 0, 0, 0))
        for index in range(3):
            vector = [0.0, 0.0, 0.0]
            vector[index] = math.pi / 2
            angles = [0.0, 0.0, 0.0]
            angles[index] = math.pi / 2
            for actual, expected in zip(rotations.from_rotation_vector(vector), rotations.from_euler(*angles)):
                self.assertAlmostEqual(actual, expected, places=14)

    def test_tiny_increment_retains_first_order_vector_part(self):
        vector = (1e-14, -2e-14, 3e-14)
        actual = rotations.from_rotation_vector(vector)
        self.assertEqual(actual[0], 1.0)
        for value, reference in zip(actual[1:], vector):
            self.assertAlmostEqual(value / reference, 0.5, places=14)
        self.assertAlmostEqual(math.hypot(*actual), 1, places=15)

    def test_oblique_axis_matches_rodrigues_formula(self):
        axis = tuple(value / math.sqrt(29) for value in (2, -3, 4))
        angle, vector = 1.1, (0.3, -1.2, 2.7)
        rotation_vector = tuple(angle * value for value in axis)
        quaternion = rotations.from_rotation_vector(rotation_vector)
        cross = (axis[1]*vector[2] - axis[2]*vector[1],
                 axis[2]*vector[0] - axis[0]*vector[2],
                 axis[0]*vector[1] - axis[1]*vector[0])
        dot = sum(a * b for a, b in zip(axis, vector))
        expected = tuple(value * math.cos(angle) + turn * math.sin(angle)
                         + direction * dot * (1 - math.cos(angle))
                         for value, turn, direction in zip(vector, cross, axis))
        for actual, reference in zip(rotations.rotate(quaternion, vector), expected):
            self.assertAlmostEqual(actual, reference, places=13)

    def test_negation_inverts_and_collinear_steps_compose(self):
        vector = (0.2, -0.3, 0.4)
        positive = rotations.from_rotation_vector(vector)
        negative = rotations.from_rotation_vector(tuple(-value for value in vector))
        doubled = rotations.from_rotation_vector(tuple(2 * value for value in vector))
        for actual, reference in zip(negative, rotations.conjugate(positive)):
            self.assertAlmostEqual(actual, reference, places=14)
        for actual, reference in zip(rotations.multiply(positive, positive), doubled):
            self.assertAlmostEqual(actual, reference, places=14)

    def test_rejects_dimensions_nonfinite_and_unrepresentable_angle(self):
        for invalid in ((), (1, 2), (1, 2, 3, 4), None, "123", (math.nan, 0, 0),
                        (0, math.inf, 0), (0, 0, -math.inf), (1.7e308,) * 3):
            with self.assertRaises(ValueError):
                rotations.from_rotation_vector(invalid)
        self.assertAlmostEqual(math.hypot(*rotations.from_rotation_vector((1e308, 0, 0))), 1, places=14)


class SlerpTests(unittest.TestCase):
    def assertSameRotation(self, actual, expected, places=13):
        self.assertAlmostEqual(math.hypot(*actual), 1, places=14)
        self.assertAlmostEqual(abs(sum(a * b for a, b in zip(actual, expected))), 1, places=places)

    def test_endpoints_and_constant_angular_speed(self):
        start, end = (2, 0, 0, 0), rotations.from_euler(0, 0, 2.4)
        for fraction in (0.0, 0.2, 0.5, 0.8, 1.0):
            actual = rotations.slerp(start, end, fraction)
            self.assertIsInstance(actual, tuple)
            self.assertSameRotation(actual, rotations.from_euler(0, 0, 2.4 * fraction))
            self.assertAlmostEqual(rotations.to_euler(actual)[2], 2.4 * fraction, places=13)

    def test_antipodal_quaternions_and_near_identical_attitudes(self):
        start = rotations.from_euler(0.3, -0.6, 1.2)
        opposite = tuple(-value for value in start)
        for fraction in (0, 0.3, 0.5, 1):
            self.assertSameRotation(rotations.slerp(start, opposite, fraction), start)
        tiny = rotations.from_euler(0, 0, 1e-10)
        midpoint = rotations.slerp((1, 0, 0, 0), tiny, 0.5)
        self.assertAlmostEqual(rotations.to_euler(midpoint)[2], 0.5e-10, places=22)

    def test_shortest_heading_arc_crosses_the_date_line(self):
        start = rotations.from_euler(0, 0, math.radians(170))
        end = rotations.from_euler(0, 0, math.radians(-170))
        for fraction in (0, 0.25, 0.5, 0.75, 1):
            expected = rotations.from_euler(0, 0, math.radians(170 + 20 * fraction))
            self.assertSameRotation(rotations.slerp(start, end, fraction), expected)
            self.assertSameRotation(rotations.slerp(start, tuple(-value for value in end), fraction), expected)

    def test_general_unit_interpolation_and_reversal_symmetry(self):
        start, end = rotations.from_euler(0.5, -0.6, 0.2), rotations.from_euler(-0.2, 0.4, -1.2)
        for fraction in (0.1, 0.4, 0.9):
            forward = rotations.slerp(start, end, fraction)
            backward = rotations.slerp(end, start, 1 - fraction)
            self.assertSameRotation(forward, backward)

    def test_rejects_invalid_fractions_and_either_bad_endpoint(self):
        identity = (1, 0, 0, 0)
        for fraction in (-0.01, 1.01, math.nan, math.inf, -math.inf, None, "bad"):
            with self.assertRaises(ValueError):
                rotations.slerp(identity, identity, fraction)
        for invalid in ((0, 0, 0, 0), (), (1, 2, 3), (1, 2, 3, 4, 5), None, "1234",
                        (1, math.nan, 0, 0), (1, 0, math.inf, 0), (1, 0, 0, -math.inf)):
            for start, end in ((invalid, identity), (identity, invalid)):
                with self.assertRaises(ValueError):
                    rotations.slerp(start, end, 0.5)


if __name__ == "__main__":
    unittest.main()
