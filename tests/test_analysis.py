"""Analytical regressions for dependency-free offline diagnostics."""

import math
import unittest

from inertialnav import analysis


class RmsErrorTests(unittest.TestCase):
    def test_known_scalar_rms_and_paired_rmse(self):
        self.assertEqual(analysis.rms([3, -3]), 3)
        self.assertAlmostEqual(analysis.rmse([2, 6], [1, 3]), math.sqrt(5))
        self.assertEqual(analysis.rmse(iter([2, 4]), iter([2, 4])), 0)

    def test_scaled_rms_does_not_overflow_intermediate_squares(self):
        self.assertAlmostEqual(analysis.rms([1e200, -1e200]) / 1e200, 1)
        self.assertEqual(analysis.rms([0, 0]), 0)

    def test_empty_mismatch_nonfinite_and_overflowed_errors_are_rejected(self):
        for values in [[], [float("nan")], [float("inf")]]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                analysis.rms(values)
        for estimates, truth in [([], []), ([1], [1, 2]), ([1e308], [-1e308]),
                                 ([1], [float("nan")])]:
            with self.subTest(estimates=estimates), self.assertRaises(ValueError):
                analysis.rmse(estimates, truth)
