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


class TrajectorySummaryTests(unittest.TestCase):
    def test_axis_bias_rmse_max_and_vector_rms(self):
        result = analysis.error_summary([[1, -2, 2], [3, 2, -2]], [[0, 0, 0], [0, 0, 0]])
        self.assertEqual(result.sample_count, 2)
        self.assertEqual(tuple(axis.bias for axis in result.axes), (2, 0, 0))
        self.assertAlmostEqual(result.axes[0].rmse, math.sqrt(5))
        self.assertEqual(tuple(axis.max_absolute for axis in result.axes), (3, 2, 2))
        self.assertAlmostEqual(result.vector_rmse, math.sqrt(13))

    def test_arbitrary_dimension_and_large_finite_values(self):
        result = analysis.error_summary(iter([[1e200], [1e200]]), iter([[0], [0]]))
        self.assertEqual(len(result.axes), 1)
        self.assertAlmostEqual(result.axes[0].bias / 1e200, 1)
        self.assertAlmostEqual(result.vector_rmse / 1e200, 1)

    def test_rejects_empty_ragged_mismatched_and_nonfinite_trajectories(self):
        cases = [([], []), ([[]], [[]]), ([[1]], [[1], [2]]),
                 ([[1, 2], [3]], [[0, 0], [0, 0]]), ([[1]], [[1, 2]]),
                 ([[float("inf")]], [[0]]), ([[1e308]], [[-1e308]])]
        for estimated, truth in cases:
            with self.subTest(estimated=estimated), self.assertRaises(ValueError):
                analysis.error_summary(estimated, truth)


class RunningMomentsTests(unittest.TestCase):
    def test_large_offset_variance_and_online_counts(self):
        moments = analysis.RunningMoments()
        for index in range(1, 6):
            self.assertIs(moments.update(1e12 + index), moments)
            self.assertEqual(moments.count, index)
        self.assertEqual(moments.mean, 1e12 + 3)
        self.assertAlmostEqual(moments.variance(), 2)
        self.assertAlmostEqual(moments.variance(sample=True), 2.5)
        self.assertAlmostEqual(moments.standard_deviation(), math.sqrt(2))

    def test_empty_singleton_and_finite_guards(self):
        moments = analysis.RunningMoments()
        with self.assertRaises(ValueError):
            _ = moments.mean
        with self.assertRaises(ValueError):
            moments.variance()
        moments.update(7)
        self.assertEqual(moments.variance(), 0)
        with self.assertRaises(ValueError):
            moments.variance(sample=True)
        for value in [float("nan"), float("inf"), -float("inf")]:
            with self.assertRaises(ValueError):
                moments.update(value)
            self.assertEqual((moments.count, moments.mean, moments.variance()), (1, 7, 0))

    def test_overflowing_update_leaves_the_accumulator_unchanged(self):
        moments = analysis.RunningMoments()
        moments.update(1e308)
        with self.assertRaises(ValueError):
            moments.update(-1e308)
        self.assertEqual((moments.count, moments.mean, moments.variance()), (1, 1e308, 0))


class InterpolationTests(unittest.TestCase):
    def test_exact_endpoints_irregular_spacing_and_singleton(self):
        self.assertEqual(analysis.interpolate([0, 1, 4], [0, 2, 8], 2.5), 5)
        self.assertEqual(analysis.interpolate([0, 1], [2, 4], 0), 2)
        self.assertEqual(analysis.interpolate([0, 1], [2, 4], 1), 4)
        self.assertEqual(analysis.interpolate([1], [7], 1), 7)
        self.assertEqual(analysis.interpolate([0, 1], [-1e308, 1e308], 0.5), 0)

    def test_no_extrapolation_sorting_or_silent_truncation(self):
        cases = [([], [], 0), ([0, 0], [1, 2], 0), ([1, 0], [1, 2], 0),
                 ([0, 1], [1], 0), ([0, 1], [0, 1], -1), ([0, 1], [0, 1], 2),
                 ([0], [1], 1), ([0, float("inf")], [0, 1], 0),
                 ([0, 1], [float("nan"), 1], 0), ([0, 1], [0, 1], float("nan"))]
        for times, values, query in cases:
            with self.subTest(times=times, query=query), self.assertRaises(ValueError):
                analysis.interpolate(times, values, query)

    def test_max_gap_rejects_only_interpolated_large_gaps(self):
        with self.assertRaises(ValueError):
            analysis.interpolate([0, 10], [0, 20], 5, max_gap=2)
        self.assertEqual(analysis.interpolate([0, 10], [0, 20], 10, max_gap=2), 20)
        self.assertEqual(analysis.interpolate([0, 2], [0, 4], 1, max_gap=2), 2)
        for gap in [0, -1, float("nan"), float("inf")]:
            with self.subTest(gap=gap), self.assertRaises(ValueError):
                analysis.interpolate([0, 1], [0, 1], 0.5, max_gap=gap)


class TimelineAlignmentTests(unittest.TestCase):
    def test_aligns_vector_reference_at_estimate_times_with_explicit_overlap(self):
        aligned = analysis.align_trajectories(
            [-1, 0, 1, 2, 3], [[-1, -2], [0, 0], [1, 2], [2, 4], [3, 6]],
            [0, 2], [[0, 0], [2, 4]])
        self.assertEqual(aligned.times, (0, 1, 2))
        self.assertEqual(aligned.estimated, aligned.reference)
        self.assertEqual(aligned.dropped_estimates, 2)
        self.assertEqual(analysis.error_summary(aligned.estimated, aligned.reference).vector_rmse, 0)

    def test_single_exact_overlap_and_gap_rejection(self):
        aligned = analysis.align_trajectories([1], [[2]], [1], [[3]])
        self.assertEqual(aligned.reference, ((3,),))
        with self.assertRaisesRegex(ValueError, "max_gap"):
            analysis.align_trajectories([1], [[2]], [0, 10], [[0], [10]], max_gap=2)

    def test_rejects_invalid_timelines_dimensions_and_no_overlap(self):
        cases = [([], [], [0], [[0]]), ([1, 1], [[0], [1]], [0, 2], [[0], [2]]),
                 ([0], [[0]], [1], [[1]]), ([0, 1], [[0]], [0], [[0]]),
                 ([0], [[0, 1]], [0], [[0]]), ([0], [[0]], [0, 1], [[0], [1, 2]]),
                 ([0], [[float("nan")]], [0], [[0]])]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                analysis.align_trajectories(*args)
