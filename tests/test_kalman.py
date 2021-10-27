"""Analytical prediction/correction and numerical safety regressions."""
import copy
import math
import unittest
from inertialnav.kalman import LinearKalman
from inertialnav import linalg as la

class KalmanTests(unittest.TestCase):
    def test_prediction_and_defensive_copies(self):
        x, p = [1, 2], [[3, 0], [0, 4]]
        kf = LinearKalman(x, p)
        x[0] = 99
        p[0][0] = 99
        kf.predict([[1, 0.5], [0, 1]], [[0.1, 0], [0, 0.2]])
        self.assertEqual(kf.x, [2, 2])
        self.assertEqual(kf.P, [[4.1, 2], [2, 4.2]])
        snapshot = copy.deepcopy((kf.x, kf.P))
        for f, q in (([[1]], [[0]]), ([[1, 0], [0, 1]], [[-1, 0], [0, 1]])):
            with self.assertRaises(ValueError):
                kf.predict(f, q)
            self.assertEqual((kf.x, kf.P), snapshot)
        with self.assertRaises(ValueError):
            LinearKalman([0], [[-1]])

class CorrectionTests(unittest.TestCase):
    def test_scalar_joseph_update(self):
        kf = LinearKalman([0], [[1]])
        report = kf.update([2], [[1]], [[1]])
        self.assertTrue(report.accepted)
        self.assertEqual(report.innovation, (2,))
        self.assertAlmostEqual(report.nis, 2)
        self.assertAlmostEqual(kf.x[0], 1)
        self.assertAlmostEqual(kf.P[0][0], 0.5)

    def test_correlated_measurement_and_repeated_updates(self):
        kf = LinearKalman([0, 0], [[2, 0.2], [0.2, 1]])
        for unused in range(100):
            kf.update([1, -1], la.identity(2), [[0.2, 0.05], [0.05, 0.3]])
            la.solve_spd(kf.P, [1, 0])
            self.assertEqual(kf.P[0][1], kf.P[1][0])
        self.assertAlmostEqual(kf.x[0], 1, delta=0.01)
        snapshot = copy.deepcopy((kf.x, kf.P))
        for z, h, r in (([float('nan')], [[1, 0]], [[1]]),
                        ([1], [[1]], [[1]]), ([1], [[1, 0]], [[-1]]),
                        ([1], [[1, 0]], [[0]])):
            with self.assertRaises(ValueError):
                kf.update(z, h, r)
            self.assertEqual((kf.x, kf.P), snapshot)


class InnovationGateTests(unittest.TestCase):
    def test_gate_reports_rejection_without_mutating_state(self):
        kf = LinearKalman([0], [[1]])
        report = kf.update([100], [[1]], [[1]], gate=9)
        self.assertFalse(report.accepted)
        self.assertEqual(kf.x, [0])
        self.assertEqual(kf.P, [[1]])
        self.assertAlmostEqual(report.nis, 5000)
        self.assertTrue(kf.update([1], [[1]], [[1]], gate=9).accepted)
        for gate in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                kf.update([1], [[1]], [[1]], gate=gate)


class ControlTests(unittest.TestCase):
    def test_control_prediction_and_argument_pairing(self):
        kf = LinearKalman([0, 1], la.identity(2))
        kf.predict([[1, 1], [0, 1]], [[0, 0], [0, 0]], control=[2], control_matrix=[[0.5], [1]])
        self.assertEqual(kf.x, [2, 3])
        before = copy.deepcopy((kf.x, kf.P))
        for kwargs in ({'control': [1]}, {'control_matrix': [[1], [1]]},
                       {'control': [1, 2], 'control_matrix': [[1], [1]]}):
            with self.assertRaises(ValueError):
                kf.predict(la.identity(2), [[0, 0], [0, 0]], **kwargs)
            self.assertEqual((kf.x, kf.P), before)


class MotionModelTests(unittest.TestCase):
    def test_exact_white_acceleration_discretization(self):
        from inertialnav.kalman import constant_velocity_model
        f, q, b = constant_velocity_model(2, 3, axes=1)
        self.assertEqual(f, [[1, 2], [0, 1]])
        self.assertEqual(q, [[8, 6], [6, 6]])
        self.assertEqual(b, [[2], [2]])
        f, q, b = constant_velocity_model(0.1, 0, axes=3)
        self.assertEqual(len(f), 6)
        self.assertEqual(f[0][3], 0.1)
        self.assertEqual(q, [[0]*6 for unused in range(6)])
        for dt, noise in ((0, 1), (-1, 1), (1, -1), (float('inf'), 1)):
            with self.assertRaises(ValueError):
                constant_velocity_model(dt, noise)


# NEXT_INCREMENT
