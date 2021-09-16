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

# NEXT_INCREMENT
