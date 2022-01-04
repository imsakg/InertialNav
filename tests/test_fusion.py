"""Aided local navigation regressions with explicit frame/noise contracts."""
import copy
import math
import unittest
from inertialnav.fusion import NavigationFilter
from inertialnav.imu import Delta, GRAVITY
from inertialnav import linalg as la

class PredictionTests(unittest.TestCase):
    def test_specific_force_drives_position_velocity_and_covariance(self):
        nav = NavigationFilter(acceleration_psd=0.2)
        nav.predict(Delta(0.5, [0]*3, [0.5, 0, -GRAVITY*0.5]))
        self.assertEqual(nav.state.position, (0.125, 0, 0))
        self.assertEqual(nav.state.velocity, (0.5, 0, 0))
        self.assertEqual(nav.time, 0.5)
        self.assertGreater(nav.filter.P[0][0], 1)
        la.solve_spd(nav.filter.P, [0]*6)
        with self.assertRaises(ValueError):
            NavigationFilter(acceleration_psd=-1)

# NEXT_INCREMENT
