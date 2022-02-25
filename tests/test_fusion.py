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

class PositionAidTests(unittest.TestCase):
    def test_position_update_and_outlier_rejection(self):
        nav = NavigationFilter()
        result = nav.update_position([1, 0, 0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertTrue(result.accepted)
        self.assertAlmostEqual(nav.state.position[0], 0.5)
        before = nav.state
        self.assertFalse(nav.update_position([100, 0, 0], la.identity(3)).accepted)
        self.assertEqual(nav.state, before)
        with self.assertRaises(ValueError):
            nav.update_position([1, 2], la.identity(3))


class VelocityAidTests(unittest.TestCase):
    def test_velocity_aiding_corrects_correlated_position(self):
        nav = NavigationFilter()
        nav.predict(Delta(0.5, [0]*3, [0, 0, -GRAVITY*0.5]))
        report = nav.update_velocity([1, 0, 0], [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
        self.assertTrue(report.accepted)
        self.assertGreater(nav.state.velocity[0], 0.9)
        self.assertGreater(nav.state.position[0], 0.4)
        self.assertFalse(nav.update_velocity([500, 0, 0], la.identity(3)).accepted)


class VerticalAndRestAidTests(unittest.TestCase):
    def test_altitude_is_up_and_stationary_requires_confirmation(self):
        nav = NavigationFilter()
        self.assertTrue(nav.update_altitude(2, 1).accepted)
        self.assertAlmostEqual(nav.state.position[2], -1)
        nav.update_velocity([0.2, 0, 0], la.identity(3))
        before = nav.state
        self.assertIsNone(nav.update_zero_velocity(False))
        self.assertEqual(nav.state, before)
        self.assertTrue(nav.update_zero_velocity(True).accepted)
        self.assertLess(abs(nav.state.velocity[0]), 0.01)
        with self.assertRaises(ValueError):
            nav.update_zero_velocity('probably')
        with self.assertRaises(ValueError):
            nav.update_altitude(1, 0)


class CompassTests(unittest.TestCase):
    def test_heading_wrap_and_preserved_roll_pitch(self):
        from inertialnav.imu import StrapdownState
        from inertialnav.rotations import from_euler, to_euler
        nav = NavigationFilter(state=StrapdownState(attitude=from_euler(0.3, -0.2, math.radians(179))))
        report = nav.update_heading(math.radians(-179), math.radians(2)**2)
        self.assertTrue(report.accepted)
        self.assertAlmostEqual(report.innovation[0], math.radians(2))
        roll, pitch, yaw = to_euler(nav.attitude)
        self.assertAlmostEqual(roll, 0.3)
        self.assertAlmostEqual(pitch, -0.2)
        self.assertAlmostEqual(yaw, math.radians(-179), delta=0.001)
        before = (nav.attitude, nav.heading_variance)
        self.assertFalse(nav.update_heading(0, 0.001).accepted)
        self.assertEqual((nav.attitude, nav.heading_variance), before)
        vertical = NavigationFilter(state=StrapdownState(attitude=from_euler(0, math.pi/2, 0)))
        with self.assertRaises(ValueError):
            vertical.update_heading(0, 0.1)


class TimingAndHealthTests(unittest.TestCase):
    def test_rejected_or_stale_data_never_refreshes_aiding(self):
        nav = NavigationFilter(max_step=0.2)
        self.assertFalse(nav.aiding_fresh('position', 1))
        nav.update_position([0]*3, la.identity(3), timestamp=0)
        self.assertTrue(nav.aiding_fresh('position', 0.1))
        nav.predict_to(0.1, Delta(0.1, [0]*3, [0, 0, -GRAVITY*0.1]))
        before = (nav.state, copy.deepcopy(nav.filter.P), nav.time)
        for stamp in (0, 0.2, float('nan')):
            with self.assertRaises(ValueError):
                nav.update_position([1]*3, la.identity(3), timestamp=stamp)
            self.assertEqual((nav.state, nav.filter.P, nav.time), before)
        self.assertFalse(nav.update_position([999]*3, la.identity(3)).accepted)
        self.assertFalse(nav.aiding_fresh('position', 0.05))
        self.assertEqual(nav.aiding_rejections['position'], 1)
        nav.update_zero_velocity(True)
        self.assertTrue(nav.aiding_fresh('zero_velocity', 0.1))
        self.assertFalse(nav.aiding_fresh('velocity', 0.1))
        before = (nav.state, copy.deepcopy(nav.filter.P), nav.time)
        for stamp, delta in ((0.3, Delta(0.1, [0]*3, [0]*3)), (0.4, Delta(0.3, [0]*3, [0]*3))):
            with self.assertRaises(ValueError):
                nav.predict_to(stamp, delta)
            self.assertEqual((nav.state, nav.filter.P, nav.time), before)


class ClockPrecisionRegressionTests(unittest.TestCase):
    def test_long_100_hz_run_retains_observed_clock(self):
        nav = NavigationFilter()
        delta = Delta(0.01, [0]*3, [0, 0, -GRAVITY*0.01])
        for index in range(1, 10001):
            timestamp = index*0.01
            nav.predict_to(timestamp, delta)
            self.assertEqual(nav.time, timestamp)
        self.assertEqual(nav.time, 100)

    def test_epoch_clock_precision_and_invalid_heading_noise_are_atomic(self):
        nav = NavigationFilter(time=1700000000.0)
        delta = Delta(0.01, [0]*3, [0, 0, -GRAVITY*0.01])
        nav.predict_to(1700000000.01, delta)
        self.assertEqual(nav.time, 1700000000.01)
        before = (nav.state, copy.deepcopy(nav.filter.P), nav.time)
        nav.gyro_noise_psd = float('inf')
        with self.assertRaises(ValueError):
            nav.predict(delta)
        self.assertEqual((nav.state, nav.filter.P, nav.time), before)
