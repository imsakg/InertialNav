"""Known-rate IMU processing tests; acceleration is body specific force."""
import math
import unittest
from inertialnav.imu import ImuSample, sample_interval

class SampleTests(unittest.TestCase):
    def test_sample_is_finite_immutable_and_time_ordered(self):
        gyro = [1, 2, 3]
        a = ImuSample(1, gyro, [0, 0, -9.80665])
        gyro[0] = 99
        self.assertEqual(a.gyro, (1, 2, 3))
        self.assertEqual(sample_interval(a, ImuSample(1.01, [0]*3, [0]*3)), 1.01-1)
        for t, g in ((float('nan'), [0]*3), (-1, [0]*3), (1, [0, 0]), (1, [0, 0, float('inf')])):
            with self.assertRaises(ValueError):
                ImuSample(t, g, [0]*3)
        for t in (0.5, 1, 2):
            with self.assertRaises(ValueError):
                sample_interval(a, ImuSample(t, [0]*3, [0]*3), max_gap=0.1)

class CalibrationTests(unittest.TestCase):
    def test_bias_scale_correction_retains_time(self):
        from inertialnav.imu import Calibration
        calibration = Calibration(gyro_bias=(1, 0, -1), accel_bias=(0, 1, 0),
                                  gyro_scale=(2, 1, 1), accel_scale=(1, 2, 1))
        raw = ImuSample(2, [2, 3, 4], [4, 5, 6])
        calibrated = calibration.correct(raw)
        self.assertEqual(calibrated, ImuSample(2, [2, 3, 5], [4, 8, 6]))
        self.assertEqual(raw.gyro, (2, 3, 4))
        for kwargs in ({'accel_scale': (1, 0, 1)}, {'gyro_bias': (0, 0)},
                       {'gyro_scale': (1, float('nan'), 1)}):
            with self.assertRaises(ValueError):
                Calibration(**kwargs)


class IntegrationTests(unittest.TestCase):
    def test_trapezoidal_rate_integrals(self):
        from inertialnav.imu import integrate_pair, Delta
        a = ImuSample(0, [1, 0, 0], [0, 2, 0])
        b = ImuSample(0.5, [3, 0, 0], [0, 4, 0])
        delta = integrate_pair(a, b)
        self.assertEqual(delta.angle, (1, 0, 0))
        self.assertEqual(delta.velocity, (0, 1.5, 0))
        self.assertEqual(delta.dt, 0.5)
        self.assertEqual(delta.velocity_frame, 'midpoint')
        for args in ((0, [0]*3, [0]*3), (1, [0]*2, [0]*3)):
            with self.assertRaises(ValueError):
                Delta(*args)
        with self.assertRaises(ValueError):
            Delta(1, [0]*3, [0]*3, velocity_frame='earth')


class ConingTests(unittest.TestCase):
    def test_two_sample_cross_terms_and_frame_contract(self):
        from inertialnav.imu import Delta, combine_increments
        a = Delta(0.01, [0.1, 0, 0], [0, 0, 1])
        b = Delta(0.01, [0, 0.2, 0], [0, 0, 1])
        result = combine_increments(a, b)
        self.assertAlmostEqual(result.angle[2], 2/3*0.02)
        self.assertAlmostEqual(result.velocity[0], 0.2-2/3*0.2)
        self.assertAlmostEqual(result.velocity[1], -0.1-2/3*0.1)
        self.assertEqual(result.velocity_frame, 'start')
        self.assertEqual(result.dt, 0.02)
        with self.assertRaises(ValueError):
            combine_increments(a, Delta(0.02, [0]*3, [0]*3))
        with self.assertRaises(ValueError):
            combine_increments(result, result)


class LowPassTests(unittest.TestCase):
    def test_exact_first_order_gain_and_reset(self):
        from inertialnav.imu import LowPass
        filt = LowPass(1)
        self.assertEqual(filt.update([0, 0, 0], 0.1), (0, 0, 0))
        value = filt.update([1, 2, 3], 1)
        for i in range(3):
            self.assertAlmostEqual(value[i], (i+1)*(1-math.exp(-2*math.pi)))
        before = filt.value
        for bad, dt in (([1, 2], 1), ([0]*3, 0), ([float('nan')]*3, 1)):
            with self.assertRaises(ValueError):
                filt.update(bad, dt)
            self.assertEqual(filt.value, before)
        filt.reset()
        self.assertIsNone(filt.value)
        self.assertEqual(filt.update([1, 2, 3], 0.1), (1, 2, 3))


class StationaryTests(unittest.TestCase):
    def test_full_window_required_and_motion_resets_decision(self):
        from inertialnav.imu import StationaryDetector
        detector = StationaryDetector(window=4)
        for i in range(3):
            self.assertFalse(detector.push(ImuSample(i, [0]*3, [0, 0, -9.80665])))
        self.assertTrue(detector.push(ImuSample(3, [0]*3, [0, 0, -9.80665])))
        self.assertFalse(detector.push(ImuSample(4, [0, 0, 0.1], [0, 0, -9.80665])))
        with self.assertRaises(ValueError):
            detector.push(ImuSample(4, [0]*3, [0, 0, -9.80665]))
        self.assertFalse(detector.push(ImuSample(5, [0]*3, [1, 0, -9.80665])))
        for window in (0, 1, True, 2.5):
            with self.assertRaises(ValueError):
                StationaryDetector(window=window)


class AttitudeIntegrationTests(unittest.TestCase):
    def test_body_rate_attitude_increments_keep_unit_norm(self):
        from inertialnav.imu import integrate_attitude
        from inertialnav.rotations import to_euler
        q = (1, 0, 0, 0)
        for unused in range(1000):
            q = integrate_attitude(q, [0, 0, math.pi/2000])
        self.assertAlmostEqual(math.hypot(*q), 1, places=14)
        self.assertAlmostEqual(to_euler(q)[2], math.pi/2, places=12)
        with self.assertRaises(ValueError):
            integrate_attitude([0, 0, 0, 0], [0]*3)


# NEXT_INCREMENT
