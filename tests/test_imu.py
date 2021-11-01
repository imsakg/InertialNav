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

# NEXT_INCREMENT
