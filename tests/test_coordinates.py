"""Known-answer and convention tests for the offline coordinate helpers."""

import math
import unittest

from inertialnav import coordinates


class GeodeticToECEFTests(unittest.TestCase):
    def assertVectorAlmostEqual(self, actual, expected, places=7):
        self.assertIsInstance(actual, tuple)
        self.assertEqual(len(actual), 3)
        for value, reference in zip(actual, expected):
            self.assertAlmostEqual(value, reference, places=places)

    def test_equator_greenwich_and_positive_height(self):
        self.assertVectorAlmostEqual(
            coordinates.geodetic_to_ecef(0.0, 0.0, 120.0),
            (6378257.0, 0.0, 0.0),
        )

    def test_equator_quadrature_and_date_line(self):
        self.assertVectorAlmostEqual(
            coordinates.geodetic_to_ecef(0.0, math.pi / 2),
            (0.0, 6378137.0, 0.0),
        )
        self.assertVectorAlmostEqual(
            coordinates.geodetic_to_ecef(0.0, math.pi),
            (-6378137.0, 0.0, 0.0),
        )

    def test_wgs84_oblique_known_answer(self):
        self.assertVectorAlmostEqual(
            coordinates.geodetic_to_ecef(math.pi / 4, math.pi / 4),
            (3194419.1450605746, 3194419.1450605746, 4487348.408865919),
            places=6,
        )

    def test_both_poles_use_semi_minor_axis_and_height(self):
        for sign in (-1, 1):
            with self.subTest(sign=sign):
                self.assertVectorAlmostEqual(
                    coordinates.geodetic_to_ecef(sign * math.pi / 2, 1.2, 50),
                    (0.0, 0.0, sign * (6356752.314245179 + 50)),
                )

    def test_rejects_nonfinite_angles_height_and_out_of_range_angles(self):
        for index in range(3):
            for invalid in (math.nan, math.inf, -math.inf, "bad", None):
                point = [0.0, 0.0, 0.0]
                point[index] = invalid
                with self.subTest(index=index, invalid=invalid):
                    with self.assertRaises(ValueError):
                        coordinates.geodetic_to_ecef(*point)
        for latitude, longitude in ((math.pi, 0), (0, 4), (0, -4)):
            with self.assertRaises(ValueError):
                coordinates.geodetic_to_ecef(latitude, longitude)


if __name__ == "__main__":
    unittest.main()
