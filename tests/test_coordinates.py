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


class ECEFToGeodeticTests(unittest.TestCase):
    def test_inverse_equator_and_date_line(self):
        self.assertEqual(coordinates.ecef_to_geodetic(6378137.0, 0, 0), (0, 0, 0))
        latitude, longitude, altitude = coordinates.ecef_to_geodetic(-6378257, 0, 0)
        self.assertEqual((latitude, longitude), (0, math.pi))
        self.assertAlmostEqual(altitude, 120, places=7)

    def test_exact_poles_have_canonical_zero_longitude(self):
        for sign in (-1, 1):
            actual = coordinates.ecef_to_geodetic(0, 0, sign * 6356852.314245179)
            self.assertEqual(actual[:2], (sign * math.pi / 2, 0))
            self.assertAlmostEqual(actual[2], 100.0, places=7)

    def test_round_trip_across_hemispheres_and_heights(self):
        for latitude in (-1.55, -0.8, 0.0, 0.7, 1.55):
            for longitude in (-math.pi, -1.4, 0.0, 2.3, math.pi):
                for altitude in (-400.0, 0.0, 5000.0, 35786000.0):
                    with self.subTest(latitude=latitude, longitude=longitude, altitude=altitude):
                        point = coordinates.geodetic_to_ecef(latitude, longitude, altitude)
                        actual = coordinates.ecef_to_geodetic(*point)
                        self.assertAlmostEqual(actual[0], latitude, places=12)
                        self.assertAlmostEqual(actual[1], longitude, places=12)
                        self.assertAlmostEqual(actual[2], altitude, places=6)

    def test_near_polar_inverse_is_not_divided_by_cosine(self):
        point = coordinates.geodetic_to_ecef(math.pi / 2 - 1e-12, 0.3, 123)
        actual = coordinates.ecef_to_geodetic(*point)
        self.assertAlmostEqual(actual[0], math.pi / 2 - 1e-12, places=13)
        self.assertAlmostEqual(actual[1], 0.3, places=12)
        self.assertAlmostEqual(actual[2], 123, places=7)

    def test_rejects_geocenter_and_nonfinite_components(self):
        with self.assertRaisesRegex(ValueError, "center"):
            coordinates.ecef_to_geodetic(0, 0, 0)
        for index in range(3):
            for invalid in (math.nan, math.inf, -math.inf, None):
                point = [1.0, 1.0, 1.0]
                point[index] = invalid
                with self.assertRaises(ValueError):
                    coordinates.ecef_to_geodetic(*point)


class LocalVectorTests(unittest.TestCase):
    def test_equatorial_axes_distinguish_vectors_from_points(self):
        self.assertEqual(coordinates.ecef_to_ned_vector((1, 2, 3), 0, 0), (3, 2, -1))
        self.assertEqual(coordinates.ned_to_ecef_vector((3, 2, -1), 0, 0), (1, 2, 3))
        self.assertEqual(coordinates.ecef_to_ned_vector((0, 0, 0), 0, 0), (0, 0, 0))

    def test_east_longitude_rotates_east_axis(self):
        actual = coordinates.ecef_to_ned_vector((1, 2, 3), 0, math.pi / 2)
        for value, reference in zip(actual, (3, -1, -2)):
            self.assertAlmostEqual(value, reference, places=12)

    def test_inverse_rotations_preserve_vector_and_norm(self):
        vector = (123.0, -45.0, 67.0)
        for latitude in (-math.pi / 2, -0.5, 0, 1.2, math.pi / 2):
            for longitude in (-math.pi, 0, 1.4, math.pi):
                local = coordinates.ecef_to_ned_vector(vector, latitude, longitude)
                inverse = coordinates.ned_to_ecef_vector(local, latitude, longitude)
                self.assertAlmostEqual(math.hypot(*local), math.hypot(*vector), places=11)
                for value, reference in zip(inverse, vector):
                    self.assertAlmostEqual(value, reference, places=11)

    def test_rejects_bad_dimensions_and_nonfinite_vectors_or_origins(self):
        for transform in (coordinates.ecef_to_ned_vector, coordinates.ned_to_ecef_vector):
            for vector in ((), (1, 2), (1, 2, 3, 4), None, "123", (1, math.nan, 3),
                           (1, 2, math.inf), (-math.inf, 2, 3)):
                with self.subTest(transform=transform.__name__, vector=vector):
                    with self.assertRaises(ValueError):
                        transform(vector, 0, 0)
            for latitude, longitude in ((math.nan, 0), (0, math.inf), (2, 0), (0, 4)):
                with self.assertRaises(ValueError):
                    transform((1, 2, 3), latitude, longitude)


class GeodeticToNEDTests(unittest.TestCase):
    def test_origin_maps_to_zero_and_height_maps_to_negative_down(self):
        origin = (0.7, 1.2, 500.0)
        self.assertEqual(coordinates.geodetic_to_ned(origin, origin), (0, 0, 0))
        actual = coordinates.geodetic_to_ned((0.7, 1.2, 525.0), origin)
        for value, reference in zip(actual, (0, 0, -25)):
            self.assertAlmostEqual(value, reference, places=8)

    def test_east_point_is_translated_before_rotation(self):
        angle = 1e-3
        actual = coordinates.geodetic_to_ned((0, angle, 0), (0, 0, 0))
        self.assertEqual(actual[0], 0)
        self.assertAlmostEqual(actual[1], 6378137 * math.sin(angle), places=8)
        self.assertAlmostEqual(actual[2], 6378137 * (1 - math.cos(angle)), places=8)

    def test_date_line_uses_short_ecef_displacement(self):
        actual = coordinates.geodetic_to_ned(
            (0, -math.pi + 1e-6, 0), (0, math.pi - 1e-6, 0)
        )
        self.assertAlmostEqual(actual[1], 12.756274, places=6)
        self.assertAlmostEqual(actual[0], 0, places=10)
        self.assertLess(abs(actual[2]), 2e-5)

    def test_point_and_origin_are_finite_geodetic_triples(self):
        for invalid in ((), (0, 0), (0, 0, 0, 0), "000", None,
                        (math.nan, 0, 0), (0, math.inf, 0), (0, 0, -math.inf), (2, 0, 0)):
            for point, origin in ((invalid, (0, 0, 0)), ((0, 0, 0), invalid)):
                with self.assertRaises(ValueError):
                    coordinates.geodetic_to_ned(point, origin)


if __name__ == "__main__":
    unittest.main()
