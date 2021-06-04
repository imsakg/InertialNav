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


class NEDToGeodeticTests(unittest.TestCase):
    def test_zero_and_vertical_displacement(self):
        origin = (0.4, -1.2, 200)
        for down in (0, -50, 75):
            actual = coordinates.ned_to_geodetic((0, 0, down), origin)
            self.assertAlmostEqual(actual[0], origin[0], places=12)
            self.assertAlmostEqual(actual[1], origin[1], places=12)
            self.assertAlmostEqual(actual[2], origin[2] - down, places=7)

    def test_local_round_trips_including_polar_and_date_line_origins(self):
        for origin in ((0, 0, 0), (0.8, 2.2, 1000), (-1.2, -3.14, -300),
                       (math.pi / 2, 1.3, 0), (-math.pi / 2, -2.0, 400),
                       (0, math.pi - 1e-7, 0)):
            for local in ((0, 0, 0), (1000, -2000, 35), (-50, 70, -200)):
                with self.subTest(origin=origin, local=local):
                    point = coordinates.ned_to_geodetic(local, origin)
                    recovered = coordinates.geodetic_to_ned(point, origin)
                    for value, reference in zip(recovered, local):
                        self.assertAlmostEqual(value, reference, places=6)

    def test_geodetic_point_round_trip(self):
        origin = (0.75, -2.0, 15)
        point = (0.7502, -1.9999, 140)
        recovered = coordinates.ned_to_geodetic(coordinates.geodetic_to_ned(point, origin), origin)
        self.assertAlmostEqual(recovered[0], point[0], places=12)
        self.assertAlmostEqual(recovered[1], point[1], places=12)
        self.assertAlmostEqual(recovered[2], point[2], places=7)

    def test_rejects_geocenter_and_invalid_local_or_origin(self):
        with self.assertRaisesRegex(ValueError, "center"):
            coordinates.ned_to_geodetic((0, 0, 6378137), (0, 0, 0))
        for invalid in ((), (1, 2), (1, 2, 3, 4), "123", None, (0, math.nan, 0),
                        (0, 0, math.inf), (0, -math.inf, 0)):
            for local, origin in ((invalid, (0, 0, 0)), ((0, 0, 0), invalid)):
                with self.assertRaises(ValueError):
                    coordinates.ned_to_geodetic(local, origin)
        with self.assertRaises(ValueError):
            coordinates.ned_to_geodetic((0, 0, 0), (2, 0, 0))


class ENUNEDTests(unittest.TestCase):
    def test_axis_order_and_up_down_sign(self):
        self.assertEqual(coordinates.enu_to_ned((4, 5, 6)), (5, 4, -6))
        self.assertEqual(coordinates.ned_to_enu((5, 4, -6)), (4, 5, 6))
        for source, expected in (((1, 0, 0), (0, 1, 0)),
                                 ((0, 1, 0), (1, 0, 0)),
                                 ((0, 0, 1), (0, 0, -1))):
            self.assertEqual(coordinates.enu_to_ned(source), expected)

    def test_zero_and_general_round_trips_preserve_norm(self):
        for vector in ((0, 0, 0), (-2.3, 1.4, -8.2), (1e100, -2e100, 3e100)):
            ned = coordinates.enu_to_ned(vector)
            self.assertEqual(coordinates.ned_to_enu(ned), vector)
            self.assertEqual(math.hypot(*ned), math.hypot(*vector))

    def test_rejects_dimensions_and_each_nonfinite_component(self):
        for transform in (coordinates.enu_to_ned, coordinates.ned_to_enu):
            for invalid in ((), (1, 2), (1, 2, 3, 4), None, "123"):
                with self.assertRaises(ValueError):
                    transform(invalid)
            for index in range(3):
                for invalid in (math.nan, math.inf, -math.inf):
                    vector = [1, 2, 3]
                    vector[index] = invalid
                    with self.assertRaises(ValueError):
                        transform(vector)


class CoordinateRegressionTests(unittest.TestCase):
    def test_nonrepresentable_ecef_and_vector_outputs_are_rejected(self):
        cases = (
            (coordinates.ecef_to_geodetic, (1.7e308, 1.7e308, 0)),
            (coordinates.ecef_to_geodetic, (1.3e308, 0, 1.3e308)),
            (coordinates.ecef_to_ned_vector, ((1.7e308, 1.7e308, 0), 0, -math.pi / 4)),
            (coordinates.ned_to_ecef_vector, ((0, 1.7e308, 1.7e308), 0, math.pi / 4)),
        )
        for transform, arguments in cases:
            with self.subTest(transform=transform.__name__, arguments=arguments):
                with self.assertRaises(ValueError):
                    transform(*arguments)

    def test_seeded_global_geodetic_round_trips(self):
        import random
        rng = random.Random(70021)
        for _ in range(500):
            point = (rng.uniform(-math.pi / 2, math.pi / 2),
                     rng.uniform(-math.pi, math.pi), rng.uniform(-500, 4e7))
            actual = coordinates.ecef_to_geodetic(*coordinates.geodetic_to_ecef(*point))
            self.assertAlmostEqual(actual[0], point[0], places=12)
            self.assertAlmostEqual(actual[1], point[1], places=12)
            self.assertAlmostEqual(actual[2], point[2], places=6)

    def test_seeded_local_point_round_trips(self):
        import random
        rng = random.Random(70022)
        for _ in range(200):
            origin = (rng.uniform(-1.5, 1.5), rng.uniform(-math.pi, math.pi),
                      rng.uniform(-400, 3000))
            local = tuple(rng.uniform(-50000, 50000) for _ in range(3))
            actual = coordinates.geodetic_to_ned(coordinates.ned_to_geodetic(local, origin), origin)
            for value, reference in zip(actual, local):
                self.assertAlmostEqual(value, reference, places=6)

    def test_north_cross_east_is_down_at_general_and_polar_origins(self):
        for latitude, longitude in ((0, 0), (0.7, -1.3), (math.pi / 2, 2.0), (-math.pi / 2, -0.4)):
            north, east, down = (coordinates.ned_to_ecef_vector(vector, latitude, longitude)
                                 for vector in ((1, 0, 0), (0, 1, 0), (0, 0, 1)))
            cross = (north[1]*east[2] - north[2]*east[1],
                     north[2]*east[0] - north[0]*east[2],
                     north[0]*east[1] - north[1]*east[0])
            for value, reference in zip(cross, down):
                self.assertAlmostEqual(value, reference, places=14)
            for first, second in ((north, east), (east, down), (down, north)):
                self.assertAlmostEqual(sum(a*b for a, b in zip(first, second)), 0, places=14)

    def test_iterables_are_supported_without_mutating_lists(self):
        point, origin, vector = [0.3, -0.2, 100], [0.2, -0.3, 50], [2, -4, 7]
        snapshot = (point[:], origin[:], vector[:])
        local = coordinates.geodetic_to_ned(iter(point), iter(origin))
        coordinates.ned_to_geodetic(iter(local), iter(origin))
        coordinates.ecef_to_ned_vector(vector, *origin[:2])
        coordinates.ned_to_ecef_vector(vector, *origin[:2])
        self.assertEqual(coordinates.enu_to_ned(iter(vector)), (-4, 2, -7))
        self.assertEqual(coordinates.ned_to_enu(iter(vector)), (-4, 2, -7))
        self.assertEqual((point, origin, vector), snapshot)


if __name__ == "__main__":
    unittest.main()
