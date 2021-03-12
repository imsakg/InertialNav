"""WGS-84 coordinates in radians/metres and right-handed NED local frames."""

import math


WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A * (1.0 - WGS84_F)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def _finite(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("{} must be a finite number".format(name)) from exc
    if not math.isfinite(value):
        raise ValueError("{} must be a finite number".format(name))
    return value


def _angles(latitude, longitude):
    latitude = _finite(latitude, "latitude")
    longitude = _finite(longitude, "longitude")
    if not -math.pi / 2 <= latitude <= math.pi / 2:
        raise ValueError("latitude must be in [-pi/2, pi/2]")
    if not -math.pi <= longitude <= math.pi:
        raise ValueError("longitude must be in [-pi, pi]")
    return latitude, longitude


def geodetic_to_ecef(latitude, longitude, altitude=0.0):
    """Return ECEF (x, y, z) metres from geodetic radians/ellipsoid height."""
    latitude, longitude = _angles(latitude, longitude)
    altitude = _finite(altitude, "altitude")
    sin_latitude = math.sin(latitude)
    cos_latitude = 0.0 if abs(latitude) == math.pi / 2 else math.cos(latitude)
    radius = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_latitude ** 2)
    return (
        (radius + altitude) * cos_latitude * math.cos(longitude),
        (radius + altitude) * cos_latitude * math.sin(longitude),
        (radius * (1.0 - WGS84_E2) + altitude) * sin_latitude,
    )
