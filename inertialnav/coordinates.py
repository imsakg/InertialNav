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


def ecef_to_geodetic(x, y, z):
    """Return (latitude, longitude, ellipsoid height); angles are radians.

    At exact poles longitude is canonically zero. The geocenter has no
    geodetic latitude and is rejected. Iteration is intended for navigation
    positions near/outside the ellipsoid, not ambiguous deep-interior points.
    """
    x, y, z = (_finite(value, name) for value, name in zip((x, y, z), "xyz"))
    horizontal = math.hypot(x, y)
    if horizontal == 0.0:
        if z == 0.0:
            raise ValueError("geodetic coordinates are undefined at Earth's center")
        return math.copysign(math.pi / 2, z), 0.0, abs(z) - WGS84_B
    longitude = math.atan2(y, x)
    latitude = math.atan2(z, horizontal * (1.0 - WGS84_E2))
    for _ in range(16):
        sin_latitude = math.sin(latitude)
        radius = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_latitude ** 2)
        updated = math.atan2(z + WGS84_E2 * radius * sin_latitude, horizontal)
        if abs(updated - latitude) <= 2e-15:
            latitude = updated
            break
        latitude = updated
    else:
        raise ValueError("geodetic latitude did not converge for this ECEF point")
    sin_latitude, cos_latitude = math.sin(latitude), math.cos(latitude)
    # Project onto the ellipsoid normal instead of dividing by cos(latitude).
    altitude = (horizontal * cos_latitude + z * sin_latitude
                - WGS84_A * math.sqrt(1.0 - WGS84_E2 * sin_latitude ** 2))
    return latitude, longitude, altitude


def _vector(values, name="vector"):
    if isinstance(values, (str, bytes)):
        raise ValueError("{} must have three finite components".format(name))
    try:
        values = tuple(values)
    except TypeError as exc:
        raise ValueError("{} must have three finite components".format(name)) from exc
    if len(values) != 3:
        raise ValueError("{} must have three finite components".format(name))
    return tuple(_finite(value, name) for value in values)


def _ned_basis(latitude, longitude):
    latitude, longitude = _angles(latitude, longitude)
    slat, clat = math.sin(latitude), math.cos(latitude)
    slon, clon = math.sin(longitude), math.cos(longitude)
    return (
        (-slat * clon, -slat * slon, clat),
        (-slon, clon, 0.0),
        (-clat * clon, -clat * slon, -slat),
    )


def ecef_to_ned_vector(vector, latitude, longitude):
    """Rotate a free ECEF vector into NED at the given geodetic angles.

    No origin translation is applied; units and magnitude are preserved.
    """
    vector = _vector(vector)
    return tuple(sum(a * b for a, b in zip(row, vector))
                 for row in _ned_basis(latitude, longitude))


def ned_to_ecef_vector(vector, latitude, longitude):
    """Rotate a free NED vector into ECEF, without translating a point."""
    vector = _vector(vector)
    basis = _ned_basis(latitude, longitude)
    return tuple(sum(basis[row][column] * vector[row] for row in range(3))
                 for column in range(3))
