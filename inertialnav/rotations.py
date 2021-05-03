"""Scalar-first Hamilton quaternions for active body-to-NED rotations.

Euler angles use the ZYX (yaw-pitch-roll / 321) sequence and radians, matching
``derivations/EulToQuat.m`` and ``derivations/Quat2Tbn.m`` in this repository.
No function mutates its inputs; quaternion results are (w, x, y, z) tuples.
"""

import math


def _finite(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("{} must be a finite number".format(name)) from exc
    if not math.isfinite(value):
        raise ValueError("{} must be a finite number".format(name))
    return value


def _components(values, length, name):
    if isinstance(values, (str, bytes)):
        raise ValueError("{} must have {} finite components".format(name, length))
    try:
        values = tuple(values)
    except TypeError as exc:
        raise ValueError("{} must have {} finite components".format(name, length)) from exc
    if len(values) != length:
        raise ValueError("{} must have {} finite components".format(name, length))
    return tuple(_finite(value, name) for value in values)


def _quaternion(q):
    q = _components(q, 4, "quaternion")
    if not any(q):
        raise ValueError("zero quaternion does not define a rotation")
    return q


def normalize(q):
    """Return a unit quaternion without changing its sign; reject zero/nonfinite.

    Scale before taking the norm so finite very large or subnormal components
    remain usable even when the unscaled Euclidean norm is not representable.
    """
    q = _quaternion(q)
    scale = max(abs(value) for value in q)
    scaled = tuple(value / scale for value in q)
    magnitude = math.hypot(*scaled)
    return tuple(value / magnitude for value in scaled)
