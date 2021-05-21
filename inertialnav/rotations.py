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


def multiply(q1, q2):
    """Return the Hamilton product q1*q2, without normalizing either operand.

    For unit attitudes q2 acts first, then q1. Nonzero finite raw operands are
    accepted, but an unrepresentable (nonfinite or underflowed zero) result is
    rejected. Normalize separately when composing accumulated attitudes.
    """
    a, b, c, d = _quaternion(q1)
    e, f, g, h = _quaternion(q2)
    return _quaternion((
        a * e - b * f - c * g - d * h,
        a * f + b * e + c * h - d * g,
        a * g - b * h + c * e + d * f,
        a * h + b * g - c * f + d * e,
    ))


def conjugate(q):
    """Return (w, -x, -y, -z); this is the inverse only for a unit quaternion."""
    w, x, y, z = _quaternion(q)
    return w, -x, -y, -z


def rotate(q, vector):
    """Actively rotate a three-vector by q (body components to NED components).

    The attitude is normalized internally; q and -q produce the same vector.
    Use ``conjugate(normalize(q))`` for the opposite frame transformation.
    """
    w, x, y, z = normalize(q)
    vector = _components(vector, 3, "vector")
    matrix = (
        (w*w + x*x - y*y - z*z, 2*(x*y - w*z), 2*(x*z + w*y)),
        (2*(x*y + w*z), w*w - x*x + y*y - z*z, 2*(y*z - w*x)),
        (2*(x*z - w*y), 2*(y*z + w*x), w*w - x*x - y*y + z*z),
    )
    result = tuple(sum(a * b for a, b in zip(row, vector)) for row in matrix)
    return _components(result, 3, "rotated vector")


def from_euler(roll, pitch, yaw):
    """Return a unit ZYX attitude: Rz(yaw)*Ry(pitch)*Rx(roll), in radians."""
    roll, pitch, yaw = (_finite(value, name)
                        for value, name in zip((roll, pitch, yaw), ("roll", "pitch", "yaw")))
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return normalize((
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ))


def to_euler(q):
    """Return canonical (roll, pitch, yaw) radians from a normalized attitude.

    Pitch lies in [-pi/2, pi/2]; roll/yaw lie in [-pi, pi]. At gimbal lock
    (|cos(pitch)| <= 1e-12), choose roll=0 and preserve the observable heading.
    The individual roll/yaw angles are not uniquely observable there.
    """
    w, x, y, z = normalize(q)
    r00 = w*w + x*x - y*y - z*z
    r10 = 2 * (x*y + w*z)
    r20 = 2 * (x*z - w*y)
    cos_pitch = math.hypot(r00, r10)
    pitch = math.atan2(-r20, cos_pitch)
    if cos_pitch <= 1e-12:
        roll = 0.0
        yaw = math.atan2(2 * (w*z - x*y), w*w - x*x + y*y - z*z)
    else:
        roll = math.atan2(2 * (y*z + w*x), w*w - x*x - y*y + z*z)
        yaw = math.atan2(r10, r00)
    return roll, pitch, yaw


def from_rotation_vector(vector):
    """Exponentiate an axis-times-angle three-vector in radians to a unit q.

    Zero maps to identity. A small-angle sinc expansion keeps tiny increments
    instead of treating them as an exactly zero rotation.
    """
    vector = _components(vector, 3, "rotation vector")
    angle = _finite(math.hypot(*vector), "rotation vector magnitude")
    if angle < 1e-6:
        squared = angle * angle
        factor = 0.5 - squared / 48.0 + squared * squared / 3840.0
        imaginary = tuple(factor * value for value in vector)
    else:
        sin_half = math.sin(angle / 2)
        imaginary = tuple(sin_half * (value / angle) for value in vector)
    return normalize((math.cos(angle / 2),) + imaginary)
