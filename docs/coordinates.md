# Coordinate and attitude transforms

These dependency-free Python 3.9+ helpers are an **offline reference companion**
to the original C++/MATLAB estimator, not a replacement flight estimator or a
certified geodesy library. Imports do not read logs, open devices or use a network.

## Units and axes

- All angles, including latitude/longitude, Euler angles and rotation vectors,
  are **radians**. All positions and ellipsoid heights are **metres**.
- Latitude is geodetic, in `[-pi/2, pi/2]`; longitude is in `[-pi, pi]`.
  Out-of-range angles are rejected, not silently wrapped.
- ECEF is Earth-centred/Earth-fixed: +X at the equator/Greenwich, +Y at the
  equator/90 degrees east, +Z toward the north pole.
- Local NED is right-handed **north, east, down**. ENU is **east, north, up**.
  A higher point directly above the origin has negative NED down.
- WGS-84 uses `a = 6378137.0`, `f = 1 / 298.257223563`,
  `b = a * (1 - f)` and `e2 = f * (2 - f)`. Height is relative to this ellipsoid,
  **not** mean sea level, a geoid model or a barometric datum.
- Functions return immutable tuples and do not mutate their inputs. Vector,
  quaternion and geodetic-triple arguments accept finite iterables of the exact
  required length, including lists and one-use iterators (but not text strings).

## Coordinate API

Import from `inertialnav.coordinates`:

- `geodetic_to_ecef(latitude, longitude, altitude=0.0)` returns `(x, y, z)`.
- `ecef_to_geodetic(x, y, z)` returns `(latitude, longitude, altitude)`.
- `ecef_to_ned_vector(vector, latitude, longitude)` rotates a free ECEF vector
  into the tangent axes at those geodetic angles, **without translation**.
- `ned_to_ecef_vector(vector, latitude, longitude)` applies the inverse rotation.
  Vector units are preserved (for example, m/s stays m/s).
- `geodetic_to_ned(point, origin)` converts a geodetic triple to a local NED
  point in metres; `point` and `origin` are `(latitude, longitude, altitude)`.
- `ned_to_geodetic(point, origin)` inverts that local point mapping; `point`
  is NED metres and `origin` is a geodetic triple.
- `enu_to_ned(vector)` and `ned_to_enu(vector)` swap the first two axes and negate
  the third. For points, both frames must already share an origin.

The point mapping subtracts ECEF origins **before** rotating:

```text
local = R_ecef_to_ned(origin) * (ecef(point) - ecef(origin))
ecef(point) = ecef(origin) + transpose(R_ecef_to_ned(origin)) * local
```

With `sphi = sin(latitude)`, `cphi = cos(latitude)`, `slam = sin(longitude)`
and `clam = cos(longitude)`, the rotation rows are:

```text
north: (-sphi*clam, -sphi*slam,  cphi)
east:  (-slam,       clam,       0)
down:  (-cphi*clam, -cphi*slam, -sphi)
```

This is an ECEF **chord** resolved in one fixed tangent frame, not a geodesic
surface distance. An equal-height distant point can have positive down because
of Earth curvature. Keep the origin explicit and avoid treating a fixed NED
frame as a globally flat Earth. ECEF subtraction avoids date-line jumps, but
retains normal floating-point cancellation limits for extremely tiny offsets.

Exact ECEF poles return longitude zero; forward conversion of an exact
geodetic pole sets X and Y to zero. At a polar local origin, the supplied
longitude still selects the otherwise arbitrary north/east axes. The geocenter
has no geodetic latitude and is rejected by the inverse. The iterative inverse
is intended for navigation positions near/outside the ellipsoid, including
moderately negative heights, not ambiguous deep-interior normal coordinates;
nonconvergence raises `ValueError`. Height is projected on the ellipsoid normal
rather than divided by `cos(latitude)`, avoiding polar division instability.

Executable point/vector examples (run `python3 -m doctest docs/coordinates.md`):

```pycon
>>> import math
>>> from inertialnav.coordinates import (geodetic_to_ecef, ecef_to_geodetic,
...     geodetic_to_ned, ned_to_geodetic, enu_to_ned, ned_to_ecef_vector)
>>> xyz = geodetic_to_ecef(0.0, 0.0)
>>> assert xyz == (6378137.0, 0.0, 0.0)
>>> assert ecef_to_geodetic(*xyz) == (0.0, 0.0, 0.0)
>>> origin = (math.radians(41.0), math.radians(29.0), 100.0)
>>> requested = (100.0, -50.0, -20.0)
>>> point = ned_to_geodetic(requested, origin)
>>> recovered = geodetic_to_ned(point, origin)
>>> assert all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(recovered, requested))
>>> assert enu_to_ned((4.0, 5.0, 6.0)) == (5.0, 4.0, -6.0)

```

## Attitude API and composition

Import from `inertialnav.rotations`. Quaternions are scalar-first **`(w, x, y, z)`**,
Hamilton products, and active body-to-NED rotations. Body axes follow the usual
forward/right/down convention. This matches the original
[`EulToQuat.m`](../derivations/EulToQuat.m),
[`QuatMult.m`](../derivations/QuatMult.m) and
[`Quat2Tbn.m`](../derivations/Quat2Tbn.m), not a scalar-last/JPL convention.

- `normalize(q)` returns a unit quaternion, preserving sign. Scaling before
  taking the norm supports very large, very small and subnormal finite inputs.
- `multiply(q1, q2)` returns the **raw**, non-normalized Hamilton product.
  For unit attitudes, `q2` acts first and `q1` second. Normalize accumulated
  attitude products explicitly. A raw product that becomes nonfinite or
  underflows entirely to zero raises `ValueError`; this is not arbitrary-
  precision quaternion algebra.
- `conjugate(q)` returns `(w, -x, -y, -z)`. It is an inverse only if `q` is unit.
- `rotate(q, vector)` normalizes the attitude and returns a three-tuple. It
  implements `q * (0, vector) * conjugate(q)` for the normalized `q`, without
  invoking the public product on a zero vector. It preserves vector magnitude
  up to rounding and maps body components to NED components. To invert, rotate
  by `conjugate(normalize(q))`. `q` and `-q` represent the same rotation.
- `from_euler(roll, pitch, yaw)` returns the unit quaternion for
  `Rz(yaw) * Ry(pitch) * Rx(roll)`: ZYX / 321 yaw-pitch-roll. Inputs may be any
  finite angles; output Euler angles need not recover a noncanonical input.
- `to_euler(q)` normalizes `q` and returns `(roll, pitch, yaw)` with pitch in
  `[-pi/2, pi/2]`, roll/yaw in `[-pi, pi]`. At `abs(cos(pitch)) <= 1e-12`, it
  chooses roll zero and retains the observable combined heading. Individual
  roll/yaw are undefined at exact gimbal lock and ill-conditioned near it;
  keep quaternions for propagation and compare represented rotations instead.
- `from_rotation_vector(vector)` returns the quaternion exponential for
  **axis times angle**, not Euler angles. Zero maps to identity; a small-angle
  sinc series retains tiny increments. A nonrepresentable vector norm is
  rejected. Very large angles remain subject to floating-point argument-
  reduction accuracy and are not useful IMU increments.
- `slerp(q0, q1, fraction)` normalizes both endpoints and interpolates the shortest
  attitude arc for finite `fraction` in `[0, 1]` (no extrapolation). It flips
  `q1` if the quaternion dot product is negative, so the final sign can differ
  from the supplied `q1`. Near-coincident inputs (adjusted dot `> 0.9995`) use
  normalized linear interpolation, an approximation to constant angular speed.
  Opposite-sign representations of one attitude remain at that attitude.
  Exactly half-turn-separated rotations have two equally short arcs; the input
  signs deterministically select the tie direction.

Positive yaw turns body forward toward NED east. Frame composition is explicit:

```pycon
>>> from inertialnav.rotations import (normalize, multiply, conjugate, rotate,
...     from_euler, to_euler, from_rotation_vector, slerp)
>>> q = from_euler(0.0, 0.0, math.pi / 2)
>>> ned_forward = rotate(q, (1.0, 0.0, 0.0))
>>> assert all(math.isclose(a, b, abs_tol=1e-14) for a, b in zip(ned_forward, (0, 1, 0)))
>>> ecef_forward = ned_to_ecef_vector(ned_forward, 0.0, 0.0)
>>> assert all(math.isclose(a, b, abs_tol=1e-14) for a, b in zip(ecef_forward, (0, 1, 0)))
>>> recovered_body = rotate(conjugate(q), ned_forward)
>>> assert all(math.isclose(a, b, abs_tol=1e-14) for a, b in zip(recovered_body, (1, 0, 0)))
>>> halfway = slerp((1, 0, 0, 0), q, 0.5)
>>> assert math.isclose(to_euler(halfway)[2], math.pi / 4, abs_tol=1e-14)

```

For an interval with constant **body-axis angular velocity**, an increment acts
on the right: `q_next = normalize(multiply(q, delta_body))`. An increment expressed
in navigation axes instead acts on the left. Rotation vectors are integrated
angular displacements, not angular rates or Euler-angle rates. This primitive
does not add bias correction, coning compensation, Earth-rate correction or
changing local-frame transport:

```pycon
>>> dt, yaw_rate, steps = 0.01, 0.5, 100
>>> attitude = (1.0, 0.0, 0.0, 0.0)
>>> delta_body = from_rotation_vector((0.0, 0.0, yaw_rate * dt))
>>> for _ in range(steps):
...     attitude = normalize(multiply(attitude, delta_body))
>>> assert math.isclose(to_euler(attitude)[2], yaw_rate * dt * steps, abs_tol=1e-12)

```

## Validation and scope

Invalid dimensions, nonnumeric/nonfinite components and zero quaternion inputs
raise `ValueError`. Zero three-vectors and a zero rotation vector are valid.
Finite inputs whose calculated coordinate/vector results are not representable
also raise `ValueError` rather than returning infinities or NaNs. Scalar function
arity still follows normal Python argument errors. There are no hidden unit
conversions, magnetic declination corrections, geoid corrections or global
frame-state mutations.

Run:

```sh
python3 -m unittest discover -s tests -v
python3 -m doctest docs/coordinates.md
python3 -m compileall -q inertialnav tests
```

Tests cover fixed WGS-84 known answers, both poles and the date line, seeded
global/local round trips, orthonormal/right-handed axes, independent Euler and
Rodrigues matrices, quaternion signs/composition, tiny and repeated increments,
Euler singularities, antipodal/half-turn interpolation and finite-output guards.
These are mathematical software checks, not sensor calibration or accuracy
validation of the original estimator against a surveyed trajectory.
