"""Timestamped body-frame IMU processing (rad/s, m/s^2, seconds)."""
import math
from dataclasses import dataclass
from collections import deque
from .linalg import vector

GRAVITY = 9.80665


def _positive(value, name, allow_zero=False):
    value = float(value)
    if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(name + " is out of range")
    return value


@dataclass(frozen=True)
class ImuSample:
    time: float
    gyro: tuple
    accel: tuple

    def __post_init__(self):
        object.__setattr__(self, 'time', _positive(self.time, 'time', allow_zero=True))
        object.__setattr__(self, 'gyro', tuple(vector(self.gyro, 3)))
        object.__setattr__(self, 'accel', tuple(vector(self.accel, 3)))


def sample_interval(first, second, max_gap=None):
    dt = _positive(second.time-first.time, 'sample interval')
    if max_gap is not None and dt > _positive(max_gap, 'max_gap'):
        raise ValueError("IMU sample gap exceeds maximum")
    return dt


@dataclass(frozen=True)
class Calibration:
    """Per-axis corrected = (raw - bias) * scale; no mounting rotation."""
    gyro_bias: tuple = (0.0, 0.0, 0.0)
    accel_bias: tuple = (0.0, 0.0, 0.0)
    gyro_scale: tuple = (1.0, 1.0, 1.0)
    accel_scale: tuple = (1.0, 1.0, 1.0)

    def __post_init__(self):
        for name in ('gyro_bias', 'accel_bias', 'gyro_scale', 'accel_scale'):
            values = tuple(vector(getattr(self, name), 3))
            if name.endswith('scale') and any(value <= 0 for value in values):
                raise ValueError("calibration scales must be positive")
            object.__setattr__(self, name, values)

    def correct(self, sample):
        gyro = [(raw-bias)*scale for raw, bias, scale in zip(sample.gyro, self.gyro_bias, self.gyro_scale)]
        accel = [(raw-bias)*scale for raw, bias, scale in zip(sample.accel, self.accel_bias, self.accel_scale)]
        return ImuSample(sample.time, gyro, accel)


@dataclass(frozen=True)
class Delta:
    """Integrated angle and force. Raw force uses midpoint-body approximation.

    Coning/sculling-corrected velocity increments are expressed in start-body.
    This distinction prevents applying a body-rotation correction twice.
    """
    dt: float
    angle: tuple
    velocity: tuple
    velocity_frame: str = 'midpoint'

    def __post_init__(self):
        object.__setattr__(self, 'dt', _positive(self.dt, 'dt'))
        object.__setattr__(self, 'angle', tuple(vector(self.angle, 3)))
        object.__setattr__(self, 'velocity', tuple(vector(self.velocity, 3)))
        if self.velocity_frame not in ('midpoint', 'start'):
            raise ValueError("velocity_frame must be midpoint or start")


def integrate_pair(first, second, max_gap=None):
    """Trapezoidal integrals between two samples; does not remove gravity."""
    dt = sample_interval(first, second, max_gap)
    angle = [(a/2+b/2)*dt for a, b in zip(first.gyro, second.gyro)]
    velocity = [(a/2+b/2)*dt for a, b in zip(first.accel, second.accel)]
    return Delta(dt, angle, velocity)


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def combine_increments(first, second):
    """Equal-duration two-sample coning/sculling approximation.

    Inputs are raw integrals, not previously compensated increments. Velocity
    output is in the body frame at the start of the combined interval.
    """
    if first.velocity_frame != 'midpoint' or second.velocity_frame != 'midpoint':
        raise ValueError("cannot compensate already-corrected increments")
    if not math.isclose(first.dt, second.dt, rel_tol=1e-9, abs_tol=0.0):
        raise ValueError("two-sample correction requires equal intervals")
    a, b, u, v = first.angle, second.angle, first.velocity, second.velocity
    ab = _cross(a, b)
    angle = [x+y+2/3*z for x, y, z in zip(a, b, ab)]
    total_a = [x+y for x, y in zip(a, b)]
    total_v = [x+y for x, y in zip(u, v)]
    rotation = _cross(total_a, total_v)
    av, ub = _cross(a, v), _cross(u, b)
    velocity = [total_v[i]+0.5*rotation[i]+2/3*(av[i]+ub[i]) for i in range(3)]
    return Delta(first.dt+second.dt, angle, velocity, velocity_frame='start')


class LowPass:
    """First-order vector low-pass, exact gain for each interval."""
    def __init__(self, cutoff_hz):
        self.cutoff_hz = _positive(cutoff_hz, 'cutoff_hz')
        self.value = None

    def reset(self):
        self.value = None

    def update(self, values, dt):
        values = vector(values, 3)
        dt = _positive(dt, 'dt')
        gain = -math.expm1(-2*math.pi*self.cutoff_hz*dt)
        if self.value is None:
            result = values
        else:
            result = vector([(1-gain)*old+gain*new for old, new in zip(self.value, values)])
        self.value = tuple(result)
        return self.value


class StationaryDetector:
    """Heuristic dwell detector, not proof of zero translational velocity."""
    def __init__(self, window=20, gyro_threshold=0.02, accel_tolerance=0.15,
                 accel_std=0.05, gravity=GRAVITY):
        if isinstance(window, bool) or not isinstance(window, int) or window < 2:
            raise ValueError("stationary window must contain at least two samples")
        self.samples = deque(maxlen=window)
        self.gyro_threshold = _positive(gyro_threshold, 'gyro_threshold')
        self.accel_tolerance = _positive(accel_tolerance, 'accel_tolerance')
        self.accel_std = _positive(accel_std, 'accel_std')
        self.gravity = _positive(gravity, 'gravity')

    def push(self, sample):
        if self.samples:
            sample_interval(self.samples[-1], sample)
        self.samples.append(sample)
        if len(self.samples) != self.samples.maxlen:
            return False
        if any(math.hypot(*s.gyro) > self.gyro_threshold for s in self.samples):
            return False
        n = len(self.samples)
        mean = [math.fsum(s.accel[i]/n for s in self.samples) for i in range(3)]
        if abs(math.hypot(*mean)-self.gravity) > self.accel_tolerance:
            return False
        for i in range(3):
            std = math.sqrt(math.fsum((s.accel[i]-mean[i])**2/n for s in self.samples))
            if std > self.accel_std:
                return False
        return True


def integrate_attitude(attitude, delta_angle):
    """Right-multiply a body angular increment onto body-to-NED attitude."""
    from .rotations import normalize, multiply, from_rotation_vector
    return normalize(multiply(normalize(attitude), from_rotation_vector(delta_angle)))


@dataclass(frozen=True)
class StrapdownState:
    """Local flat-Earth state; no Earth rotation, Coriolis or transport rates."""
    attitude: tuple = (1.0, 0.0, 0.0, 0.0)
    position: tuple = (0.0, 0.0, 0.0)
    velocity: tuple = (0.0, 0.0, 0.0)

    def __post_init__(self):
        from .rotations import normalize
        object.__setattr__(self, 'attitude', tuple(normalize(self.attitude)))
        object.__setattr__(self, 'position', tuple(vector(self.position, 3)))
        object.__setattr__(self, 'velocity', tuple(vector(self.velocity, 3)))

    def propagate(self, delta, gravity=GRAVITY):
        from .rotations import rotate
        gravity = _positive(gravity, 'gravity')
        end = integrate_attitude(self.attitude, delta.angle)
        if delta.velocity_frame == 'start':
            force_attitude = self.attitude
        else:
            force_attitude = integrate_attitude(self.attitude, [angle/2 for angle in delta.angle])
        dv = list(rotate(force_attitude, delta.velocity))
        dv[2] += gravity*delta.dt
        velocity = vector([v+change for v, change in zip(self.velocity, dv)])
        position = vector([p+(v/2+w/2)*delta.dt for p, v, w in zip(self.position, self.velocity, velocity)])
        return StrapdownState(end, tuple(position), tuple(velocity))
