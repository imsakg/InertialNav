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


# NEXT_INCREMENT
