"""Loosely coupled local navigation reference, not a full error-state EKF.

Six stochastic states are NED position followed by velocity. Attitude is a
separate strapdown estimate; translational covariance does not include attitude
or bias uncertainty, which must be accounted for in practical tuning.
"""
import math
from . import linalg as la
from .imu import StrapdownState, _positive, GRAVITY
from .kalman import LinearKalman, constant_velocity_model


class NavigationFilter:
    def __init__(self, state=None, covariance=None, time=0.0, acceleration_psd=0.1):
        state = state if state is not None else StrapdownState()
        self.filter = LinearKalman(state.position+state.velocity,
                                   covariance if covariance is not None else la.identity(6))
        self.attitude = state.attitude
        self.time = _positive(time, 'time', allow_zero=True)
        self.acceleration_psd = _positive(acceleration_psd, 'acceleration_psd', allow_zero=True)

    @property
    def state(self):
        return StrapdownState(self.attitude, tuple(self.filter.x[:3]), tuple(self.filter.x[3:]))

    def predict(self, delta):
        next_time = _positive(self.time+delta.dt, 'next time')
        propagated = self.state.propagate(delta)
        acceleration = [(a-b)/delta.dt for a, b in zip(propagated.velocity, self.state.velocity)]
        f, q, b = constant_velocity_model(delta.dt, self.acceleration_psd)
        self.filter.predict(f, q, control=acceleration, control_matrix=b)
        self.attitude, self.time = propagated.attitude, next_time
        return self.state

    # CLASS_INCREMENT


# NEXT_INCREMENT
