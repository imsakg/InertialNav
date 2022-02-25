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
    def __init__(self, state=None, covariance=None, time=0.0, acceleration_psd=0.1, max_step=1.0):
        state = state if state is not None else StrapdownState()
        self.filter = LinearKalman(state.position+state.velocity,
                                   covariance if covariance is not None else la.identity(6))
        self.attitude = state.attitude
        self.heading_variance = 1.0
        self.gyro_noise_psd = 0.001
        self.time = _positive(time, 'time', allow_zero=True)
        self.max_step = _positive(max_step, 'max_step')
        self.aiding_times = {}
        self.aiding_rejections = {}
        self.acceleration_psd = _positive(acceleration_psd, 'acceleration_psd', allow_zero=True)

    @property
    def state(self):
        return StrapdownState(self.attitude, tuple(self.filter.x[:3]), tuple(self.filter.x[3:]))

    def predict(self, delta):
        if delta.dt > self.max_step:
            raise ValueError("IMU propagation interval exceeds max_step")
        next_time = _positive(self.time+delta.dt, 'next time')
        if next_time <= self.time:
            raise ValueError("IMU interval is below clock precision")
        propagated = self.state.propagate(delta)
        acceleration = [(a-b)/delta.dt for a, b in zip(propagated.velocity, self.state.velocity)]
        f, q, b = constant_velocity_model(delta.dt, self.acceleration_psd)
        heading_variance = _positive(self.heading_variance+self.gyro_noise_psd*delta.dt, "heading variance")
        self.filter.predict(f, q, control=acceleration, control_matrix=b)
        self.attitude, self.time = propagated.attitude, next_time
        self.heading_variance = heading_variance
        return self.state

    def update_position(self, position_ned, covariance, gate=16.27, timestamp=None):
        """Fuse a co-located local NED fix with full 3x3 covariance (m^2)."""
        self._check_measurement_time(timestamp)
        z = la.vector(position_ned, 3)
        h = [[float(i == j) for j in range(6)] for i in range(3)]
        return self._record("position", self.filter.update(z, h, covariance, gate=gate))


    def update_velocity(self, velocity_ned, covariance, gate=16.27, timestamp=None):
        """Fuse NED velocity with full covariance in (m/s)^2."""
        self._check_measurement_time(timestamp)
        z = la.vector(velocity_ned, 3)
        h = [[float(i+3 == j) for j in range(6)] for i in range(3)]
        return self._record("velocity", self.filter.update(z, h, covariance, gate=gate))


    def update_altitude(self, height_up, variance, gate=9.0, timestamp=None):
        """Height is up-positive metres relative to the NED origin datum."""
        self._check_measurement_time(timestamp)
        height = la.vector([height_up])[0]
        result = self.filter.update([-height], [[0, 0, 1, 0, 0, 0]], [[variance]], gate=gate)
        return self._record("altitude", result)

    def update_zero_velocity(self, stationary, variance=0.01, gate=16.27, timestamp=None):
        """Apply a pseudo-observation only after explicit stationary confirmation."""
        self._check_measurement_time(timestamp)
        if not isinstance(stationary, bool):
            raise ValueError("stationary confirmation must be a boolean")
        if not stationary:
            return None
        variance = _positive(variance, 'velocity variance')
        covariance = [[variance*float(i == j) for j in range(3)] for i in range(3)]
        h = [[float(i+3 == j) for j in range(6)] for i in range(3)]
        return self._record("zero_velocity", self.filter.update([0]*3, h, covariance, gate=gate))


    def update_heading(self, heading, variance, gate=9.0, timestamp=None):
        """Fuse a declination-corrected NED heading (radians) in a separate scalar filter.

        Heading covariance is not coupled to translation or tilt. Raw magnetic
        axes are not headings; tilt compensation/calibration belongs upstream.
        """
        self._check_measurement_time(timestamp)
        from .analysis import wrapped_angle_error
        from .rotations import to_euler, from_euler, multiply, normalize
        roll, pitch, yaw = to_euler(self.attitude)
        if abs(math.cos(pitch)) < 1e-3:
            raise ValueError("Euler heading is unobservable near vertical pitch")
        innovation = wrapped_angle_error(heading, yaw)
        heading_filter = LinearKalman([0], [[self.heading_variance]])
        result = heading_filter.update([innovation], [[1]], [[variance]], gate=gate)
        if result.accepted:
            attitude = normalize(multiply(from_euler(0, 0, heading_filter.x[0]), self.attitude))
            self.attitude, self.heading_variance = attitude, heading_filter.P[0][0]
        return self._record("heading", result)


    def _check_measurement_time(self, timestamp):
        if timestamp is not None:
            timestamp = _positive(timestamp, 'measurement time', allow_zero=True)
            if not math.isclose(timestamp, self.time, rel_tol=0, abs_tol=1e-9):
                raise ValueError("measurement must be aligned to the current state time")

    def _record(self, sensor, result):
        if result.accepted:
            self.aiding_times[sensor] = self.time
        else:
            self.aiding_rejections[sensor] = self.aiding_rejections.get(sensor, 0)+1
        return result

    def aiding_fresh(self, sensor, max_age):
        max_age = _positive(max_age, 'max_age', allow_zero=True)
        last = self.aiding_times.get(sensor)
        return last is not None and self.time-last <= max_age

    def predict_to(self, timestamp, delta):
        timestamp = _positive(timestamp, 'prediction time', allow_zero=True)
        tolerance = max(1e-12, 1e-9*delta.dt, 2*math.ulp(timestamp), 2*math.ulp(self.time))
        if timestamp <= self.time or abs((timestamp-self.time)-delta.dt) > tolerance:
            raise ValueError("prediction timestamp must match the increment interval")
        state = self.predict(delta)
        # Retain the observation clock, not an accumulating floating-point sum.
        self.time = timestamp
        return state
