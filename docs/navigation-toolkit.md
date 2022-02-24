# Offline navigation reference toolkit

The original 21/22/23/24-state C++ and MATLAB filters remain the prototypes
described in the root README. `inertialnav/` adds a dependency-free Python 3.9+
reference toolkit for offline experiments and analysis, not a replacement flight
controller, full error-state EKF, production-certified estimator or accuracy claim.

## Run and verify

From the repository root:

```sh
make verify                         # unittest, compileall, diff checks, examples
python3 -m unittest discover -s tests -v
python3 -m examples.simulate         # explicitly synthetic seeded experiment
python3 -m examples.replay_csv examples/level_imu.csv
python3 code/validate_output.py /path/to/StateDataOut.txt
```

No pip installation, NumPy, network service or downloaded test data is needed.
The test suite builds the original C++11 harness in a temporary directory when
`c++` and `make` are available; otherwise that one test is explicitly skipped.
`TMPDIR` can select the temporary build location. MATLAB/Simulink is not exercised
by these checks. The original `code/Makefile` and archives remain usable.

The simulation includes a biased IMU, six-second GPS outage and one 250 m bad fix.
Seed 7 produced 1.53850 m aided position RMSE versus 8.39517 m dead-reckoned RMSE,
a 0.84155 m final position error and one rejected position fix. These are outcomes
of this particular **synthetic** experiment, not flight performance statistics.

## Explicit conventions

- Navigation axes are north/east/down (NED), in metres and metres/second.
- Body gyro samples are radians/second; integrated body angles are radians.
- Accelerometer data is **specific force**, in metres/second squared. A level,
  stationary sensor reads `(0, 0, -9.80665)`, not zero. Propagation adds NED gravity.
- Attitudes are scalar-first Hamilton quaternions `(w,x,y,z)`, body to NED,
  active rotations; body increments multiply on the right. Euler is ZYX radians.
- Time is seconds on a caller-supplied nonnegative monotonic clock. Frames,
  sensor mounting and clocks must already agree; no implicit unit conversion.
- WGS-84 geographic APIs use latitude/longitude **degrees** and ellipsoid height
  metres. NED point transformations require an explicit fixed origin. See
  [coordinate and rotation contracts](coordinates.md).

## Kalman filtering

`LinearKalman(x, P)` accepts any finite state dimension and symmetric positive
semidefinite covariance, copies inputs, and provides:

```python
from inertialnav.kalman import LinearKalman, constant_velocity_model

kf = LinearKalman([0, 1], [[1, 0], [0, 1]])
F, Q, B = constant_velocity_model(dt=0.1, acceleration_psd=0.2, axes=1)
kf.predict(F, Q, control=[0.5], control_matrix=B)
report = kf.update([0.11], [[1, 0]], [[0.25]], gate=9.0)
assert report.accepted
```

Prediction uses `x=F*x+B*u`, `P=F*P*F.T+Q`. `control` and `control_matrix`
must appear together. Corrections solve positive-definite innovation systems via
Cholesky rather than explicitly inverting them. Measurement noise `R` must be
positive definite, not zero. Posterior covariance uses Joseph form:

`P=(I-KH)*P*(I-KH).T+K*R*K.T`.

Results expose acceptance, residual, full innovation covariance and NIS. A gate is
a positive **squared Mahalanobis** threshold, not a raw sigma multiplier. Rejected
or invalid data does not mutate state/covariance. `gate=None` disables rejection.
Symmetry is checked with scale-relative tolerance; rank-deficient process noise is
allowed, but singular measurement/innovation solves are not regularized silently.
These small dense routines prioritize inspectability over throughput and may
reject extreme or ill-conditioned floating-point inputs.

For position/velocity states, `constant_velocity_model` integrates continuous
white acceleration spectral density `q`:

- `F=[[I, dt*I],[0,I]]`, `B=[[dt**2/2*I],[dt*I]]`.
- `Q=q*[[dt**3/3*I, dt**2/2*I],[dt**2/2*I, dt*I]]`.

`q` has units `(m/s^2)^2/Hz`. It is not a per-sample acceleration variance;
converting noise density or discrete variance requires the correct bandwidth and
sampling convention. Do not confuse this Q with a held random-acceleration model.

## IMU processing and strapdown

```python
from inertialnav.imu import ImuSample, Calibration, integrate_pair, StrapdownState

calibration = Calibration(gyro_bias=(0.001, 0, 0))
a = calibration.correct(ImuSample(0, (0.001,0,0), (0,0,-9.80665)))
b = calibration.correct(ImuSample(0.01, (0.001,0,0), (0,0,-9.80665)))
delta = integrate_pair(a, b, max_gap=0.02)
state = StrapdownState().propagate(delta)
```

- Calibration applies `(raw-bias)*scale` per axis; scale must be positive. It does
  not estimate bias or correct mounting misalignment, lever arms or temperature.
- `integrate_pair` performs trapezoidal integration over irregular intervals.
- `combine_increments` applies equal-duration two-sample coning/sculling cross
  terms. Raw inputs have `velocity_frame='midpoint'`; corrected output is in the
  **start-body frame**. Propagation respects that flag to avoid double rotation.
  Do not feed corrected increments back into the two-sample correction.
- `LowPass(cutoff_hz)` uses interval-dependent gain `1-exp(-2*pi*fc*dt)`. The first
  valid sample initializes the vector; `reset()` removes that history.
- `StationaryDetector` requires a full low-gyro, near-gravity, low-variance dwell
  window. Constant-velocity motion can look stationary to an IMU: it is a heuristic,
  never independent proof for zero-velocity constraints.
- `StrapdownState.propagate` returns a new state, normalizes quaternion attitude,
  rotates force and adds gravity. Position uses average endpoint velocity. This
  is a local flat-Earth approximation; no Earth/transport rotation or Coriolis.

## Loosely coupled aiding

```python
from inertialnav.fusion import NavigationFilter
from inertialnav.imu import Delta

nav = NavigationFilter(acceleration_psd=0.1, max_step=0.2)
nav.predict_to(0.1, Delta(0.1, (0,0,0), (0,0,-0.980665)))
report = nav.update_position((0,0,0), [[4,0,0],[0,4,0],[0,0,9]], timestamp=0.1)
assert nav.aiding_fresh('position', max_age=0.5)
```

`NavigationFilter` tracks six stochastic states `[pn,pe,pd,vn,ve,vd]`, using
strapdown attitude to drive acceleration. It accepts full 3x3 position/velocity
covariance, up-positive altitude relative to the origin datum, confirmed stationary
zero-velocity updates, and wrapped/declination-corrected heading in radians.
Default translation gates are 16.27 (3D) and scalar altitude/heading gates are 9.
Choose statistical gates for the dimension and desired false-rejection rate;
these defaults are not calibrated against your sensor.

Attitude propagation is separate from translational covariance; roll/pitch and
bias uncertainty do **not** feed the six-state covariance. Heading uses its own
scalar variance (initial 1 rad^2; gyro PSD default 0.001 rad^2/s), a wrapped residual,
and a navigation-frame yaw correction that preserves tilt. Raw magnetometer axes
must be calibrated, tilt-compensated and corrected for declination upstream.
Near-vertical Euler heading is rejected. There is no magnetic disturbance model.

Aiding must be aligned with the current state time (optional `timestamp` is
checked); delayed/future data is rejected, not retroactively fused. `predict_to`
checks that the target time matches the interval. Gaps above `max_step` are errors;
do not invent zero-filled data to bridge them. Rejected aids do not refresh
freshness. A ZUPT refreshes `zero_velocity`, not the independent `velocity` sensor.
`aiding_rejections` counts gated observations; invalid inputs raise exceptions.

For real flights, add a complete error-state attitude/bias model, calibrated
noise, sensor latency handling, datum and lever-arm compensation, ground truth
and failure handling before relying on this reference design.

## CSV replay

```sh
python3 -m examples.replay_csv imu.csv --output trajectory.csv \
  --position-sigma 2.5 --velocity-sigma 0.2 --max-step 0.2 \
  --roll-deg 0 --pitch-deg 0 --yaw-deg 0
```

Required header: `time,gx,gy,gz,ax,ay,az`. Optional complete groups are `pn,pe,pd`
and `vn,ve,vd`. A group may be entirely blank on a row; partial groups, duplicate
columns, unknown columns, nonfinite data and nonincreasing timestamps are errors.
The first row initializes time and any available position/velocity. Later aids
apply at their row time after trapezoidal IMU propagation. Attitude defaults to
level/north and must be supplied correctly; no alignment is inferred. Noise flags
are standard deviations, converted to diagonal covariances; real sensors can be
correlated. Output CSV includes NED state and scalar-first attitude. This small
reference replay holds output in memory and is not a streaming flight logger.

## Original C++ output and error diagnostics

The placeholder `code/validate_output.py` no longer reports unearned success.
It checks **every** row of the actual headerless `time + 22 states` format,
requires finite values, monotonic time and near-unit quaternions, and exits nonzero
on invalid/missing files. It does not compare against ground truth or certify
navigation accuracy. See [log formats and statistical diagnostics](analysis.md)
for RMSE, time alignment, NIS, NEES, ACF and overlapping Allan deviation contracts.

With the bundled `code/InputFilesPX4.zip`, the native C++ replay completed and
validated 12,165 rows over 1.003–89.997 seconds, with maximum quaternion norm error
`1.88602067e-7`. These are structural/numerical checks only. Native compilation
still exposes legacy warning categories; numerical prototype limitations remain.

## History provenance

The checked-out upstream history contained 458 reachable commits. It is preserved;
55 additive commits bring this branch to 513. New commits have user-requested
historical author/committer timestamps between March 2021 and February 2022. Those
dates are curated git metadata, **not evidence that these implementations or test
runs occurred then**. The pre-change commit is preserved under the local branch
`backup/inertialnav-original-037c5c3`. No push is part of this work.
