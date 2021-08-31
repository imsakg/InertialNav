# Offline log validation and error analysis

Python 3.9+ and the standard library only. These modules are independent of the
reference filter/linalg modules and do not change the original estimator. They
provide **sanity checks and diagnostics, not flight qualification or a navigation
accuracy certificate**. Comparisons require an independent, appropriate reference.

## Actual C++ state-log contract

The source of truth is `WriteFilterOutput()` in
[`code/main_closed_loop_float.cpp`](../code/main_closed_loop_float.cpp): it writes
`float(IMUmsec * 0.001f)` followed by every element of `_ekf->states`, using `%e`.
[`code/estimator_22states.h`](../code/estimator_22states.h) declares
`EKF_STATE_ESTIMATES = 22`. Therefore the current format is **23 numeric fields:
time plus 22 states**, with no header or footer. Older archives/prototypes can
have other widths; they are deliberately not guessed or silently truncated.

State indices (zero-based, excluding the time column), as documented in
[`code/estimator_22states.cpp`](../code/estimator_22states.cpp):

- 0–3: scalar-first quaternion `(q0, q1, q2, q3)`.
- 4–6: north/east/down velocity, m/s.
- 7–9: north/east/down position, m.
- 10–12: delta-angle bias, rad, body X/Y/Z.
- 13: delta-velocity bias, m/s, body Z.
- 14–15: north/east wind velocity, m/s.
- 16–18: earth magnetic field, north/east/down, gauss.
- 19–21: body magnetic-field bias, X/Y/Z, gauss.

`parse_state_data(lines)` consumes an iterable of text lines and returns a list
of immutable `StateSample(time, states, line_number)` records. `read_state_data(path)`
opens a UTF-8 file and calls the same parser. Blank lines are allowed; all other
rows must have exactly 23 floating-point fields. Headers, comments, ragged rows,
nonnumeric fields and empty logs raise `LogValidationError` (a `ValueError`).
The first and last samples are never skipped. Physical line numbers include
blank lines. Missing/unreadable paths raise `OSError`; decoding errors propagate.

Parsing checks structure, **not semantics**: Python can parse `nan` and `inf`.
Call `validate_state_data(samples, quaternion_tolerance=1e-3)` to require:

- Exactly 22 states and finite values in every state and the timestamp.
- Strictly increasing time; duplicate/decreasing time fails, rather than sorting.
- Absolute quaternion norm error `abs(norm(q) - 1)` at most the specified finite,
  nonnegative tolerance. This is a norm check, not an orientation-error bound.

The returned immutable `LogSummary` contains `sample_count`, `state_count`,
`start_time`, `end_time`, and `max_quaternion_norm_error`. One row is allowed but
cannot establish a time progression. The API never normalizes quaternions or
repairs data. It consumes the whole supplied iterable, raising on the first error.

```pycon
>>> from inertialnav.logs import parse_state_data, validate_state_data
>>> line = " 0.000000e+00 1.000000e+00" + " 0.000000e+00" * 21
>>> rows = parse_state_data([line])
>>> rows[0].quaternion
(1.0, 0.0, 0.0, 0.0)
>>> rows[0].position_ned
(0.0, 0.0, 0.0)
>>> summary = validate_state_data(rows)
>>> (summary.sample_count, summary.state_count, summary.max_quaternion_norm_error)
(1, 22, 0.0)

```

## Legacy executable checker

From repository root:

```sh
python3 code/validate_output.py
python3 code/validate_output.py /path/to/StateDataOut.txt --expected-rows 1000
python3 code/validate_output.py /path/to/StateDataOut.txt --json
```

From `code/`, use `python3 validate_output.py`. **With no positional path, both
invocations always read `code/StateDataOut.txt`, beside the script.** A root-level
`StateDataOut.txt` does not shadow that default. An explicitly supplied relative
path is relative to the caller's working directory. No installation or NumPy is
needed; tests invoke the script with `python3 -S` (site packages disabled).

Exit status is 0 only when all requested checks pass; a bad/missing/empty log or
wrong `--expected-rows` produces 1. Invalid command-line syntax/options produce
argparse's status 2. `--quaternion-tolerance` sets the absolute norm-error limit.
`--expected-rows` is optional, must be positive, and checks the **actual** row
count; without it a complete-row truncation cannot be distinguished from an
intentionally shorter log.

The normal report gives the real sample count, state count, time endpoints and
maximum norm error, followed by an explicit accuracy disclaimer. `--json` emits
one JSON object on stdout: success has `status: "pass"`, all five summary fields,
and `accuracy_assessed: false`; log failure has `status: "fail"`, an `error`
string and `accuracy_assessed: false`, with no misleading partial sample count.
Argument-parse errors retain normal argparse diagnostics rather than JSON.

### Real archived data and known original-code concerns

The checked-in `ValidationOutputData.zip` contains 40,471 rows of the current
23-field format. Strict validation fails honestly: its first 50 rows (times
0 through 0.999 s) have zero-norm quaternions. An **explicit** selection of times
at least 1 s contains 40,421 rows, from 1.02 to 847.904 s, and passes these sanity
checks with the default tolerance. That selected result is not a validation of
the omitted startup rows or proof of navigation accuracy. The CLI does not omit
them automatically. If selecting an initialized interval, record the selection
and omitted count externally and validate the selected file explicitly.

`OutputFiles.zip` instead contains 25-field rows from a different historical
format and is rejected, not mislabelled as current 22-state output. Original
C++ compilation on current Apple Clang also exposes two pointer/integer ordered
comparisons and an out-of-bounds `posNE[2]` warning. This companion does not patch
those independent original-code issues. Replaying a build requires resolving
compiler compatibility separately; do not report a failed build as a replay.

## Error summaries and online statistics

All errors use **estimated minus reference**. The caller must establish common
units, origin, reference frame, clocks, validity and (where relevant) independence.
The analytical examples below test mathematics; they are not measurements of the
original estimator's accuracy.

- `rms(errors)`: scalar root mean square, using scaled squares to avoid avoidable
  overflow. Requires at least one finite value.
- `rmse(estimated, reference)`: paired scalar RMSE. Equal nonzero lengths are
  required; mismatched sequences are never silently zip-truncated.
- `error_summary(estimated, reference)`: paired, rectangular N-by-D trajectories
  of any positive dimension. Returns `TrajectoryErrorSummary(sample_count, axes,
  vector_rmse)`. Each `AxisErrorSummary` has `bias`, `rmse`, `max_absolute`.
  `vector_rmse` is RMS Euclidean error, equal to the norm of axis RMSEs, **not**
  their average. Position/velocity NED extraction is available on `StateSample`.
- `RunningMoments()`: scalar Welford accumulator in constant memory. `update(x)`
  returns the accumulator; `count` and `mean` are read-only properties.
  `variance()` and `standard_deviation()` default to population statistics;
  `sample=True` uses n-1 and needs two samples. Empty statistics raise `ValueError`.
  Invalid/overflowing updates leave the accumulator unchanged.

```pycon
>>> from inertialnav import analysis as a
>>> a.rms([3, -3])
3.0
>>> a.rmse([1, 3], [0, 2])
1.0
>>> stats = a.error_summary([[1, 0], [3, 0]], [[0, 0], [0, 0]])
>>> (stats.sample_count, stats.axes[0].bias, stats.axes[0].max_absolute)
(2, 2.0, 3.0)
>>> moments = a.RunningMoments()
>>> for value in [1, 2, 3]:
...     _ = moments.update(value)
>>> (moments.count, moments.mean, round(moments.variance(), 6))
(3, 2.0, 0.666667)

```

## Interpolation and timeline alignment

`interpolate(times, values, query_time, max_gap=None)` is piecewise-linear scalar
interpolation. Times must be finite, strictly increasing and match the nonempty
values. Extrapolation is an error. Exact samples, including a singleton timeline,
are returned directly. An optional finite positive `max_gap` limits only the
interval actually interpolated, in the timeline's units.

`align_trajectories(estimate_times, estimated, reference_times, reference,
max_gap=None)` accepts N-by-D and M-by-D trajectories. It interpolates reference
components at each estimate timestamp within the reference range. The immutable
`AlignedTrajectory` gives `times`, `estimated`, `reference`, and
`dropped_estimates` (the number outside that range). No overlap is an error.
Invalid gaps fail the call instead of quietly removing inconvenient samples.
It does not estimate a clock offset or interpolate quaternion/angle manifolds.

```pycon
>>> a.interpolate([0, 1, 4], [0, 2, 8], 2.5)
5.0
>>> paired = a.align_trajectories([0, 1, 2], [[1], [2], [3]], [0, 2], [[0], [2]])
>>> (paired.times, paired.dropped_estimates)
((0.0, 1.0, 2.0), 0)
>>> a.error_summary(paired.estimated, paired.reference).vector_rmse
1.0

```

## Consistency and correlation diagnostics

`nis(innovation, variances)` computes the diagonal/independent-component sum
`innovation[i]**2 / variances[i]`. All variances must be finite and strictly
positive, with one per innovation. C++ `VelPosFuse.txt` and `MagFuse.txt` emit
alternating scalar innovation/variance fields after time; they do **not** provide
full cross-covariances. Summing correlated components as if independent is an
assumption, not a validated joint NIS.

`nees(error, covariance)` computes `error.T @ covariance^-1 @ error` without an
explicit inverse, using unit scaling and a Cholesky solve. The full covariance
must be finite, square, dimension-matched, symmetric (1e-12 relative tolerance,
no absolute floor), and positive definite. The lower triangle is used within
that tolerance. Singular/indefinite matrices are rejected, never regularized.
The same quadratic-form API can evaluate full-covariance NIS when such a matrix
is available. C++ `CovDataOut.txt` contains diagonals only, so full NEES cannot be
recovered from it without an explicit additional covariance assumption/source.

These functions return a score, not a statistical pass/fail classification.
Chi-square interpretation requires appropriate Gaussian/covariance assumptions,
degrees of freedom and independence; no confidence thresholds are invented here.

`autocorrelation(innovations, max_lag)` returns demeaned scalar ACF at sample lags
0 through the integer `max_lag`, which must be less than the sample count:
`rho[k] = sum((x[i]-mean)*(x[i+k]-mean)) / sum((x[i]-mean)**2)`.
This biased normalization shares one denominator across lags, without n/(n-k)
correction. Constant sequences have undefined ACF and raise `ValueError`.
Uniform sample spacing is the caller's responsibility; no whiteness certificate
or significance band is inferred.

```pycon
>>> round(a.nis([2, -3], [4, 9]), 6)
2.0
>>> round(a.nees([1, 1], [[2, 1], [1, 2]]), 6)
0.666667
>>> a.autocorrelation([1, -1, 1, -1], 3)
(1.0, -0.75, 0.5, -0.25)

```

## Overlapping Allan deviation and wrapped angles

`overlapping_allan_deviation(rate_samples, sample_interval, cluster_sizes)` uses
the **rate/frequency**, not phase/angle, convention. For cluster length m, take
all `N - 2*m + 1` adjacent m-sample mean pairs and compute
`sqrt(mean((second_mean - first_mean)**2) / 2)`. Input must be uniformly spaced
rate samples; the interval must be finite and positive. Cluster sizes are
strictly increasing integers from 1 to N//2 (booleans are not sizes).
Each immutable `AllanPoint` reports `cluster_size`, `tau = m*sample_interval`,
`deviation` in input-rate units and `pair_count`. Compensated, scaled prefix sums
give O(N) work per requested cluster and O(N) memory. Do not infer bias stability
or noise coefficients from a few points without further assumptions and analysis.

`wrapped_angle_error(estimated, reference, period=2*pi)` returns the shortest
signed difference in `[-period/2, period/2)`; both half-turn ties map to the
negative endpoint. Use `period=360` for degrees. `wrapped_angle_errors` handles
paired series; pass its tuple to `rms` for angular RMSE. This is a scalar heading
metric, not a 3-D attitude/quaternion distance.

```pycon
>>> points = a.overlapping_allan_deviation([0, 1, 2, 3], 0.5, [1, 2])
>>> tuple((p.tau, p.pair_count) for p in points)
((0.5, 3), (1.0, 1))
>>> tuple(round(p.deviation, 6) for p in points)
(0.707107, 1.414214)
>>> errors = a.wrapped_angle_errors([-179, 179], [179, -179], period=360)
>>> errors
(2.0, -2.0)
>>> a.rms(errors)
2.0

```

## Limits and regression gates

Batch readers/summaries materialize data; only `RunningMoments` is streaming in
constant memory. ACF is O(N * max_lag), dense NEES is O(D^3), and all arithmetic
uses binary floating point. Overflow is rejected where checked; poorly
conditioned matrices, nearly constant sequences, long prefix differences and
very large angles can still lose precision. No resampling, uncertainty bounds,
reference-trajectory loader or performance/accuracy threshold is supplied.

Run from repository root:

```sh
python3 -m unittest discover -s tests -v
python3 -m compileall -q inertialnav tests
git diff --check
```

Tests cover malformed/empty/ragged logs, every nonfinite state column, duplicate
and decreasing times, invalid quaternions/covariances, analytical diagnostic
answers, documented examples, and CLI subprocess success/failure/default-path
semantics without site packages. They do not substitute for an actual C++ replay.
