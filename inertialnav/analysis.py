"""Dependency-free offline navigation error and consistency diagnostics.

All comparisons require caller-supplied references in common units/frames.
No function here supplies ground truth or certifies estimator accuracy.
"""

import math
from bisect import bisect_left
from dataclasses import dataclass
from typing import Iterable, Tuple


def _finite_values(values: Iterable[float], name: str) -> Tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + " must contain finite numeric values") from exc
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError(name + " must be nonempty and finite")
    return result


def rms(errors: Iterable[float]) -> float:
    """Root mean square of finite scalar errors, with scaled squaring."""
    values = _finite_values(errors, "errors")
    scale = max(abs(value) for value in values)
    if scale == 0.0:
        return 0.0
    return scale * math.sqrt(math.fsum((value / scale) ** 2 for value in values) / len(values))


def rmse(estimated: Iterable[float], reference: Iterable[float]) -> float:
    """Scalar RMSE; require equal nonzero sample counts (no zip truncation)."""
    estimates = _finite_values(estimated, "estimated")
    truth = _finite_values(reference, "reference")
    if len(estimates) != len(truth):
        raise ValueError("estimated and reference lengths must match")
    return rms(estimate - target for estimate, target in zip(estimates, truth))


@dataclass(frozen=True)
class AxisErrorSummary:
    bias: float
    rmse: float
    max_absolute: float


@dataclass(frozen=True)
class TrajectoryErrorSummary:
    sample_count: int
    axes: Tuple[AxisErrorSummary, ...]
    vector_rmse: float


def _trajectory(values, name):
    rows = tuple(_finite_values(row, name + " row") for row in values)
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError(name + " must be nonempty and rectangular")
    return rows


def error_summary(estimated, reference) -> TrajectoryErrorSummary:
    """Summarize paired N-by-D trajectories, using estimated minus reference.

    Axes are caller-defined; NED position logs use metres and velocity logs use
    metres/second. vector_rmse is RMS Euclidean error, not mean axis RMSE.
    """
    estimates = _trajectory(estimated, "estimated")
    truth = _trajectory(reference, "reference")
    if len(estimates) != len(truth) or len(estimates[0]) != len(truth[0]):
        raise ValueError("trajectory shapes must match")
    errors = tuple(_finite_values((a - b for a, b in zip(e, t)), "errors")
                   for e, t in zip(estimates, truth))
    count = len(errors)
    axes = tuple(AxisErrorSummary(math.fsum(v / count for v in column),
                                  rms(column), max(abs(v) for v in column))
                 for column in zip(*errors))
    vector_rmse = math.hypot(*(axis.rmse for axis in axes))
    if not math.isfinite(vector_rmse):
        raise ValueError("vector RMSE exceeds floating-point range")
    return TrajectoryErrorSummary(count, axes, vector_rmse)


class RunningMoments:
    """Welford scalar count/mean/variance in O(1) memory.

    Finite updates that overflow the accumulator are rejected atomically.
    Variance defaults to population variance; sample=True uses n-1.
    """

    def __init__(self):
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0

    @property
    def count(self) -> int:
        return self._count

    @property
    def mean(self) -> float:
        if not self._count:
            raise ValueError("mean requires at least one sample")
        return self._mean

    def update(self, value: float):
        value = _finite_values((value,), "value")[0]
        count = self._count + 1
        delta = value - self._mean
        mean = self._mean + delta / count
        m2 = self._m2 + delta * (value - mean)
        if not all(math.isfinite(v) for v in (mean, m2)):
            raise ValueError("running moments exceed floating-point range")
        self._count, self._mean, self._m2 = count, mean, m2
        return self

    def variance(self, *, sample: bool = False) -> float:
        divisor = self._count - int(sample)
        if divisor <= 0:
            raise ValueError("insufficient samples for variance")
        return self._m2 / divisor

    def standard_deviation(self, *, sample: bool = False) -> float:
        return math.sqrt(self.variance(sample=sample))


def _time_axis(times, name="times"):
    times = _finite_values(times, name)
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError(name + " must be strictly increasing")
    return times


def _check_gap(max_gap):
    if max_gap is not None:
        max_gap = _finite_values((max_gap,), "max_gap")[0]
        if max_gap <= 0:
            raise ValueError("max_gap must be positive")
    return max_gap


def _bracket(times, query, max_gap):
    if query < times[0] or query > times[-1]:
        raise ValueError("query is outside the reference time range")
    right = bisect_left(times, query)
    if times[right] == query:
        return right, right, 0.0
    left = right - 1
    width = times[right] - times[left]
    if not math.isfinite(width):
        raise ValueError("time interval exceeds floating-point range")
    if max_gap is not None and width > max_gap:
        raise ValueError("interpolation interval exceeds max_gap")
    return left, right, (query - times[left]) / width


def interpolate(times: Iterable[float], values: Iterable[float], query_time: float,
                *, max_gap=None) -> float:
    """Piecewise linear scalar interpolation, with no extrapolation or sorting.

    Exact samples (including singleton timelines) require no interpolation.
    max_gap optionally bounds the bracketing interval, in the time axis's units.
    """
    times = _time_axis(times)
    values = _finite_values(values, "values")
    if len(times) != len(values):
        raise ValueError("times and values lengths must match")
    query = _finite_values((query_time,), "query_time")[0]
    left, right, weight = _bracket(times, query, _check_gap(max_gap))
    return (1 - weight) * values[left] + weight * values[right]


@dataclass(frozen=True)
class AlignedTrajectory:
    times: Tuple[float, ...]
    estimated: Tuple[Tuple[float, ...], ...]
    reference: Tuple[Tuple[float, ...], ...]
    dropped_estimates: int


def align_trajectories(estimate_times, estimated, reference_times, reference,
                       *, max_gap=None) -> AlignedTrajectory:
    """Interpolate a vector reference at estimate times inside its time range.

    Out-of-range estimates are counted as dropped, never extrapolated. Invalid
    timelines or interpolation gaps fail the whole call, rather than silently
    selecting an easier subset. Clocks, units and frames must already agree.
    This is componentwise interpolation, not quaternion/angle interpolation.
    """
    et = _time_axis(estimate_times, "estimate_times")
    rt = _time_axis(reference_times, "reference_times")
    estimates = _trajectory(estimated, "estimated")
    truth = _trajectory(reference, "reference")
    if len(et) != len(estimates) or len(rt) != len(truth):
        raise ValueError("each timeline must match its trajectory length")
    if len(estimates[0]) != len(truth[0]):
        raise ValueError("trajectory dimensions must match")
    max_gap = _check_gap(max_gap)
    times, paired_estimates, paired_truth = [], [], []
    for time, estimate in zip(et, estimates):
        if time < rt[0] or time > rt[-1]:
            continue
        left, right, weight = _bracket(rt, time, max_gap)
        target = _finite_values(((1 - weight) * a + weight * b
                                 for a, b in zip(truth[left], truth[right])), "interpolated reference")
        times.append(time)
        paired_estimates.append(estimate)
        paired_truth.append(target)
    if not times:
        raise ValueError("timelines have no estimate samples in common")
    return AlignedTrajectory(tuple(times), tuple(paired_estimates), tuple(paired_truth),
                             len(et) - len(times))


def _squared_norm(values, name):
    norm = math.hypot(*values)
    score = norm * norm
    if not math.isfinite(score):
        raise ValueError(name + " exceeds floating-point range")
    return score


def nis(innovation: Iterable[float], variances: Iterable[float]) -> float:
    """Normalized innovation squared for independent/diagonal innovations.

    Return sum(innovation[i]**2 / variances[i]). The C++ VelPosFuse/MagFuse
    logs contain scalar variances, not full innovation covariances. Do not use
    this diagonal approximation for correlated components without justification.
    """
    innovation = _finite_values(innovation, "innovation")
    variances = _finite_values(variances, "variances")
    if len(innovation) != len(variances):
        raise ValueError("innovation and variances lengths must match")
    if any(variance <= 0 for variance in variances):
        raise ValueError("innovation variances must be positive")
    return _squared_norm((v / math.sqrt(s) for v, s in zip(innovation, variances)), "NIS")
