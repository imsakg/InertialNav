"""Dependency-free offline navigation error and consistency diagnostics.

All comparisons require caller-supplied references in common units/frames.
No function here supplies ground truth or certifies estimator accuracy.
"""

import math
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
