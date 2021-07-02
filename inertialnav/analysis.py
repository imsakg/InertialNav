"""Dependency-free offline navigation error and consistency diagnostics.

All comparisons require caller-supplied references in common units/frames.
No function here supplies ground truth or certifies estimator accuracy.
"""

import math
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
