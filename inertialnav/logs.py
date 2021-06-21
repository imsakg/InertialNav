"""Strict readers for the original C++ estimator's StateDataOut.txt.

The current writer emits time in seconds followed by exactly 22 states, with
no header or footer. Parsing is structural; use validate_state_data for physical
sanity checks. This module does not depend on the reference filter package.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Union

STATE_COUNT = 22


class LogValidationError(ValueError):
    """A malformed log or a failed state-log sanity check."""


@dataclass(frozen=True)
class StateSample:
    time: float
    states: Tuple[float, ...]
    line_number: int = 0

    @property
    def quaternion(self) -> Tuple[float, ...]:
        """Scalar-first quaternion (q0, q1, q2, q3)."""
        return self.states[:4]

    @property
    def velocity_ned(self) -> Tuple[float, ...]:
        """North/east/down velocity in metres per second."""
        return self.states[4:7]

    @property
    def position_ned(self) -> Tuple[float, ...]:
        """North/east/down position in metres."""
        return self.states[7:10]


def parse_state_data(lines: Iterable[str]) -> List[StateSample]:
    """Read whitespace-separated numeric rows without discarding endpoints.

    Blank lines are ignored, but comments, headers and ragged rows are errors.
    NaN/Inf tokens can be parsed; semantic validation rejects them separately.
    """
    samples = []
    for line_number, line in enumerate(lines, 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != STATE_COUNT + 1:
            raise LogValidationError(
                "line %d: expected 23 fields (time + 22 states), got %d"
                % (line_number, len(fields)))
        try:
            values = tuple(float(field) for field in fields)
        except ValueError as exc:
            raise LogValidationError("line %d: nonnumeric field" % line_number) from exc
        samples.append(StateSample(values[0], values[1:], line_number))
    if not samples:
        raise LogValidationError("state log is empty")
    return samples


def read_state_data(path: Union[str, Path]) -> List[StateSample]:
    """Read a UTF-8 state log; missing/unreadable paths raise OSError."""
    with open(path, encoding="utf-8") as stream:
        return parse_state_data(stream)


@dataclass(frozen=True)
class LogSummary:
    sample_count: int
    state_count: int
    start_time: float
    end_time: float
    max_quaternion_norm_error: float


def validate_state_data(samples: Iterable[StateSample], *,
                        quaternion_tolerance: float = 1e-3) -> LogSummary:
    """Require finite states, strictly increasing time and a near-unit quaternion.

    The absolute norm tolerance is a sanity check, not an attitude accuracy
    threshold. No normalization, sorting, interpolation or silent repair occurs.
    A single sample is valid but says nothing about time progression.
    """
    if not math.isfinite(quaternion_tolerance) or quaternion_tolerance < 0:
        raise ValueError("quaternion_tolerance must be finite and nonnegative")
    count = 0
    start = previous = None
    max_error = 0.0
    for count, sample in enumerate(samples, 1):
        label = "line %d" % (sample.line_number or count)
        if len(sample.states) != STATE_COUNT:
            raise LogValidationError(label + ": expected 22 states")
        if not all(math.isfinite(value) for value in (sample.time,) + tuple(sample.states)):
            raise LogValidationError(label + ": time and all states must be finite")
        if previous is not None and sample.time <= previous:
            raise LogValidationError(label + ": time must be strictly increasing")
        error = abs(math.hypot(*sample.quaternion) - 1.0)
        if error > quaternion_tolerance:
            raise LogValidationError(label + ": quaternion norm error %.9g exceeds %.9g"
                                     % (error, quaternion_tolerance))
        max_error = max(max_error, error)
        if start is None:
            start = sample.time
        previous = sample.time
    if count == 0:
        raise LogValidationError("state log is empty")
    assert start is not None and previous is not None
    return LogSummary(count, STATE_COUNT, start, previous, max_error)
