#!/usr/bin/env python3
"""Sanity-check real C++ state logs without NumPy or accuracy claims."""

import argparse
import math
from pathlib import Path
import sys

# Make the legacy script runnable from both repository root and code/.
CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR.parent))
from inertialnav.logs import LogValidationError, read_state_data, validate_state_data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path,
                        default=CODE_DIR / "StateDataOut.txt",
                        help="log path (default: StateDataOut.txt beside this script)")
    parser.add_argument("--quaternion-tolerance", type=float, default=1e-3,
                        help="maximum absolute quaternion norm error (default: 0.001)")
    parser.add_argument("--expected-rows", type=int,
                        help="optionally require this exact positive sample count")
    args = parser.parse_args(argv)
    if not math.isfinite(args.quaternion_tolerance) or args.quaternion_tolerance < 0:
        parser.error("--quaternion-tolerance must be finite and nonnegative")
    if args.expected_rows is not None and args.expected_rows <= 0:
        parser.error("--expected-rows must be positive")
    try:
        summary = validate_state_data(read_state_data(args.path),
                                      quaternion_tolerance=args.quaternion_tolerance)
        if args.expected_rows is not None and summary.sample_count != args.expected_rows:
            raise LogValidationError("expected %d samples, got %d"
                                     % (args.expected_rows, summary.sample_count))
    except (OSError, UnicodeError, ValueError) as exc:
        print("validation failed: %s: %s" % (args.path, exc), file=sys.stderr)
        return 1
    print("PASS: samples=%d states=%d time=[%.9g, %.9g] max_quaternion_norm_error=%.9g"
          % (summary.sample_count, summary.state_count, summary.start_time,
             summary.end_time, summary.max_quaternion_norm_error))
    print("Structural and numerical sanity checks only; not an accuracy assessment.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
