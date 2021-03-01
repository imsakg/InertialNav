# Navigation reference toolkit implementation plan

> **For Hermes:** Use the subagent-driven-development workflow for isolated math and analysis work; integrate and verify on the main checkout.

**Goal:** Add a dependency-free, testable reference toolkit for navigation experiments and meaningful validation of the existing C++ estimator output.

**Architecture:** Preserve the original C++/MATLAB prototypes and their provenance. Add an offline Python package with explicit SI/NED conventions, small reusable filtering/IMU components, WGS-84/attitude transforms and statistical diagnostics. These are reference and analysis tools, not a certified autopilot or a replacement for the 22-state estimator.

**Tech stack:** Python 3.9+, standard-library unittest, existing C++11 compiler/test harness.

## Boundaries and acceptance

- Do not push. Preserve all 458 existing reachable commits and their metadata.
- Add 55 non-empty, coherent commits. The requested approximate initial count (11) differs from the actual checkout; the resulting reachable count is 513, not 65.
- Date only the newly authored commits in March 2021 through February 2022, with randomized weekday evening author/committer dates; retain the pre-change backup branch.
- Commit dates are user-requested historical metadata, not evidence that this implementation or its tests existed at those dates.
- Observe red tests before each behavior addition; run the integrated suite and executable examples at completion.
- Re-run the original C++ build and replay a provided data archive in an isolated build directory.

## Workstream A: coordinate and attitude transforms (14 commits)

Owned paths: `inertialnav/coordinates.py`, `inertialnav/rotations.py`, `tests/test_coordinates.py`, `tests/test_rotations.py`, `docs/coordinates.md`.

Implement WGS-84 forward/inverse transforms, ECEF/NED/ENU point and vector transforms; scalar-first Hamilton quaternions, Euler conversions, rotation vectors and interpolation. Cover origin, poles, date line, inverse round trips, singularities, finite-input guards and convention examples. Each behavior is tested before implementation and each coherent increment is committed in an isolated worktree.

## Workstream B: error analysis and log validation (14 commits)

Owned paths: `inertialnav/analysis.py`, `inertialnav/logs.py`, `tests/test_analysis.py`, `tests/test_logs.py`, `docs/analysis.md`, `code/validate_output.py`.

Add robust readers for the actual StateDataOut text format, finite/quaternion/time validation, real nonzero-exit CLI checks; RMSE, error summaries, interpolation/alignment, innovation/NIS/NEES, running moments, autocorrelation and Allan-deviation analysis. Test malformed/empty/truncated input and mathematical known answers. No invented reference trajectory or unsupported accuracy claims. Keep the legacy CLI runnable without NumPy.

## Workstream C: filtering, IMU and integrated fusion (27 commits)

Owned paths: package/test bootstrap; `inertialnav/linalg.py`, `kalman.py`, `imu.py`, `fusion.py`, focused tests, `examples/`, package/usage docs and build/test entry points.

1. Bootstrap the reference package and this plan.
2. Add validated dense small-matrix operations and positive-definite solves, with tests first.
3. Implement linear Kalman prediction, Joseph-form corrections, innovation gating, control inputs and missing-data/shape guards.
4. Add constant-velocity motion/noise models and diagnostic innovations.
5. Add timestamped IMU samples, bias/scale calibration, trapezoidal integration, two-sample coning/sculling, low-pass filtering and stationary detection.
6. Propagate body-to-NED attitude with quaternion increments and compensate specific force for gravity; document the local-frame approximations.
7. Fuse independent NED position and velocity aiding, barometric altitude, stationary zero-velocity updates and wrapped heading; preserve state on rejected data.
8. Add deterministic simulation/replay examples and regression scenarios (outliers, dropouts, tilted gravity, covariance positivity).
9. Add usage/tuning/limitations docs, run full tests, inspect the integrated diff and obtain independent review.

## Per-increment execution

Write a focused test of the desired API/known answer; run it and verify an expected missing-behavior failure. Implement the smallest coherent feature, run the focused and current full tests, inspect diff, and commit explicitly named files. Do not stage generated caches, logs or binaries.

## Final gates

- `python3 -m unittest discover -s tests -v`
- `python3 -m compileall -q inertialnav tests examples`
- Execute all documented examples and the actual log validator against a real C++ replay.
- Compile/replay the original C++ estimator in scratch; report warnings and numerical limitations honestly.
- `git diff --check`, clean status, no accidental runtime artifacts, exactly 55 added nonempty commits.
- Use `git commit --date` for authored commits and rebase only the new range to normalize author/committer metadata; programmatically verify month/weekday/evening/order constraints and unchanged original ancestry.
