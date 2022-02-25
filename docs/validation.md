# Navigation toolkit validation

Validation was performed locally on 2026-10-15, separately from the requested
historical git timestamps. No flight-accuracy or certification claim is made.

## Executed gates

- `make verify`: **148 unittest tests passed**; package/tests/examples compiled;
  `git diff --check` passed; both documented executable examples completed.
- Host compiler: Apple Clang 21.0.0, C++11; Python 3.9.6.
- The native build regression uses `-Werror=array-bounds`, copies sources into a
  temporary directory, and replays the real bundled `InputFilesPX4.zip` dataset.
- The original C++ replay completed with exit 0. Its actual StateDataOut log
  validated **12,165 samples**, time **1.003–89.997 seconds**, 22 states per row,
  maximum quaternion norm error **1.88602067e-7**. Every diagnostic down-height
  sample is also checked against RefVelPosDataOut; no out-of-bounds posNE read.
- Negative cases verify missing/empty/malformed/undecodable logs and CSV, nonfinite
  measurements, covariance failures, wrong shapes, time reversal, rejected fixes,
  stale aiding, large sample gaps and proper nonzero CLI exits.

## Independent numerical review follow-ups

A bounded read-only reviewer reproduced two bugs, then regression tests were run
red before the repairs and green afterward:

1. Accumulated addition drift rejected valid 100 Hz timestamped propagation, and
   Unix-epoch subtraction lost precision. `predict_to` now uses clock-resolution-
   aware interval validation and retains the observed target timestamp. A 10,000-
   sample 100 Hz regression and an epoch-time regression both pass.
2. A global covariance tolerance allowed an unrelated huge diagonal to mask an
   indefinite small block. Validation now equilibrates axes before Cholesky,
   checks correlation-scale symmetry/PSD, and always rejects negative diagonals.
   Mixed-scale indefinite blocks and 1e-200/1e200 SPD solve tests pass.

Native compilation also exposed an out-of-bounds diagnostic access to the third
axis of a two-axis `posNE` array. The writer now logs measured down-height, the
plotting convention is corrected with it, and the compiler regression forbids
array-bounds warnings. Optional FILE pointers use portable null comparisons.

## Synthetic experiment (seed 7)

- 600 simulated samples with accelerometer bias, a six-second GPS dropout and a
  single 250 m bad position fix.
- Aided 3D position RMSE: **1.5385033536675397 m**.
- Dead-reckoned 3D position RMSE: **8.395168727115685 m**.
- Final position error: **0.8415509267304847 m**.
- Rejected position fixes: **1**.

These numbers are computed against this explicitly synthetic trajectory only.
No reference trajectory was invented for the historical flight data.

## Remaining limitations

The toolkit is an offline local-frame reference with loosely coupled translation
and heading, not a full attitude/bias error-state EKF. No MATLAB/Simulink run,
real-time/HIL qualification or new flight test was performed. Legacy C++ warnings
about an unsupported optimization flag and constructor member initialization order
remain. Historical archives use multiple state formats/startup conventions; the
validator intentionally does not silently normalize or guess incompatible input.
See [the toolkit guide](navigation-toolkit.md) and [analysis contracts](analysis.md).
