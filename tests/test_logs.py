"""Tests of the source-derived, headerless 22-state C++ text format."""

import io
import tempfile
import unittest
from pathlib import Path

from inertialnav import logs


def state_line(time=0.0, states=None):
    if states is None:
        states = [1.0] + [0.0] * 21
    return " " + " ".join("%e" % value for value in [time] + list(states)) + "\n"


class StateReaderTests(unittest.TestCase):
    def test_headerless_scientific_notation_and_named_state_groups(self):
        states = [1.0] + [float(i) for i in range(1, 22)]
        rows = logs.parse_state_data(io.StringIO("\n" + state_line(1.25, states) + state_line(2.5)))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].time, 1.25)
        self.assertEqual(rows[0].states, tuple(states))
        self.assertEqual(rows[0].quaternion, (1.0, 1.0, 2.0, 3.0))
        self.assertEqual(rows[0].velocity_ned, (4.0, 5.0, 6.0))
        self.assertEqual(rows[0].position_ned, (7.0, 8.0, 9.0))
        self.assertEqual(rows[0].line_number, 2)
        self.assertEqual(rows[-1].time, 2.5)  # no header/footer is discarded

    def test_tabs_and_final_line_without_newline(self):
        text = state_line().strip().replace(" ", "\t")
        self.assertEqual(len(logs.parse_state_data([text])), 1)

    def test_rejects_empty_ragged_wrong_count_and_nonnumeric_rows(self):
        cases = ["", "\n \t\n", "time q0 q1\n", " ".join(["0"] * 22),
                 " ".join(["0"] * 24), state_line() + "0 1", state_line().replace("1.000000e+00", "oops")]
        for text in cases:
            with self.subTest(text=text), self.assertRaises(logs.LogValidationError):
                logs.parse_state_data(io.StringIO(text))

    def test_error_reports_physical_line(self):
        with self.assertRaisesRegex(logs.LogValidationError, "line 3.*23"):
            logs.parse_state_data([state_line(), "\n", "0 1\n"])

    def test_file_reader_and_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "StateDataOut.txt"
            path.write_text(state_line(), encoding="utf-8")
            self.assertEqual(logs.read_state_data(path)[0].time, 0.0)
            with self.assertRaises(OSError):
                logs.read_state_data(Path(tmp) / "missing")


class StateValidationTests(unittest.TestCase):
    def validate(self, text, **kwargs):
        return logs.validate_state_data(logs.parse_state_data(io.StringIO(text)), **kwargs)

    def test_summary_counts_every_sample_and_bounds_quaternion_norm_error(self):
        q = [1.0002] + [0.0] * 21
        summary = self.validate(state_line(4) + state_line(5, q) + state_line(6))
        self.assertEqual(summary.sample_count, 3)
        self.assertEqual(summary.state_count, 22)
        self.assertEqual((summary.start_time, summary.end_time), (4, 6))
        self.assertAlmostEqual(summary.max_quaternion_norm_error, 0.0002)

    def test_empty_and_wrong_state_count_are_errors_for_direct_callers(self):
        for samples in [[], [logs.StateSample(0, (1, 0, 0, 0))]]:
            with self.subTest(samples=samples), self.assertRaises(logs.LogValidationError):
                logs.validate_state_data(samples)

    def test_nonfinite_time_and_each_state_are_rejected(self):
        for value in [float("nan"), float("inf"), -float("inf")]:
            with self.subTest(value=value), self.assertRaisesRegex(logs.LogValidationError, "finite"):
                self.validate(state_line(value))
            for axis in range(22):
                states = [1.0] + [0.0] * 21
                states[axis] = value
                with self.subTest(value=value, axis=axis), self.assertRaisesRegex(logs.LogValidationError, "finite"):
                    self.validate(state_line(0, states))

    def test_duplicate_and_decreasing_times_report_the_row(self):
        for time in [1.0, 0.5]:
            with self.subTest(time=time), self.assertRaisesRegex(logs.LogValidationError, "line 2.*strictly increasing"):
                self.validate(state_line(1) + state_line(time))

    def test_quaternion_check_uses_norm_and_configurable_tolerance(self):
        for q in [0.0, 2.0, 1.0011, 1e308]:
            with self.subTest(q=q), self.assertRaisesRegex(logs.LogValidationError, "quaternion"):
                self.validate(state_line(0, [q] + [0.0] * 21))
        self.validate(state_line(0, [-1] + [0] * 21))
        self.validate(state_line(0, [1.01] + [0] * 21), quaternion_tolerance=0.02)
        for tolerance in [-1, float("nan"), float("inf")]:
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                self.validate(state_line(), quaternion_tolerance=tolerance)
