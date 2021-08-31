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


class ValidatorCliTests(unittest.TestCase):
    def run_cli(self, *args, cwd=None, script=None):
        import subprocess
        import sys
        root = Path(__file__).resolve().parents[1]
        script = script or root / "code" / "validate_output.py"
        return subprocess.run([sys.executable, "-S", str(script)] + list(args),
                              cwd=str(cwd or root), capture_output=True, text=True)

    def test_valid_file_reports_real_count_and_no_accuracy_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "states.txt"
            path.write_text(state_line(0) + state_line(1) + state_line(2))
            result = self.run_cli(str(path), "--expected-rows", "3")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("samples=3", result.stdout)
            self.assertIn("states=22", result.stdout)
            self.assertIn("not an accuracy assessment", result.stdout)
            mismatch = self.run_cli(str(path), "--expected-rows", "4")
            self.assertNotEqual(mismatch.returncode, 0)
            self.assertIn("expected 4 samples, got 3", mismatch.stderr)

    def test_bad_missing_and_empty_logs_exit_nonzero_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "states.txt"
            for text in ["", "0 1", state_line(0) + state_line(0),
                         state_line(float("nan")), state_line(0, [2] + [0] * 21)]:
                path.write_text(text)
                result = self.run_cli(str(path))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("validation failed", result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertNotIn("PASS", result.stdout)
            result = self.run_cli(str(Path(tmp) / "missing"))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("validation failed", result.stderr)

    def test_legacy_default_from_root_and_code_directory(self):
        import shutil
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            fake_root = Path(tmp)
            (fake_root / "code").mkdir()
            (fake_root / "inertialnav").mkdir()
            shutil.copy(root / "code" / "validate_output.py", fake_root / "code")
            shutil.copy(root / "inertialnav" / "__init__.py", fake_root / "inertialnav")
            shutil.copy(root / "inertialnav" / "logs.py", fake_root / "inertialnav")
            (fake_root / "code" / "StateDataOut.txt").write_text(state_line())
            (fake_root / "StateDataOut.txt").write_text("bad")
            for cwd in [fake_root, fake_root / "code"]:
                result = self.run_cli(cwd=cwd, script=fake_root / "code" / "validate_output.py")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("samples=1", result.stdout)

    def test_explicit_relative_path_is_relative_to_working_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "local.txt").write_text(state_line())
            result = self.run_cli("local.txt", cwd=Path(tmp))
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_options_fail(self):
        for args in [("--quaternion-tolerance", "nan"), ("--expected-rows", "0")]:
            result = self.run_cli(*args)
            self.assertNotEqual(result.returncode, 0)


class JsonValidationIntegrationTests(unittest.TestCase):
    run_cli = ValidatorCliTests.run_cli

    def test_json_report_agrees_with_reader_and_preserves_endpoints(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "StateDataOut.txt"
            path.write_text(state_line(0) + state_line(0.01) + state_line(0.02))
            result = self.run_cli(str(path), "--json", "--expected-rows", "3")
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            summary = logs.validate_state_data(logs.read_state_data(path))
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["sample_count"], summary.sample_count)
            self.assertEqual(report["state_count"], 22)
            self.assertEqual((report["start_time"], report["end_time"]), (0, 0.02))
            self.assertEqual(report["max_quaternion_norm_error"], 0)
            self.assertIs(report["accuracy_assessed"], False)
            self.assertEqual(result.stderr, "")

    def test_json_failure_is_parseable_and_never_claims_accuracy(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "StateDataOut.txt"
            for content in [state_line(0) + state_line(0), "", state_line(0, [0] * 22)]:
                path.write_text(content)
                result = self.run_cli(str(path), "--json")
                self.assertNotEqual(result.returncode, 0)
                report = json.loads(result.stdout)
                self.assertEqual(report["status"], "fail")
                self.assertIs(report["accuracy_assessed"], False)
                self.assertTrue(report["error"])
                self.assertNotIn("sample_count", report)

    def test_decoding_failures_are_handled_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "StateDataOut.txt"
            path.write_bytes(b"\xff\xfe")
            result = self.run_cli(str(path))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("validation failed", result.stderr)
            self.assertNotIn("Traceback", result.stderr)


class ArchivedStateRegressionTests(unittest.TestCase):
    run_cli = ValidatorCliTests.run_cli

    def test_archived_current_format_requires_explicit_startup_selection(self):
        import json
        from zipfile import ZipFile
        root = Path(__file__).resolve().parents[1]
        with ZipFile(root / "code" / "ValidationOutputData.zip") as archive:
            text = archive.read("StateDataOut.txt").decode("utf-8")
        rows = logs.parse_state_data(text.splitlines())
        self.assertEqual(len(rows), 40471)
        with self.assertRaisesRegex(logs.LogValidationError, "line 1: quaternion"):
            logs.validate_state_data(rows)
        selected = [row for row in rows if row.time >= 1.0]
        self.assertEqual(len(rows) - len(selected), 50)
        summary = logs.validate_state_data(selected)
        self.assertEqual((summary.sample_count, summary.start_time, summary.end_time), (40421, 1.02, 847.904))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "explicitly-selected-after-1s.txt"
            path.write_text("".join(line for line in text.splitlines(keepends=True)
                                    if float(line.split()[0]) >= 1.0), encoding="utf-8")
            result = self.run_cli(str(path), "--json", "--expected-rows", "40421")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["sample_count"], summary.sample_count)
            self.assertIs(json.loads(result.stdout)["accuracy_assessed"], False)

    def test_different_historical_archive_width_is_not_guessed(self):
        from zipfile import ZipFile
        root = Path(__file__).resolve().parents[1]
        with ZipFile(root / "code" / "OutputFiles.zip") as archive:
            with archive.open("StateDataOut.txt") as stream:
                line = stream.readline().decode("utf-8")
        with self.assertRaisesRegex(logs.LogValidationError, "expected 23.*got 25"):
            logs.parse_state_data([line])
