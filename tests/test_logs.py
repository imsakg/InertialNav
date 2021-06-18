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
