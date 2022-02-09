"""Executable deterministic reference scenarios; all data here is synthetic."""
import unittest
from examples.simulate import run_simulation

class SimulationTests(unittest.TestCase):
    def test_gps_aiding_improves_biased_dead_reckoning(self):
        result = run_simulation(seed=7)
        self.assertEqual(result['samples'], 600)
        self.assertGreaterEqual(result['rejected_position_fixes'], 1)
        self.assertLess(result['position_rmse_m'], result['dead_reckoning_rmse_m'])
        self.assertEqual(result, run_simulation(seed=7))

class ReplayTests(unittest.TestCase):
    def test_csv_replay_and_strict_rows(self):
        import io
        from examples.replay_csv import replay
        text = 'time,gx,gy,gz,ax,ay,az,pn,pe,pd,vn,ve,vd\n0,0,0,0,0,0,-9.80665,10,20,0,0,0,0\n0.1,0,0,0,0,0,-9.80665,,,,,,\n'
        rows = replay(io.StringIO(text))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1][1:4], (10, 20, 0))
        for bad in ('', 'time,gx,gy,gz,ax,ay\n',
                    text+'0.1,0,0,0,0,0,-9.80665,,,,,,\n',
                    text.replace(',,,,,,', ',1,,,,,'),
                    text.replace('-9.80665', 'nan'),
                    text.replace('0.1,0', '2,0')):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                replay(io.StringIO(bad))

    def test_replay_cli_exits_nonzero_on_bad_input(self):
        import os
        import tempfile
        import subprocess
        import sys
        from pathlib import Path
        with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as directory:
            path = Path(directory)/'imu.csv'
            path.write_text('time,gx,gy,gz,ax,ay,az\n0,0,0,0,0,0,-9.80665\n0.1,0,0,0,0,0,-9.80665\n')
            result = subprocess.run([sys.executable, '-m', 'examples.replay_csv', str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(result.stdout.splitlines()), 3)
            path.write_text('invalid')
            result = subprocess.run([sys.executable, '-m', 'examples.replay_csv', str(path)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('invalid', result.stderr.lower())


# NEXT_INCREMENT
