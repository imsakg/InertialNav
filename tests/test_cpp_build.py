"""Compile the original replay harness without polluting its source directory."""
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(shutil.which('c++') and shutil.which('make'), 'C++ compiler and make are required')
class OriginalHarnessBuildTests(unittest.TestCase):
    def test_cpp11_harness_builds_with_native_compiler(self):
        source = Path(__file__).resolve().parents[1]/'code'
        with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as directory:
            target = Path(directory)
            for path in source.iterdir():
                if path.suffix in ('.cpp', '.h') or path.name == 'Makefile':
                    shutil.copy2(path, target/path.name)
            result = subprocess.run(['make', '-C', str(target), '-j2', 'CXX=c++ -Werror=array-bounds'], capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertTrue((target/'estimator_closed_loop_test').is_file())
            import zipfile
            from inertialnav.logs import read_state_data, validate_state_data
            with zipfile.ZipFile(source/'InputFilesPX4.zip') as archive:
                for name in archive.namelist():
                    if Path(name).name != name or name.endswith('/'):
                        raise ValueError('unexpected archive path: '+name)
                    archive.extract(name, target)
            replay = subprocess.run([str(target/'estimator_closed_loop_test')], cwd=target,
                                    capture_output=True, text=True, timeout=120)
            self.assertEqual(replay.returncode, 0, replay.stdout+replay.stderr)
            summary = validate_state_data(read_state_data(target/'StateDataOut.txt'))
            self.assertGreater(summary.sample_count, 10000)
            glitch = (target/'GlitchOffsetOut.txt').read_text().splitlines()
            reference = (target/'RefVelPosDataOut.txt').read_text().splitlines()
            self.assertEqual(len(glitch), len(reference))
            for observed, expected in zip(glitch, reference):
                fields, truth = observed.split(), expected.split()
                self.assertEqual(len(fields), 11)
                self.assertEqual(float(fields[6]), float(truth[6]))
