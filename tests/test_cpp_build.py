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
            result = subprocess.run(['make', '-C', str(target), '-j2'], capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertTrue((target/'estimator_closed_loop_test').is_file())
