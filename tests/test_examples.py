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

# NEXT_INCREMENT
