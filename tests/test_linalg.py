"""Known-answer linear algebra contracts for small reference filters."""
import math
import unittest
from inertialnav import linalg as la

class LinearAlgebraTests(unittest.TestCase):
    def test_matrix_products_and_input_validation(self):
        self.assertEqual(la.matmul([[1, 2], [3, 4]], [[2], [1]]), [[4], [10]])
        self.assertEqual(la.matvec([[1, 2], [3, 4]], [2, 1]), [4, 10])
        self.assertEqual(la.transpose([[1, 2]]), [[1], [2]])
        self.assertEqual(la.add([[1, 2]], [[3, 4]]), [[4, 6]])
        self.assertEqual(la.identity(2), [[1, 0], [0, 1]])
        self.assertEqual(la.outer([1, 2], [3]), [[3], [6]])
        for bad in ([], [[1], [2, 3]], [[float('nan')]], [[float('inf')]]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                la.matrix(bad)
        with self.assertRaises(ValueError):
            la.matmul([[1, 2]], [[1, 2]])
        with self.assertRaises(ValueError):
            la.add([[1, 2]], [[3]])
        with self.assertRaises(ValueError):
            la.identity(0)
        with self.assertRaises(ValueError):
            la.matvec([[1, 2]], [1])

class CovarianceTests(unittest.TestCase):
    def test_cholesky_solution_and_semidefinite_covariance(self):
        a = [[4, 2], [2, 3]]
        for value in la.solve_spd(a, [6, 5]):
            self.assertAlmostEqual(value, 1.0)
        self.assertEqual(la.covariance([[0, 0], [0, 2]]), [[0, 0], [0, 2]])
        for bad in ([[1, 2], [2, 1]], [[1, 1], [0, 1]], [[-1]], [[0, 1], [1, 2]]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                la.covariance(bad)
        for bad in ([[0]], [[1, 1], [1, 1]]):
            with self.assertRaises(ValueError):
                la.solve_spd(bad, [1]*len(bad))
        original = [[2, 0], [0, 1]]
        copied = la.covariance(original)
        copied[0][0] = 9
        self.assertEqual(original[0][0], 2)


class MixedScaleRegressionTests(unittest.TestCase):
    def test_covariance_tolerances_do_not_hide_indefinite_small_blocks(self):
        for bad in ([[1, 2, 0], [2, 1, 0], [0, 0, 1e14]],
                    [[-1e-20, 0], [0, 1e14]],
                    [[1e-20, 1e-10, 0], [0, 1e-20, 0], [0, 0, 1e14]]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                la.covariance(bad)
        solved = la.solve_spd([[1e-200, 0], [0, 1e200]], [1e-200, 1e200])
        for value in solved:
            self.assertAlmostEqual(value, 1)
        with self.assertRaises(ValueError):
            la.vector('123')
