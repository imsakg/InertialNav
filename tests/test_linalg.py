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

# NEXT_INCREMENT
