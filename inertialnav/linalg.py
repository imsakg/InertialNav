"""Validated small dense matrices using only the Python standard library.

These routines favor reviewable formulas over throughput. Inputs and return values
are copied lists; dimensions and finite values are checked at every boundary.
"""
import math


def vector(values, size=None):
    try:
        result = [float(value) for value in values]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("expected a finite numeric vector") from exc
    if not result or (size is not None and len(result) != size):
        raise ValueError("vector has the wrong size")
    if not all(math.isfinite(value) for value in result):
        raise ValueError("vector must be finite")
    return result


def matrix(rows, shape=None):
    try:
        result = [vector(row) for row in rows]
    except TypeError as exc:
        raise ValueError("expected a rectangular matrix") from exc
    if not result or any(len(row) != len(result[0]) for row in result):
        raise ValueError("matrix must be nonempty and rectangular")
    if shape is not None and (len(result), len(result[0])) != shape:
        raise ValueError("matrix has the wrong shape")
    return result


def identity(size):
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("identity size must be a positive integer")
    return [[float(i == j) for j in range(size)] for i in range(size)]


def transpose(a):
    return [list(column) for column in zip(*matrix(a))]


def matmul(a, b):
    a, b = matrix(a), matrix(b)
    if len(a[0]) != len(b):
        raise ValueError("incompatible matrix dimensions")
    return matrix([[math.fsum(x*y for x, y in zip(row, col))
                    for col in zip(*b)] for row in a])


def matvec(a, x):
    a = matrix(a)
    x = vector(x, len(a[0]))
    return vector([math.fsum(u*v for u, v in zip(row, x)) for row in a])


def add(a, b):
    a = matrix(a)
    b = matrix(b, (len(a), len(a[0])))
    return matrix([[x+y for x, y in zip(ar, br)] for ar, br in zip(a, b)])


def outer(a, b):
    a, b = vector(a), vector(b)
    return matrix([[x*y for y in b] for x in a])


# NEXT_INCREMENT
