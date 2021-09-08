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


def covariance(rows, size=None):
    """Return a finite symmetric positive-semidefinite covariance copy."""
    a = matrix(rows)
    n = len(a)
    if len(a[0]) != n or (size is not None and n != size):
        raise ValueError("covariance must be square with the requested size")
    scale = max(abs(value) for row in a for value in row)
    tol = 1e-12*scale
    for i in range(n):
        for j in range(i):
            if abs(a[i][j]-a[j][i]) > tol:
                raise ValueError("covariance must be symmetric")
            a[i][j] = a[j][i] = 0.5*a[i][j]+0.5*a[j][i]
    # A rank-deficient covariance is valid; zero Cholesky pivots must
    # have zero residual coupling to all later rows.
    lower = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1):
            residual = a[i][j]-math.fsum(lower[i][k]*lower[j][k] for k in range(j))
            if i == j:
                if residual < -tol:
                    raise ValueError("covariance must be positive semidefinite")
                lower[i][j] = math.sqrt(max(0.0, residual))
            elif lower[j][j] > 0.0:
                lower[i][j] = residual/lower[j][j]
            elif abs(residual) > tol:
                raise ValueError("singular covariance has nonzero coupling")
    return a


def solve_spd(a, b):
    """Solve A x = b by Cholesky, rejecting non-positive-definite A."""
    a = covariance(a)
    n = len(a)
    b = vector(b, n)
    lower = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1):
            r = a[i][j]-math.fsum(lower[i][k]*lower[j][k] for k in range(j))
            if i == j:
                if r <= 0.0:
                    raise ValueError("matrix must be positive definite")
                lower[i][j] = math.sqrt(r)
            else:
                lower[i][j] = r/lower[j][j]
    y = [0.0]*n
    for i in range(n):
        y[i] = (b[i]-math.fsum(lower[i][k]*y[k] for k in range(i)))/lower[i][i]
    x = [0.0]*n
    for i in reversed(range(n)):
        x[i] = (y[i]-math.fsum(lower[k][i]*x[k] for k in range(i+1,n)))/lower[i][i]
    return vector(x)


# NEXT_INCREMENT
