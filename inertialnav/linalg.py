"""Validated small dense matrices using only the Python standard library.

These routines favor reviewable formulas over throughput. Inputs and return values
are copied lists; dimensions and finite values are checked at every boundary.
"""
import math


def vector(values, size=None):
    if isinstance(values, (str, bytes)):
        raise ValueError("expected numeric components, not serialized text")
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


def _scaled_covariance(rows, size=None):
    """Symmetrize within dimensionless tolerance and equilibrate by axis SD."""
    a = matrix(rows)
    n = len(a)
    if len(a[0]) != n or (size is not None and n != size):
        raise ValueError("covariance must be square with the requested size")
    if any(a[i][i] < 0 for i in range(n)):
        raise ValueError("covariance diagonals must be nonnegative")
    scales = [math.sqrt(a[i][i]) for i in range(n)]
    correlation = [[float(i == j and scales[i] > 0) for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i):
            if scales[i] == 0 or scales[j] == 0:
                if a[i][j] != 0 or a[j][i] != 0:
                    raise ValueError("zero-variance axes must have zero cross covariance")
                continue
            lower, upper = (a[i][j]/scales[i])/scales[j], (a[j][i]/scales[i])/scales[j]
            if not math.isfinite(lower) or not math.isfinite(upper) or abs(lower-upper) > 1e-12:
                raise ValueError("covariance must be symmetric in axis-scaled units")
            value = lower/2+upper/2
            if abs(value) > 1+1e-12:
                raise ValueError("cross covariance violates Cauchy-Schwarz")
            correlation[i][j] = correlation[j][i] = value
            a[i][j] = a[j][i] = a[i][j]/2+a[j][i]/2
    return a, correlation, scales


def _cholesky(correlation, definite=False):
    n = len(correlation)
    lower = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1):
            r = correlation[i][j]-math.fsum(lower[i][k]*lower[j][k] for k in range(j))
            if i == j:
                if r < -1e-12 or (definite and r <= 0):
                    raise ValueError("covariance must be positive " + ("definite" if definite else "semidefinite"))
                lower[i][j] = math.sqrt(max(0.0, r))
            elif lower[j][j] > 0:
                lower[i][j] = r/lower[j][j]
            elif abs(r) > 1e-12:
                raise ValueError("singular covariance has nonzero coupling")
    return lower


def covariance(rows, size=None):
    """Validate PSD after axis scaling; tolerate 1e-12 correlation roundoff.

    A large unrelated variance must never mask an indefinite smaller block.
    Negative diagonals are always rejected, regardless of their magnitude.
    """
    a, correlation, unused = _scaled_covariance(rows, size)
    _cholesky(correlation)
    return a


def solve_spd(a, b):
    """Solve A x = b by axis-scaled Cholesky without explicit inversion."""
    a, correlation, scales = _scaled_covariance(a)
    n = len(a)
    b = vector(b, n)
    lower = _cholesky(correlation, definite=True)
    y = [0.0]*n
    for i in range(n):
        y[i] = (b[i]/scales[i]-math.fsum(lower[i][k]*y[k] for k in range(i)))/lower[i][i]
    z = [0.0]*n
    for i in reversed(range(n)):
        z[i] = (y[i]-math.fsum(lower[k][i]*z[k] for k in range(i+1,n)))/lower[i][i]
    return vector([value/scale for value, scale in zip(z, scales)])
