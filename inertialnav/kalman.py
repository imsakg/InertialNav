"""Small linear Kalman filter with explicit process and measurement models."""
import math
from dataclasses import dataclass
from . import linalg as la


class LinearKalman:
    """x and P are state/covariance copies; failed operations do not mutate them."""
    def __init__(self, state, covariance):
        self.x = la.vector(state)
        self.P = la.covariance(covariance, len(self.x))

    def predict(self, transition, process_noise):
        n = len(self.x)
        f = la.matrix(transition, (n, n))
        q = la.covariance(process_noise, n)
        x = la.matvec(f, self.x)
        p = la.covariance(la.add(la.matmul(la.matmul(f, self.P), la.transpose(f)), q), n)
        self.x, self.P = x, p
        return list(x)

    # CLASS_INCREMENT


# NEXT_INCREMENT
