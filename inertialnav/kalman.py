"""Small linear Kalman filter with explicit process and measurement models."""
import math
from dataclasses import dataclass
from . import linalg as la


class LinearKalman:
    """x and P are state/covariance copies; failed operations do not mutate them."""
    def __init__(self, state, covariance):
        self.x = la.vector(state)
        self.P = la.covariance(covariance, len(self.x))

    def predict(self, transition, process_noise, control=None, control_matrix=None):
        n = len(self.x)
        f = la.matrix(transition, (n, n))
        q = la.covariance(process_noise, n)
        x = la.matvec(f, self.x)
        if (control is None) != (control_matrix is None):
            raise ValueError("provide both control and control_matrix")
        if control is not None:
            u = la.vector(control)
            b = la.matrix(control_matrix, (n, len(u)))
            change = la.matvec(b, u)
            x = la.vector([value+delta for value, delta in zip(x, change)])
        p = la.covariance(la.add(la.matmul(la.matmul(f, self.P), la.transpose(f)), q), n)
        self.x, self.P = x, p
        return list(x)

    def update(self, measurement, observation, measurement_noise, gate=None):
        """Correct using a solve, never a matrix inverse, and Joseph covariance."""
        if gate is not None and (not math.isfinite(gate) or gate <= 0):
            raise ValueError("NIS gate must be finite and positive")
        z = la.vector(measurement)
        m, n = len(z), len(self.x)
        h = la.matrix(observation, (m, n))
        r = la.covariance(measurement_noise, m)
        la.solve_spd(r, [0.0]*m)
        hx = la.matvec(h, self.x)
        innovation = la.vector([value-pred for value, pred in zip(z, hx)])
        pht = la.matmul(self.P, la.transpose(h))
        s = la.covariance(la.add(la.matmul(h, pht), r), m)
        weighted = la.solve_spd(s, innovation)
        nis = math.fsum(a*b for a, b in zip(innovation, weighted))
        if not math.isfinite(nis):
            raise ValueError("innovation statistic overflow")
        accepted = gate is None or nis <= gate
        result = UpdateResult(accepted, tuple(innovation), tuple(tuple(row) for row in s), nis)
        if not accepted:
            return result
        gain = [la.solve_spd(s, row) for row in pht]
        delta = la.matvec(gain, innovation)
        x = la.vector([value+change for value, change in zip(self.x, delta)])
        kh = la.matmul(gain, h)
        a = [[float(i == j)-kh[i][j] for j in range(n)] for i in range(n)]
        p = la.add(la.matmul(la.matmul(a, self.P), la.transpose(a)),
                   la.matmul(la.matmul(gain, r), la.transpose(gain)))
        p = la.covariance(p, n)
        self.x, self.P = x, p
        return result


    # CLASS_INCREMENT


@dataclass(frozen=True)
class UpdateResult:
    accepted: bool
    innovation: tuple
    innovation_covariance: tuple
    nis: float


# NEXT_INCREMENT
