"""Deterministic synthetic IMU/GPS experiment with outage and one bad fix.

Run from the repository root: python3 -m examples.simulate
Metrics describe this simulation only, never the original flight datasets.
"""
import argparse
import json
import math
import random
from inertialnav.imu import Delta, GRAVITY, StrapdownState
from inertialnav.fusion import NavigationFilter
from inertialnav.analysis import error_summary


def run_simulation(seed=7, steps=600, dt=0.05):
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not math.isfinite(dt) or not 0 < dt <= 0.2:
        raise ValueError("dt must lie in (0, 0.2] seconds")
    rng = random.Random(seed)
    nav = NavigationFilter(acceleration_psd=0.01, max_step=0.2)
    dead = StrapdownState()
    position, velocity = [0.0]*3, [0.0]*3
    truth, fused, unassisted = [], [], []
    for index in range(1, steps+1):
        time = index*dt
        acceleration = 0.2 if time <= 10 else 0.0
        position[0] += velocity[0]*dt+0.5*acceleration*dt*dt
        velocity[0] += acceleration*dt
        specific_force = (acceleration+0.04+rng.gauss(0, 0.03),
                          rng.gauss(0, 0.03), -GRAVITY+rng.gauss(0, 0.03))
        delta = Delta(dt, [0]*3, [value*dt for value in specific_force])
        dead = dead.propagate(delta)
        nav.predict_to(time, delta)
        # A six-second GPS dropout tests covariance growth and reacquisition.
        if index % 20 == 0 and not 12 <= time <= 18:
            fix = [value+rng.gauss(0, 2.5) for value in position]
            if index == 200:
                fix[0] += 250
            nav.update_position(fix, [[6.25*float(i == j) for j in range(3)] for i in range(3)])
            measured_velocity = [value+rng.gauss(0, 0.2) for value in velocity]
            nav.update_velocity(measured_velocity, [[0.04*float(i == j) for j in range(3)] for i in range(3)])
        truth.append(tuple(position))
        fused.append(nav.state.position)
        unassisted.append(dead.position)
    return {'source': 'synthetic', 'seed': seed, 'samples': steps,
            'position_rmse_m': error_summary(fused, truth).vector_rmse,
            'dead_reckoning_rmse_m': error_summary(unassisted, truth).vector_rmse,
            'final_position_error_m': math.dist(fused[-1], truth[-1]),
            'rejected_position_fixes': nav.aiding_rejections.get('position', 0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(run_simulation(seed=args.seed), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
