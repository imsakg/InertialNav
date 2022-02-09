"""Replay timestamped SI-unit body IMU CSV and optional co-located NED aids.

Required fields: time,gx,gy,gz,ax,ay,az. Optional complete groups are pn,pe,pd
and vn,ve,vd. An entire optional group may be blank on a row. Acceleration is
specific force (level stationary az=-9.80665), not gravity-removed acceleration.
"""
import argparse
import csv
import math
import sys
from contextlib import nullcontext
from inertialnav.imu import ImuSample, integrate_pair, StrapdownState, _positive
from inertialnav.fusion import NavigationFilter
from inertialnav.rotations import from_euler

BASE = ('time', 'gx', 'gy', 'gz', 'ax', 'ay', 'az')
POSITION = ('pn', 'pe', 'pd')
VELOCITY = ('vn', 've', 'vd')
OUTPUT = ('time', 'pn', 'pe', 'pd', 'vn', 've', 'vd', 'qw', 'qx', 'qy', 'qz')


def _optional(row, names):
    if names[0] not in row:
        return None
    values = [row[name] for name in names]
    if all(value is not None and not value.strip() for value in values):
        return None
    if any(value is None or not value.strip() for value in values):
        raise ValueError('partial aiding group: '+','.join(names))
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError('aiding must be finite')
    return result


def replay(stream, position_sigma=2.5, velocity_sigma=0.2, max_step=0.2,
           initial_attitude=(1, 0, 0, 0)):
    position_sigma = _positive(position_sigma, 'position_sigma')
    velocity_sigma = _positive(velocity_sigma, 'velocity_sigma')
    max_step = _positive(max_step, 'max_step')
    reader = csv.DictReader(stream)
    fields = reader.fieldnames or []
    if len(set(fields)) != len(fields) or not set(BASE).issubset(fields) or set(fields)-set(BASE+POSITION+VELOCITY):
        raise ValueError('invalid CSV header; expected '+','.join(BASE)+' and optional NED aiding')
    for group in (POSITION, VELOCITY):
        if set(group).intersection(fields) and not set(group).issubset(fields):
            raise ValueError('header has a partial aiding group')
    nav = previous = None
    output = []
    position_cov = [[position_sigma**2*float(i == j) for j in range(3)] for i in range(3)]
    velocity_cov = [[velocity_sigma**2*float(i == j) for j in range(3)] for i in range(3)]
    for row in reader:
        try:
            if None in row or any(value is None for value in row.values()):
                raise ValueError('ragged CSV row')
            values = [float(row[name]) for name in BASE]
            sample = ImuSample(values[0], values[1:4], values[4:7])
            position, velocity = _optional(row, POSITION), _optional(row, VELOCITY)
            if nav is None:
                state = StrapdownState(initial_attitude, position or (0, 0, 0), velocity or (0, 0, 0))
                diagonal = ([position_sigma**2]*3 if position else [100.0]*3)+([velocity_sigma**2]*3 if velocity else [1.0]*3)
                covariance = [[diagonal[i]*float(i == j) for j in range(6)] for i in range(6)]
                nav = NavigationFilter(state=state, covariance=covariance, time=sample.time, max_step=max_step)
            else:
                nav.predict_to(sample.time, integrate_pair(previous, sample, max_gap=max_step))
                if position is not None:
                    nav.update_position(position, position_cov, timestamp=sample.time)
                if velocity is not None:
                    nav.update_velocity(velocity, velocity_cov, timestamp=sample.time)
            output.append((sample.time,)+nav.state.position+nav.state.velocity+nav.attitude)
            previous = sample
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError('line %d: %s' % (reader.line_num, exc)) from exc
    if not output:
        raise ValueError('CSV contains no samples')
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    parser.add_argument('--output', help='CSV output path; stdout if omitted')
    parser.add_argument('--position-sigma', type=float, default=2.5)
    parser.add_argument('--velocity-sigma', type=float, default=0.2)
    parser.add_argument('--max-step', type=float, default=0.2)
    parser.add_argument('--roll-deg', type=float, default=0)
    parser.add_argument('--pitch-deg', type=float, default=0)
    parser.add_argument('--yaw-deg', type=float, default=0)
    args = parser.parse_args(argv)
    try:
        attitude = from_euler(*(math.radians(value) for value in (args.roll_deg, args.pitch_deg, args.yaw_deg)))
        with open(args.input, newline='', encoding='utf-8') as stream:
            rows = replay(stream, args.position_sigma, args.velocity_sigma, args.max_step, attitude)
        context = open(args.output, 'w', newline='', encoding='utf-8') if args.output else nullcontext(sys.stdout)
        with context as stream:
            writer = csv.writer(stream)
            writer.writerow(OUTPUT)
            writer.writerows(rows)
    except (OSError, ValueError, OverflowError) as exc:
        print('Invalid replay input: '+str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
