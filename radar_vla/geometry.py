"""Sampled-time counterfactual labels in the fixed current-ego coordinate frame.

These labels are supervision, not continuous-time collision guarantees. Agent
trajectories must describe every relevant tracked agent; untracked entrants are
not invented. Distances are between oriented rectangle boundaries in metres.
"""
from __future__ import annotations

import math
import numpy as np


def nominal_rollout(velocity, yaw_rate, times, deceleration=0.):
    """CV/CTRV rollout; braking stops translation and rotation without reversing."""
    velocity = np.asarray(velocity, dtype=float)
    times = np.asarray(times, dtype=float)
    if velocity.shape != (2,) or not np.isfinite(velocity).all():
        raise ValueError('velocity must be a finite 2-vector')
    if (not np.isfinite(times).all() or np.any(times < 0)
            or not math.isfinite(yaw_rate) or not math.isfinite(deceleration)
            or deceleration < 0):
        raise ValueError('rollout times, yaw rate and deceleration must be finite and valid')
    speed = float(np.linalg.norm(velocity))
    t = np.minimum(times, speed / deceleration) if deceleration > 0 else times
    if speed < 1e-12:
        return np.zeros((*times.shape, 2)), np.zeros_like(times)
    unit = complex(*velocity) / speed
    if abs(yaw_rate) * float(np.max(t, initial=0.)) < 1e-3:
        # Integrate exp(i*w*t) in series near w=0. The closed form divides by
        # w**2, causing metre-scale cancellation errors during near-straight braking.
        integral = np.zeros_like(t, dtype=complex)
        coefficient = 1. + 0.j
        for k in range(8):
            integral += coefficient * (speed * t ** (k + 1) / (k + 1)
                                       - deceleration * t ** (k + 2) / (k + 2))
            coefficient *= 1j * yaw_rate / (k + 1)
        displacement = unit * integral
    else:
        w = float(yaw_rate)
        phase = np.exp(1j * w * t)
        i0 = (phase - 1) / (1j * w)
        i1 = phase * (t / (1j * w) + 1 / w ** 2) - 1 / w ** 2
        displacement = unit * (speed * i0 - deceleration * i1)
    return np.stack((displacement.real, displacement.imag), -1), yaw_rate * t


def _corners(center, heading, size):
    center, size = np.asarray(center, float), np.asarray(size, float)
    if (center.shape != (2,) or size.shape != (2,) or np.any(size <= 0)
            or not np.isfinite(center).all() or not np.isfinite(size).all()
            or not math.isfinite(heading)):
        raise ValueError('box center, heading and positive size must be finite')
    local = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]]) * size / 2
    c, s = math.cos(heading), math.sin(heading)
    return local @ np.array([[c, s], [-s, c]]) + center


def _point_segment_distance(point, start, end):
    edge = end - start
    u = np.clip(np.dot(point - start, edge) / np.dot(edge, edge), 0., 1.)
    return float(np.linalg.norm(point - start - u * edge))


def box_distance(center_a, heading_a, size_a, center_b, heading_b, size_b):
    """Exact Euclidean distance between two oriented rectangles (zero if touching)."""
    a, b = _corners(center_a, heading_a, size_a), _corners(center_b, heading_b, size_b)
    separated = False
    for polygon in (a, b):
        for i in (0, 1):
            edge = polygon[(i + 1) % 4] - polygon[i]
            axis = np.array([-edge[1], edge[0]])
            aa, bb = a @ axis, b @ axis
            if aa.max() < bb.min() or bb.max() < aa.min():
                separated = True
    if not separated:
        return 0.
    return min(_point_segment_distance(p, q[j], q[(j + 1) % 4])
               for pset, q in ((a, b), (b, a)) for p in pset for j in range(4))


def build_risk_labels(record, safety_margin_m=.5, max_deceleration=8.,
                      deceleration_step=.5, distance_cap_m=80., sample_dt_s=.05):
    """Compute cumulative 1/2/3 s conflicts, clipped dmin and grid-search areq.

    Positive conflict observations are valid despite missing later observations;
    absence of conflict requires full available future coverage. areq is masked
    when future coverage is incomplete or no grid value is safe. The zero stored
    with an invalid target is a placeholder, never a supervised zero-braking GT.
    """
    if (not all(math.isfinite(v) for v in (safety_margin_m, max_deceleration,
                                         deceleration_step, distance_cap_m, sample_dt_s))
            or safety_margin_m < 0 or max_deceleration < 0
            or deceleration_step <= 0 or distance_cap_m <= 0 or sample_dt_s <= 0):
        raise ValueError('invalid risk label configuration')
    future = np.asarray(record['future_times_s'], float)
    if (future.ndim != 1 or len(future) == 0 or not np.isfinite(future).all()
            or np.any(np.diff(future) <= 0) or future[0] <= 0):
        raise ValueError('future_times_s must be finite, positive and strictly increasing')
    ego, agents = record['ego'], record.get('risk_agents', record.get('agents', []))
    # Dense sampled interpolation catches conflicts between sparse annotation knots.
    # It is still explicitly sampled-time, not a swept-volume guarantee.
    end = min(3., float(future[-1]))
    times = np.unique(np.r_[np.arange(0., end, sample_dt_s), end, future[future <= 3.],
                             [h for h in (1., 2., 3.) if h <= future[-1]]])
    coverage = np.asarray(record.get('tracking_coverage_valid', [record.get('schema_version') != 'radar_vla_v2'] * len(future)), bool)
    if coverage.shape != future.shape:
        raise ValueError('tracking_coverage_valid must match future_times_s')
    coverage_knots = np.r_[0., future]
    coverage_values = np.r_[record.get('agent_supervision_available', True), coverage]
    dense_coverage = []
    for time in times:
        right = np.searchsorted(coverage_knots, time)
        if right < len(coverage_knots) and abs(coverage_knots[right]-time) < 1e-8:
            dense_coverage.append(coverage_values[right])
        else:
            dense_coverage.append(coverage_values[right-1] and coverage_values[right])
    dense_coverage = np.asarray(dense_coverage, bool)
    trajectories = []
    for agent in agents:
        xy = agent.get('future_xy')
        valid = np.asarray(agent.get('future_valid', [xy is not None] * len(future)), bool)
        if xy is None:
            valid[:] = False
            xy = np.zeros((len(future), 2))
        xy = np.asarray(xy, float)
        headings = np.asarray(agent.get('future_yaw', [agent['heading']] * len(future)), float)
        if xy.shape != (len(future), 2) or valid.shape != future.shape or headings.shape != future.shape:
            raise ValueError('agent future shape must match future_times_s')
        if not np.isfinite(xy).all() or not np.isfinite(headings).all():
            raise ValueError('agent futures must be finite; use future_valid for missing labels')
        knots = np.r_[0., future]
        positions = np.vstack((agent['position'], xy))
        angles = np.unwrap(np.r_[agent['heading'], headings])
        observed = np.r_[agent.get('present_valid', True), valid]
        interp_xy = np.stack([np.interp(times, knots, positions[:, i]) for i in range(2)], -1)
        interp_yaw = np.interp(times, knots, angles)
        interp_valid = []
        for t in times:
            right = np.searchsorted(knots, t)
            if right < len(knots) and abs(knots[right] - t) < 1e-8:
                interp_valid.append(bool(observed[right]))
            else:
                interp_valid.append(bool(observed[right - 1] and observed[right]))
        trajectories.append((agent, interp_xy, interp_yaw, np.asarray(interp_valid)))

    def distances(deceleration):
        ego_xy, ego_yaw = nominal_rollout(ego['velocity'], ego['yaw_rate'], times, deceleration)
        values = np.full((len(agents), len(times)), np.nan)
        for i, (agent, xy, yaw, valid) in enumerate(trajectories):
            for j in np.flatnonzero(valid):
                values[i, j] = box_distance(ego_xy[j], ego_yaw[j], ego['box_size'],
                                             xy[j], yaw[j], agent['size'])
        return values

    def violation(values):
        return (values <= 1e-9) | (values < safety_margin_m)

    nominal = distances(0.)
    pcol, masks = [], []
    for horizon in (1., 2., 3.):
        chosen = times <= horizon
        hit = bool(violation(nominal[:, chosen]).any())
        covered = future[-1] >= horizon and bool(np.isfinite(nominal[:, chosen]).all()) and bool(dense_coverage[chosen].all())
        pcol.append(float(hit))
        masks.append(hit or covered)
    complete = bool(future[-1] >= 3. and np.isfinite(nominal).all() and dense_coverage.all())
    valid_distances = nominal[np.isfinite(nominal)]
    minimum = min(float(valid_distances.min()), distance_cap_m) if valid_distances.size else distance_cap_m
    # Zero observed separation is already the global minimum despite missing data.
    dmin_mask = complete or bool(valid_distances.size and minimum == 0.)
    braking_feasible = None
    areq, areq_mask = 0., False
    already_violating = bool(violation(nominal[:, 0]).any())
    stationary_violation = np.linalg.norm(ego['velocity']) < 1e-12 and bool(violation(nominal).any())
    if already_violating or stationary_violation:
        # Missing later GT cannot make an observed unavoidable violation unknown.
        # Longitudinal braking changes neither the current box nor a stopped ego.
        braking_feasible = False
    elif complete:
        grid = np.unique(np.r_[np.arange(0., max_deceleration + 1e-9, deceleration_step), max_deceleration])
        # Full grid evaluation deliberately makes no monotonicity assumption.
        braking_feasible = False
        for deceleration in grid:
            if not violation(nominal if deceleration == 0 else distances(deceleration)).any():
                areq, areq_mask, braking_feasible = float(deceleration), True, True
                break  # first safe ordered grid point; no monotonicity assumption
    return {'pcol': pcol, 'pcol_mask': masks, 'dmin': minimum, 'dmin_mask': dmin_mask,
            'areq': areq, 'areq_mask': areq_mask, 'braking_feasible': braking_feasible,
            'distance_cap_m': distance_cap_m, 'safety_margin_m': safety_margin_m,
            'max_deceleration_mps2': max_deceleration, 'deceleration_step_mps2': deceleration_step,
            'sample_dt_s': sample_dt_s,
            'label_semantics': 'sampled_time_counterfactual_cv_ctrv; dmin capped; no swept-volume guarantee'}
