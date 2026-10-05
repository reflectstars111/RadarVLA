"""Offline, coverage-aware action metrics in the fixed current-ego frame.

Counterfactual risk supervision and action evaluation deliberately differ: risk
uses nominal ego motion; action metrics use the actual generated trajectory.
No offline metric here is a closed-loop route-completion or safety guarantee.
"""
from __future__ import annotations

import math
import numpy as np
from .geometry import box_distance, _corners


def translational_box_ttc(center_a, heading_a, size_a, velocity_a,
                          center_b, heading_b, size_b, velocity_b, horizon_s=10.):
    """Exact first contact of fixed-heading rectangles under constant velocity.

    Intersects swept separating-axis intervals, rather than dividing range by
    closing speed (which falsely flags lateral near misses). ``None`` means no
    contact within the specified lookahead, not a numeric infinite TTC.
    """
    if not math.isfinite(horizon_s) or horizon_s <= 0:
        raise ValueError('TTC horizon must be finite and positive')
    a, b = _corners(center_a, heading_a, size_a), _corners(center_b, heading_b, size_b)
    relative = np.asarray(velocity_b, float) - np.asarray(velocity_a, float)
    if relative.shape != (2,) or not np.isfinite(relative).all():
        raise ValueError('TTC velocities must be finite 2-vectors')
    entry, exit_time = 0., horizon_s
    for polygon in (a, b):
        for i in (0, 1):
            edge = polygon[i + 1] - polygon[i]
            axis = np.array([-edge[1], edge[0]])
            aa, bb = a @ axis, b @ axis
            speed = float(relative @ axis)
            if abs(speed) < 1e-12:
                if aa.max() < bb.min() or bb.max() < aa.min():
                    return None
                continue
            first, last = sorted(((aa.min() - bb.max()) / speed,
                                  (aa.max() - bb.min()) / speed))
            entry, exit_time = max(entry, first), min(exit_time, last)
            if entry > exit_time + 1e-10:
                return None
    return float(max(entry, 0.)) if entry <= horizon_s else None


def _interpolate(knots, values, valid, times):
    """Linear interpolation only when both enclosing samples are observed."""
    values, valid = np.asarray(values, float), np.asarray(valid, bool)
    result = np.empty((len(times),) + values.shape[1:], float)
    mask = np.zeros(len(times), bool)
    for i, moment in enumerate(times):
        right = int(np.searchsorted(knots, moment))
        if right < len(knots) and abs(knots[right] - moment) < 1e-7:
            result[i], mask[i] = values[right], valid[right]
        elif 0 < right < len(knots):
            left = right - 1
            alpha = (moment - knots[left]) / (knots[right] - knots[left])
            result[i] = (1 - alpha) * values[left] + alpha * values[right]
            mask[i] = valid[left] and valid[right]
        else:
            result[i] = 0.
    return result, mask & np.isfinite(result.reshape(len(times), -1)).all(-1)


def _yaw_from_path(xy):
    """Tangent headings with reverse motion and stationary heading continuity."""
    yaw = np.zeros(len(xy))
    for i, delta in enumerate(np.diff(xy, axis=0), 1):
        if np.linalg.norm(delta) < 1e-8:
            yaw[i] = yaw[i - 1]
            continue
        tangent = math.atan2(delta[1], delta[0])
        # Position alone cannot determine forward vs reverse body heading. Use
        # the branch continuous with current ego yaw=0, explicitly reported.
        candidates = tangent + np.arange(-3, 4) * math.pi
        yaw[i] = candidates[np.argmin(abs(candidates - yaw[i - 1]))]
    return yaw


def _route_coordinate(point, polyline):
    edge = np.diff(polyline, axis=0)
    length = np.linalg.norm(edge, axis=-1)
    if np.any(length <= 1e-8):
        raise ValueError('route centerline requires distinct consecutive points')
    u = np.clip(np.sum((point - polyline[:-1]) * edge, axis=-1) / length ** 2, 0., 1.)
    closest = polyline[:-1] + u[:, None] * edge
    index = int(np.linalg.norm(closest - point, axis=-1).argmin())
    return float(np.r_[0., length.cumsum()][index] + u[index] * length[index])


def spline_plan_sampler(plan):
    """Return the shared differentiable-spline sampler for complete controls.

    Callable ``sample(times, derivative=0)`` returns a numpy [T,2] array. No
    extrapolation, missing-control interpolation, or alternate spline is used.
    ``None`` denotes a legacy waypoint plan or unavailable complete controls.
    """
    controls = plan.get('ego_control_points')
    if controls is None or any(point is None for point in controls):
        return None
    import torch
    from .curves import spline_basis
    points = torch.as_tensor(controls, dtype=torch.float64)
    times = torch.as_tensor(plan.get('control_times_s'), dtype=torch.float64)
    if (points.ndim != 2 or points.shape[1] != 2 or times.shape != points.shape[:1]
            or len(times) < 2 or not torch.isfinite(points).all()
            or abs(float(times[0])) > 1e-8 or not torch.allclose(points[0], torch.zeros(2, dtype=points.dtype), atol=1e-8)):
        raise ValueError('ego spline must start at current ego origin at t=0 with finite [K,2] controls')
    def sample(query, derivative=0):
        return (spline_basis(times, torch.as_tensor(query, dtype=torch.float64), derivative) @ points).numpy()
    sample([0.])  # Validate knot ordering before any metrics are emitted.
    return sample


def evaluate_plan(record, plan, risk_config=None, d_safe=.5):
    """Evaluate one decoded plan against observed GT and recorded route.

    Config keys: ``safety_margin_m``, ``evaluation_step_s`` (default .05),
    ``ttc_horizon_s`` (10), ``brake_threshold_mps2`` (.5), and
    ``hard_brake_threshold_mps2`` (3). Collision is sampled oriented-box contact;
    safety violation also includes the margin. Missing coverage yields None
    unless a collision has actually been observed. TTC instead assumes local
    constant velocity/fixed headings and is labeled independently.
    """
    cfg = dict(risk_config or {})
    margin = float(cfg.get('safety_margin_m', d_safe))
    step = float(cfg.get('evaluation_step_s', .05))
    ttc_horizon = float(cfg.get('ttc_horizon_s', 10.))
    brake_threshold = float(cfg.get('brake_threshold_mps2', .5))
    hard_threshold = float(cfg.get('hard_brake_threshold_mps2', 3.))
    if (not all(math.isfinite(v) for v in (margin, step, ttc_horizon, brake_threshold, hard_threshold))
            or margin < 0 or min(step, ttc_horizon, brake_threshold, hard_threshold) <= 0):
        raise ValueError('invalid action-evaluation configuration')
    names = ('collision', 'safety_violation', 'first_safety_violation_s', 'minimum_distance_m',
             'minimum_ttc_s', 'brake_onset_s', 'hard_brake_fraction', 'hard_brake_event',
             'jerk_rms_mps3', 'initial_velocity_error_mps', 'mean_speed_mps', 'path_length_m', 'forward_progress_m',
             'route_completion', 'route_progress_m', 'ego_ade_m', 'ego_final_horizon_fde_m')
    result = {name: None for name in names}
    result.update(valid_plan=False, planned_horizon_s=0., safety_coverage=0.,
                  ego_point_coverage=0., ego_ade_points=0, ttc_pair_count=0,
                  ttc_contact_pair_count=0, evaluation_step_s=step,
                  ttc_horizon_s=ttc_horizon, hard_brake_threshold_mps2=hard_threshold,
                  semantics='offline sampled oriented boxes; local constant-velocity TTC; no closed-loop claim')
    if not plan.get('valid'):
        return result
    points, time_values = plan.get('ego_trajectory', []), plan.get('ego_times_s')
    if not points or time_values is None:
        return result
    times = np.asarray(time_values, float)
    if (times.ndim != 1 or len(times) != len(points) or not np.isfinite(times).all()
            or np.any(times <= 0) or np.any(np.diff(times) <= 0)):
        raise ValueError('generated plan needs strictly increasing positive ego_times_s')
    xy, known = [], []
    for point in points:
        if point is None:
            xy.append([0., 0.]); known.append(False)
        else:
            point = np.asarray(point, float)
            if point.shape != (2,) or not np.isfinite(point).all():
                raise ValueError('generated ego points must be finite 2-vectors or None')
            xy.append(point); known.append(True)
    xy, known = np.asarray(xy), np.asarray(known, bool)
    horizon = float(times[-1])
    result.update(valid_plan=True, planned_horizon_s=horizon)
    future = np.asarray(record['future_times_s'], float)
    ego = record['ego']
    if future.ndim != 1 or not len(future) or np.any(np.diff(future) <= 0) or future[0] <= 0:
        raise ValueError('record future_times_s must increase strictly from positive time')
    knots, positions, valid = np.r_[0., times], np.vstack(([0., 0.], xy)), np.r_[True, known]
    if 'ego_future' in plan:
        # Do not silently allow two competing action arrays.
        raise ValueError('plan must provide ego_trajectory, not a second ego_future')
    if plan.get('ego_yaw') is not None:
        yaw = np.asarray(plan['ego_yaw'], float)
        if yaw.shape != times.shape or not np.isfinite(yaw).all():
            raise ValueError('ego_yaw must match generated ego times')
        headings = np.unwrap(np.r_[0., yaw])
        result['ego_heading_source'] = 'generated_yaw'
        heading_valid = valid.copy()
    else:
        headings = _yaw_from_path(positions)
        result['ego_heading_source'] = 'trajectory_tangent_continuous_forward_or_reverse_approximation'
        # A missing waypoint cannot define the next incoming tangent.
        heading_valid = valid & np.r_[True, valid[:-1]]

    spline_sample = spline_plan_sampler(plan)
    if spline_sample is not None and abs(float(plan['control_times_s'][-1]) - horizon) > 1e-6:
        raise ValueError('spline control horizon must match generated ego trajectory horizon')

    # Score only recorded GT times covered by valid interpolated action points.
    truth_xy = np.asarray(ego.get('future_xy', np.zeros((len(future), 2))), float)
    truth_valid = np.asarray(ego.get('future_valid', ['future_xy' in ego] * len(future)), bool)
    gt_pred, predicted_valid = _interpolate(knots, positions, valid, future)
    if spline_sample is not None:
        gt_pred[predicted_valid] = spline_sample(future[predicted_valid])
    if truth_xy.shape != (len(future), 2) or truth_valid.shape != future.shape:
        raise ValueError('ego future shape must match record future_times_s')
    truth_valid &= np.isfinite(truth_xy).all(-1)
    scoring = predicted_valid & truth_valid
    result['ego_point_coverage'] = float(scoring.sum() / truth_valid.sum()) if truth_valid.any() else None
    result['ego_ade_points'] = int(scoring.sum())
    if scoring.any():
        error = np.linalg.norm(gt_pred - truth_xy, axis=-1)
        result['ego_ade_m'] = float(error[scoring].mean())
        if scoring[-1]:
            result['ego_final_horizon_fde_m'] = float(error[-1])

    # Finite differences on actual, potentially nonuniform sample times.
    dt = np.diff(knots)
    velocity = np.diff(positions, axis=0) / dt[:, None]
    interval_valid = valid[:-1] & valid[1:]
    midpoints = .5 * (knots[:-1] + knots[1:])
    vel_sequence = np.vstack((ego['velocity'], velocity))
    vel_times = np.r_[0., midpoints]
    accel_dt = np.diff(vel_times)
    acceleration = np.diff(vel_sequence, axis=0) / accel_dt[:, None]
    speed = np.linalg.norm(vel_sequence, axis=-1)
    longitudinal_acceleration = np.diff(speed) / accel_dt
    accel_valid = interval_valid & np.r_[True, interval_valid[:-1]]
    if interval_valid.any():
        covered_duration = dt[interval_valid].sum()
        result['path_length_m'] = float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=-1)[interval_valid]))
        result['mean_speed_mps'] = result['path_length_m'] / float(covered_duration)
    if valid.all():
        result['forward_progress_m'] = float(positions[-1, 0])
    if accel_valid.any():
        deceleration = longitudinal_acceleration < -brake_threshold
        hard = longitudinal_acceleration < -hard_threshold
        if (deceleration & accel_valid).any():
            result['brake_onset_s'] = float(knots[:-1][deceleration & accel_valid][0])
        result['hard_brake_fraction'] = float(np.sum(dt[accel_valid] * hard[accel_valid]) / dt[accel_valid].sum())
        result['hard_brake_event'] = int((hard & accel_valid).any())
    if len(acceleration) > 1:
        acc_times = .5 * (vel_times[:-1] + vel_times[1:])
        jerk = np.diff(acceleration, axis=0) / np.diff(acc_times)[:, None]
        jerk_valid = accel_valid[1:] & accel_valid[:-1]
        if jerk_valid.any():
            result['jerk_rms_mps3'] = float(np.sqrt(np.mean(np.sum(jerk[jerk_valid] ** 2, axis=-1))))

    dense_times = np.unique(np.r_[0., np.arange(step, horizon, step), times, future[future < horizon]])
    ego_xy, ego_valid = _interpolate(knots, positions, valid, dense_times)
    ego_yaw, yaw_valid = _interpolate(knots, headings, heading_valid, dense_times)
    ego_valid &= yaw_valid
    ego_v = np.vstack((ego['velocity'], velocity))[np.minimum(np.searchsorted(knots, dense_times), len(knots) - 1)]
    if spline_sample is not None:
        ego_xy = spline_sample(dense_times)
        ego_v = spline_sample(dense_times, derivative=1)
        # Body yaw is not an independent output: use the branch continuous
        # with current yaw and disclose this approximation in metadata.
        if plan.get('ego_yaw') is None:
            ego_yaw = _yaw_from_path(np.vstack(([0., 0.], np.cumsum(ego_v[1:], axis=0))))
        bounds = np.unique(np.r_[dense_times, plan['control_times_s']])
        durations = np.diff(bounds)
        moments = (bounds[1:] + bounds[:-1]) / 2
        actual_velocity = spline_sample(moments, derivative=1)
        actual_acceleration = spline_sample(moments, derivative=2)
        actual_jerk = spline_sample(moments, derivative=3)
        actual_speed = np.linalg.norm(actual_velocity, axis=-1)
        speed_change = np.sum(actual_velocity * actual_acceleration, axis=-1) / np.maximum(actual_speed, 1e-8)
        braking, hard = speed_change < -brake_threshold, speed_change < -hard_threshold
        result['brake_onset_s'] = float(bounds[:-1][braking][0]) if braking.any() else None
        result['hard_brake_fraction'] = float(np.sum(durations * hard) / horizon)
        result['hard_brake_event'] = int(hard.any())
        result['path_length_m'] = float(np.sum(durations * actual_speed))
        result['mean_speed_mps'] = result['path_length_m'] / horizon
        result['jerk_rms_mps3'] = float(np.sqrt(np.sum(durations * np.sum(actual_jerk ** 2, axis=-1)) / horizon))
        result['initial_velocity_error_mps'] = float(np.linalg.norm(spline_sample([0.], derivative=1)[0] - np.asarray(ego['velocity'])))
        result['trajectory_sampling'] = 'shared natural cubic spline; analytic derivatives; sampled collision checks'
    else:
        result['trajectory_sampling'] = 'piecewise linear waypoint interpolation; finite-difference dynamics'
    distances, time_indices, ttc = [], [], []
    agents = record.get('risk_agents', record.get('agents'))
    if not isinstance(agents, list):
        raise ValueError('evaluation requires annotated agents; [] explicitly means observed empty')
    default_coverage = record.get('schema_version') != 'radar_vla_v2'
    coverage = np.asarray(record.get('tracking_coverage_valid', [default_coverage] * len(future)), bool)
    if coverage.shape != future.shape:
        raise ValueError('tracking_coverage_valid must match record future_times_s')
    coverage_knots = np.r_[0., future]
    coverage_values = np.r_[record.get('agent_supervision_available', record.get('agent_supervision', True)), coverage]
    _, tracking_valid = _interpolate(coverage_knots, np.zeros(len(coverage_knots)), coverage_values, dense_times)
    all_valid = ego_valid & tracking_valid
    for agent in agents:
        agent_knots = np.r_[0., future]
        gt = np.asarray(agent.get('future_xy', np.zeros((len(future), 2))), float)
        gt_mask = np.asarray(agent.get('future_valid', ['future_xy' in agent] * len(future)), bool)
        if gt.shape != (len(future), 2) or gt_mask.shape != future.shape:
            raise ValueError('agent future shape must match record future_times_s')
        gt_mask &= np.isfinite(gt).all(-1)
        agent_positions = np.vstack((agent['position'], gt))
        agent_mask = np.r_[agent.get('present_valid', True), gt_mask]
        observed, observed_mask = _interpolate(agent_knots, agent_positions, agent_mask, dense_times)
        agent_heading = np.unwrap(np.r_[agent['heading'], agent.get('future_yaw', [agent['heading']] * len(future))])
        if agent_heading.shape != agent_knots.shape:
            raise ValueError('agent future_yaw must match record future_times_s')
        observed_yaw, yaw_mask = _interpolate(agent_knots, agent_heading, agent_mask, dense_times)
        observed_mask &= yaw_mask
        agent_velocity = np.vstack((agent['velocity'], np.diff(agent_positions, axis=0) / np.diff(agent_knots)[:, None]))
        velocity_index = np.minimum(np.searchsorted(agent_knots, dense_times), len(agent_knots) - 1)
        observed_velocity = agent_velocity[velocity_index]
        # Velocity from differences requires both bracketing GT positions.
        velocity_valid = np.r_[agent.get('present_valid', True), agent_mask[:-1] & agent_mask[1:]][velocity_index]
        pair_valid = ego_valid & observed_mask
        all_valid &= observed_mask
        for j in np.flatnonzero(pair_valid):
            distance = box_distance(ego_xy[j], ego_yaw[j], ego['box_size'], observed[j], observed_yaw[j], agent['size'])
            distances.append(distance); time_indices.append(j)
            if velocity_valid[j]:
                result['ttc_pair_count'] += 1
                value = translational_box_ttc(ego_xy[j], ego_yaw[j], ego['box_size'], ego_v[j],
                                              observed[j], observed_yaw[j], agent['size'], observed_velocity[j], ttc_horizon)
                if value is not None:
                    ttc.append(value)
    coverage_complete = bool(all_valid.all()) and bool(record.get('agent_supervision_available', record.get('agent_supervision', True)))
    result['safety_coverage'] = float(all_valid.mean()) if record.get('agent_supervision_available', record.get('agent_supervision', True)) else 0.
    if distances:
        distances = np.asarray(distances)
        hit, violation = distances <= 1e-9, (distances <= 1e-9) | (distances < margin)
        result['collision'] = int(hit.any()) if hit.any() or coverage_complete else None
        result['safety_violation'] = int(violation.any()) if violation.any() or coverage_complete else None
        result['minimum_distance_m'] = float(distances.min()) if coverage_complete or hit.any() else None
        if violation.any():
            result['first_safety_violation_s'] = float(dense_times[np.asarray(time_indices)[violation]].min())
    elif coverage_complete:
        result.update(collision=0, safety_violation=0)
    if ttc:
        result['minimum_ttc_s'] = float(min(ttc))
        result['ttc_contact_pair_count'] = len(ttc)
    route = (record.get('map') or {}).get('route')
    if route is not None and valid.all():
        centerline = np.asarray(route.get('centerline', route.get('polyline')), float)
        if centerline.ndim != 2 or centerline.shape[1] not in (2, 3) or len(centerline) < 2 or not np.isfinite(centerline).all():
            raise ValueError('explicit map.route needs a finite centerline polyline [N,2] or [N,3]')
        centerline = centerline[:, :2]  # Ground-plane action evaluation; preserve schema's map height elsewhere.
        start, end = _route_coordinate(np.zeros(2), centerline), _route_coordinate(positions[-1], centerline)
        goal = float(route.get('goal_s_m', np.linalg.norm(np.diff(centerline, axis=0), axis=-1).sum()))
        if not math.isfinite(goal) or goal <= start:
            raise ValueError('route goal must be beyond current projected position')
        result['route_progress_m'] = float(end - start)
        result['route_completion'] = float(np.clip((end - start) / (goal - start), 0., 1.))
    return result


def aggregate_plan_metrics(rows):
    """Aggregate per-scene metrics with explicit observed denominators."""
    fields = ('collision', 'safety_violation', 'first_safety_violation_s', 'minimum_distance_m',
              'minimum_ttc_s', 'brake_onset_s', 'hard_brake_fraction', 'hard_brake_event',
              'jerk_rms_mps3', 'initial_velocity_error_mps', 'mean_speed_mps', 'path_length_m', 'forward_progress_m',
              'route_completion', 'route_progress_m', 'ego_ade_m', 'ego_final_horizon_fde_m',
              'safety_coverage', 'ego_point_coverage', 'planned_horizon_s')
    result = dict(samples=len(rows), valid_plans=sum(bool(row['valid_plan']) for row in rows),
                  semantics='scene means; observed denominators; offline only; TTC mean excludes no-contact cases')
    for field in fields:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        key = {'collision': 'collision_rate', 'safety_violation': 'safety_violation_rate',
               'hard_brake_event': 'hard_brake_frequency'}.get(field, field)
        result[key] = float(np.mean(values)) if values else None
        result[key + '_count'] = len(values)
    result['ttc_checked_pairs'] = sum(row.get('ttc_pair_count', 0) for row in rows)
    result['ttc_contact_pairs'] = sum(row.get('ttc_contact_pair_count', 0) for row in rows)
    return result
