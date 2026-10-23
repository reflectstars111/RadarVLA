"""Measured inference latency plus estimated effective ego-action onset.

Offline sampled geometry only: no closed-loop vehicle execution is available.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import numpy as np

from .evaluation import evaluate_plan, _yaw_from_path, _interpolate
from .geometry import box_distance, _corners
from .execution import (execution_plan_sampler, sample_baseline, sample_executed_plan,
                        prepare_adoption)
from .adoption import AdoptionTransitionError
from .response_clock import response_clock


@dataclass
class ResponseRewardConfig:
    safety_horizon_s: float = 3.
    response_deadline_s: float = 1.
    evaluation_step_s: float = .05
    sustain_s: float = .1
    safety_margin_m: float = .5
    hazard_distance_m: float = 1.
    clearance_improvement_m: float = .25
    brake_threshold_mps2: float = .5
    lateral_threshold_m: float = .3
    initial_velocity_tolerance_mps: float = .5
    max_acceleration_mps2: float = 8.
    max_jerk_mps3: float = 20.
    latency_weight: float = 1.
    comfort_weight: float = .05
    # Zero preserves historical checkpoints and execution semantics. Positive
    # values request a fixed C2 adoption transition, never a longer horizon.
    adoption_transition_s: float = 0.

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == 'adoption_transition_s':
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError('adoption_transition_s must be finite and nonnegative')
            elif not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if self.evaluation_step_s > self.sustain_s or self.sustain_s >= self.safety_horizon_s:
            raise ValueError('Require evaluation_step <= sustain < safety horizon')
        if self.safety_margin_m > self.hazard_distance_m:
            raise ValueError('hazard distance must include the safety margin')


def _plan(xy, times):
    return dict(valid=True, ego_trajectory=xy[1:].tolist(), ego_times_s=times[1:].tolist())


def _metrics(record, xy, times, cfg):
    return evaluate_plan(record, _plan(xy, times), dict(
        safety_margin_m=cfg.safety_margin_m, evaluation_step_s=cfg.evaluation_step_s))


def _road_polygons(record):
    """Union of explicitly linked lane boundaries; never infer missing width."""
    road = record.get('map') or {}
    boundaries = {str(b['id']): np.asarray(b['points'], float)[:, :2]
                  for b in road.get('lane_boundaries', [])}
    polygons = []
    for lane in road.get('lane_centerlines', []):
        left = boundaries.get(str(lane.get('left_boundary_id')))
        right = boundaries.get(str(lane.get('right_boundary_id')))
        if left is None or right is None or min(len(left), len(right)) < 2:
            continue
        # Boundary polylines may be stored in opposite directions.
        same = np.linalg.norm(left[0] - right[0]) + np.linalg.norm(left[-1] - right[-1])
        opposite = np.linalg.norm(left[0] - right[-1]) + np.linalg.norm(left[-1] - right[0])
        polygon = np.vstack((left, right[::-1] if same <= opposite else right))
        area = abs(np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1)
                          - polygon[:, 1] * np.roll(polygon[:, 0], -1))) / 2
        if np.isfinite(polygon).all() and area > 1e-8:
            polygons.append(polygon)
    return polygons


def _inside(point, polygon):
    start, end = polygon, np.roll(polygon, -1, axis=0)
    edge, delta = end - start, point - start
    length = np.sum(edge * edge, axis=-1)
    parameter = np.clip(np.sum(delta * edge, axis=-1) / np.maximum(length, 1e-16), 0., 1.)
    if np.min(np.linalg.norm(point - start - parameter[:, None] * edge, axis=-1)) <= 1e-7:
        return True
    crosses = (start[:, 1] > point[1]) != (end[:, 1] > point[1])
    selected = np.flatnonzero(crosses)
    x = start[selected, 0] + (point[1] - start[selected, 1]) * edge[selected, 0] / edge[selected, 1]
    return bool(np.count_nonzero(x > point[0]) % 2)


def _inside_road(record, xy, polygons):
    # Sample corners, edge midpoints and center against the lane union. This is
    # a discrete footprint check, not a continuous swept-body guarantee.
    for position, yaw in zip(xy, _yaw_from_path(xy)):
        corners = _corners(position, yaw, record['ego']['box_size'])
        probes = np.vstack((corners, (corners + np.roll(corners, -1, axis=0)) / 2, position))
        if any(not any(_inside(point, polygon) for polygon in polygons) for point in probes):
            return False
    return True


def _time_grid(horizon, step, extra=()):
    return np.unique(np.r_[np.arange(0., horizon, step), horizon, extra])


def event_baseline(record, target_agent_id, config=None, *, baseline_plan=None):
    """Threat under the current executing plan, or explicit CV/CTRV fallback.

    Reward-only GT never enters the model prompt or mode routing. A supplied
    baseline is expressed in the observation's current ego frame at time zero.
    """
    cfg = config or ResponseRewardConfig()
    agents = record.get('risk_agents', record.get('agents', []))
    targets = [a for a in agents if a['id'] == target_agent_id and a.get('present_valid', True)]
    if len(targets) != 1:
        raise ValueError('Event requires one present annotated target_agent_id')
    times = _time_grid(cfg.safety_horizon_s, cfg.evaluation_step_s)
    nominal = sample_baseline(record, times, baseline_plan=baseline_plan)
    target_record = {**record, 'agents': targets, 'risk_agents': targets}
    target = _metrics(target_record, nominal, times, cfg)
    full = _metrics(record, nominal, times, cfg)
    distance = target['minimum_distance_m']
    known = target['safety_coverage'] == 1. and full['safety_coverage'] == 1.
    hazard = distance is not None and distance < cfg.hazard_distance_m
    return dict(hazard=hazard, coverage_complete=known, target_distance_m=distance,
                times=times, nominal_xy=nominal, target_record=target_record,
                road_available=bool(_road_polygons(record)),
                other_hazard=full['safety_violation'] == 1 and not hazard)


def _target_distances(record, xy, times, headings=None):
    """Observed target clearance for component-level causal attribution.

    Counterfactuals hold ego heading fixed and remove only the longitudinal or
    lateral position change. Thus a beneficial late steer cannot confer credit
    on earlier harmful braking. This is a local geometric ablation, not a
    causal model of how another driver would react in closed loop.
    """
    agent = record.get('risk_agents', record['agents'])[0]
    future = np.asarray(record['future_times_s'], float)
    knots = np.r_[0., future]
    valid = np.r_[agent.get('present_valid', True), agent.get('future_valid', [True] * len(future))]
    positions, observed = _interpolate(knots, np.vstack((agent['position'], agent['future_xy'])), valid, times)
    yaw, yaw_valid = _interpolate(knots, np.unwrap(np.r_[agent['heading'],
        agent.get('future_yaw', [agent['heading']] * len(future))]), valid, times)
    headings = _yaw_from_path(xy) if headings is None else headings
    distance = np.full(len(times), np.nan)
    for index in np.flatnonzero(observed & yaw_valid):
        distance[index] = box_distance(xy[index], headings[index], record['ego']['box_size'],
                                       positions[index], yaw[index], agent['size'])
    return distance


def _first_sustained(times, changed, sustain):
    for index in np.flatnonzero(changed):
        end = int(np.searchsorted(times, times[index] + sustain - 1e-8))
        if end < len(times) and changed[index:end + 1].all():
            return float(times[index])
    return None


def _action_onset(record, target_record, plan, ready, local, cfg, baseline_plan):
    times = ready + local
    actual = sample_executed_plan(record, plan, ready, times, baseline_plan=baseline_plan,
        max_acceleration_mps2=cfg.max_acceleration_mps2, max_jerk_mps3=cfg.max_jerk_mps3)
    nominal = sample_baseline(record, times, baseline_plan=baseline_plan)
    baseline_velocity = sample_baseline(record, times, baseline_plan=baseline_plan, derivative=1)
    baseline_acceleration = sample_baseline(record, times, baseline_plan=baseline_plan, derivative=2)
    acceleration = sample_executed_plan(record, plan, ready, times,
        baseline_plan=baseline_plan, derivative=2,
        max_acceleration_mps2=cfg.max_acceleration_mps2, max_jerk_mps3=cfg.max_jerk_mps3)
    velocity = sample_executed_plan(record, plan, ready, times,
        baseline_plan=baseline_plan, derivative=1,
        max_acceleration_mps2=cfg.max_acceleration_mps2, max_jerk_mps3=cfg.max_jerk_mps3)
    baseline_heading = _yaw_from_path(np.vstack(([0., 0.], nominal)))[1:]
    moving = np.linalg.norm(baseline_velocity, axis=-1) > 1e-8
    baseline_heading[moving] = np.arctan2(baseline_velocity[moving, 1], baseline_velocity[moving, 0])
    tangent = np.column_stack((np.cos(baseline_heading), np.sin(baseline_heading)))
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    offset = actual - nominal
    longitudinal_offset = np.sum(offset * tangent, axis=-1)
    lateral_offset = np.sum(offset * normal, axis=-1)
    relative_acceleration = np.sum((acceleration - baseline_acceleration) * tangent, axis=-1)
    relative_velocity = np.sum((velocity - baseline_velocity) * tangent, axis=-1)
    longitudinal_change = np.abs(relative_acceleration) >= cfg.brake_threshold_mps2
    steering = np.abs(lateral_offset) >= cfg.lateral_threshold_m
    changed_onset = _first_sustained(local, longitudinal_change | steering, cfg.sustain_s)
    headings = _yaw_from_path(np.vstack(([0., 0.], actual)))[1:]
    actual_gap = _target_distances(target_record, actual, times, headings)
    without_longitudinal_change = actual - longitudinal_offset[:, None] * tangent
    without_steer = actual - lateral_offset[:, None] * normal
    longitudinal_gain = actual_gap - _target_distances(target_record, without_longitudinal_change, times, headings)
    steer_gain = actual_gap - _target_distances(target_record, without_steer, times, headings)
    # Only measured positive contribution counts. The full action must also
    # satisfy clearance_improvement_m; this epsilon only rejects numerical noise.
    # Displacement and velocity must agree with the current change's direction:
    # earlier acceleration cannot confer braking credit, or vice versa.
    causal_longitudinal = (longitudinal_change
        & (relative_acceleration * longitudinal_offset > 1e-6)
        & (relative_acceleration * relative_velocity > 1e-6))
    effective_change = ((causal_longitudinal & (longitudinal_gain > 1e-6)) | (steering & (steer_gain > 1e-6)))
    effective_onset = _first_sustained(local, effective_change, cfg.sustain_s)
    return changed_onset, effective_onset


def response_reward(record, plan, target_agent_id, generation_s, *, waiting_s=0.,
                    perception_s=0., config=None, baseline_plan=None):
    """Measure trajectory-change latency, then assess response quality separately.

    Queueing starts at the first detector report. Perception shifts execution
    from observation time but precedes that reaction-time origin. The shared
    execution adapter follows the currently executing baseline until adoption,
    rebases the new plan there, and exposes any explicit SHORT stop continuation.
    The clock reads no future GT. Incomplete future GT cannot certify safety,
    but does not erase the clock or its response-time learning signal.
    """
    cfg = config or ResponseRewardConfig()
    if any(not math.isfinite(x) or x < 0 for x in (generation_s, waiting_s, perception_s)):
        raise ValueError('Latency components must be finite and nonnegative')
    ready = perception_s + waiting_s + generation_s
    wait_cost = cfg.latency_weight * (waiting_s + generation_s) / cfg.response_deadline_s
    preparation_error = None
    try:
        plan = prepare_adoption(record, plan, ready, baseline_plan,
            adoption_transition_s=cfg.adoption_transition_s,
            max_acceleration_mps2=cfg.max_acceleration_mps2, max_jerk_mps3=cfg.max_jerk_mps3)
    except AdoptionTransitionError:
        preparation_error = 'invalid_adoption_transition'
    except (ValueError, TypeError, RuntimeError):
        preparation_error = 'invalid_plan'
    clock = response_clock(record, plan, generation_s, waiting_s=waiting_s,
                           perception_s=perception_s, config=cfg, baseline_plan=baseline_plan)
    base = event_baseline(record, target_agent_id, cfg, baseline_plan=baseline_plan)
    result = dict(reward=-6. - wait_cost, reason='invalid_plan', effective=False, hazard=base['hazard'],
                  generation_s=float(generation_s), waiting_s=float(waiting_s),
                  perception_s=float(perception_s), **clock,
                  effective_action_onset_s=None, effective_reaction_s=None,
                  collision=None, safety_coverage=0., safety_status='unknown',
                  semantics='GT-free trajectory-change clock; separate observed safety and target-effectiveness diagnostics')

    def finish(reason, reward=None):
        if reason in ('unknown_coverage', 'unknown_road') and reward is None:
            # Absence of evidence about the suffix is not an absent response.
            # Known harmless controls still must not learn to always brake.
            if not base['hazard'] and base['coverage_complete'] and not base['other_hazard']:
                reward = (-1. if clock['response_detected'] else 1.) - wait_cost
            else:
                reward = clock['response_time_reward']
        if reason == 'collision':
            result['safety_status'] = 'collision'
        elif reason in ('invalid_plan', 'short_horizon', 'invalid_dynamics', 'invalid_adoption_transition',
                        'road_departure', 'unsafe_clearance'):
            result['safety_status'] = reason
        result.update(reason=reason, reward=float(-6. - wait_cost if reward is None else reward))
        result['quality_adjustment'] = result['reward'] - clock['response_time_reward']
        return result

    # Inspect known execution before checking coverage or malformed output.
    # Missing late GT must never erase an already observed collision.
    if ready > 1e-8:
        prefix_times = _time_grid(ready, cfg.evaluation_step_s)
        prefix_xy = sample_baseline(record, prefix_times, baseline_plan=baseline_plan)
        prefix_metrics = _metrics(record, prefix_xy, prefix_times, cfg)
        if prefix_metrics['collision'] == 1:
            result.update(collision=1, safety_coverage=prefix_metrics['safety_coverage'],
                          minimum_distance_m=prefix_metrics['minimum_distance_m'],
                          evaluated_horizon_s=float(ready))
            return finish('collision')
    if preparation_error:
        return finish(preparation_error)
    if not plan.get('valid') or plan.get('ego_complete') is False:
        return finish('invalid_plan')
    limits = dict(max_acceleration_mps2=cfg.max_acceleration_mps2, max_jerk_mps3=cfg.max_jerk_mps3)
    try:
        sampler, execution = execution_plan_sampler(plan, **limits)
    except (ValueError, TypeError, RuntimeError):
        return finish('invalid_plan')
    if sampler is None:
        return finish('invalid_plan')
    native_horizon = float(execution['native_horizon_s'])
    local_horizon = max(cfg.safety_horizon_s, float(execution['execution_horizon_s']))
    if native_horizon < cfg.safety_horizon_s - 1e-7 and execution.get('fallback') is None:
        return finish('short_horizon')
    adoption_times = np.asarray(plan.get('adoption_times_s', []), float)
    dynamics_times = np.r_[np.asarray(plan.get('dynamics_times_s', []), float),
                           np.asarray(execution.get('dynamics_times_s', []), float)]
    adoption_velocity_errors = np.asarray(plan.get('adoption_velocity_errors_mps', []), float)
    adoption_acceleration_jumps = np.asarray(plan.get('adoption_acceleration_jumps_mps2', []), float)
    if (any(value.ndim != 1 or not np.isfinite(value).all() or (value < 0).any()
            for value in (adoption_times, dynamics_times, adoption_velocity_errors, adoption_acceleration_jumps))
            or (adoption_times > local_horizon).any() or (dynamics_times > local_horizon).any()):
        return finish('invalid_plan')
    knots = np.asarray(plan.get('control_times_s', []), float)
    stop_midpoint = native_horizon + .5 * float(execution.get('stop_duration_s', 0.))
    end = ready + local_horizon
    local = _time_grid(local_horizon, cfg.evaluation_step_s,
                       np.r_[native_horizon, float(execution['execution_horizon_s']),
                             stop_midpoint, knots, adoption_times, dynamics_times])
    times = _time_grid(end, cfg.evaluation_step_s, ready + local)
    xy = sample_executed_plan(record, plan, ready, times, baseline_plan=baseline_plan, **limits)
    nominal = sample_baseline(record, times, baseline_plan=baseline_plan)
    metrics = _metrics(record, xy, times, cfg)
    result.update(collision=metrics['collision'], safety_coverage=metrics['safety_coverage'],
                  minimum_distance_m=metrics['minimum_distance_m'], evaluated_horizon_s=float(end),
                  execution=execution)
    if metrics['collision'] == 1:
        return finish('collision')
    if plan.get('execution_failure'):
        return finish(str(plan['execution_failure']))
    if metrics['safety_violation'] == 1:
        return finish('unsafe_clearance')
    velocity, acceleration, jerk = (sampler(local, derivative=i) for i in (1, 2, 3))
    actual_initial = sample_executed_plan(record, plan, ready, np.array([ready]),
        baseline_plan=baseline_plan, derivative=1, **limits)[0]
    expected_initial = sample_baseline(record, np.array([ready]), baseline_plan=baseline_plan, derivative=1)[0]
    initial_error = float(max(np.linalg.norm(actual_initial - expected_initial),
                              adoption_velocity_errors.max(initial=0.)))
    actual_acceleration = sample_executed_plan(record, plan, ready, np.array([ready]),
        baseline_plan=baseline_plan, derivative=2, **limits)[0]
    expected_acceleration = sample_baseline(record, np.array([ready]), baseline_plan=baseline_plan, derivative=2)[0]
    acceleration_jump = float(max(np.linalg.norm(actual_acceleration - expected_acceleration),
                                  adoption_acceleration_jumps.max(initial=0.)))
    max_accel = float(np.linalg.norm(acceleration, axis=-1).max())
    max_jerk = float(np.linalg.norm(jerk, axis=-1).max())
    result.update(initial_velocity_error_mps=initial_error, max_acceleration_mps2=max_accel,
                  max_jerk_mps3=max_jerk, adoption_acceleration_jump_mps2=acceleration_jump)
    if (initial_error > cfg.initial_velocity_tolerance_mps or max_accel > cfg.max_acceleration_mps2 + 1e-7
            or max_jerk > cfg.max_jerk_mps3 + 1e-7 or acceleration_jump > 1e-5):
        return finish('invalid_dynamics')
    polygons = _road_polygons(record)
    if not polygons:
        return finish('unknown_road')
    if not _inside_road(record, xy, polygons):
        return finish('road_departure')
    if metrics['safety_coverage'] < 1. or not base['coverage_complete']:
        return finish('unknown_coverage')
    if base['other_hazard']:
        return finish('unrelated_hazard')
    result['safety_status'] = 'safe_within_evaluated_horizon'
    changed_onset, onset = _action_onset(record, base['target_record'], plan, ready, local, cfg, baseline_plan)
    result['effective_action_onset_s'] = onset
    result['changed_onset_s'] = changed_onset
    if not base['hazard']:
        return finish('unnecessary_response', -1. - wait_cost) if clock['response_detected'] else finish('safe_control', 1. - wait_cost)
    if onset is None:
        return finish('ineffective_change' if clock['response_detected'] else 'no_response')
    reaction = waiting_s + generation_s + onset
    result['effective_reaction_s'] = reaction
    if reaction > cfg.response_deadline_s:
        return finish('timeout')
    target_metrics = _metrics(base['target_record'], xy, times, cfg)
    target_baseline = _metrics(base['target_record'], nominal, times, cfg)
    if target_baseline['safety_coverage'] < 1.:
        return finish('unknown_coverage')
    improvement = target_metrics['minimum_distance_m'] - target_baseline['minimum_distance_m']
    result['clearance_improvement_m'] = improvement
    if improvement < cfg.clearance_improvement_m:
        return finish('ineffective_change')
    result['effective'] = True
    comfort = cfg.comfort_weight * float(np.mean(np.linalg.norm(jerk, axis=-1))) / cfg.max_jerk_mps3
    return finish('effective_response', clock['response_time_reward'] - comfort)
