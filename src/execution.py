"""Shared deployment clock and explicit delayed plan-execution semantics.

The physical adapter is an offline controller approximation, not a vehicle
controller or a safety certificate. Annotation coverage is never extrapolated.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from .evaluation import spline_plan_sampler, _yaw_from_path
from .geometry import nominal_rollout
from .adoption import (AdoptionTransitionError, quintic_coefficients,
                       quintic_dynamics, sample_quintic)


@torch.no_grad()
def measure_plan(planner, radar, risk, instructions, ids=None, max_new_tokens=1024,
                 future_times_s=None, clock=None, observed_ego=None):
    """Time deployment generation plus decoding, excluding policy diagnostics.

    Supplied ids replay a sampled action through the identical forward/grammar/
    argmax path, with no sampling RNG or probability work in the timing window.
    This is a deployment-equivalent latency estimate for that sampled action;
    offline sampling itself is additional training work, not measured response.
    """
    if planner.training or radar.shape[0] != 1:
        raise ValueError('Deployment timing requires eval mode and batch size one')
    now = time.perf_counter if clock is None else clock
    def sync():
        if radar.device.type == 'cuda':
            torch.cuda.synchronize(radar.device)
    sync()
    start = now()
    parallel = getattr(planner.config, 'action_representation', 'token_spline_v1') == 'parallel_acceleration_v1'
    if parallel:
        if ids is not None:
            raise ValueError('continuous action timing cannot replay legacy token ids')
        plan = planner.plan_action(radar, risk, instructions, observed_ego, future_times_s)[0]
        rows = [[]]
    elif ids is None:
        rows = planner.generate(radar, risk, instructions, max_new_tokens=max_new_tokens)
    else:
        rows = planner._generate_ids(radar, risk, instructions, max_new_tokens,
                                     forced_ids=[ids])
    if not parallel:
        plan = planner.decode(rows[0], future_times_s=future_times_s)
    sync()
    elapsed = float(now() - start)
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError('Deployment clock must advance monotonically')
    return dict(ids=rows[0], plan=plan, generation_s=elapsed, generation_seconds=elapsed)


def _query(times, derivative):
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or not np.isfinite(t).all() or (t < 0).any():
        raise ValueError('Execution times must be finite, nonnegative and one-dimensional')
    if derivative not in (0, 1, 2, 3):
        raise ValueError('Execution derivative must be 0, 1, 2 or 3')
    return t


def execution_plan_sampler(plan, *, max_acceleration_mps2=8., max_jerk_mps3=20.):
    """Return the native spline and, for SHORT, an explicit bounded stop.

    Native natural splines have zero terminal acceleration. A cubic velocity
    easing preserves position, velocity and acceleration at that boundary,
    reaches rest with zero acceleration, and bounds acceleration and jerk.
    No native control or supervised trajectory is enlarged. Native infeasibility
    still has to be rejected by the caller; this does not repair unsafe outputs.
    """
    if not plan.get('valid') or plan.get('ego_complete') is False:
        raise ValueError('Cannot execute an invalid or incomplete plan')
    if any(not math.isfinite(x) or x <= 0 for x in (max_acceleration_mps2, max_jerk_mps3)):
        raise ValueError('Execution dynamics limits must be finite and positive')
    if callable(plan.get('_sampler')):
        horizon = float(plan.get('execution_horizon_s', 0.))
        native_horizon = float(plan.get('native_horizon_s', horizon))
        if not math.isfinite(horizon) or horizon <= 0 or not 0 < native_horizon <= horizon:
            raise ValueError('Composed execution needs an explicit finite positive horizon')
        return plan['_sampler'], dict(native_horizon_s=native_horizon,
            execution_horizon_s=horizon, stop_duration_s=0., fallback=None)
    # A prepared adoption replaces only the native prefix. Keep the shared
    # SHORT continuation here so later LONG-as-baseline continuation works too.
    native = plan.get('_native_sampler')
    if not callable(native):
        native = spline_plan_sampler(plan)
    if native is None:
        raise ValueError('Execution requires complete metric-time spline controls')
    horizon = float(plan['control_times_s'][-1])
    metadata = dict(native_horizon_s=horizon, execution_horizon_s=horizon,
                    stop_duration_s=0., fallback=None)
    if plan.get('mode') != 'short':
        return native, metadata
    if plan.get('schema') == 'radar_vla_parallel_acceleration_v1':
        from .action_head import action_stopping_continuation, action_limits
        # Limits are part of the serialized action protocol, including its
        # continuation; default helper arguments must not change the path.
        # Stricter caller bounds remain validation/rejection criteria.
        acceleration_limit, jerk_limit = action_limits(plan)
        return action_stopping_continuation(native, horizon, acceleration_limit, jerk_limit)
    endpoint = native([horizon])[0]
    velocity = native([horizon], derivative=1)[0]
    if np.linalg.norm(native([horizon], derivative=2)[0]) > 1e-6:
        raise ValueError('Stopping continuation requires zero terminal acceleration')
    speed = float(np.linalg.norm(velocity))
    duration = max(1.5 * speed / max_acceleration_mps2,
                   math.sqrt(6 * speed / max_jerk_mps3), .05) if speed > 1e-10 else 0.
    metadata.update(execution_horizon_s=horizon + duration,
                    stop_duration_s=duration, fallback='jerk_limited_stop')

    def sample(times, derivative=0):
        query = _query(times, derivative)
        result = np.empty((len(query), 2), dtype=float)
        before = query <= horizon
        if before.any():
            result[before] = native(query[before], derivative=derivative)
        if (~before).any():
            if duration == 0.:
                result[~before] = endpoint if derivative == 0 else 0.
            else:
                phase = np.clip((query[~before] - horizon) / duration, 0., 1.)
                if derivative == 0:
                    result[~before] = endpoint + duration * (phase - phase**3 + .5*phase**4)[:, None] * velocity
                else:
                    factors = {1: 1-3*phase**2+2*phase**3,
                               2: (-6*phase+6*phase**2)/duration,
                               3: (-6+12*phase)/duration**2}
                    factor = factors[derivative]
                    factor[query[~before] >= horizon + duration] = 0.
                    result[~before] = factor[:, None] * velocity
        return result
    return sample, metadata


def prepare_adoption(record, plan, delay_s, baseline_plan=None, *, adoption_transition_s=0.,
                     max_acceleration_mps2=8., max_jerk_mps3=20.):
    """Build an opt-in C2 transition at the actual adoption state, without GT.

    The fixed transition ends on the original plan at the same local time;
    native horizon, suffix and terminal stopping rules are not extended. The
    caller must still evaluate the suffix, road and obstacle constraints.
    Adapter construction is offline work, excluded from measure_plan's timing.
    """
    duration = float(adoption_transition_s)
    if not math.isfinite(duration) or duration < 0:
        raise ValueError('adoption_transition_s must be finite and nonnegative')
    if duration == 0:
        return plan
    if not math.isfinite(delay_s) or delay_s < 0:
        raise ValueError('Plan adoption delay must be finite and nonnegative')
    if plan.get('_adoption_prepared'):
        if (abs(plan['_adoption_duration_s'] - duration) > 1e-10
                or abs(plan['_adoption_ready_s'] - delay_s) > 1e-10):
            raise ValueError('Prepared adoption belongs to a different clock or configuration')
        return plan
    _, metadata = execution_plan_sampler(plan, max_acceleration_mps2=max_acceleration_mps2,
                                          max_jerk_mps3=max_jerk_mps3)
    if callable(plan.get('_sampler')):
        raise ValueError('Opaque execution must be prepared at each original adoption')
    if plan.get('schema') == 'radar_vla_parallel_acceleration_v1':
        from .action_head import action_limits
        action_acceleration, action_jerk = action_limits(plan)
        max_acceleration_mps2 = min(max_acceleration_mps2, action_acceleration)
        max_jerk_mps3 = min(max_jerk_mps3, action_jerk)
    native = spline_plan_sampler(plan)
    horizon = metadata['native_horizon_s']
    diagnostics = dict(schema='quintic_adoption_v1', duration_s=duration, status='prepared',
                       native_horizon_s=horizon)
    if duration > horizon:
        raise AdoptionTransitionError('transition_exceeds_native_horizon', diagnostics)
    yaw = float(baseline_heading(record, [delay_s], baseline_plan)[0])
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s], [s, c]])
    velocity = sample_baseline(record, [delay_s], baseline_plan, derivative=1)[0] @ rotation
    acceleration = sample_baseline(record, [delay_s], baseline_plan, derivative=2)[0] @ rotation
    diagnostics.update(raw_position_jump_m=float(np.linalg.norm(native([0.])[0])),
        raw_initial_velocity_error_mps=float(np.linalg.norm(native([0.], derivative=1)[0] - velocity)),
        raw_acceleration_jump_mps2=float(np.linalg.norm(native([0.], derivative=2)[0] - acceleration)))
    prepared_state = dict(_adoption_prepared=True, _adoption_duration_s=duration,
                          _adoption_ready_s=float(delay_s))
    if max(diagnostics[name] for name in ('raw_position_jump_m', 'raw_initial_velocity_error_mps',
                                          'raw_acceleration_jump_mps2')) <= 1e-10:
        # An already continuous trajectory needs no new interpolation, even
        # when the requested transition would cross multiple native knots.
        diagnostics.update(status='not_needed', requested_duration_s=duration, duration_s=0.)
        return dict(plan, adoption_transition=diagnostics, **prepared_state)
    coefficients = quintic_coefficients(velocity, acceleration, native([duration])[0],
        native([duration], derivative=1)[0], native([duration], derivative=2)[0], duration)
    diagnostics.update(quintic_dynamics(coefficients, duration))
    diagnostics['coefficients_normalized_time'] = coefficients.tolist()
    if (diagnostics['max_acceleration_mps2'] > max_acceleration_mps2 + 1e-8
            or diagnostics['max_jerk_mps3'] > max_jerk_mps3 + 1e-8):
        raise AdoptionTransitionError('transition_dynamics_exceeded', diagnostics)

    def prepared_native(times, derivative=0):
        query = _query(times, derivative)
        result = np.empty((len(query), 2), float)
        bridge = query < duration
        if bridge.any():
            result[bridge] = sample_quintic(coefficients, duration, query[bridge], derivative)
        if (~bridge).any():
            result[~bridge] = native(query[~bridge], derivative=derivative)
        return result

    # Include both sides of the join: jerk can jump while p/v/a stay continuous.
    times = set(plan.get('dynamics_times_s', [])) | set(diagnostics['dynamics_times_s'])
    times.update((float(np.nextafter(duration, 0.)), duration))
    return dict(plan, _native_sampler=prepared_native, adoption_transition=diagnostics,
                dynamics_times_s=sorted(times), **prepared_state)


def _continued_baseline_plan(baseline_plan):
    previous = dict(baseline_plan, valid=baseline_plan.get('valid', True), mode='short')
    if (previous.get('ego_control_points') is None and not callable(previous.get('_sampler'))
            and previous.get('schema') != 'radar_vla_parallel_acceleration_v1'):
        knots = np.asarray(previous.get('ego_times_s'), dtype=float)
        points = np.asarray(previous.get('ego_trajectory'), dtype=float)
        if knots.ndim != 1 or points.shape != (len(knots), 2) or not len(knots):
            raise ValueError('Current plan requires aligned ego times and positions')
        if knots[0] > 0:
            knots, points = np.r_[0., knots], np.vstack((np.zeros(2), points))
        previous.update(control_times_s=knots.tolist(), ego_control_points=points.tolist())
    return previous


def sample_baseline(record, times, baseline_plan=None, derivative=0):
    """Continue the current ego-frame plan, or explicit CV/CTRV if none exists.

    Prior finite plans receive the same terminal stopping continuation. Online
    state can supply ``_sampler`` to preserve an adopted piecewise path exactly.
    Legacy sampled paths are explicitly interpolated as natural splines.
    """
    query = _query(times, derivative)
    if baseline_plan is None:
        velocity = np.asarray(record['ego']['velocity'], dtype=float)
        yaw_rate = float(record['ego'].get('yaw_rate', 0.))
        if derivative == 0:
            return nominal_rollout(velocity, yaw_rate, query)[0]
        phase = np.exp(1j * yaw_rate * query)
        values = complex(*velocity) * phase * (1j*yaw_rate)**(derivative-1)
        return np.column_stack((values.real, values.imag))
    if callable(baseline_plan.get('_sampler')):
        return np.asarray(baseline_plan['_sampler'](query, derivative=derivative), dtype=float)
    sampler, _ = execution_plan_sampler(_continued_baseline_plan(baseline_plan))
    return sampler(query, derivative=derivative)


def _plan_heading(plan, times):
    """Tangent heading, retaining the last direction after an explicit stop."""
    query = _query(times, 0)
    if callable(plan.get('_heading')):
        return np.asarray(plan['_heading'](query), dtype=float)
    if callable(plan.get('_sampler')):
        sampler = plan['_sampler']
        native_horizon = float(plan.get('native_horizon_s', plan.get('execution_horizon_s', query.max(initial=0.))))
    else:
        sampler, metadata = execution_plan_sampler(plan)
        native_horizon = metadata['native_horizon_s']
        if plan.get('schema') == 'radar_vla_parallel_acceleration_v1':
            # The nonzero-a release can change the tangent before stopping.
            native_horizon = metadata['execution_horizon_s']
    # Natural-spline stops retain the native terminal tangent; continuous
    # acceleration release uses its full extension before retaining direction.
    clipped = np.minimum(query, native_horizon)
    end = clipped.max(initial=0.)
    knots = np.asarray(plan.get('control_times_s', [0., end]), dtype=float)
    probes = np.unique(np.r_[np.linspace(0., end, 65), knots[knots <= end], clipped])
    velocities = sampler(probes, derivative=1)
    # Match the evaluator's continuous forward/reverse branch. Integrating the
    # sampled tangents retains analytic local direction while avoiding a pi flip
    # during reverse motion. This costs a bounded native-horizon probe count.
    headings = _yaw_from_path(np.vstack((np.zeros(2), np.cumsum(velocities, axis=0))))[1:]
    return headings[np.searchsorted(probes, clipped)]


def baseline_heading(record, times, baseline_plan=None):
    """Nominal body yaw, or the executing plan's retained tangent approximation."""
    query = _query(times, 0)
    if baseline_plan is None:
        return nominal_rollout(record['ego']['velocity'], record['ego'].get('yaw_rate', 0.), query)[1]
    if callable(baseline_plan.get('_heading')):
        return np.asarray(baseline_plan['_heading'](query), dtype=float)
    previous = _continued_baseline_plan(baseline_plan)
    return _plan_heading(previous, query)


def executed_plan_heading(record, plan, delay_s, times, baseline_plan=None):
    """Heading companion for an adopted path; suitable for composing replay state."""
    query = _query(times, 0)
    result = baseline_heading(record, query, baseline_plan)
    active = query >= delay_s
    if active.any():
        result[active] = (_plan_heading(plan, query[active]-delay_s)
                          + baseline_heading(record, [delay_s], baseline_plan)[0])
    return result


def sample_executed_plan(record, plan, delay_s, times, baseline_plan=None, derivative=0,
                         *, max_acceleration_mps2=8., max_jerk_mps3=20.):
    """Continue the prior path until ready, then rebase at its actual pose.

    Input/output use the observation's ego frame and seconds since observation.
    Position is continuous; incompatible initial dynamics remain detectable by
    reward validation rather than being silently blended into another action.
    """
    if not math.isfinite(delay_s) or delay_s < 0:
        raise ValueError('Plan adoption delay must be finite and nonnegative')
    query = _query(times, derivative)
    output = sample_baseline(record, query, baseline_plan, derivative=derivative)
    active = query >= delay_s
    if active.any():
        sampler, _ = execution_plan_sampler(plan, max_acceleration_mps2=max_acceleration_mps2,
                                            max_jerk_mps3=max_jerk_mps3)
        origin = sample_baseline(record, [delay_s], baseline_plan)[0]
        yaw = float(baseline_heading(record, [delay_s], baseline_plan)[0])
        c, s = math.cos(yaw), math.sin(yaw)
        rotation = np.array([[c, -s], [s, c]])
        output[active] = sampler(query[active] - delay_s, derivative=derivative) @ rotation.T
        if derivative == 0:
            output[active] += origin
    return output


def execution_payload(record, plan, delay_s, step=.05, baseline=None, *,
                      max_acceleration_mps2=8., max_jerk_mps3=20.,
                      initial_velocity_tolerance_mps=.5, adoption_transition_s=0.):
    """Serialize the proposed physical execution separately from native tokens.

    An invalid payload is not an adopted command. When a proposed path exists,
    retain its samples and infeasibility diagnostics for inspection. A complete
    payload is still only an offline trajectory, not a claim of collision safety.
    """
    if not math.isfinite(step) or step <= 0 or not math.isfinite(delay_s) or delay_s < 0:
        raise ValueError('Execution sampling step and adoption delay must be finite and valid')
    result = dict(valid=False, adoption_s=float(delay_s), ego_times_s=[], ego_trajectory=[],
                  reason='invalid_plan', coordinate_frame='observation_current_ego',
                  semantics='proposed delayed execution; sampled dynamics; no safety certification')
    limits = dict(max_acceleration_mps2=max_acceleration_mps2, max_jerk_mps3=max_jerk_mps3)
    try:
        plan = prepare_adoption(record, plan, delay_s, baseline,
                                adoption_transition_s=adoption_transition_s, **limits)
        if 'adoption_transition' in plan:
            result['adoption_transition'] = plan['adoption_transition']
        sampler, metadata = execution_plan_sampler(plan, **limits)
        horizon = float(metadata['execution_horizon_s'])
        end = delay_s + horizon
        times = np.unique(np.r_[0., np.arange(step, end, step), delay_s,
                                delay_s + metadata['native_horizon_s'], end])
        xy = sample_executed_plan(record, plan, delay_s, times, baseline, **limits)
        knots = np.asarray(plan.get('control_times_s', []), dtype=float)
        local = np.unique(np.r_[0., np.arange(step, horizon, step), knots,
            (knots[:-1] + knots[1:])/2, metadata['native_horizon_s'],
            metadata['native_horizon_s'] + metadata['stop_duration_s']/2, horizon,
            plan.get('dynamics_times_s', []), metadata.get('dynamics_times_s', [])])
        acceleration = sampler(local, derivative=2)
        jerk = sampler(local, derivative=3)
        expected = sample_baseline(record, [delay_s], baseline, derivative=1)[0]
        adopted = sample_executed_plan(record, plan, delay_s, [delay_s], baseline,
                                       derivative=1, **limits)[0]
        initial_error = float(np.linalg.norm(adopted - expected))
        expected_a = sample_baseline(record, [delay_s], baseline, derivative=2)[0]
        adopted_a = sample_executed_plan(record, plan, delay_s, [delay_s], baseline,
                                         derivative=2, **limits)[0]
        acceleration_jump = float(np.linalg.norm(adopted_a - expected_a))
        max_acceleration = float(np.linalg.norm(acceleration, axis=-1).max())
        max_jerk = float(np.linalg.norm(jerk, axis=-1).max())
        valid = (initial_error <= initial_velocity_tolerance_mps
                 and acceleration_jump <= 1e-5
                 and max_acceleration <= max_acceleration_mps2 + 1e-8
                 and max_jerk <= max_jerk_mps3 + 1e-8)
        result.update(metadata, valid=valid, reason='executable_candidate' if valid else 'invalid_dynamics',
                      evaluated_horizon_s=end, ego_times_s=times[1:].tolist(), ego_trajectory=xy[1:].tolist(),
                      initial_velocity_error_mps=initial_error, max_acceleration_mps2=max_acceleration,
                      max_jerk_mps3=max_jerk, adoption_acceleration_jump_mps2=acceleration_jump)
    except AdoptionTransitionError as error:
        result.update(reason='invalid_adoption_transition', adoption_transition=error.diagnostics)
    except (ValueError, TypeError, RuntimeError):
        pass
    return result
