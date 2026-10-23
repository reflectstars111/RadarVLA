"""Compose chronological actions over recorded observations (not a simulator)."""
from __future__ import annotations

import math
import numpy as np

from .execution import (execution_plan_sampler, sample_baseline, sample_executed_plan,
                        baseline_heading, executed_plan_heading, prepare_adoption)
from .adoption import AdoptionTransitionError


def compose_execution(record, decisions, config):
    """Keep adopted motion through each subsequent plan's measured ready time.

    Coordinates are anchored to the first observation. Each new vehicle-relative
    action is rebased at the currently executing pose, using the deployment
    adapter. Switching discontinuities are retained for reward rejection.
    """
    if not decisions:
        raise ValueError('Replay requires at least one decision')
    ready_times = np.asarray([decision['ready_s'] for decision in decisions], dtype=float)
    if not np.isfinite(ready_times).all() or (ready_times < 0).any():
        raise ValueError('Decision ready times must be finite and nonnegative')
    if (np.diff(ready_times) <= 0).any():
        raise ValueError('Decision ready times must be strictly chronological')
    first = float(ready_times[0])
    limits = dict(max_acceleration_mps2=config.max_acceleration_mps2,
                  max_jerk_mps3=config.max_jerk_mps3)
    prior, switches, velocity_errors, acceleration_jumps = None, [], [], []
    failures, dynamics, transitions = [], [], []
    end = first
    for decision in decisions:
        delay, plan = float(decision['ready_s']), decision['plan']
        if delay < first or (switches and delay < first + switches[-1]):
            raise ValueError('Plan ready times must be chronological')
        try:
            plan = prepare_adoption(record, plan, delay, prior,
                                    adoption_transition_s=config.adoption_transition_s, **limits)
            if 'adoption_transition' in plan:
                transitions.append(dict(ready_s=delay, **plan['adoption_transition']))
            _, metadata = execution_plan_sampler(plan, **limits)
            if (metadata['native_horizon_s'] < config.safety_horizon_s - 1e-7
                    and metadata.get('fallback') is None):
                failures.append('short_horizon')
                continue
            expected_v = sample_baseline(record, [delay], prior, derivative=1)[0]
            expected_a = sample_baseline(record, [delay], prior, derivative=2)[0]
            new_v = sample_executed_plan(record, plan, delay, [delay], prior, derivative=1, **limits)[0]
            new_a = sample_executed_plan(record, plan, delay, [delay], prior, derivative=2, **limits)[0]
        except AdoptionTransitionError as error:
            transitions.append(dict(ready_s=delay, **error.diagnostics))
            failures.append('invalid_adoption_transition')
            continue
        except (ValueError, TypeError, RuntimeError):
            failures.append('invalid_plan')
            continue
        velocity_errors.append(float(np.linalg.norm(new_v - expected_v)))
        acceleration_jumps.append(float(np.linalg.norm(new_a - expected_a)))
        switches.append(delay - first)
        # Only the prefix actually executed before a new adoption remains.
        dynamics = [t for t in dynamics if t < delay - first]
        dynamics.extend(delay - first + float(t) for t in plan.get('control_times_s', []))
        dynamics.extend(delay - first + float(t) for t in plan.get('dynamics_times_s', []))
        dynamics.extend(delay - first + float(t) for t in metadata.get('dynamics_times_s', []))
        dynamics.extend([delay-first, delay-first+metadata['native_horizon_s'] + metadata['stop_duration_s']/2,
                         delay-first+metadata['execution_horizon_s']])
        previous = prior
        # When a finite action is a prior plan, its terminal stop is explicit,
        # exactly as sample_baseline does for native previous-plan controls.
        continuation = dict(plan, mode='short')
        def continued(query, derivative=0, p=continuation, d=delay, old=previous):
            return sample_executed_plan(record, p, d, query, old, derivative=derivative, **limits)
        def continued_heading(query, p=continuation, d=delay, old=previous):
            return executed_plan_heading(record, p, d, query, old)
        prior = dict(_sampler=continued, _heading=continued_heading)
        end = delay + max(config.safety_horizon_s, metadata['execution_horizon_s'])
        final_plan, final_prior, final_delay = plan, previous, delay
    if not switches:
        horizon = decisions[-1]['ready_s'] - first + config.safety_horizon_s
        final_delay = first
        final_prior = None
        final_plan = dict(valid=True, _sampler=lambda q, derivative=0: sample_baseline(
            record, q, derivative=derivative), _heading=lambda q: baseline_heading(record, q),
            execution_horizon_s=horizon)
        end = first + horizon
    elif failures:
        # Continue the last adopted action, including its explicit terminal stop,
        # through the rejected output's arrival. Keep the failure for the reward.
        final_plan = dict(final_plan, mode='short')
        _, fallback = execution_plan_sampler(final_plan, **limits)
        end = max(end, decisions[-1]['ready_s'] + config.safety_horizon_s,
                  final_delay + fallback['execution_horizon_s'])
        dynamics.extend([final_delay-first+fallback['native_horizon_s'],
                         final_delay-first+fallback['native_horizon_s']+fallback['stop_duration_s']/2,
                         final_delay-first+fallback['execution_horizon_s']])
        dynamics.extend(final_delay-first+float(t) for t in fallback.get('dynamics_times_s', []))
    origin = sample_baseline(record, [first])[0]
    yaw = float(baseline_heading(record, [first])[0])
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s], [s, c]])
    def composed(query, derivative=0):
        global_times = first + np.asarray(query, dtype=float)
        xy = sample_executed_plan(record, final_plan, final_delay, global_times,
                                  final_prior, derivative=derivative, **limits)
        return (xy - origin if derivative == 0 else xy) @ rotation
    def composed_heading(query):
        return executed_plan_heading(record, final_plan, final_delay,
            first + np.asarray(query, dtype=float), final_prior) - yaw
    horizon = end - first
    prepared = (dict(_adoption_prepared=True, _adoption_duration_s=config.adoption_transition_s,
                     _adoption_ready_s=first, adoption_transition=dict(
                         schema='quintic_adoption_replay_v1', duration_s=config.adoption_transition_s,
                         decisions=transitions)) if config.adoption_transition_s else {})
    return dict(valid=True, ego_complete=True, _sampler=composed, _heading=composed_heading,
                native_horizon_s=horizon, execution_horizon_s=horizon,
                adoption_times_s=switches, adoption_velocity_errors_mps=velocity_errors,
                adoption_acceleration_jumps_mps2=acceleration_jumps,
                dynamics_times_s=[float(t) for t in dynamics if 0 <= t <= horizon],
                execution_failure=failures[0] if failures else None,
                fallback='composed_execution', **prepared), first
