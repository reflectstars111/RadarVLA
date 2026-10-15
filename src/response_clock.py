"""Measure planned motion response without consulting future annotations.

This clock reports a sustained change from the currently executing trajectory.
It does not certify that the change is dynamically feasible, avoids an obstacle,
or would actually be executed by a vehicle controller.
"""
from __future__ import annotations

import math
import numpy as np

from .execution import (baseline_heading, execution_plan_sampler,
                        sample_baseline, sample_executed_plan)


def response_clock(record, plan, generation_s, *, waiting_s=0., perception_s=0.,
                   config=None, baseline_plan=None):
    """Return the first sustained longitudinal or lateral motion change.

    Only ``record['ego']`` and the proposed/current plans enter the calculation.
    Positive and negative longitudinal acceleration changes both count. The
    origin of ``reaction_s`` is the first detector report; perception shifts the
    action's adoption in the observation frame but precedes that clock origin.
    Missing future annotations and later safety failures cannot erase this
    measurement. Safety and motion feasibility are separate reward diagnostics.
    """
    if config is None:
        # The reward integrates this clock, so keep the default import lazy.
        from .response_reward import ResponseRewardConfig
        config = ResponseRewardConfig()
    cfg = config
    if any(not math.isfinite(x) or x < 0 for x in (generation_s, waiting_s, perception_s)):
        raise ValueError('Latency components must be finite and nonnegative')
    latency = float(waiting_s + generation_s)
    ready = float(perception_s + latency)
    result = dict(response_detected=False, action_onset_s=None, reaction_s=None,
                  response_time_reward=-6. - cfg.latency_weight * latency / cfg.response_deadline_s,
                  response_clock_status='invalid_plan')
    limits = dict(max_acceleration_mps2=cfg.max_acceleration_mps2,
                  max_jerk_mps3=cfg.max_jerk_mps3)
    try:
        _, metadata = execution_plan_sampler(plan, **limits)
        horizon = float(metadata['execution_horizon_s'])
        # Explicit SHORT stopping/holding is executable beyond its native
        # horizon. Finite LONG plans receive no implicit extrapolation here.
        if metadata.get('fallback') is not None:
            horizon = max(horizon, cfg.safety_horizon_s)
        extras = []
        for key in ('control_times_s', 'adoption_times_s', 'dynamics_times_s'):
            values = np.asarray(plan.get(key, []), dtype=float)
            if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError('Motion clock knots must be finite nonnegative vectors')
            extras.extend(values[values <= horizon])
        extras.extend((metadata['native_horizon_s'],
                       metadata['native_horizon_s'] + .5 * metadata['stop_duration_s']))
        local = np.unique(np.r_[np.arange(0., horizon, cfg.evaluation_step_s), horizon, extras])
        times = ready + local
        actual = sample_executed_plan(record, plan, ready, times, baseline_plan, **limits)
        actual_acceleration = sample_executed_plan(record, plan, ready, times,
            baseline_plan, derivative=2, **limits)
        nominal = sample_baseline(record, times, baseline_plan)
        velocity = sample_baseline(record, times, baseline_plan, derivative=1)
        acceleration = sample_baseline(record, times, baseline_plan, derivative=2)
        if any(value.shape != (len(times), 2) or not np.isfinite(value).all()
               for value in (actual, actual_acceleration, nominal, velocity, acceleration)):
            raise ValueError('Motion samples must contain finite XY values')
        heading = baseline_heading(record, times, baseline_plan)
        moving = np.linalg.norm(velocity, axis=-1) > 1e-8
        heading[moving] = np.arctan2(velocity[moving, 1], velocity[moving, 0])
        tangent = np.column_stack((np.cos(heading), np.sin(heading)))
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        acceleration_change = np.sum((actual_acceleration - acceleration) * tangent, axis=-1)
        lateral_change = np.sum((actual - nominal) * normal, axis=-1)
        changed = ((np.abs(acceleration_change) >= cfg.brake_threshold_mps2 - 1e-10)
                   | (np.abs(lateral_change) >= cfg.lateral_threshold_m - 1e-10))
    except (ValueError, TypeError, RuntimeError, KeyError):
        return result
    result['response_clock_status'] = 'no_change'
    for index in np.flatnonzero(changed):
        end = int(np.searchsorted(local, local[index] + cfg.sustain_s - 1e-8))
        if end < len(local) and changed[index:end + 1].all():
            onset = float(local[index])
            reaction = latency + onset
            result.update(response_detected=True, action_onset_s=onset, reaction_s=reaction,
                          response_clock_status='changed', response_time_reward=float(
                              2. - cfg.latency_weight * reaction / cfg.response_deadline_s))
            break
    return result
