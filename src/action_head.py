"""Versioned parallel ego actions, with exact piecewise-linear-a integration.

This kinematic adapter bounds Cartesian acceleration and jerk norms. It is not
a vehicle dynamics/controller or collision-safety certificate. Future labels
are used only by the loss below, never by observed_state or the action query.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

REPRESENTATION = 'parallel_acceleration_v1'
SCHEMA = 'radar_vla_parallel_acceleration_v1'


def action_limits(plan):
    limits = plan.get('action_limits', {})
    values = tuple(float(limits.get(name, float('nan'))) for name in
                   ('max_acceleration_mps2', 'max_jerk_mps3'))
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError('continuous action requires explicit finite positive protocol limits')
    return values


def validate_future_times(times, batch, device):
    expected = torch.arange(1, 7, device=device, dtype=torch.float32) * .5
    if times is None:
        raise ValueError('parallel actions require explicit future times 0.5,1,...,3.0 s')
    supplied = torch.as_tensor(times, device=device, dtype=torch.float32)
    if supplied.ndim == 1:
        supplied = supplied[None].expand(batch, -1)
    if supplied.shape != (batch, 6) or not torch.allclose(supplied, expected[None].expand(batch, -1), atol=1e-6, rtol=0):
        raise ValueError('parallel actions require six future times 0.5,1,...,3.0 s')
    return supplied


def observed_state(ego, batch, device, max_acceleration):
    if not isinstance(ego, dict) or 'velocity' not in ego or 'yaw_rate' not in ego:
        raise ValueError('parallel actions require observed ego velocity and yaw_rate')
    velocity = torch.as_tensor(ego['velocity'], device=device, dtype=torch.float32)
    yaw = torch.as_tensor(ego['yaw_rate'], device=device, dtype=torch.float32)
    if velocity.shape != (batch, 2) or yaw.shape != (batch,):
        raise ValueError('observed velocity must be [B,2] and yaw_rate [B]')
    if not torch.isfinite(velocity).all() or not torch.isfinite(yaw).all():
        raise ValueError('observed velocity and yaw_rate must be finite')
    velocity_valid = torch.as_tensor(ego.get('velocity_valid', torch.ones(batch, device=device)), device=device).bool()
    if velocity_valid.ndim == 2:
        velocity_valid = velocity_valid.all(-1)
    if velocity_valid.shape != (batch,) or not velocity_valid.all():
        raise ValueError('parallel actions require valid observed velocity, not missing-label zero')
    fallback = torch.stack((-velocity[:, 1], velocity[:, 0]), -1) * yaw[:, None]
    observed = torch.zeros(batch, dtype=torch.bool, device=device)
    acceleration = fallback
    if ego.get('acceleration') is not None:
        supplied = torch.as_tensor(ego['acceleration'], device=device, dtype=torch.float32)
        if supplied.shape != (batch, 2):
            raise ValueError('observed acceleration must have shape [B,2]')
        observed = torch.as_tensor(ego.get('acceleration_valid', torch.ones(batch, device=device)), device=device).bool()
        if observed.ndim == 2:
            observed = observed.all(-1)
        if observed.shape != (batch,):
            raise ValueError('acceleration_valid must have shape [B] or [B,2]')
        if (observed & ~torch.isfinite(supplied).all(-1)).any():
            raise ValueError('acceleration marked observed must be finite')
        acceleration = torch.where(observed[:, None], torch.nan_to_num(supplied), fallback)
    valid = acceleration.double().norm(dim=-1) <= max_acceleration
    return velocity, acceleration, yaw, observed, valid


def _norm_project(value, limit):
    return value * (limit / value.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp(max=1.)


class ParallelAccelerationHead(nn.Module):
    def __init__(self, hidden_dim, config):
        super().__init__()
        self.config = config
        self.mode = nn.Linear(hidden_dim, 2)
        self.residual = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 12))

    def forward(self, hidden, state, times):
        velocity, acceleration, yaw, observed, valid = state
        residual = self.residual(hidden.float()).reshape(-1, 6, 2)
        # Only six small physical segments use float64. FP32 projections can
        # overshoot exact norm limits by ~1e-6 when saturated, while execution
        # deliberately retains its stricter 1e-8 numerical tolerance. Keeping
        # this integration precise also makes JSON/native replay identical.
        velocity, acceleration, yaw = velocity.double(), acceleration.double(), yaw.double()
        angle = yaw[:, None] * times.double()
        c, s = angle.cos(), angle.sin()
        vx, vy = velocity.unbind(-1)
        nominal_v = torch.stack((c*vx[:, None]-s*vy[:, None], s*vx[:, None]+c*vy[:, None]), -1)
        nominal_a = torch.stack((-nominal_v[..., 1], nominal_v[..., 0]), -1) * yaw[:, None, None]
        desired = _norm_project(nominal_a + residual.double(), self.config.action_max_acceleration_mps2)
        p, v, a = torch.zeros_like(velocity), velocity, acceleration
        positions, velocities, accelerations, jerks = [], [], [a], []
        dt = .5
        for k in range(6):
            # Both endpoints lie in the acceleration ball for a valid initial
            # state. Convex interpolation therefore bounds the entire segment.
            next_a = a + _norm_project(desired[:, k] - a, self.config.action_max_jerk_mps3 * dt)
            j = (next_a - a) / dt
            p = p + v*dt + a*(dt**2/2) + j*(dt**3/6)
            v = v + a*dt + j*(dt**2/2)
            positions.append(p); velocities.append(v); accelerations.append(next_a); jerks.append(j)
            a = next_a
        return dict(mode_logits=self.mode(hidden.float()), acceleration_residuals=residual,
                    positions=torch.stack(positions, 1), velocities=torch.stack(velocities, 1),
                    accelerations=torch.stack(accelerations, 1), jerks=torch.stack(jerks, 1),
                    initial_velocity=velocity, initial_state_valid=valid,
                    acceleration_observed=observed, future_times_s=times)


def parallel_action_loss(output, batch, config):
    """Masked ego position and independent mode supervision, with no GT prefix.

    Finite differences of sparse position labels are not treated as exact
    instantaneous velocity labels. SHORT labels supervise only the native 1 s;
    unknown mode receives no mode loss but may have valid trajectory labels.
    """
    prediction = output['positions']
    target = torch.as_tensor(batch['ego_future'], device=prediction.device, dtype=prediction.dtype)
    mask = torch.as_tensor(batch['ego_future_mask'], device=prediction.device).bool()
    if target.shape != prediction.shape or mask.shape != prediction.shape[:2]:
        raise ValueError('parallel ego supervision must have shape [B,6,2] and mask [B,6]')
    if (mask & ~torch.isfinite(target).all(-1)).any():
        raise ValueError('ego future marked valid must be finite')
    labels = batch.get('action_mode_target', batch.get('mode_target'))
    known = batch.get('action_mode_mask', batch.get('mode_supervision_mask'))
    if labels is None or known is None:
        labels = torch.full((len(prediction),), -1, device=prediction.device, dtype=torch.long)
        known = torch.zeros_like(labels, dtype=torch.bool)
    labels = torch.as_tensor(labels, device=prediction.device, dtype=torch.long)
    known = torch.as_tensor(known, device=prediction.device).bool()
    if labels.shape != (len(prediction),) or known.shape != labels.shape:
        raise ValueError('action mode supervision must have shape [B]')
    if (known & ((labels < 0) | (labels > 1))).any():
        raise ValueError('known action mode must be SHORT=0 or LONG=1')
    mask = mask & output['initial_state_valid'][:, None]
    mask = mask & (~(known & (labels == 0))[:, None] | (output['future_times_s'] <= 1. + 1e-6))
    clean = torch.where(mask[..., None], target, prediction.detach())
    per_point = F.smooth_l1_loss(prediction, clean, reduction='none').sum(-1)
    position_loss = (per_point * mask).sum() / mask.sum().clamp_min(1)
    safe_labels = labels.clamp(0, 1)
    mode_values = F.cross_entropy(output['mode_logits'], safe_labels, reduction='none')
    mode_loss = (mode_values * known).sum() / known.sum().clamp_min(1)
    total = config.ego_trajectory_weight * position_loss + config.action_mode_loss_weight * mode_loss
    return total, dict(action_position_loss=float(position_loss.detach()), action_mode_loss=float(mode_loss.detach()),
                      trajectory_count=int(mask.sum()), mode_count=int(known.sum()),
                      invalid_initial_state_count=int((~output['initial_state_valid']).sum()))


def action_plans(output, config):
    if not all(torch.isfinite(output[name]).all() for name in
               ('positions', 'velocities', 'accelerations', 'mode_logits', 'initial_velocity')):
        raise ValueError('continuous action output must be finite')
    mode_ids = output['mode_logits'].argmax(-1).tolist()
    if config.reasoning_policy != 'adaptive':
        mode_ids = [int(config.reasoning_policy == 'always_long')] * len(mode_ids)
    plans = []
    for b, mode_id in enumerate(mode_ids):
        n = 2 if mode_id == 0 else 6
        times = output['future_times_s'][b, :n].detach().cpu().tolist()
        positions = output['positions'][b, :n].detach().cpu().tolist()
        acceleration = output['accelerations'][b, :n+1].detach().cpu().tolist()
        # Capture both sides of each piecewise-constant-jerk boundary.
        dynamics = [0.] + [u for t in times for u in (float(np.nextafter(t, 0.)), t)]
        valid = bool(output['initial_state_valid'][b])
        plans.append(dict(schema=SCHEMA, action_representation=REPRESENTATION,
            valid=valid, ego_complete=valid, mode='short' if mode_id == 0 else 'long',
            failure_reason=None if valid else 'initial_acceleration_exceeds_limit',
            ego_times_s=times, ego_trajectory=positions, control_times_s=[0.] + times,
            initial_velocity_mps=output['initial_velocity'][b].detach().cpu().tolist(),
            acceleration_knots_mps2=acceleration,
            acceleration_source='observed' if bool(output['acceleration_observed'][b]) else 'ctrv_fallback',
            mode_probabilities=output['mode_logits'][b].softmax(-1).detach().cpu().tolist(),
            dynamics_times_s=dynamics, native_horizon_s=times[-1],
            action_limits=dict(max_acceleration_mps2=config.action_max_acceleration_mps2,
                               max_jerk_mps3=config.action_max_jerk_mps3),
            description_available=False))
    return plans


def native_action_sampler(plan):
    """JSON-reconstructible native action, never a natural-spline reinterpretation."""
    if plan.get('schema') != SCHEMA:
        raise ValueError('unknown continuous action schema')
    action_limits(plan)
    times = np.asarray(plan.get('control_times_s'), dtype=float)
    a = np.asarray(plan.get('acceleration_knots_mps2'), dtype=float)
    v0 = np.asarray(plan.get('initial_velocity_mps'), dtype=float)
    # Baseline execution explicitly converts a completed LONG action to SHORT
    # stopping semantics; native knots retain their original 3 s horizon.
    expected_count = len(times) if times.ndim == 1 else 0
    if (expected_count not in (3, 7) or a.shape != (expected_count, 2) or v0.shape != (2,)
            or not np.isfinite(a).all() or not np.isfinite(v0).all()
            or not np.allclose(times, np.arange(expected_count)*.5, atol=1e-7, rtol=0)):
        raise ValueError('invalid parallel action knots or metric time grid')
    p, v = [np.zeros(2)], [v0]
    j = np.diff(a, axis=0) / .5
    for k in range(len(j)):
        p.append(p[-1]+v[-1]*.5+a[k]*.5**2/2+j[k]*.5**3/6)
        v.append(v[-1]+a[k]*.5+j[k]*.5**2/2)
    p, v = np.asarray(p), np.asarray(v)
    def sample(query, derivative=0):
        query = np.asarray(query, dtype=float)
        if (query.ndim != 1 or not np.isfinite(query).all() or (query < -1e-8).any()
                or (query > times[-1]+1e-8).any()):
            raise ValueError('continuous action query exceeds native horizon')
        if derivative not in (0, 1, 2, 3):
            raise ValueError('derivative must be 0,1,2,3')
        index = np.clip(np.searchsorted(times, query, side='right')-1, 0, len(j)-1)
        t = (query-times[index])[:, None]
        if derivative == 0:
            return p[index]+v[index]*t+a[index]*t*t/2+j[index]*t*t*t/6
        if derivative == 1:
            return v[index]+a[index]*t+j[index]*t*t/2
        if derivative == 2:
            return a[index]+j[index]*t
        return j[index]
    return sample


def action_stopping_continuation(native, horizon, max_acceleration, max_jerk):
    """Release terminal acceleration with bounded jerk, then stop with C2 join.

    The release and stop are execution-only; they neither extend supervised
    future labels nor claim a collision-free continuation beyond observations.
    """
    p0, v0, a0 = (native([horizon], derivative=d)[0] for d in (0, 1, 2))
    magnitude = float(np.linalg.norm(a0))
    if magnitude > max_acceleration + 1e-6:
        raise ValueError('terminal action acceleration exceeds continuation limit')
    release = magnitude / max_jerk if magnitude > 1e-10 else 0.
    j0 = -a0/release if release else np.zeros(2)
    p1 = p0+v0*release+a0*release**2/2+j0*release**3/6
    v1 = v0+a0*release+j0*release**2/2
    speed = float(np.linalg.norm(v1))
    stop = max(1.5*speed/max_acceleration, math.sqrt(6*speed/max_jerk), .05) if speed > 1e-10 else 0.
    start = horizon+release
    end = start+stop
    def sample(query, derivative=0):
        query = np.asarray(query, dtype=float)
        if query.ndim != 1 or not np.isfinite(query).all() or (query < 0).any() or derivative not in (0,1,2,3):
            raise ValueError('invalid continuous execution query')
        result = np.empty((len(query), 2), float)
        before = query <= horizon
        ramp = (query > horizon) & (query < start)
        after = query >= start
        if before.any():
            result[before] = native(query[before], derivative=derivative)
        if ramp.any():
            t = (query[ramp]-horizon)[:, None]
            values = (p0+v0*t+a0*t*t/2+j0*t*t*t/6, v0+a0*t+j0*t*t/2, a0+j0*t, np.broadcast_to(j0, (len(t),2)))
            result[ramp] = values[derivative]
        # With no release, native owns the exact boundary (including jerk).
        after &= ~before
        if after.any():
            if stop == 0:
                result[after] = p1 if derivative == 0 else 0.
            else:
                s = np.clip((query[after]-start)/stop, 0., 1.)
                if derivative == 0:
                    result[after] = p1+stop*(s-s**3+.5*s**4)[:, None]*v1
                else:
                    factor = {1: 1-3*s*s+2*s**3, 2: (-6*s+6*s*s)/stop, 3: (-6+12*s)/stop**2}[derivative]
                    factor[query[after] >= end] = 0.
                    result[after] = factor[:, None]*v1
        return result
    critical = [horizon, float(np.nextafter(horizon, math.inf)), start,
                float(np.nextafter(start, 0.)), start+stop/2, float(np.nextafter(end, 0.)), end]
    return sample, dict(native_horizon_s=horizon, execution_horizon_s=end,
                        stop_duration_s=release+stop, acceleration_release_duration_s=release,
                        fallback='acceleration_release_then_jerk_limited_stop', dynamics_times_s=critical)
