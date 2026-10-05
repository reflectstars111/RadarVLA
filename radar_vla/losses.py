"""Masked risk supervision and permutation-invariant physical object losses."""

import math

import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F


DEFAULT_WEIGHTS = dict(collision=1., distance=1., brake=1., monotonicity=.1,
                       objectness=1., position=1., velocity=1., radial=1.,
                       agent_future=1., future_velocity=1., doppler=.1, kinematic=.01, smooth=.001,
                       scale_position=10., scale_velocity=10.,
                       scale_distance=10., scale_brake=5.)


def _mean_masked(values, mask, zero):
    mask = mask.bool()
    return values[mask].mean() if mask.any() else zero


def _regression(prediction, target, mask, scale, zero):
    mask = mask.bool() & torch.isfinite(target)
    if not mask.any():
        return zero
    return F.smooth_l1_loss(prediction[mask] / scale, target[mask] / scale)


def loss_grounding(output: dict, batch: dict, weights: dict | None = None) -> dict[str, torch.Tensor]:
    """Return scalar components and ``total`` with missing labels ignored.

    ``agent_mask=False`` denotes observed absence/padding, so it still supervises
    objectness. For an unannotated scene, set ``agent_supervision_mask=False``.
    ``agent_state_mask`` optionally marks missing state components. Risk masks
    must mark censored braking labels as absent. Regression scales express units
    (meters, meters/sec, meters/sec²); they never change predictions' units.
    """
    cfg = {**DEFAULT_WEIGHTS, **(weights or {})}
    unknown = set(cfg) - set(DEFAULT_WEIGHTS)
    if unknown:
        raise ValueError(f'Unknown loss weights: {sorted(unknown)}')
    for name, value in cfg.items():
        if not math.isfinite(value) or value < 0 or (name.startswith('scale_') and value == 0):
            raise ValueError(f'{name} must be nonnegative (scales strictly positive)')
    # All heads stay connected even when the entire batch has no labels.
    zero = sum(value.sum() * 0 for value in output.values())
    losses = {name: zero for name in DEFAULT_WEIGHTS if not name.startswith('scale_')}
    risk = output['risk']
    if 'risk_target' in batch:
        target = batch['risk_target']
        if target.shape != risk.shape:
            raise ValueError('risk_target must match risk [B,5]')
        mask = batch.get('risk_mask', torch.ones_like(target, dtype=torch.bool)).bool() & torch.isfinite(target)
        if mask[:, :3].any():
            logits = output['risk_logits'][mask[:, :3]]
            truth = target[:, :3][mask[:, :3]]
            if ((truth < 0) | (truth > 1)).any():
                raise ValueError('collision targets must lie in [0,1]')
            bce = F.binary_cross_entropy_with_logits(logits, truth, reduction='none')
            prob = logits.sigmoid()
            pt = truth * prob + (1 - truth) * (1 - prob)
            alpha = .25 * truth + .75 * (1 - truth)
            losses['collision'] = (alpha * (1 - pt).square() * bce).mean()
        losses['distance'] = _regression(risk[:, 3], target[:, 3], mask[:, 3], cfg['scale_distance'], zero)
        losses['brake'] = _regression(risk[:, 4], target[:, 4], mask[:, 4], cfg['scale_brake'], zero)
        losses['monotonicity'] = _mean_masked(F.relu(risk[:, :2] - risk[:, 1:3]), mask[:, :2] & mask[:, 1:3], zero)

    if 'agent_mask' in batch:
        state = output['agent_state']
        truth = batch['agent_state']
        active = batch['agent_mask'].bool()
        supervision = batch.get('agent_supervision_mask', torch.ones(state.shape[0], dtype=torch.bool, device=state.device)).bool()
        component_mask = batch.get('agent_state_mask', torch.ones_like(truth, dtype=torch.bool)).bool() & torch.isfinite(truth)
        if truth.shape[:2] != active.shape or truth.shape[-1] != 5:
            raise ValueError('agent_state/agent_mask shapes must be [B,S,5]/[B,S]')
        object_target = torch.zeros_like(output['object_logits'])
        matches = []
        for b in range(state.shape[0]):
            if not supervision[b]:
                continue
            targets = torch.nonzero(active[b], as_tuple=False).flatten()
            if targets.numel() > state.shape[1]:
                raise ValueError('More annotated agents than object queries; increase max_agents')
            if targets.numel() == 0:
                continue
            valid = component_mask[b, targets]
            if not valid[:, :2].all():
                raise ValueError('Every active agent needs finite x/y for spatial matching')
            target_state = torch.where(valid, truth[b, targets], 0.)
            with torch.no_grad():
                pos_cost = torch.cdist(state[b, :, :2], target_state[:, :2], p=1) / cfg['scale_position']
                velocity_error = (state[b, :, None, 3:5] - target_state[None, :, 3:5]).abs()
                vel_cost = (velocity_error * valid[None, :, 3:5]).sum(-1) / cfg['scale_velocity']
                pred_indices, target_indices = linear_sum_assignment((pos_cost + vel_cost).cpu().numpy())
            pred_indices = torch.as_tensor(pred_indices, device=state.device)
            target_indices = targets[torch.as_tensor(target_indices, device=state.device)]
            object_target[b, pred_indices] = 1
            matches.extend((b, int(p), int(t)) for p, t in zip(pred_indices, target_indices))
        object_mask = supervision[:, None].expand_as(object_target)
        losses['objectness'] = _mean_masked(
            F.binary_cross_entropy_with_logits(output['object_logits'], object_target, reduction='none'), object_mask, zero)
        if matches:
            bi, pi, ti = (torch.tensor(items, device=state.device) for items in zip(*matches))
            pred, target = state[bi, pi], truth[bi, ti]
            valid = component_mask[bi, ti]
            losses['position'] = _regression(pred[:, :2], target[:, :2], valid[:, :2], cfg['scale_position'], zero)
            losses['velocity'] = _regression(pred[:, 3:5], target[:, 3:5], valid[:, 3:5], cfg['scale_velocity'], zero)
            losses['radial'] = _regression(pred[:, 2], target[:, 2], valid[:, 2], cfg['scale_velocity'], zero)
            # Physical supervision must use an actual radar observation. A
            # self-consistency loss between two predicted heads is not evidence
            # of Doppler grounding, and missing radar returns remain masked.
            if 'agent_radial_observed' in batch:
                measured = batch['agent_radial_observed'][bi, ti]
                measured_valid = batch['agent_radial_mask'][bi, ti].bool()
                sensor_velocity = batch.get('agent_sensor_velocity')
                sensor_velocity = sensor_velocity[bi, ti] if sensor_velocity is not None else batch['ego_velocity'][bi]
                has_projection = 'agent_doppler_projection' in batch
                los = batch.get('agent_doppler_projection', batch['agent_los'])[bi, ti]
                physical_mask = (measured_valid & torch.isfinite(measured)
                                 & torch.isfinite(sensor_velocity).all(-1)
                                 & torch.isfinite(los).all(-1))
                if not has_projection:
                    physical_mask &= los.norm(dim=-1) > 1e-6
                sensor_velocity = torch.nan_to_num(sensor_velocity)
                los = torch.nan_to_num(los)
                if not has_projection:
                    los = F.normalize(los, dim=-1, eps=1e-6)
                projected_relative = ((pred[:, 3:5] - sensor_velocity) * los).sum(-1)
                losses['doppler'] = _regression(projected_relative, measured, physical_mask, cfg['scale_velocity'], zero)
                losses['radial'] = _regression(pred[:, 2], measured, physical_mask, cfg['scale_velocity'], zero)
            if 'agent_future' in batch and 'agent_future_mask' in batch:
                future, future_truth = output['agent_future'][bi, pi], batch['agent_future'][bi, ti]
                future_valid = batch['agent_future_mask'][bi, ti].bool()[..., None].expand_as(future_truth) & torch.isfinite(future_truth)
                losses['agent_future'] = _regression(future, future_truth, future_valid, cfg['scale_position'], zero)
                if 'agent_future_velocity' in output and 'future_times_s' in batch:
                    times = batch['future_times_s'][bi]
                    dt = torch.diff(torch.cat((torch.zeros_like(times[:, :1]), times), -1), dim=-1)
                    if (dt <= 0).any() or not torch.isfinite(dt).all():
                        raise ValueError('future_times_s must be finite, positive and increasing')
                    velocity = output['agent_future_velocity'][bi, pi]
                    if velocity.shape != future.shape:
                        raise ValueError('agent_future_velocity must match agent_future shape')
                    position_sequence = torch.cat((pred[:, None, :2], future), 1)
                    velocity_sequence = torch.cat((pred[:, None, 3:5], velocity), 1)
                    # Trapezoidal integration is second order for acceleration;
                    # unlike a constant-velocity anchor it constrains every
                    # interval without disallowing acceleration and turning.
                    residual = (position_sequence[:, 1:] - position_sequence[:, :-1]
                                - .5 * (velocity_sequence[:, 1:] + velocity_sequence[:, :-1]) * dt[..., None])
                    point_valid = torch.cat((valid[:, None, :2].all(-1), future_valid.all(-1)), 1)
                    pair_valid = point_valid[:, 1:] & point_valid[:, :-1]
                    pair_valid[:, 0] &= valid[:, 3:5].all(-1)
                    losses['kinematic'] = _mean_masked(
                        residual.abs().sum(-1) / cfg['scale_position'], pair_valid, zero)
                    acceleration = torch.diff(velocity_sequence, dim=1) / dt[..., None]
                    losses['smooth'] = _mean_masked(acceleration.square().sum(-1), pair_valid, zero)
                    if acceleration.shape[1] > 1:
                        midpoint_dt = .5 * (dt[:, 1:] + dt[:, :-1])
                        jerk = torch.diff(acceleration, dim=1) / midpoint_dt[..., None]
                        losses['smooth'] = losses['smooth'] + _mean_masked(
                            jerk.square().sum(-1), pair_valid[:, 1:] & pair_valid[:, :-1], zero)
                    if 'agent_future_velocity' in batch:
                        velocity_truth = batch['agent_future_velocity'][bi, ti]
                        velocity_valid = batch.get('agent_future_velocity_mask')
                        if velocity_valid is None:
                            raise ValueError('agent_future_velocity supervision requires an explicit mask')
                        velocity_valid = velocity_valid[bi, ti].bool()[..., None].expand_as(velocity_truth)
                        losses['future_velocity'] = _regression(
                            velocity, velocity_truth, velocity_valid, cfg['scale_velocity'], zero)
    losses['total'] = sum(cfg[name] * value for name, value in losses.items())
    return losses
