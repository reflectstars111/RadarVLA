"""Versioned, prediction-only physical features for the Risk readout.

The detector state is [x, y, radial_doppler, ground_vx, ground_vy]. Radial
Doppler is a relative sensor measurement; vx/vy are absolute ground velocity
in the current ego frame. No GT, annotation coverage, decode threshold, NMS,
class selection or description length enters this module.

Geometry is deliberately stopped before the learned readout. Otherwise Risk
supervision could train the detector's purported physical outputs as an
unconstrained communication channel instead of improving physical grounding.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def nominal_ego_motion(velocity, yaw_rate, times):
    """Unbraked CV/CTRV positions/velocities in the fixed current-ego frame.

    The sinc form has no division by yaw rate, including at zero. As in the
    label contract, this is a counterfactual reference, not the future ego GT.
    """
    phase = yaw_rate[:, None] * times
    along = times * torch.sinc(phase / math.pi)
    across = times * (phase / 2) * torch.sinc(phase / (2 * math.pi)).square()
    vx, vy = velocity[:, :1], velocity[:, 1:2]
    position = torch.stack((vx * along - vy * across, vx * across + vy * along), -1)
    rotated = torch.stack((vx * phase.cos() - vy * phase.sin(),
                           vx * phase.sin() + vy * phase.cos()), -1)
    return position, rotated


def point_to_oriented_box_distance(relative_xy, heading, size):
    """Distance from a point to a filled target rectangle (zero inside).

    relative_xy is point minus box center (its negation gives the same result).
    This is NOT the distance between two vehicle bodies: the input contract
    contains no ego dimensions. Future target orientation is held at its
    predicted current heading; no future GT heading is supplied.
    """
    c, s = heading.cos(), heading.sin()
    local = torch.stack((relative_xy[..., 0] * c + relative_xy[..., 1] * s,
                         -relative_xy[..., 0] * s + relative_xy[..., 1] * c), -1)
    return (local.abs() - size / 2).clamp_min(0).norm(dim=-1)


class PredictedGeometryRiskFusion(nn.Module):
    """Shared candidate MLP, one Risk query, and residual radar/geometry fusion.

    Every legal dense candidate participates, regardless of objectness. The
    sigmoid objectness is a feature, not a hard gate. A zero sentinel is valid
    only for rows with no legal candidates, preventing all-masked attention.

    Feature order (20 + 10*H): current xy, absolute/relative ground velocity,
    radial Doppler, size, cos/sin heading, objectness, observed ego v/a/yaw,
    current center-to-box distance, range, closing speed; then for each future
    knot: predicted xy, relative xy to ego CV/CTRV, absolute/relative velocity,
    ego-center-to-target-box distance, time. Fixed units normalize position by
    range_scale, speed by doppler_scale, size by 10m, acceleration by 5m/s²,
    yaw rate by 1rad/s, and time by 3s.
    """

    def __init__(self, hidden_dim, num_heads, horizon_steps, range_scale=100.,
                 doppler_scale=15., use_ego=True):
        super().__init__()
        self.horizon_steps = horizon_steps
        self.range_scale, self.doppler_scale, self.use_ego = range_scale, doppler_scale, use_ego
        self.feature_dim = 20 + 10 * horizon_steps
        self.candidate_encoder = nn.Sequential(nn.Linear(self.feature_dim, hidden_dim),
                                               nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.fusion = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(),
                                    nn.Linear(hidden_dim, hidden_dim))

    @torch.no_grad()
    def physical_features(self, predictions, batch):
        logits = predictions['detection_logits']
        if logits.ndim != 2:
            raise ValueError('Risk geometry objectness must have shape [B,K]')
        b, k = logits.shape
        h = self.horizon_steps
        mask = predictions.get('detection_candidate_mask', torch.ones_like(logits, dtype=torch.bool))
        if mask.shape != (b, k) or mask.dtype != torch.bool:
            raise ValueError('Risk geometry candidate mask must be boolean [B,K]')

        def stopped(name, suffix, fill=0.):
            value = predictions[name].detach().float()
            if value.shape != (b, k, *suffix):
                raise ValueError(f'Risk geometry {name} has an incompatible shape')
            expanded_mask = mask.reshape(b, k, *([1] * len(suffix)))
            # Mask BEFORE nonlinear geometry; invalid padding may contain NaN.
            value = torch.where(expanded_mask, value, fill)
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f'Risk geometry {name} contains non-finite valid predictions')
            return value

        state = stopped('detection_state', (5,))
        size = stopped('detection_size', (2,), fill=1.)
        if not bool((size > 0).all()):
            raise ValueError('Risk geometry target dimensions must be positive')
        heading = stopped('detection_heading', ())
        confidence = stopped('detection_logits', ()).sigmoid()
        future = stopped('detection_future', (h, 2))
        future_velocity = stopped('detection_future_velocity', (h, 2))
        observed = batch['ego'].detach().to(state)
        velocity = batch.get('ego_velocity', torch.stack((observed[:, 0], torch.zeros_like(observed[:, 0])), -1))
        acceleration = batch.get('ego_acceleration', torch.stack((observed[:, 1], torch.zeros_like(observed[:, 1])), -1))
        ego_state = torch.cat((velocity.detach().to(state), acceleration.detach().to(state), observed[:, 2:3]), -1)
        if not self.use_ego:
            ego_state = torch.zeros_like(ego_state)
        times = batch['future_times_s'].detach().to(state)
        if (ego_state.shape != (b, 5) or times.shape != (b, h)
                or not bool(torch.isfinite(ego_state).all()) or not bool(torch.isfinite(times).all())):
            raise ValueError('Risk geometry requires finite observed Ego and future sample times')
        ego_xy, ego_velocity = nominal_ego_motion(ego_state[:, :2], ego_state[:, 4], times)
        position, absolute_velocity = state[..., :2], state[..., 3:5]
        relative_velocity = absolute_velocity - ego_state[:, None, :2]
        distance = position.norm(dim=-1)
        closing = -(position * relative_velocity).sum(-1) / distance.clamp_min(1e-6)
        clearance = point_to_oriented_box_distance(position, heading, size)
        current = torch.cat((position / self.range_scale,
                             absolute_velocity / self.doppler_scale,
                             relative_velocity / self.doppler_scale,
                             state[..., 2:3] / self.doppler_scale,
                             size / 10., torch.stack((heading.cos(), heading.sin()), -1),
                             confidence[..., None],
                             (ego_state / ego_state.new_tensor([self.doppler_scale, self.doppler_scale, 5., 5., 1.]))[:, None].expand(-1, k, -1),
                             clearance[..., None] / self.range_scale,
                             distance[..., None] / self.range_scale,
                             closing[..., None] / self.doppler_scale), -1)
        relative_future = future - ego_xy[:, None]
        future_clearance = point_to_oriented_box_distance(relative_future, heading[..., None], size[:, :, None])
        future_features = torch.cat((future / self.range_scale, relative_future / self.range_scale,
                                     future_velocity / self.doppler_scale,
                                     (future_velocity - ego_velocity[:, None]) / self.doppler_scale,
                                     future_clearance[..., None] / self.range_scale,
                                     times[:, None, :, None].expand(-1, k, -1, -1) / 3.), -1)
        features = torch.cat((current, future_features.reshape(b, k, 10 * h)), -1)
        return torch.where(mask[..., None], features, 0.), mask

    def forward(self, radar_context, risk_query, predictions, batch):
        features, mask = self.physical_features(predictions, batch)
        # Only empty rows may attend to the sentinel. Nonempty rows attend to
        # precisely the complete legal set, with no softmax over invalid cells.
        features = torch.cat((features, features.new_zeros(features.shape[0], 1, self.feature_dim)), 1)
        valid = torch.cat((mask, ~mask.any(1, keepdim=True)), 1)
        encoded = self.candidate_encoder(features.to(radar_context.dtype))
        geometry = self.attention(risk_query, encoded, encoded, key_padding_mask=~valid,
                                  need_weights=False)[0][:, 0]
        return radar_context + self.fusion(torch.cat((radar_context, geometry), -1))
