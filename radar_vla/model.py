"""Compact multi-frame Power/Doppler encoder and physical grounding heads.

Coordinates are in the current ego frame: x forward, y left; agent vx/vy
are absolute ground velocities resolved on that frame, whereas radial Doppler
is relative to the ego velocity. Only observation fields are read in forward.
"""

from dataclasses import dataclass
from math import gcd

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    hidden_dim: int = 64
    num_heads: int = 4
    num_queries: int = 8
    max_agents: int = 8
    horizon_steps: int = 6
    power_scale: float = 1.0
    doppler_scale: float = 15.0
    use_doppler: bool = True
    use_ego: bool = True
    max_grid_size: int = 8
    range_scale: float = 100.0

    def __post_init__(self):
        if self.hidden_dim < 4 or self.num_heads < 1 or self.hidden_dim % self.num_heads:
            raise ValueError('hidden_dim must be >= 4 and divisible by num_heads')
        for name in ('num_queries', 'max_agents', 'horizon_steps', 'max_grid_size'):
            if getattr(self, name) < 1:
                raise ValueError(f'{name} must be positive')
        for name in ('power_scale', 'doppler_scale', 'range_scale'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')


class RadarVLAGrounder(nn.Module):
    """One spatial backbone, temporal query compression, and lightweight heads."""

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config.hidden_dim
        half = c // 2
        self.encoder = nn.Sequential(
            nn.Conv2d(2, half, 5, stride=2, padding=2),
            nn.GroupNorm(gcd(8, half), half), nn.GELU(),
            nn.Conv2d(half, c, 3, stride=2, padding=1),
            nn.GroupNorm(gcd(8, c), c), nn.GELU(),
            nn.Conv2d(c, c, 3, padding=1), nn.GELU(),
        )
        self.position_encoder = nn.Sequential(nn.Linear(4, c), nn.GELU(), nn.Linear(c, c))
        self.radar_queries = nn.Parameter(torch.randn(self.config.num_queries, c) * .02)
        self.temporal_attention = nn.MultiheadAttention(c, self.config.num_heads, batch_first=True)
        self.token_norm = nn.LayerNorm(c)
        self.token_mlp = nn.Sequential(nn.Linear(c, 2 * c), nn.GELU(), nn.Linear(2 * c, c))
        self.output_norm = nn.LayerNorm(c)
        self.ego_encoder = nn.Sequential(nn.Linear(3, c), nn.GELU(), nn.Linear(c, c))
        self.risk_query = nn.Parameter(torch.randn(1, c) * .02)
        self.risk_attention = nn.MultiheadAttention(c, self.config.num_heads, batch_first=True)
        self.risk_head = nn.Sequential(nn.Linear(2 * c, c), nn.GELU(), nn.Linear(c, 5))
        self.object_queries = nn.Parameter(torch.randn(self.config.max_agents, c) * .02)
        self.object_attention = nn.MultiheadAttention(c, self.config.num_heads, batch_first=True)
        self.object_norm = nn.LayerNorm(c)
        self.object_head = nn.Linear(c, 1)
        self.state_head = nn.Sequential(nn.Linear(c, c), nn.GELU(), nn.Linear(c, 5))
        self.future_head = nn.Sequential(
            nn.Linear(c, c), nn.GELU(), nn.Linear(c, self.config.horizon_steps * 2))

    def normalize_radar(self, radar: torch.Tensor) -> torch.Tensor:
        """Fixed train/inference normalization; Doppler sign is never removed."""
        power = torch.log1p(radar[:, :, 0].clamp_min(0) * self.config.power_scale)
        doppler = radar[:, :, 1] / self.config.doppler_scale
        if not self.config.use_doppler:
            doppler = torch.zeros_like(doppler)
        return torch.stack((power, doppler), dim=2)

    def _validate_inputs(self, batch):
        radar = batch['radar']
        if radar.ndim != 5 or radar.shape[2] != 2:
            raise ValueError('radar must have shape [B,T,2,R,A]')
        b, t, _, r, a = radar.shape
        if min(b, t, r, a) < 1:
            raise ValueError('radar axes must be nonempty')
        shapes = dict(range_m=(b, r), azimuth_rad=(b, a), time_offsets_s=(b, t),
                      ego=(b, 3), future_times_s=(b, self.config.horizon_steps))
        for name, shape in shapes.items():
            if name not in batch or tuple(batch[name].shape) != shape:
                raise ValueError(f'{name} must have shape {shape}')
            if not torch.isfinite(batch[name]).all():
                raise ValueError(f'{name} contains non-finite values')
        if not torch.isfinite(radar).all():
            raise ValueError('radar contains non-finite values')
        if (batch['range_m'] < 0).any() or (batch['range_m'].diff(dim=1) <= 0).any():
            raise ValueError('range_m must be nonnegative and strictly increasing')
        if (batch['azimuth_rad'].diff(dim=1) <= 0).any():
            raise ValueError('azimuth_rad must be strictly increasing')
        if (batch['time_offsets_s'] > 1e-6).any() or (batch['time_offsets_s'].diff(dim=1) <= 0).any():
            raise ValueError('time_offsets_s must be increasing historical times ending at zero')
        if not torch.allclose(batch['time_offsets_s'][:, -1], torch.zeros_like(batch['time_offsets_s'][:, -1]), atol=1e-6):
            raise ValueError('time_offsets_s must end at the current frame (zero)')
        future = batch['future_times_s']
        if (future <= 0).any() or (future.diff(dim=1) <= 0).any():
            raise ValueError('future_times_s must be strictly positive and increasing')

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self._validate_inputs(batch)
        radar = self.normalize_radar(batch['radar'])
        b, t, _, r, a = radar.shape
        feature = self.encoder(radar.reshape(b * t, 2, r, a))
        nr = min(feature.shape[-2], self.config.max_grid_size)
        na = min(feature.shape[-1], self.config.max_grid_size)
        # Bound the key/value sequence length for real high-resolution RA maps.
        feature = F.adaptive_avg_pool2d(feature, (nr, na))
        feature = feature.reshape(b, t, self.config.hidden_dim, nr, na).permute(0, 1, 3, 4, 2)
        ranges = F.adaptive_avg_pool1d(batch['range_m'][:, None], nr)[:, 0]
        angles = F.adaptive_avg_pool1d(batch['azimuth_rad'][:, None], na)[:, 0]
        shape = (b, t, nr, na)
        coords = torch.stack((
            (ranges / self.config.range_scale)[:, None, :, None].expand(shape),
            angles.sin()[:, None, None, :].expand(shape),
            angles.cos()[:, None, None, :].expand(shape),
            batch['time_offsets_s'][:, :, None, None].expand(shape),
        ), dim=-1)
        feature = (feature + self.position_encoder(coords)).reshape(b, t * nr * na, -1)
        queries = self.radar_queries[None].expand(b, -1, -1)
        tokens = self.token_norm(queries + self.temporal_attention(queries, feature, feature, need_weights=False)[0])
        tokens = self.output_norm(tokens + self.token_mlp(tokens))
        ego = batch['ego'] if self.config.use_ego else torch.zeros_like(batch['ego'])
        ego = self.ego_encoder(ego / ego.new_tensor([15., 5., 1.]))
        risk_query = self.risk_query[None].expand(b, -1, -1)
        risk_feature = self.risk_attention(risk_query, tokens, tokens, need_weights=False)[0][:, 0]
        raw_risk = self.risk_head(torch.cat((risk_feature, ego), dim=-1))
        risk = torch.cat((raw_risk[:, :3].sigmoid(), F.softplus(raw_risk[:, 3:])), dim=-1)
        objects = self.object_queries[None].expand(b, -1, -1)
        objects = self.object_norm(objects + self.object_attention(objects, tokens, tokens, need_weights=False)[0] + ego[:, None])
        state = self.state_head(objects)
        offsets = self.future_head(objects).reshape(b, self.config.max_agents, self.config.horizon_steps, 2)
        # A learned residual permits acceleration and turning, unlike a fixed CV decoder.
        future = state[:, :, None, :2] + state[:, :, None, 3:5] * batch['future_times_s'][:, None, :, None] + offsets
        return dict(radar_tokens=tokens, risk_logits=raw_risk[:, :3], risk=risk,
                    object_logits=self.object_head(objects).squeeze(-1),
                    agent_state=state, agent_future=future)
