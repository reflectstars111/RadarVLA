"""Compact multi-frame Power/Doppler encoder and physical grounding heads.

Coordinates are in the current ego frame: x forward, y left; agent vx/vy
are absolute ground velocities resolved on that frame, whereas radial Doppler
is relative to the ego velocity. Only observation fields are read in forward.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd, isfinite, radians

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    hidden_dim: int = 64
    num_heads: int = 4
    num_queries: int = 8
    max_agents: int = 8  # Legacy query head only; dense detection has no such cap.
    horizon_steps: int = 6
    power_scale: float = 1.0
    doppler_scale: float = 15.0
    use_doppler: bool = True
    use_ego: bool = True
    use_temporal: bool = True
    # Opt in for newly trained planners; absent in legacy checkpoint configs.
    # Reuses ego_encoder weights without changing the grounding architecture.
    planner_ego_context: bool = False
    max_grid_size: int = 8
    range_scale: float = 100.0
    use_detection: bool = True
    detection_class_schema: str = 'legacy4_v1'
    detection_slots_per_cell: int = 2
    detection_confidence: float = .5
    detection_class_confidence: float = .5
    detection_nms_iou: float = .3
    detection_max_candidates: int | None = None
    detection_max_objects: int | None = None
    detection_match_distance_m: float = 2.
    backbone: str = 'cnn2d'
    backbone_options: dict | None = None
    # Versioned opt-in: legacy checkpoints retain their historical candidate set.
    detection_candidate_policy: str = 'legacy_v1'
    detection_candidate_range_m: float = 50.
    detection_candidate_half_fov_deg: float = 53.
    risk_horizon_schema: str = '1_2_3_v1'

    def __post_init__(self):
        from .risk_contract import contract
        contract(self.risk_horizon_schema)
        if self.detection_candidate_policy not in ('legacy_v1', 'supervised_sector_v1'):
            raise ValueError('Unknown detection candidate policy')
        if (not isfinite(self.detection_candidate_range_m) or self.detection_candidate_range_m <= 0
                or not isfinite(self.detection_candidate_half_fov_deg)
                or not 0 < self.detection_candidate_half_fov_deg <= 90):
            raise ValueError('Invalid detection candidate sector')
        if self.detection_candidate_policy != 'legacy_v1' and not self.use_detection:
            raise ValueError('Candidate sector policy requires dense detection')
        if self.backbone not in ('cnn2d', 'rodnet_hg', 'trodnet'):
            raise ValueError('Unknown backbone; expected cnn2d, rodnet_hg or trodnet')
        expected = {'cnn2d': {}, 'rodnet_hg': {'stacks': 1, 'temporal_kernel': 9},
                    'trodnet': {'embed_dim': 64, 'window_size': 4, 'depths': [2,2,2],
                                'num_heads': [2,4,8], 'drop_path_rate': .1}}[self.backbone]
        if self.backbone_options is not None:
            if not isinstance(self.backbone_options, dict):
                raise ValueError('backbone_options must be an object or null')
            for key,value in self.backbone_options.items():
                if key not in expected or type(value) is not type(expected[key]) or value != expected[key]:
                    raise ValueError(f'Unsupported {self.backbone} backbone option {key}={value!r}')
        from .categories import detection_classes
        detection_classes(self.detection_class_schema)
        if self.detection_class_schema == 'agent_v1' and not self.use_detection:
            raise ValueError('agent_v1 requires the dense detector')
        if self.hidden_dim < 4 or self.num_heads < 1 or self.hidden_dim % self.num_heads:
            raise ValueError('hidden_dim must be >= 4 and divisible by num_heads')
        for name in ('num_queries', 'max_agents', 'horizon_steps', 'max_grid_size'):
            if getattr(self, name) < 1:
                raise ValueError(f'{name} must be positive')
        for name in ('power_scale', 'doppler_scale', 'range_scale'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        for name in ('detection_slots_per_cell', 'detection_max_candidates', 'detection_max_objects'):
            if name != 'detection_slots_per_cell' and getattr(self, name) is None:
                continue
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('detection_confidence', 'detection_class_confidence', 'detection_nms_iou'):
            if not isfinite(getattr(self, name)) or not 0 < getattr(self, name) <= 1:
                raise ValueError(f'{name} must lie in (0,1]')
        if not isfinite(self.detection_match_distance_m) or self.detection_match_distance_m <= 0:
            raise ValueError('detection_match_distance_m must be positive')
        if type(self.use_detection) is not bool:
            raise ValueError('use_detection must be boolean')
        if type(self.planner_ego_context) is not bool:
            raise ValueError('planner_ego_context must be boolean')


class RadarVLAGrounder(nn.Module):
    """One spatial backbone, temporal query compression, and lightweight heads."""

    def load_state_dict(self, state_dict, strict=True, assign=False):
        incoming = state_dict.get('_risk_horizon_contract')
        expected = getattr(self, '_risk_horizon_contract', None)
        if (incoming is None) != (expected is None) or incoming is not None and not torch.equal(incoming.cpu(),expected.cpu()):
            raise ValueError('Checkpoint Risk horizons incompatible; explicit retraining required even with strict=False')
        incoming_sector = state_dict.get('_candidate_sector')
        expected_sector = getattr(self, '_candidate_sector', None)
        if ((incoming_sector is None) != (expected_sector is None)
                or incoming_sector is not None and (incoming_sector.shape != expected_sector.shape
                    or not torch.equal(incoming_sector.detach().cpu(), expected_sector.detach().cpu()))):
            raise ValueError('Checkpoint candidate policy/sector is incompatible; '
                             'retraining is required, including when strict=False')
        marker = state_dict.get('_backbone_id')
        incoming = 'cnn2d' if marker is None and 'encoder.0.weight' in state_dict else (
            {1: 'rodnet_hg', 2: 'trodnet'}.get(int(marker)) if marker is not None else None)
        if incoming != self.config.backbone:
            raise ValueError(f'Checkpoint backbone {incoming!r} is incompatible with {self.config.backbone!r}; '
                             'cross-backbone and implicit partial loading are forbidden')
        if self.config.use_detection:
            has_classifier = 'detector.class_head.weight' in state_dict
            if has_classifier != (self.config.detection_class_schema == 'legacy4_v1'):
                raise ValueError('Checkpoint detection class schema is incompatible with this model; '
                                 'legacy4_v1 and agent_v1 require separate checkpoints, no implicit transfer')
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        from .risk_contract import SHORT_HORIZONS
        if self.config.risk_horizon_schema == SHORT_HORIZONS:
            self.register_buffer('_risk_horizon_contract',torch.tensor([.5,1.,2.,2.],dtype=torch.float64))
        if self.config.detection_candidate_policy == 'supervised_sector_v1':
            half_fov = radians(self.config.detection_candidate_half_fov_deg)
            self.register_buffer('_candidate_sector', torch.tensor(
                [self.config.detection_candidate_range_m, -half_fov, half_fov], dtype=torch.float64))
        c = self.config.hidden_dim
        half = c // 2
        if self.config.backbone == 'cnn2d':
            self.encoder = nn.Sequential(
                nn.Conv2d(2, half, 5, stride=2, padding=2),
                nn.GroupNorm(gcd(8, half), half), nn.GELU(),
                nn.Conv2d(half, c, 3, stride=2, padding=1),
                nn.GroupNorm(gcd(8, c), c), nn.GELU(),
                nn.Conv2d(c, c, 3, padding=1), nn.GELU(),
            )
        elif self.config.backbone == 'rodnet_hg':
            from .backbone_hg import RODNetHGFeatures
            self.encoder = RODNetHGFeatures(c)
            self.register_buffer('_backbone_id', torch.tensor(1, dtype=torch.int64))
        else:
            from .backbone_trodnet import T_RODNetFeatures
            self.encoder = T_RODNetFeatures(c)
            self.register_buffer('_backbone_id', torch.tensor(2, dtype=torch.int64))
        self.validity_encoder = nn.Linear(1, c)
        self.position_encoder = nn.Sequential(nn.Linear(10, c), nn.GELU(), nn.Linear(c, c))
        self.radar_queries = nn.Parameter(torch.randn(self.config.num_queries, c) * .02)
        self.temporal_attention = nn.MultiheadAttention(c, self.config.num_heads, batch_first=True)
        self.token_norm = nn.LayerNorm(c)
        self.token_mlp = nn.Sequential(nn.Linear(c, 2 * c), nn.GELU(), nn.Linear(2 * c, c))
        self.output_norm = nn.LayerNorm(c)
        self.ego_encoder = nn.Sequential(nn.Linear(5, c), nn.GELU(), nn.Linear(c, c))
        self.risk_query = nn.Parameter(torch.randn(1, c) * .02)
        self.risk_attention = nn.MultiheadAttention(c, self.config.num_heads, batch_first=True)
        self.risk_head = nn.Sequential(nn.Linear(2 * c, c), nn.GELU(), nn.Linear(c, 5))
        if self.config.use_detection:
            from .detection import DenseRadarDetector
            self.detector = DenseRadarDetector(c, self.config.horizon_steps, self.config.detection_slots_per_cell,
                                               self.config.detection_class_schema)
        else:
            # Preserve the exact architecture of pre-detector checkpoints.
            self.object_queries = nn.Parameter(torch.randn(self.config.max_agents, c) * .02)
            self.object_attention = nn.MultiheadAttention(c, self.config.num_heads, batch_first=True)
            self.object_norm = nn.LayerNorm(c)
            self.object_head = nn.Linear(c, 1)
            self.state_head = nn.Sequential(nn.Linear(c, c), nn.GELU(), nn.Linear(c, 5))
            self.future_head = nn.Sequential(
                nn.Linear(c, c), nn.GELU(), nn.Linear(c, self.config.horizon_steps * 2))
            self.future_velocity_head = nn.Sequential(
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
        optional_shapes = dict(ego_velocity=(b, 2), ego_acceleration=(b, 2),
                               radar_cartesian=(b, t, r, a, 2), radar_los=(b, t, r, a, 2),
                               radar_pose=(b, t, 4, 4), radar_sensor_velocity=(b, t, 2),
                               radar_doppler_valid=(b, t, r, a))
        for name, shape in optional_shapes.items():
            if name in batch and (tuple(batch[name].shape) != shape or not torch.isfinite(batch[name]).all()):
                raise ValueError(f'{name} must be finite with shape {shape}')
        if 'radar_doppler_valid' in batch and not ((batch['radar_doppler_valid'] == 0) | (batch['radar_doppler_valid'] == 1)).all():
            raise ValueError('radar_doppler_valid must contain boolean validity, not continuous confidences')
        if not torch.isfinite(radar).all():
            raise ValueError('radar contains non-finite values')
        if (batch['range_m'] < 0).any() or (batch['range_m'].diff(dim=1) <= 0).any():
            raise ValueError('range_m must be nonnegative and strictly increasing')
        if (batch['azimuth_rad'].diff(dim=1) <= 0).any():
            raise ValueError('azimuth_rad must be strictly increasing')
        if (batch['time_offsets_s'] > 1e-6).any() or (batch['time_offsets_s'].diff(dim=1) <= 0).any():
            raise ValueError('time_offsets_s must be increasing historical observation times at or before current ego time')
        future = batch['future_times_s']
        if (future <= 0).any() or (future.diff(dim=1) <= 0).any():
            raise ValueError('future_times_s must be strictly positive and increasing')

    def physical_coordinates(self, batch, nr, na):
        """Encode current-frame geometry without resampling scalar radial Doppler.

        Each history cell carries current-ego radius/bearing, observation age,
        its actual sensor line of sight/origin, and sensor ground velocity.
        This distinguishes translated/rotated observations of the same world
        location and makes the relative-Doppler measurement frame explicit.
        """
        b, t, _, r, a = batch['radar'].shape
        radius = batch['range_m'][:, None, :, None]
        angle = batch['azimuth_rad'][:, None, None, :]
        fallback_xy = torch.stack((radius * angle.cos(), radius * angle.sin()), -1).expand(-1, t, -1, -1, -1)
        xy = batch.get('radar_cartesian', fallback_xy)
        fallback_los = torch.stack((angle.cos().expand(b, t, r, a), angle.sin().expand(b, t, r, a)), -1)
        los = batch.get('radar_los', fallback_los)
        def pool(value):
            value = value.reshape(b * t, r, a, 2).permute(0, 3, 1, 2)
            return F.adaptive_avg_pool2d(value, (nr, na)).permute(0, 2, 3, 1).reshape(b, t, nr, na, 2)
        xy, los = pool(xy), pool(los)
        los = F.normalize(los, dim=-1, eps=1e-6)
        distance = xy.norm(dim=-1)
        direction = xy / distance.clamp_min(1e-6)[..., None]
        shape = (b, t, nr, na)
        origin = batch['radar_pose'][..., :2, 3] if 'radar_pose' in batch else xy.new_zeros(b, t, 2)
        sensor_velocity = batch.get('radar_sensor_velocity', xy.new_zeros(b, t, 2))
        if not self.config.use_ego:
            # no-ego ablation removes velocity conditioning as well as its MLP.
            sensor_velocity = torch.zeros_like(sensor_velocity)
        return torch.cat((torch.stack((distance / self.config.range_scale,
                                      direction[..., 1], direction[..., 0],
                                      batch['time_offsets_s'][:, :, None, None].expand(shape)), -1),
                          los, origin[:, :, None, None].expand(*shape, 2) / self.config.range_scale,
                          sensor_velocity[:, :, None, None].expand(*shape, 2) / self.config.doppler_scale), -1)

    def risk_from_context(self, tokens: torch.Tensor, ego: torch.Tensor) -> dict[str, torch.Tensor]:
        """Canonical physical head, shared with frozen-feature Risk refinement."""
        if (tokens.ndim != 3 or tokens.shape[1:] != (self.config.num_queries, self.config.hidden_dim)
                or ego.shape != (tokens.shape[0], self.config.hidden_dim)):
            raise ValueError('Invalid risk attention tokens or encoded Ego context')
        risk_query = self.risk_query[None].expand(tokens.shape[0], -1, -1)
        risk_feature = self.risk_attention(risk_query, tokens, tokens, need_weights=False)[0][:, 0]
        raw = self.risk_head(torch.cat((risk_feature, ego), dim=-1))
        risk = torch.cat((raw[:, :3].sigmoid(), F.softplus(raw[:, 3:])), dim=-1)
        return dict(risk_logits=raw[:, :3], risk=risk)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        from .risk_contract import validate_batch
        validate_batch(batch,self.config.risk_horizon_schema)
        if self.config.use_detection and 'detection_class_schema_id' in batch:
            expected = int(self.config.detection_class_schema == 'agent_v1')
            if not bool((batch['detection_class_schema_id'] == expected).all()):
                raise ValueError('Detector input class schema differs from checkpoint/model configuration')
        self._validate_inputs(batch)
        if not self.config.use_temporal:
            batch = dict(batch)
            for key in ('radar', 'time_offsets_s', 'radar_cartesian', 'radar_los',
                        'radar_pose', 'radar_sensor_velocity', 'radar_doppler_valid'):
                if key in batch:
                    batch[key] = batch[key][:, -1:]
        radar = self.normalize_radar(batch['radar'])
        b, t, _, r, a = radar.shape
        doppler_valid = batch.get('radar_doppler_valid', torch.ones((b, t, r, a), device=radar.device, dtype=torch.bool)).bool()
        radar = torch.stack((radar[:, :, 0], torch.where(doppler_valid, radar[:, :, 1], 0.)), dim=2)
        if self.config.backbone == 'cnn2d':
            feature = self.encoder(radar.reshape(b * t, 2, r, a))
        else:
            temporal = self.encoder(radar)
            feature = temporal.reshape(b * t, self.config.hidden_dim, *temporal.shape[-2:])
        # Detection keeps stride-four detail; language/risk attention stays compact.
        detection_spatial = feature.reshape(b, t, self.config.hidden_dim, *feature.shape[-2:])[:, -1]
        nr = min(feature.shape[-2], self.config.max_grid_size)
        na = min(feature.shape[-1], self.config.max_grid_size)
        # Bound the key/value sequence length for real high-resolution RA maps.
        feature = F.adaptive_avg_pool2d(feature, (nr, na))
        feature = feature.reshape(b, t, self.config.hidden_dim, nr, na).permute(0, 1, 3, 4, 2)
        if self.config.use_doppler:
            fraction = F.adaptive_avg_pool2d(doppler_valid.reshape(b * t, 1, r, a).to(feature.dtype), (nr, na))
            fraction = fraction.reshape(b, t, nr, na, 1)
            feature = feature + self.validity_encoder(fraction)
        coords = self.physical_coordinates(batch, nr, na)
        feature = (feature + self.position_encoder(coords)).reshape(b, t * nr * na, -1)
        queries = self.radar_queries[None].expand(b, -1, -1)
        tokens = self.token_norm(queries + self.temporal_attention(queries, feature, feature, need_weights=False)[0])
        tokens = self.output_norm(tokens + self.token_mlp(tokens))
        # v2 provides full signed vectors. Legacy v1 tensors remain readable,
        # with longitudinal-only acceleration when that vector was not stored.
        velocity = batch.get('ego_velocity', torch.stack((batch['ego'][:, 0], torch.zeros_like(batch['ego'][:, 0])), -1))
        acceleration = batch.get('ego_acceleration', torch.stack((batch['ego'][:, 1], torch.zeros_like(batch['ego'][:, 1])), -1))
        ego_state = torch.cat((velocity, acceleration, batch['ego'][:, 2:3]), -1)
        if not self.config.use_ego:
            ego_state = torch.zeros_like(ego_state)
        ego = self.ego_encoder(ego_state / ego_state.new_tensor([15., 15., 5., 5., 1.]))
        risk_output = self.risk_from_context(tokens, ego)
        risk, raw_logits = risk_output['risk'], risk_output['risk_logits']
        planner_tokens = tokens
        if self.config.planner_ego_context:
            # The planner must observe acceleration/yaw and signed velocity
            # directly, rather than infer them through five risk scalars. Keep
            # this extra token out of the pretrained detection/risk attentions.
            context = ego if self.config.use_ego else torch.zeros_like(ego)
            planner_tokens = torch.cat((tokens, context[:, None]), dim=1)
        if self.config.use_detection:
            output = self.detector(detection_spatial, tokens, ego, batch)
            if self.config.detection_candidate_policy == 'supervised_sector_v1':
                from .annotation_domain import configured_candidate_mask
                limits = self._candidate_sector.to(output['detection_anchors']).expand(b, -1)
                output['detection_candidate_mask'] = configured_candidate_mask(
                    output['detection_anchors'], batch, limits)
                output['detection_candidate_limits'] = limits
            # Tensor aliases support existing raw physical metrics. Public JSON
            # exports these fields after confidence/NMS, aligned to detections.
            output.update(radar_tokens=planner_tokens, risk_logits=raw_logits, risk=risk,
                          object_logits=output['detection_logits'], agent_state=output['detection_state'],
                          agent_future=output['detection_future'],
                          agent_future_velocity=output['detection_future_velocity'])
            return output
        objects = self.object_queries[None].expand(b, -1, -1)
        objects = self.object_norm(objects + self.object_attention(objects, tokens, tokens, need_weights=False)[0] + ego[:, None])
        state = self.state_head(objects)
        offsets = self.future_head(objects).reshape(b, self.config.max_agents, self.config.horizon_steps, 2)
        # A learned residual permits acceleration and turning, unlike a fixed CV decoder.
        future = state[:, :, None, :2] + state[:, :, None, 3:5] * batch['future_times_s'][:, None, :, None] + offsets
        future_velocity = state[:, :, None, 3:5] + self.future_velocity_head(objects).reshape(b, self.config.max_agents, self.config.horizon_steps, 2)
        output = dict(radar_tokens=planner_tokens, risk_logits=raw_logits, risk=risk,
                    object_logits=self.object_head(objects).squeeze(-1),
                    agent_state=state, agent_future=future, agent_future_velocity=future_velocity)
        return output
