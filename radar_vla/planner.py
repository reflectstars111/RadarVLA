"""One causal physical-language planner with a learned reasoning budget.

Both backends use the same variable-agent grammar, continuous metric teachers,
compact spline controls, and differentiable physical losses. Ground truth is
used only to construct supervised targets, never to route inference.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .curves import decode_spline, fit_control_points
from .tokenizer import PhysicalTokenizer, token_cross_entropy


@dataclass
class PlannerConfig:
    hidden_dim: int = 64
    radar_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    max_length: int = 512
    bins: int = 64
    coordinate_limit_m: float = 80.
    velocity_limit_mps: float = 40.
    max_agents: int = 8
    horizon_steps: int = 6
    short_horizon_steps: int | None = None  # Optional legacy regular-grid alias, validated below.
    trajectory_controls: int = 4
    road_controls: int = 4
    short_max_agents: int = 2
    short_horizon_s: float = 1.
    future_step_s: float = .5
    short_collision_threshold: float = .5
    short_brake_threshold_mps2: float = 3.
    short_clearance_threshold_m: float = .5
    reasoning_policy: str = 'adaptive'
    risk_source: str = 'predicted'
    use_risk_token: bool = True
    ego_trajectory_weight: float = 1.
    agent_trajectory_weight: float = 1.
    road_loss_weight: float = 1.
    doppler_loss_weight: float = .2
    kinematic_loss_weight: float = .1
    smooth_loss_weight: float = .001
    jerk_weight: float = .1
    max_instruction_bytes: int = 96

    def __post_init__(self):
        if self.hidden_dim <= 0 or self.num_heads <= 0 or self.hidden_dim % self.num_heads:
            raise ValueError('hidden_dim must be positive and divisible by num_heads')
        if min(self.radar_dim, self.num_layers, self.max_length, self.max_agents,
               self.horizon_steps, self.short_max_agents,
               self.max_instruction_bytes) <= 0:
            raise ValueError('model, sequence, agent, and horizon dimensions must be positive')
        if self.short_horizon_steps is not None:
            if self.short_horizon_steps < 1:
                raise ValueError('short_horizon_steps must be positive when supplied')
            legacy_horizon = self.short_horizon_steps * self.future_step_s
            if self.short_horizon_s != 1. and not math.isclose(self.short_horizon_s, legacy_horizon):
                raise ValueError('short_horizon_steps conflicts with short_horizon_s')
            self.short_horizon_s = legacy_horizon
        if min(self.trajectory_controls, self.road_controls) < 2:
            raise ValueError('spline representations require at least two controls')
        if self.reasoning_policy not in ('adaptive', 'always_long', 'always_short'):
            raise ValueError('reasoning_policy must be adaptive, always_long, or always_short')
        if self.risk_source not in ('predicted', 'none', 'oracle'):
            raise ValueError('risk_source must be predicted, none, or oracle')
        if self.risk_source == 'none':
            self.use_risk_token = False
        for name in ('short_horizon_s', 'future_step_s', 'short_brake_threshold_mps2'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not 0 < self.short_collision_threshold <= 1:
            raise ValueError('short_collision_threshold must lie in (0,1]')
        for name in ('short_clearance_threshold_m', 'ego_trajectory_weight', 'agent_trajectory_weight',
                     'road_loss_weight', 'doppler_loss_weight', 'kinematic_loss_weight',
                     'smooth_loss_weight', 'jerk_weight'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative')


def _layout(mode: str, config: PlannerConfig, agent_count: int | None = None):
    """Return typed fields for a specified number of present agents."""
    if mode not in ('short', 'long'):
        raise ValueError('unknown reasoning mode')
    maximum = min(config.short_max_agents, config.max_agents) if mode == 'short' else config.max_agents
    agents = maximum if agent_count is None else agent_count
    if not 0 <= agents <= maximum:
        raise ValueError('agent count exceeds the configured grammar')
    layout = ['<SHORT>' if mode == 'short' else '<LONG>']
    if mode == 'long':
        layout += ['<ROAD>'] + ['position'] * (2 * config.road_controls) + ['width', '<AGENTS>']
    else:
        layout += ['<CRITICAL_DYNAMICS>']
    for _ in range(agents):
        layout += ['<AGENT>', 'position', 'position', 'velocity', 'velocity', 'velocity']
    layout += ['<END_AGENTS>']
    if mode == 'long':
        layout += ['<AGENT_FUTURE>']
        for _ in range(agents):
            layout += ['<TRAJECTORY>'] + ['position'] * (2 * config.trajectory_controls)
    layout += ['<IMMEDIATE_ACTION>' if mode == 'short' else '<EGO_FUTURE>', '<TRAJECTORY>']
    # Ego t=0 is the known current-frame origin, encoded implicitly.
    layout += ['position'] * (2 * (config.trajectory_controls - 1)) + ['<EOS>']
    return layout, agents, config.trajectory_controls


def _is_hazard(risk: Tensor, mask: Tensor, config: PlannerConfig) -> Tensor:
    known = mask.bool() & torch.isfinite(risk)
    return (((risk[..., :2] >= config.short_collision_threshold) & known[..., :2]).any(-1)
            | ((risk[..., 3] <= config.short_clearance_threshold_m) & known[..., 3])
            | ((risk[..., 4] >= config.short_brake_threshold_mps2) & known[..., 4]))


def _teacher_mode(batch: dict, index: int, config: PlannerConfig) -> str | None:
    if config.reasoning_policy != 'adaptive':
        return 'long' if config.reasoning_policy == 'always_long' else 'short'
    if 'risk_target' not in batch or 'risk_mask' not in batch:
        return None
    risk, mask = batch['risk_target'][index], batch['risk_mask'][index]
    mask = mask.bool() & torch.isfinite(risk)
    if bool(_is_hazard(risk, mask, config)):
        return 'short'
    # Both imminent collision horizons must be observed. Continuous dimensions
    # are also required unless the observation explicitly contains no agents.
    present = bool(batch['agent_mask'][index].any())
    if bool(mask[:2].all()) and (not present or bool(mask[3:5].all())):
        return 'long'
    return None


def _time_grid(batch: dict, index: int, config: PlannerConfig) -> Tensor:
    if 'future_times_s' in batch:
        times = batch['future_times_s']
        times = times[index] if times.ndim == 2 else times
    else:
        times = torch.arange(1, config.horizon_steps + 1, device=batch['agent_state'].device) * config.future_step_s
    times = times.to(device=batch['agent_state'].device, dtype=torch.float32)
    if times.ndim != 1 or not len(times) or not bool(torch.isfinite(times).all()):
        raise ValueError('future_times_s must be a nonempty finite vector per sample')
    if bool((times <= 0).any()) or bool((times[1:] <= times[:-1]).any()):
        raise ValueError('future_times_s must be positive and strictly increasing')
    return times


def _critical_agents(batch: dict, index: int, config: PlannerConfig) -> list[int]:
    active = batch['agent_mask'][index].bool()
    if not bool(active.any()):
        return []
    if 'agent_risk' in batch and 'agent_risk_mask' in batch:
        critical = _is_hazard(batch['agent_risk'][index], batch['agent_risk_mask'][index], config) & active
    elif 'critical_agent_mask' in batch:
        critical = batch['critical_agent_mask'][index].bool() & active
    else:
        raise ValueError('SHORT teachers require counterfactual per-agent risk labels or critical_agent_mask')
    selected = critical.nonzero().flatten().tolist()
    if 'critical_agent_order' in batch:
        order = batch['critical_agent_order'][index].tolist()
        if len(set(order)) != len(order) or set(order) != set(range(len(active))):
            raise ValueError('critical_agent_order must be a permutation of all agent slots')
        selected = [a for a in order if a in selected]
    elif selected:
        # Collision at the earliest horizon has priority; then required braking
        # and signed box clearance. Unknown quantities do not become zero risk.
        if 'agent_risk' not in batch:
            raise ValueError('critical_agent_mask requires a counterfactual critical_agent_order')
        values, masks = batch['agent_risk'][index], batch['agent_risk_mask'][index]
        def key(a):
            collision = [h for h in range(3) if bool(masks[a, h])
                         and float(values[a, h]) >= config.short_collision_threshold]
            brake = -float(values[a, 4]) if bool(masks[a, 4]) else float('inf')
            clearance = float(values[a, 3]) if bool(masks[a, 3]) else float('inf')
            return (min(collision, default=3), brake, clearance, a)
        selected.sort(key=key)
    return selected[:min(config.short_max_agents, config.max_agents)]


def create_supervision(batch: dict, tokenizer: PhysicalTokenizer, config: PlannerConfig) -> dict:
    """Build compact structured teachers and retain their original physical GT.

    Only present agents enter the semantic sequence. Unavailable labels are PAD
    masked, not UNKNOWN states or zero-valued physical teachers. Metadata holds
    source indices solely for supervised loss alignment; generated indices at
    inference are local object hypotheses, not fabricated persistent track IDs.
    """
    state = batch['agent_state']
    device = state.device
    token_rows, value_rows, metadata = [], [], []
    for b in range(len(state)):
        ids, values = [], []
        row = {'mode': _teacher_mode(batch, b, config), 'agents': [], 'road': None, 'ego': None}
        def tag(name):
            ids.append(tokenizer.token(name)); values.append(float('nan'))
        def physical(value, valid, kind='position'):
            index = len(ids)
            number = float(value.detach()) if isinstance(value, Tensor) else float(value)
            if bool(valid) and not math.isfinite(number):
                raise ValueError('a physical target marked valid is not finite')
            ids.append(tokenizer.encode_scalar(number, kind) if bool(valid) else tokenizer.pad_id)
            values.append(number if bool(valid) else float('nan'))
            return index
        def trajectory(points, times, valid, origin_anchored=False):
            controls, knots, fitted = fit_control_points(points, times, valid, config.trajectory_controls)
            tag('<TRAJECTORY>')
            emitted = controls[1:] if origin_anchored else controls
            indices = [[physical(value, fitted) for value in point] for point in emitted]
            return {'indices': indices, 'control_times': knots, 'query_times': times[1:],
                    'origin_anchored': origin_anchored,
                    'target': points[1:], 'mask': valid[1:].bool(), 'fitted': fitted}
        mode = row['mode']
        if mode is None:
            token_rows.append([tokenizer.pad_id]); value_rows.append([float('nan')]); metadata.append(row)
            continue
        times = _time_grid(batch, b, config)
        future_mask = times <= config.short_horizon_s + 1e-6 if mode == 'short' else torch.ones_like(times, dtype=torch.bool)
        times = times[future_mask]
        if not len(times):
            raise ValueError('short_horizon_s must include at least one observed future time')
        full_times = torch.cat((times.new_zeros(1), times))
        tag('<SHORT>' if mode == 'short' else '<LONG>')
        if mode == 'long':
            tag('<ROAD>')
            required = ('road_centerline', 'road_centerline_mask', 'road_width', 'road_mask')
            if any(name not in batch for name in required):
                raise ValueError('LONG teachers require map-derived road centerline, width and validity masks')
            observed = batch['road_centerline_mask'][b].bool()
            centerline = batch['road_centerline'][b][observed]
            width_known = bool(batch['road_mask'][b].all())
            fitted = False
            road_times = torch.linspace(0, 1, config.road_controls, device=device)
            controls = state.new_zeros(config.road_controls, 2)
            if len(centerline) >= 2:
                arc = torch.cat((state.new_zeros(1), torch.linalg.vector_norm(centerline.diff(dim=0), dim=-1).cumsum(0)))
                # Exact repeated vertices add no geometric information.
                keep = torch.cat((torch.ones(1, dtype=torch.bool, device=device), arc.diff() > 1e-6))
                centerline, arc = centerline[keep], arc[keep]
                if len(arc) >= 2:
                    controls, road_times, fitted = fit_control_points(centerline, arc, num_controls=config.road_controls)
            indices = [[physical(v, fitted) for v in point] for point in controls]
            width = batch['road_width'][b].reshape(-1)[0]
            if width_known and float(width) <= 0:
                raise ValueError('observed road width must be positive')
            width_index = physical(width, width_known)
            row['road'] = {'indices': indices, 'control_times': road_times, 'fitted': fitted,
                           'target': centerline, 'query_times': arc if len(centerline) >= 2 else None,
                           'width_index': width_index, 'width_target': width, 'width_mask': width_known}
            tag('<AGENTS>')
            order = batch['agent_mask'][b].nonzero().flatten().tolist()
        else:
            tag('<CRITICAL_DYNAMICS>')
            order = _critical_agents(batch, b, config)
        if len(order) > config.max_agents:
            raise ValueError('annotated agent count exceeds planner max_agents')
        for a in order:
            tag('<AGENT>')
            known = batch.get('agent_state_mask')
            indices = [physical(state[b, a, k], True if known is None else known[b, a, k], kind)
                       for k, kind in enumerate(('position', 'position', 'velocity', 'velocity', 'velocity'))]
            row['agents'].append({'source_index': a, 'state_indices': indices, 'trajectory': None})
        tag('<END_AGENTS>')
        if mode == 'long':
            tag('<AGENT_FUTURE>')
            for agent in row['agents']:
                a = agent['source_index']
                current_valid = (batch['agent_state_mask'][b, a, :2].all()
                                 if 'agent_state_mask' in batch else batch['agent_mask'][b, a])
                positions = torch.cat((state[b, a, :2][None], batch['agent_future'][b, a][future_mask]), 0)
                valid = torch.cat((current_valid.reshape(1), batch['agent_future_mask'][b, a][future_mask]))
                agent['trajectory'] = trajectory(positions, full_times, valid)
        tag('<IMMEDIATE_ACTION>' if mode == 'short' else '<EGO_FUTURE>')
        # In the current ego coordinate frame, its present position is exactly
        # the origin. This coordinate identity is not a missing-label fallback.
        ego_positions = torch.cat((state.new_zeros(1, 2), batch['ego_future'][b][future_mask]), 0)
        ego_valid = torch.cat((torch.ones(1, device=device, dtype=torch.bool), batch['ego_future_mask'][b][future_mask]))
        row['ego'] = trajectory(ego_positions, full_times, ego_valid, origin_anchored=True)
        tag('<EOS>')
        token_rows.append(ids); value_rows.append(values); metadata.append(row)
    length = max(map(len, token_rows))
    ids = torch.full((len(state), length), tokenizer.pad_id, device=device, dtype=torch.long)
    continuous = torch.full((len(state), length), float('nan'), device=device, dtype=torch.float32)
    for b, (tokens, values) in enumerate(zip(token_rows, value_rows)):
        ids[b, :len(tokens)] = torch.tensor(tokens, device=device)
        continuous[b, :len(values)] = torch.tensor(values, device=device)
    return {'target_ids': ids, 'continuous_targets': continuous, 'rows': metadata,
            'schema': 'radar_vla_physical_supervision_v2'}


def create_targets(batch: dict, tokenizer: PhysicalTokenizer, config: PlannerConfig) -> Tensor:
    """Compatibility interface; SFT should retain ``create_supervision`` metadata."""
    return create_supervision(batch, tokenizer, config)['target_ids']
class RiskConditionedPlanner(nn.Module):
    def __init__(self, config: PlannerConfig):
        super().__init__()
        self.config = config
        self.tokenizer = PhysicalTokenizer(config.bins, config.coordinate_limit_m, config.velocity_limit_mps)
        self.radar_projector = nn.Linear(config.radar_dim, config.hidden_dim)
        self.risk_projector = nn.Sequential(nn.Linear(5, config.hidden_dim), nn.GELU(), nn.Linear(config.hidden_dim, config.hidden_dim))
        self.embedding = nn.Embedding(self.tokenizer.vocab_size, config.hidden_dim, padding_idx=self.tokenizer.pad_id)
        self.position_embedding = nn.Embedding(config.max_length, config.hidden_dim)
        layer = nn.TransformerEncoderLayer(config.hidden_dim, config.num_heads, config.hidden_dim * 4,
                                           dropout=0., activation='gelu', batch_first=True, norm_first=True)
        self.decoder = nn.TransformerEncoder(layer, config.num_layers, norm=nn.LayerNorm(config.hidden_dim), enable_nested_tensor=False)
        self.lm_head = nn.Linear(config.hidden_dim, self.tokenizer.vocab_size)

    def encode_prefix(self, radar_tokens: Tensor, risk: Tensor, instructions: list[str]):
        """Return continuous prefix and padding mask for future LLM integration."""
        if radar_tokens.ndim != 3 or radar_tokens.shape[-1] != self.config.radar_dim:
            raise ValueError('radar_tokens must have shape [batch, tokens, radar_dim]')
        batch_size = radar_tokens.shape[0]
        if risk.shape != (batch_size, 5) or len(instructions) != batch_size:
            raise ValueError('risk [batch,5] and instructions must match the radar batch')
        if not bool(torch.isfinite(radar_tokens).all()) or not bool(torch.isfinite(risk).all()):
            raise ValueError('radar and risk conditions must be finite')
        words = [self.tokenizer.encode_instruction(text, self.config.max_instruction_bytes) for text in instructions]
        width = max(1, max(map(len, words)))
        ids = torch.full((batch_size, width), self.tokenizer.pad_id, dtype=torch.long, device=radar_tokens.device)
        for i, values in enumerate(words):
            if values:
                ids[i, :len(values)] = torch.tensor(values, device=ids.device)
        # Preserve continuous KRS; scale physical dimensions for the small MLP.
        scale = risk.new_tensor([1., 1., 1., self.config.coordinate_limit_m, 10.])
        pieces = [self.radar_projector(radar_tokens)]
        if self.config.use_risk_token:
            pieces.append(self.risk_projector(risk / scale).unsqueeze(1))
        pieces.append(self.embedding(ids))
        prefix = torch.cat(pieces, dim=1)
        mask = torch.cat([torch.zeros(batch_size, radar_tokens.shape[1] + int(self.config.use_risk_token), dtype=torch.bool, device=ids.device), ids == self.tokenizer.pad_id], dim=1)
        return prefix, mask

    def _decode_inputs(self, prefix, prefix_mask, input_ids):
        length = prefix.shape[1] + input_ids.shape[1]
        if length > self.config.max_length:
            raise ValueError(f'sequence length {length} exceeds max_length={self.config.max_length}')
        hidden = torch.cat([prefix, self.embedding(input_ids)], dim=1)
        hidden = hidden + self.position_embedding(torch.arange(length, device=hidden.device))[None]
        mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=hidden.device), diagonal=1)
        padding = torch.cat([prefix_mask, input_ids == self.tokenizer.pad_id], dim=1)
        states = self.decoder(hidden, mask=mask, src_key_padding_mask=padding)
        return self.lm_head(states[:, prefix.shape[1]:])

    def forward(self, radar_tokens: Tensor, risk: Tensor, instructions: list[str], target_ids: Tensor) -> Tensor:
        if target_ids.ndim != 2 or target_ids.shape[0] != radar_tokens.shape[0] or target_ids.shape[1] == 0:
            raise ValueError('target_ids must be a nonempty [batch,length] tensor')
        prefix, mask = self.encode_prefix(radar_tokens, risk, instructions)
        inputs = torch.full_like(target_ids, self.tokenizer.bos_id)
        inputs[:, 1:] = target_ids[:, :-1]
        return self._decode_inputs(prefix, mask, inputs)

    def _allowed_next(self, result: list[int]) -> list[int]:
        tok, cfg = self.tokenizer, self.config
        if not result:
            modes = ('<SHORT>', '<LONG>') if cfg.reasoning_policy == 'adaptive' else (
                '<LONG>' if cfg.reasoning_policy == 'always_long' else '<SHORT>',)
            return [tok.token(mode) for mode in modes]
        mode = 'short' if result[0] == tok.token('<SHORT>') else 'long'
        empty_layout, _, _ = _layout(mode, cfg, 0)
        prefix_length = empty_layout.index('<END_AGENTS>')
        if len(result) < prefix_length:
            field = empty_layout[len(result)]
        else:
            maximum = min(cfg.short_max_agents, cfg.max_agents) if mode == 'short' else cfg.max_agents
            index, count = prefix_length, 0
            while True:
                if index == len(result):
                    return [tok.token('<END_AGENTS>')] + ([tok.token('<AGENT>')] if count < maximum else [])
                if result[index] == tok.token('<END_AGENTS>'):
                    break
                count += 1
                if len(result) < index + 6:
                    kind = ('position', 'position', 'velocity', 'velocity', 'velocity')[len(result) - index - 1]
                    return tok.physical_ids(kind)
                index += 6
            layout, _, _ = _layout(mode, cfg, count)
            field = layout[len(result)]
        if field in ('position', 'velocity'):
            return tok.physical_ids(field)
        if field == 'width':
            return [i for i in tok.physical_ids('position') if tok.decode_scalar(i) > 0]
        return [tok.token(field)]

    @torch.no_grad()
    def generate(self, radar_tokens: Tensor, risk: Tensor, instructions: list[str],
                 max_new_tokens: int = 256) -> list[list[int]]:
        """One-model greedy generation with learned mode and agent cardinality.

        Grammar limits types and maximum length; it does not substitute agent
        states, map values, safety actions, or GT risk. Only explicitly selected
        Q5 ablations force SHORT/LONG. A valid format is not a safety guarantee.
        """
        if max_new_tokens <= 0:
            raise ValueError('max_new_tokens must be positive')
        prefix, mask = self.encode_prefix(radar_tokens, risk, instructions)
        rows = []
        for b in range(radar_tokens.shape[0]):
            result = []
            for _ in range(min(max_new_tokens, self.config.max_length - prefix.shape[1])):
                inputs = torch.tensor([[self.tokenizer.bos_id] + result], device=radar_tokens.device)
                logits = self._decode_inputs(prefix[b:b + 1], mask[b:b + 1], inputs)[0, -1]
                allowed = self._allowed_next(result)
                selected = allowed[int(logits[allowed].argmax())]
                result.append(selected)
                if selected == self.tokenizer.eos_id:
                    break
            rows.append(result)
        return rows

    def decode(self, ids: list[int], future_times_s=None) -> dict:
        """Parse physical controls and reconstruct trajectories on metric times.

        ``future_times_s`` is the protocol's full future time grid, shared with
        training. Without it the configured regular grid is used. Agent numbers
        are local hypotheses in this output, not persistent dataset track IDs.
        """
        tok, cfg = self.tokenizer, self.config
        ids = [int(i) for i in ids]
        while ids and ids[-1] == tok.pad_id:
            ids.pop()
        failure = {'valid': False, 'mode': None, 'ego_trajectory': [], 'errors': []}
        if not ids or ids[0] not in (tok.token('<SHORT>'), tok.token('<LONG>')):
            failure['errors'] = ['missing SHORT/LONG mode token']
            return failure
        mode = 'short' if ids[0] == tok.token('<SHORT>') else 'long'
        failure['mode'] = mode
        count = ids.count(tok.token('<AGENT>'))
        try:
            layout, _, controls_count = _layout(mode, cfg, count)
        except ValueError as error:
            failure['errors'] = [str(error)]
            return failure
        if len(ids) != len(layout):
            failure['errors'] = [f'incomplete or trailing sequence: expected {len(layout)} tokens, got {len(ids)}']
            return failure
        values = {}
        for index, (token_id, field) in enumerate(zip(ids, layout)):
            if field in ('position', 'velocity', 'width'):
                if token_id == tok.pad_id:
                    values[index] = None
                    continue
                try:
                    values[index] = tok.decode_scalar(token_id, 'position' if field == 'width' else field)
                    if field == 'width' and values[index] <= 0:
                        raise ValueError('road width must be positive')
                except ValueError as error:
                    failure['errors'] = [str(error)]
                    return failure
            elif token_id != tok.token(field):
                failure['errors'] = [f'expected {field} at token {index}']
                return failure
        times = ([cfg.future_step_s * i for i in range(1, cfg.horizon_steps + 1)]
                 if future_times_s is None else [float(t) for t in future_times_s])
        if not times or any(not math.isfinite(t) or t <= 0 for t in times) or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError('future_times_s must be positive and strictly increasing')
        if mode == 'short':
            times = [t for t in times if t <= cfg.short_horizon_s + 1e-6]
        if not times:
            raise ValueError('future_times_s must include the short planning horizon')
        control_times = torch.linspace(0, times[-1], controls_count, dtype=torch.float64).tolist()
        def points_at(index, number):
            result = []
            for n in range(number):
                x, y = values[index + 2 * n], values[index + 2 * n + 1]
                result.append(None if x is None or y is None else [x, y])
            return result
        def dense(points):
            return ([None] * len(times) if any(p is None for p in points) else
                    decode_spline(torch.tensor(points, dtype=torch.float64), control_times, times).tolist())
        road = None
        if mode == 'long':
            road_controls = points_at(layout.index('<ROAD>') + 1, cfg.road_controls)
            width = values[layout.index('<ROAD>') + 1 + 2 * cfg.road_controls]
            road = {'centerline_controls': road_controls, 'width_m': width,
                    'parameter': 'normalized_arc_length',
                    'control_parameters': torch.linspace(0, 1, cfg.road_controls, dtype=torch.float64).tolist(),
                    'complete': all(p is not None for p in road_controls) and width is not None}
        agent_positions = [i for i, field in enumerate(layout) if field == '<AGENT>']
        states = [dict(local_id=a, **dict(zip(('x', 'y', 'vr', 'vx', 'vy'),
                                             (values[i + k] for k in range(1, 6)))))
                  for a, i in enumerate(agent_positions)]
        trajectory_positions = [i + 1 for i, field in enumerate(layout) if field == '<TRAJECTORY>']
        agent_controls = [points_at(index, controls_count) for index in trajectory_positions[:-1]]
        ego_controls = [[0., 0.]] + points_at(trajectory_positions[-1], controls_count - 1)
        ego = dense(ego_controls)
        complete = (all(point is not None for point in ego_controls)
                    and all(value is not None for state in states for key, value in state.items() if key != 'local_id')
                    and all(p is not None for controls in agent_controls for p in controls)
                    and (mode == 'short' or road['complete']))
        return {'valid': True, 'physically_complete': complete, 'mode': mode, 'road': road,
                'agent_states': states, 'agent_control_points': agent_controls,
                'agent_trajectories': [dense(points) for points in agent_controls],
                'ego_control_points': ego_controls, 'control_times_s': control_times,
                'ego_trajectory': ego, 'ego_times_s': times,
                'ego_complete': all(p is not None for p in ego), 'token_count': len(ids),
                'errors': [], 'schema': 'radar_vla_structured_v2'}


def planner_physics_loss(logits: Tensor, supervision: dict, batch: dict,
                         tokenizer: PhysicalTokenizer, config: PlannerConfig) -> dict[str, Tensor]:
    """Physical losses on generated-token expectations and decoded splines.

    They constrain the language planner itself, not just the perception head.
    Missing GT remains masked. Doppler labels must be actual associated radar
    observations; annotation velocities never masquerade as radar observations.
    State/future kinematics compare the spline's present velocity and position
    with generated Agent fields, allowing nonconstant future motion.
    """
    from .curves import spline_basis
    if logits.shape[:-1] != supervision['target_ids'].shape:
        raise ValueError('planner physics logits do not align with supervision')
    zero = logits.float().sum() * 0
    components = {name: [] for name in ('ego_trajectory', 'agent_trajectory', 'road', 'doppler', 'kinematic', 'smooth')}
    def expected(b, indices, kind='position'):
        index = torch.as_tensor(indices, device=logits.device, dtype=torch.long)
        offset = tokenizer.position_offset if kind == 'position' else tokenizer.velocity_offset
        probabilities = logits[b, index, offset:offset + tokenizer.bins].float().softmax(-1)
        return (probabilities * tokenizer.centers(kind, device=logits.device)).sum(-1)
    def trajectory(b, item):
        controls = expected(b, item['indices'])
        if item.get('origin_anchored'):
            controls = torch.cat((controls.new_zeros(1, 2), controls), 0)
        dense = decode_spline(controls, item['control_times'], item['query_times'])
        mask = item['mask'] & torch.isfinite(item['target']).all(-1)
        loss = F.smooth_l1_loss(dense[mask], item['target'][mask].float()) if bool(mask.any()) else zero
        return controls, dense, loss
    for b, row in enumerate(supervision['rows']):
        if row['mode'] is None:
            continue
        if row['road'] is not None:
            road = row['road']
            if road['fitted']:
                controls = expected(b, road['indices'])
                decoded = decode_spline(controls, road['control_times'], road['query_times'])
                components['road'].append(F.smooth_l1_loss(decoded, road['target'].float()))
            if road['width_mask']:
                components['road'].append(F.smooth_l1_loss(expected(b, road['width_index']), road['width_target'].float()))
        for agent in row['agents']:
            a = agent['source_index']
            index = agent['state_indices']
            position = expected(b, index[:2])
            velocity = expected(b, index[2:], 'velocity')
            vr, xy_velocity = velocity[0], velocity[1:]
            if ('agent_radial_observed' in batch and 'agent_radial_mask' in batch
                    and bool(batch['agent_radial_mask'][b, a])):
                if 'agent_los' not in batch or 'ego_velocity' not in batch:
                    raise ValueError('Doppler consistency requires measured LOS and ego velocity')
                # A power-weighted average of radial cells projects onto the
                # same weighted average of LOS vectors. Its norm can be <1;
                # normalizing would change the measured physical quantity.
                projection = batch.get('agent_doppler_projection', batch['agent_los'])
                los = projection[b, a].float()
                if not bool(torch.isfinite(los).all()) or float(torch.linalg.vector_norm(los)) > 1.001:
                    raise ValueError('associated radar projection must be finite with norm <=1')
                observer_velocity = (batch['agent_sensor_velocity'][b, a] if 'agent_sensor_velocity' in batch
                                     else batch['ego_velocity'][b]).float()
                predicted_radial = ((xy_velocity - observer_velocity) * los).sum()
                observed = batch['agent_radial_observed'][b, a].float()
                if not bool(torch.isfinite(observed)):
                    raise ValueError('radar observation marked valid must be finite')
                components['doppler'].append((F.smooth_l1_loss(predicted_radial, observed)
                                              + F.smooth_l1_loss(vr, observed)) / 2)
            item = agent['trajectory']
            if item is not None:
                controls, _, loss = trajectory(b, item)
                if bool(item['mask'].any()):
                    components['agent_trajectory'].append(loss)
                basis = spline_basis(item['control_times'], item['control_times'][:1], derivative=1).to(controls)
                initial_velocity = (basis @ controls)[0]
                components['kinematic'].append((F.smooth_l1_loss(controls[0], position)
                                                 + F.smooth_l1_loss(initial_velocity, xy_velocity)) / 2)
        ego = row['ego']
        if ego is not None:
            controls, _, loss = trajectory(b, ego)
            if bool(ego['mask'].any()):
                components['ego_trajectory'].append(loss)
            # Time derivatives are analytic and measured in m/s² and m/s³.
            # Evaluate at interval midpoints to avoid ambiguous knot jerk.
            knots = ego['control_times']
            queries = (knots[1:] + knots[:-1]) / 2
            acceleration = spline_basis(knots, queries, derivative=2).to(controls) @ controls
            jerk = spline_basis(knots, queries, derivative=3).to(controls) @ controls
            durations = knots.diff().to(controls)
            smooth = ((acceleration.square().sum(-1) + config.jerk_weight * jerk.square().sum(-1))
                      * durations).sum() / durations.sum()
            components['smooth'].append(smooth)
    losses = {name: torch.stack(values).mean() if values else zero for name, values in components.items()}
    losses['total'] = (config.ego_trajectory_weight * losses['ego_trajectory']
                       + config.agent_trajectory_weight * losses['agent_trajectory']
                       + config.road_loss_weight * losses['road']
                       + config.doppler_loss_weight * losses['doppler']
                       + config.kinematic_loss_weight * losses['kinematic']
                       + config.smooth_loss_weight * losses['smooth'])
    return losses
