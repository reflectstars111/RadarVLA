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
    road_format: str = 'legacy'
    max_road_segments: int = 8
    short_max_agents: int = 2
    short_horizon_s: float = 1.
    future_step_s: float = .5
    short_collision_threshold: float = .5
    short_brake_threshold_mps2: float = 3.
    short_clearance_threshold_m: float = .5
    reasoning_policy: str = 'adaptive'
    risk_source: str = 'predicted'
    risk_projection: str = 'none'  # Explicit opt-in; old checkpoints preserve their inputs.
    risk_horizon_schema: str = '1_2_3_v1'
    use_risk_token: bool = True
    ego_trajectory_weight: float = 1.
    agent_trajectory_weight: float = 1.
    road_loss_weight: float = 1.
    doppler_loss_weight: float = .2
    kinematic_loss_weight: float = .1
    # Zero preserves the objective for configs stored before these additions.
    ego_initial_velocity_weight: float = 0.
    mode_loss_weight: float = 0.
    smooth_loss_weight: float = .001
    jerk_weight: float = .1
    max_instruction_bytes: int = 96
    action_representation: str = 'token_spline_v1'
    action_max_acceleration_mps2: float = 8.
    action_max_jerk_mps3: float = 20.
    action_mode_loss_weight: float = 1.
    action_description_loss_weight: float = .1
    action_adoption_transition_s: float = .5

    def __post_init__(self):
        from .risk_contract import contract
        contract(self.risk_horizon_schema)
        if self.action_representation not in ('token_spline_v1', 'parallel_acceleration_v1'):
            raise ValueError('unknown action_representation')
        for name in ('action_max_acceleration_mps2', 'action_max_jerk_mps3'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not math.isfinite(self.action_mode_loss_weight) or self.action_mode_loss_weight < 0:
            raise ValueError('action_mode_loss_weight must be finite and nonnegative')
        if not math.isfinite(self.action_description_loss_weight) or self.action_description_loss_weight < 0:
            raise ValueError('action_description_loss_weight must be finite and nonnegative')
        if not math.isfinite(self.action_adoption_transition_s) or self.action_adoption_transition_s < 0:
            raise ValueError('action_adoption_transition_s must be finite and nonnegative')
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
        if self.action_representation == 'parallel_acceleration_v1' and (
                self.horizon_steps != 6 or not math.isclose(self.future_step_s, .5)
                or not math.isclose(self.short_horizon_s, 1.)):
            raise ValueError('parallel actions require six 0.5 s steps and a 1 s SHORT horizon')
        if min(self.trajectory_controls, self.road_controls) < 2:
            raise ValueError('spline representations require at least two controls')
        if self.reasoning_policy not in ('adaptive', 'always_long', 'always_short'):
            raise ValueError('reasoning_policy must be adaptive, always_long, or always_short')
        if self.risk_source not in ('predicted', 'none', 'oracle'):
            raise ValueError('risk_source must be predicted, none, or oracle')
        if self.risk_projection not in ('none', 'isotonic'):
            raise ValueError('risk_projection must be none or isotonic')
        if self.road_format not in ('legacy', 'segments') or self.max_road_segments < 1:
            raise ValueError('invalid road_format or max_road_segments')
        if self.risk_source == 'none':
            self.use_risk_token = False
        for name in ('short_horizon_s', 'future_step_s', 'short_brake_threshold_mps2'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not 0 < self.short_collision_threshold <= 1:
            raise ValueError('short_collision_threshold must lie in (0,1]')
        for name in ('short_clearance_threshold_m', 'ego_trajectory_weight', 'agent_trajectory_weight',
                     'road_loss_weight', 'doppler_loss_weight', 'kinematic_loss_weight',
                     'ego_initial_velocity_weight', 'mode_loss_weight',
                     'smooth_loss_weight', 'jerk_weight'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative')


def _layout(mode: str, config: PlannerConfig, agent_count: int | None = None,
            road_count: int | None = None):
    """Return typed fields for a specified number of present agents."""
    if mode not in ('short', 'long'):
        raise ValueError('unknown reasoning mode')
    maximum = min(config.short_max_agents, config.max_agents) if mode == 'short' else config.max_agents
    agents = maximum if agent_count is None else agent_count
    if not 0 <= agents <= maximum:
        raise ValueError('agent count exceeds the configured grammar')
    layout = ['<SHORT>' if mode == 'short' else '<LONG>']
    if mode == 'long':
        if config.road_format == 'segments':
            count = config.max_road_segments if road_count is None else road_count
            if not 0 <= count <= config.max_road_segments:
                raise ValueError('road segment count exceeds configured grammar')
            layout += ['<ROAD>']
            for _ in range(count):
                layout += ['<ROAD_SEGMENT>'] + ['position'] * (2 * config.road_controls) + ['width']
            layout += ['<END_ROADS>', '<AGENTS>']
        else:
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
    end = 3 if config.risk_horizon_schema == '0p5_1_2_v2' else 2
    return (((risk[..., :end] >= config.short_collision_threshold) & known[..., :end]).any(-1)
            | ((risk[..., 3] <= config.short_clearance_threshold_m) & known[..., 3])
            | ((risk[..., 4] >= config.short_brake_threshold_mps2) & known[..., 4]))


def _teacher_mode(batch: dict, index: int, config: PlannerConfig) -> str | None:
    if 'planner_supervision_available' in batch and not bool(batch['planner_supervision_available'][index]):
        return None
    if 'taskwise_protocol_id' in batch and int(batch['taskwise_protocol_id'][index]) == 1:
        from .task_supervision import local_mode, MODE_INDEX
        if config.reasoning_policy != 'adaptive':raise ValueError('Taskwise Local teachers require adaptive reasoning policy')
        if not bool(batch['mode_supervision_mask'][index]):return None
        r,m=batch['risk_target'][index].tolist(),batch['risk_mask'][index].tolist()
        label=dict(pcol=r[:3],pcol_mask=m[:3],dmin=r[3],dmin_mask=m[3],areq=r[4],areq_mask=m[4])
        if config.risk_horizon_schema == '0p5_1_2_v2':label['pcol_horizons_s']=[.5,1.,2.]
        mode,_=local_mode(label,set_known=bool(batch['local_set_known'][index]),agent_count=int(batch['agent_mask'][index].sum()),
            collision_threshold=config.short_collision_threshold,clearance_threshold=config.short_clearance_threshold_m,brake_threshold=config.short_brake_threshold_mps2)
        if int(batch['mode_target'][index]) != MODE_INDEX[mode]:raise ValueError('Local Mode target differs from shared policy')
        return None if mode=='UNKNOWN' else mode.lower()
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
    end = 3 if config.risk_horizon_schema == '0p5_1_2_v2' else 2
    if bool(mask[:end].all()) and (not present or bool(mask[3:5].all())):
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
    if 'agent_planner_mask' in batch:
        active = active & batch['agent_planner_mask'][index].bool()
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
    from .risk_contract import validate_batch
    validate_batch(batch,config.risk_horizon_schema)
    state = batch['agent_state']
    device = state.device
    token_rows, value_rows, metadata = [], [], []
    for b in range(len(state)):
        ids, values = [], []
        row = {'mode': _teacher_mode(batch, b, config), 'agents': [], 'road': None, 'ego': None}
        taskwise = 'taskwise_protocol_id' in batch and int(batch['taskwise_protocol_id'][b]) == 1
        row['taskwise']=taskwise
        row['masked_out_of_range_fields']=0
        def tag(name, valid=True):
            ids.append(tokenizer.token(name) if valid else tokenizer.pad_id); values.append(float('nan'))
        def physical(value, valid, kind='position'):
            index = len(ids)
            number = float(value.detach()) if isinstance(value, Tensor) else float(value)
            if bool(valid) and not math.isfinite(number):
                raise ValueError('a physical target marked valid is not finite')
            if taskwise and bool(valid) and abs(number)>tokenizer._limits(kind)[0]:
                valid=False;row['masked_out_of_range_fields']+=1
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
            road_section_start=len(ids)
            tag('<ROAD>')
            if config.road_format == 'segments':
                if 'road_segments' not in batch:
                    raise ValueError('LONG segmented teachers require road_segments')
                source_segments = batch['road_segments'][b]
                if not source_segments and not taskwise:
                    raise ValueError('LONG segmented teacher needs at least one sourced road segment')
                if len(source_segments) > config.max_road_segments:
                    if not taskwise:raise ValueError('road segment count exceeds configured maximum; never truncate')
                    source_segments=[]
            else:
                required = ('road_centerline', 'road_centerline_mask', 'road_width', 'road_mask')
                if any(name not in batch for name in required):
                    raise ValueError('LONG teachers require map-derived road centerline, width and validity masks')
                observed = batch['road_centerline_mask'][b].bool()
                source_segments = [{'centerline_ego_m': batch['road_centerline'][b][observed],
                                    'width_m': batch['road_width'][b].reshape(-1)[0],
                                    'width_valid': bool(batch['road_mask'][b].all())}]
            segment_rows = []
            road_any_valid=False
            for segment_index, segment in enumerate(source_segments):
                segment_start=len(ids)
                task_mask = batch['road_task_masks'][b][segment_index] if taskwise else None
                if config.road_format == 'segments':
                    tag('<ROAD_SEGMENT>')
                centerline = torch.as_tensor(segment['centerline_ego_m'], device=device, dtype=state.dtype)
                geometry_known = torch.as_tensor(task_mask['geometry'],device=device,dtype=torch.bool) if taskwise else torch.ones(len(centerline),device=device,dtype=torch.bool)
                complete_geometry=bool(geometry_known.all()) and len(centerline)>=2
                centerline=centerline[geometry_known]
                if len(centerline)>=2:
                    arc = torch.cat((state.new_zeros(1), torch.linalg.vector_norm(centerline.diff(dim=0), dim=-1).cumsum(0)))
                    keep = torch.cat((torch.ones(1, dtype=torch.bool, device=device), arc.diff() > 1e-6))
                    centerline, arc = centerline[keep], arc[keep]
                else:arc=state.new_zeros(len(centerline))
                geometry_usable=len(centerline)>=2
                if geometry_usable:
                    controls, road_times, fitted = fit_control_points(centerline, arc, num_controls=config.road_controls)
                elif taskwise:
                    controls=state.new_zeros(config.road_controls,2);road_times=torch.linspace(0,1,config.road_controls,device=device);fitted=False
                else:raise ValueError('road segment has no nonzero arc length')
                indices = [[physical(v, fitted and complete_geometry) for v in point] for point in controls]
                width_known = bool(task_mask['width']) if taskwise else bool(segment['width_valid'])
                width = 0. if segment['width_m'] is None else segment['width_m']
                width = torch.as_tensor(width, device=device, dtype=state.dtype)
                if width_known and float(width) <= 0:raise ValueError('observed road width must be positive')
                width_index = physical(width, width_known)
                segment_valid=geometry_usable or width_known
                road_any_valid=road_any_valid or segment_valid
                if taskwise and not segment_valid:
                    for index in range(segment_start,len(ids)):ids[index]=tokenizer.pad_id;values[index]=float('nan')
                segment_rows.append({'indices': indices, 'control_times': road_times, 'fitted': fitted and complete_geometry,
                                     'geometry_usable': geometry_usable,'target': centerline, 'query_times': arc,
                                     'width_index': width_index, 'width_target': width,'width_mask': width_known})
            if config.road_format == 'segments':
                tag('<END_ROADS>')
                row['road'] = {'segments': segment_rows}
            else:
                row['road'] = segment_rows[0]
            if taskwise and not road_any_valid:
                for index in range(road_section_start,len(ids)):ids[index]=tokenizer.pad_id;values[index]=float('nan')
            tag('<AGENTS>')
            planner_active = batch['agent_mask'][b].bool()
            if 'agent_planner_mask' in batch:
                planner_active = planner_active & batch['agent_planner_mask'][b].bool()
            order = planner_active.nonzero().flatten().tolist()
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
        ego_section_start=len(ids)
        tag('<IMMEDIATE_ACTION>' if mode == 'short' else '<EGO_FUTURE>')
        # In the current ego coordinate frame, its present position is exactly
        # the origin. This coordinate identity is not a missing-label fallback.
        ego_positions = torch.cat((state.new_zeros(1, 2), batch['ego_future'][b][future_mask]), 0)
        ego_valid = torch.cat((torch.ones(1, device=device, dtype=torch.bool), batch['ego_future_mask'][b][future_mask]))
        row['ego'] = trajectory(ego_positions, full_times, ego_valid, origin_anchored=True)
        if taskwise and not bool(ego_valid[1:].any()):
            for index in range(ego_section_start,len(ids)):ids[index]=tokenizer.pad_id;values[index]=float('nan')
        tag('<EOS>')
        token_rows.append(ids); value_rows.append(values); metadata.append(row)
    length = max(map(len, token_rows))
    ids = torch.full((len(state), length), tokenizer.pad_id, device=device, dtype=torch.long)
    continuous = torch.full((len(state), length), float('nan'), device=device, dtype=torch.float32)
    for b, (tokens, values) in enumerate(zip(token_rows, value_rows)):
        ids[b, :len(tokens)] = torch.tensor(tokens, device=device)
        continuous[b, :len(values)] = torch.tensor(values, device=device)
    return {'target_ids': ids, 'continuous_targets': continuous, 'rows': metadata,
            'schema': 'radar_vla_physical_supervision_v3' if config.road_format == 'segments' else 'radar_vla_physical_supervision_v2'}


def create_targets(batch: dict, tokenizer: PhysicalTokenizer, config: PlannerConfig) -> Tensor:
    """Compatibility interface; SFT should retain ``create_supervision`` metadata."""
    return create_supervision(batch, tokenizer, config)['target_ids']
class RiskConditionedPlanner(nn.Module):
    def load_state_dict(self,state_dict,strict=True,assign=False):
        incoming_action=state_dict.get('_action_representation_contract')
        expected_action=getattr(self,'_action_representation_contract',None)
        if ((incoming_action is None)!=(expected_action is None)
                or incoming_action is not None and not torch.equal(incoming_action.cpu(),expected_action.cpu())):
            raise ValueError('Planner checkpoint action representation incompatible')
        incoming=state_dict.get('_risk_horizon_contract');expected=getattr(self,'_risk_horizon_contract',None)
        if (incoming is None)!=(expected is None) or incoming is not None and not torch.equal(incoming.cpu(),expected.cpu()):
            raise ValueError('Planner checkpoint Risk horizons incompatible')
        return super().load_state_dict(state_dict,strict=strict,assign=assign)

    def __init__(self, config: PlannerConfig):
        super().__init__()
        self.config = config
        if config.risk_horizon_schema == '0p5_1_2_v2':
            self.register_buffer('_risk_horizon_contract',torch.tensor([.5,1.,2.,2.],dtype=torch.float64))
        self.tokenizer = PhysicalTokenizer(config.bins, config.coordinate_limit_m, config.velocity_limit_mps)
        self.radar_projector = nn.Linear(config.radar_dim, config.hidden_dim)
        self.risk_projector = nn.Sequential(nn.Linear(5, config.hidden_dim), nn.GELU(), nn.Linear(config.hidden_dim, config.hidden_dim))
        self.embedding = nn.Embedding(self.tokenizer.vocab_size, config.hidden_dim, padding_idx=self.tokenizer.pad_id)
        self.position_embedding = nn.Embedding(config.max_length, config.hidden_dim)
        layer = nn.TransformerEncoderLayer(config.hidden_dim, config.num_heads, config.hidden_dim * 4,
                                           dropout=0., activation='gelu', batch_first=True, norm_first=True)
        self.decoder = nn.TransformerEncoder(layer, config.num_layers, norm=nn.LayerNorm(config.hidden_dim), enable_nested_tensor=False)
        self.lm_head = nn.Linear(config.hidden_dim, self.tokenizer.vocab_size)
        self._initialize_action_head(config.hidden_dim)

    def _initialize_action_head(self, hidden_dim):
        if self.config.action_representation != 'parallel_acceleration_v1':
            return
        from .action_head import ParallelAccelerationHead
        cfg = self.config
        self.register_buffer('_action_representation_contract', torch.tensor(
            [1., 6., .5, 1., cfg.action_max_acceleration_mps2, cfg.action_max_jerk_mps3], dtype=torch.float64))
        self.action_query = nn.Parameter(torch.randn(hidden_dim) * .02)
        self.action_ego_projector = nn.Sequential(nn.Linear(6, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.action_head = ParallelAccelerationHead(hidden_dim, cfg)

    def _action_query_state(self, prefix, padding, ego_token):
        batch = len(prefix)
        hidden = torch.cat((prefix, ego_token[:, None].to(prefix.dtype),
                            self.action_query[None, None].expand(batch, 1, -1).to(prefix.dtype)), 1)
        length = hidden.shape[1]
        if length > self.config.max_length:
            raise ValueError('action query exceeds configured context length')
        padding = torch.cat((padding, padding.new_zeros(batch, 2)), 1)
        positions = ((~padding).long().cumsum(-1)-1).clamp_min(0)
        hidden = hidden + self.position_embedding(positions)
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=hidden.device), diagonal=1)
        return self.decoder(hidden, mask=causal, src_key_padding_mask=padding)[:, -1]

    def _parallel_forward(self, prefix, padding, observed_ego, future_times_s):
        from .action_head import observed_state, validate_future_times
        times = validate_future_times(future_times_s, len(prefix), prefix.device)
        state = observed_state(observed_ego, len(prefix), prefix.device, self.config.action_max_acceleration_mps2)
        velocity, acceleration, yaw, observed, _ = state
        features = torch.cat((velocity/self.config.velocity_limit_mps,
                              acceleration/self.config.action_max_acceleration_mps2,
                              yaw[:, None], observed[:, None].float()), -1)
        ego_token = self.action_ego_projector(features.to(self.action_ego_projector[0].weight.dtype))
        hidden = self._action_query_state(prefix, padding, ego_token)
        return self.action_head(hidden, state, times)

    @torch.no_grad()
    def plan_action(self, radar_tokens, risk, instructions, observed_ego, future_times_s):
        if self.config.action_representation != 'parallel_acceleration_v1':
            raise ValueError('plan_action requires parallel_acceleration_v1')
        from .action_head import action_plans
        output = self(radar_tokens, risk, instructions, observed_ego=observed_ego, future_times_s=future_times_s)
        return action_plans(output, self.config)

    def condition_risk(self, risk: Tensor) -> Tensor:
        """Return the actual physical Risk condition before unit scaling."""
        if self.config.risk_projection == 'none' or not self.config.use_risk_token:
            return risk
        if self.config.risk_projection != 'isotonic':
            raise ValueError('risk_projection must be none or isotonic')
        from .risk_state import project_cumulative_risk
        return project_cumulative_risk(risk)

    def encode_prefix(self, radar_tokens: Tensor, risk: Tensor, instructions: list[str]):
        """Return continuous prefix and padding mask for future LLM integration."""
        if radar_tokens.ndim != 3 or radar_tokens.shape[-1] != self.config.radar_dim:
            raise ValueError('radar_tokens must have shape [batch, tokens, radar_dim]')
        batch_size = radar_tokens.shape[0]
        if risk.shape != (batch_size, 5) or len(instructions) != batch_size:
            raise ValueError('risk [batch,5] and instructions must match the radar batch')
        if not bool(torch.isfinite(radar_tokens).all()) or not bool(torch.isfinite(risk).all()):
            raise ValueError('radar and risk conditions must be finite')
        risk = self.condition_risk(risk)
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

    def forward(self, radar_tokens: Tensor, risk: Tensor, instructions: list[str], target_ids: Tensor | None = None,
                *, observed_ego=None, future_times_s=None):
        if self.config.action_representation == 'parallel_acceleration_v1':
            prefix, mask = self.encode_prefix(radar_tokens, risk, instructions)
            output = self._parallel_forward(prefix, mask, observed_ego, future_times_s)
            if target_ids is not None:
                if target_ids.ndim != 2 or len(target_ids) != len(radar_tokens) or not target_ids.shape[1]:
                    raise ValueError('description teacher must have shape [B,L]')
                inputs = torch.full_like(target_ids, self.tokenizer.bos_id)
                inputs[:, 1:] = target_ids[:, :-1]
                output['description_logits'] = self._decode_inputs(prefix, mask, inputs)
            return output
        if target_ids is None:
            raise ValueError('token planner requires target_ids')
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
        road_count = 0
        if mode == 'long' and cfg.road_format == 'segments':
            if len(result) == 1:
                return [tok.token('<ROAD>')]
            if result[1] != tok.token('<ROAD>'):
                raise ValueError('invalid road grammar prefix')
            cursor = 2
            segment_length = 2 + 2 * cfg.road_controls
            while True:
                if len(result) == cursor:
                    options = [tok.token('<END_ROADS>')] if road_count else []
                    if road_count < cfg.max_road_segments:
                        options.append(tok.token('<ROAD_SEGMENT>'))
                    return options
                if result[cursor] == tok.token('<END_ROADS>'):
                    break
                if result[cursor] != tok.token('<ROAD_SEGMENT>') or road_count >= cfg.max_road_segments:
                    raise ValueError('invalid road segment sequence')
                offset = len(result) - cursor
                if offset < segment_length:
                    if offset == segment_length - 1:
                        return [i for i in tok.physical_ids('position') if tok.decode_scalar(i) > 0]
                    return tok.physical_ids('position')
                cursor += segment_length
                road_count += 1
        empty_layout, _, _ = _layout(mode, cfg, 0, road_count)
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
            layout, _, _ = _layout(mode, cfg, count, road_count)
            field = layout[len(result)]
        if field in ('position', 'velocity'):
            return tok.physical_ids(field)
        if field == 'width':
            return [i for i in tok.physical_ids('position') if tok.decode_scalar(i) > 0]
        return [tok.token(field)]

    def _generation_decoder(self, prefix, prefix_mask):
        """Create an observation-local next-token decoder for the shared loop."""
        return lambda input_ids: self._decode_inputs(prefix, prefix_mask, input_ids)

    @torch.no_grad()
    def _generate_ids(self, radar_tokens: Tensor, risk: Tensor, instructions: list[str],
                      max_new_tokens: int = 256, *, token_selector=None,
                      forced_ids=None) -> list[list[int]]:
        """Shared deployment, sampling and fixed-action latency replay engine.

        Replay performs the same forward, grammar and argmax work as deployment,
        then follows the supplied action. Probability diagnostics are provided
        only by the optional offline sampling callback, never by replay.
        """
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError('max_new_tokens must be a positive integer')
        if forced_ids is not None and (len(forced_ids) != radar_tokens.shape[0] or token_selector is not None):
            raise ValueError('Replay needs one action per observation and no sampling callback')
        prefix, mask = self.encode_prefix(radar_tokens, risk, instructions)
        rows = []
        for b in range(radar_tokens.shape[0]):
            result = []
            decode = self._generation_decoder(prefix[b:b + 1], mask[b:b + 1])
            budget = min(max_new_tokens, self.config.max_length - prefix.shape[1])
            replay = None if forced_ids is None else forced_ids[b]
            if replay is not None:
                if not replay or len(replay) > budget:
                    raise ValueError('Replay action is empty or exceeds the generation context')
                budget = len(replay)
            for step in range(budget):
                inputs = torch.tensor([[self.tokenizer.bos_id] + result], device=radar_tokens.device)
                logits = decode(inputs)[0, -1]
                allowed = self._allowed_next(result)
                selected = (allowed[int(logits[allowed].argmax())] if token_selector is None
                            else token_selector(logits, allowed))
                if replay is not None:
                    selected = int(replay[step])
                if selected not in allowed:
                    raise ValueError('Action violates the policy grammar')
                result.append(selected)
                if selected == self.tokenizer.eos_id:
                    if replay is not None and step != len(replay) - 1:
                        raise ValueError('Tokens after EOS violate the grammar')
                    break
            rows.append(result)
        return rows

    @torch.no_grad()
    def generate(self, radar_tokens: Tensor, risk: Tensor, instructions: list[str],
                 max_new_tokens: int = 256) -> list[list[int]]:
        """One-model greedy generation with learned mode and agent cardinality.

        Grammar limits format, never physical states or safety. This is the same
        autoregressive engine used for offline policy sampling and timed replay.
        """
        return self._generate_ids(radar_tokens, risk, instructions, max_new_tokens)

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
        road_count = ids.count(tok.token('<ROAD_SEGMENT>')) if cfg.road_format == 'segments' else None
        if mode == 'long' and cfg.road_format == 'segments' and road_count == 0:
            failure['errors'] = ['LONG output requires at least one road segment']
            return failure
        try:
            layout, _, controls_count = _layout(mode, cfg, count, road_count)
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
            if cfg.road_format == 'segments':
                segments = []
                for ordinal, index in enumerate(i for i, field in enumerate(layout) if field == '<ROAD_SEGMENT>'):
                    controls = points_at(index + 1, cfg.road_controls)
                    width = values[index + 1 + 2 * cfg.road_controls]
                    complete_points = all(point is not None for point in controls)
                    centerline = (decode_spline(
                        torch.tensor(controls, dtype=torch.float64),
                        torch.linspace(0, 1, cfg.road_controls, dtype=torch.float64),
                        torch.linspace(0, 1, 16, dtype=torch.float64)).tolist()
                        if complete_points else None)
                    segments.append({'ordinal': ordinal, 'centerline_controls': controls,
                                     'centerline_points': centerline,
                                     'width_m': width, 'parameter': 'normalized_arc_length',
                                     'complete': complete_points and width is not None})
                road = {'segments': segments, 'complete': bool(segments) and all(s['complete'] for s in segments)}
            else:
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
    components = {name: [] for name in ('ego_trajectory', 'ego_initial_velocity',
                                      'agent_trajectory', 'road', 'doppler', 'kinematic', 'smooth')}
    ego_velocity, ego_velocity_known = None, None
    if config.ego_initial_velocity_weight > 0 and 'ego_velocity' in batch:
        ego_velocity = batch['ego_velocity'].float()
        if ego_velocity.shape != (logits.shape[0], 2):
            raise ValueError('ego_velocity must have shape [batch,2]')
        ego_velocity_known = torch.isfinite(ego_velocity)
        if 'ego_velocity_mask' in batch:
            known = batch['ego_velocity_mask'].bool()
            if known.shape == (logits.shape[0],):
                known = known[:, None].expand_as(ego_velocity)
            if known.shape != ego_velocity.shape:
                raise ValueError('ego_velocity_mask must have shape [batch] or [batch,2]')
            ego_velocity_known = ego_velocity_known & known
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
            for road in row['road'].get('segments', [row['road']]):
                if road.get('geometry_usable',road['fitted']):
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
                state_known=batch.get('agent_state_mask')
                motion_known=state_known is None or bool(state_known[b,a,[0,1,3,4]].all())
                if bool(item['mask'].any()) and motion_known:components['kinematic'].append((F.smooth_l1_loss(controls[0], position)
                                                 + F.smooth_l1_loss(initial_velocity, xy_velocity)) / 2)
        ego = row['ego']
        if ego is not None:
            controls, _, loss = trajectory(b, ego)
            if bool(ego['mask'].any()):
                components['ego_trajectory'].append(loss)
            if (ego_velocity is not None and ego['fitted']
                    and bool(ego_velocity_known[b].any())):
                # Anchor the planned initial motion to measured ego odometry.
                # Unknown axes and unavailable spline teachers stay masked;
                # legacy scalar speed is not an invented signed velocity.
                initial_basis = spline_basis(ego['control_times'], ego['control_times'][:1],
                                             derivative=1).to(controls)
                initial_velocity = (initial_basis @ controls)[0]
                known = ego_velocity_known[b]
                components['ego_initial_velocity'].append(F.smooth_l1_loss(
                    initial_velocity[known], ego_velocity[b, known]))
            # Time derivatives are analytic and measured in m/s² and m/s³.
            # Evaluate at interval midpoints to avoid ambiguous knot jerk.
            knots = ego['control_times']
            queries = (knots[1:] + knots[:-1]) / 2
            acceleration = spline_basis(knots, queries, derivative=2).to(controls) @ controls
            jerk = spline_basis(knots, queries, derivative=3).to(controls) @ controls
            durations = knots.diff().to(controls)
            regularizer_valid=torch.ones_like(queries,dtype=torch.bool)
            if row.get('taskwise'):
                observed=torch.cat((torch.ones(1,device=logits.device,dtype=torch.bool),ego['mask'].bool()))
                observed_times=torch.cat((ego['query_times'].new_zeros(1),ego['query_times']))
                pairs=observed[:-1]&observed[1:]
                regularizer_valid=((queries[:,None]>=observed_times[:-1][None,:]-1e-6)&
                    (queries[:,None]<=observed_times[1:][None,:]+1e-6)&pairs[None,:]).any(-1)
            if bool(ego['mask'].any()) and bool(regularizer_valid.any()):
                smooth = ((acceleration.square().sum(-1) + config.jerk_weight * jerk.square().sum(-1))[regularizer_valid]
                          * durations[regularizer_valid]).sum() / durations[regularizer_valid].sum()
                components['smooth'].append(smooth)
    losses = {name: torch.stack(values).mean() if values else zero for name, values in components.items()}
    losses['total'] = (config.ego_trajectory_weight * losses['ego_trajectory']
                       + config.ego_initial_velocity_weight * losses['ego_initial_velocity']
                       + config.agent_trajectory_weight * losses['agent_trajectory']
                       + config.road_loss_weight * losses['road']
                       + config.doppler_loss_weight * losses['doppler']
                       + config.kinematic_loss_weight * losses['kinematic']
                       + config.smooth_loss_weight * losses['smooth'])
    return losses
