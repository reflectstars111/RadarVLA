"""Offline, randomly initialized causal RadarVLA planner prototype.

One decoder consumes continuous radar/risk tokens plus UTF-8 instructions. SHORT
versus LONG is an autoregressively generated token; no risk threshold routes
inference between models. This is not a pretrained LLM or a driving controller.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

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
    short_horizon_steps: int = 2
    max_instruction_bytes: int = 96

    def __post_init__(self):
        if self.hidden_dim <= 0 or self.num_heads <= 0 or self.hidden_dim % self.num_heads:
            raise ValueError('hidden_dim must be positive and divisible by num_heads')
        if min(self.radar_dim, self.num_layers, self.max_length, self.max_agents,
               self.horizon_steps, self.short_horizon_steps, self.max_instruction_bytes) <= 0:
            raise ValueError('model, sequence, agent, and horizon dimensions must be positive')


def _layout(mode: str, config: PlannerConfig):
    """A bounded physical grammar. None denotes a structural tag elsewhere."""
    agents = min(2, config.max_agents) if mode == 'short' else config.max_agents
    steps = min(config.short_horizon_steps, config.horizon_steps) if mode == 'short' else config.horizon_steps
    layout = ['<SHORT>' if mode == 'short' else '<LONG>']
    layout += ['<CRITICAL_DYNAMICS>'] if mode == 'short' else ['<ROAD_UNKNOWN>', '<AGENTS>']
    layout += ['position', 'position', 'velocity', 'velocity', 'velocity'] * agents
    if mode == 'long':
        layout += ['<AGENT_FUTURE>'] + ['position'] * (agents * steps * 2)
    layout += ['<IMMEDIATE_ACTION>' if mode == 'short' else '<EGO_FUTURE>']
    layout += ['position'] * (steps * 2)
    layout += ['<EOS>']
    return layout, agents, steps


def create_targets(batch: dict, tokenizer: PhysicalTokenizer, config: PlannerConfig) -> Tensor:
    """Create teacher sequences without inventing missing road or trajectory GT.

    Bootstrap format policy (not human reasoning): a valid collision label at
    1s or 2s selects SHORT; LONG requires both horizons to be valid and negative.
    Ambiguous/missing mode labels yield an all-PAD row, excluding the sample
    from SFT rather than treating unknown risk as low risk. Grounding losses
    can still use its valid physical labels. Only teachers use this policy.
    At inference the shared causal model generates its own mode token. Agent
    order is supplied by the data adapter and must be stable across a sequence.
    """
    state = batch['agent_state']
    device = state.device
    batch_size = state.shape[0]
    rows = []

    def physical(value, valid, kind):
        if not bool(valid) or not bool(torch.isfinite(value)):
            return tokenizer.unknown_id
        return tokenizer.encode_scalar(float(value.detach()), kind)

    for b in range(batch_size):
        risk = batch.get('risk_target')
        risk_mask = batch.get('risk_mask')
        risk_available = risk is not None and risk_mask is not None
        known = (risk_mask[b, :2].bool() & torch.isfinite(risk[b, :2])) if risk_available else None
        short = risk_available and bool(((risk[b, :2] >= .5) & known).any())
        long = risk_available and bool((known & (risk[b, :2] < .5)).all())
        if not short and not long:
            rows.append([tokenizer.pad_id] * len(_layout('long', config)[0]))
            continue
        mode = 'short' if short else 'long'
        _, slots, steps = _layout(mode, config)
        ids = [tokenizer.token('<SHORT>' if short else '<LONG>')]
        ids += [tokenizer.token('<CRITICAL_DYNAMICS>')] if short else [tokenizer.token('<ROAD_UNKNOWN>'), tokenizer.token('<AGENTS>')]
        for a in range(slots):
            for k, kind in enumerate(['position', 'position', 'velocity', 'velocity', 'velocity']):
                if a >= state.shape[1]:
                    ids.append(tokenizer.unknown_id)
                    continue
                valid = batch['agent_mask'][b, a]
                # A field mask, when available, prevents unobserved velocity
                # components from becoming fabricated zero-valued teachers.
                if 'agent_state_mask' in batch:
                    valid = valid & batch['agent_state_mask'][b, a, k]
                ids.append(physical(state[b, a, k], valid, kind))
        if not short:
            ids.append(tokenizer.token('<AGENT_FUTURE>'))
            future = batch.get('agent_future')
            future_mask = batch.get('agent_future_mask')
            for a in range(slots):
                for t in range(steps):
                    for k in range(2):
                        if future is None or future_mask is None or a >= future.shape[1] or t >= future.shape[2]:
                            ids.append(tokenizer.unknown_id)
                        else:
                            ids.append(physical(future[b, a, t, k], future_mask[b, a, t], 'position'))
        ids.append(tokenizer.token('<IMMEDIATE_ACTION>' if short else '<EGO_FUTURE>'))
        future = batch.get('ego_future')
        mask = batch.get('ego_future_mask')
        for t in range(steps):
            for k in range(2):
                if future is None or mask is None or t >= future.shape[1]:
                    ids.append(tokenizer.unknown_id)
                else:
                    ids.append(physical(future[b, t, k], mask[b, t], 'position'))
        ids.append(tokenizer.eos_id)
        rows.append(ids)
    length = max(map(len, rows))
    result = torch.full((batch_size, length), tokenizer.pad_id, device=device, dtype=torch.long)
    for i, ids in enumerate(rows):
        result[i, :len(ids)] = torch.tensor(ids, device=device)
    return result


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
        risk_token = self.risk_projector(risk / scale).unsqueeze(1)
        prefix = torch.cat([self.radar_projector(radar_tokens), risk_token, self.embedding(ids)], dim=1)
        mask = torch.cat([torch.zeros(batch_size, radar_tokens.shape[1] + 1, dtype=torch.bool, device=ids.device), ids == self.tokenizer.pad_id], dim=1)
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

    @torch.no_grad()
    def generate(self, radar_tokens: Tensor, risk: Tensor, instructions: list[str], max_new_tokens: int = 256) -> list[list[int]]:
        """Greedy grammar-constrained generation; format validity is not safety.

        Every physical value and SHORT/LONG decision is predicted by this same
        model. Only fixed tags and token types are grammar constrained. A token
        budget can truncate output; decode reports that explicitly.
        """
        if max_new_tokens <= 0:
            raise ValueError('max_new_tokens must be positive')
        prefix, mask = self.encode_prefix(radar_tokens, risk, instructions)
        rows = []
        for b in range(radar_tokens.shape[0]):
            result = []
            layout = None
            for step in range(min(max_new_tokens, self.config.max_length - prefix.shape[1])):
                inputs = torch.tensor([[self.tokenizer.bos_id] + result], device=radar_tokens.device)
                logits = self._decode_inputs(prefix[b:b + 1], mask[b:b + 1], inputs)[0, -1]
                if not result:
                    allowed = [self.tokenizer.token('<SHORT>'), self.tokenizer.token('<LONG>')]
                else:
                    field = layout[step]
                    allowed = self.tokenizer.physical_ids(field) + [self.tokenizer.unknown_id] if field in ('position', 'velocity') else [self.tokenizer.token(field)]
                selected = allowed[int(logits[allowed].argmax())]
                result.append(selected)
                if len(result) == 1:
                    layout, _, _ = _layout('short' if selected == self.tokenizer.token('<SHORT>') else 'long', self.config)
                if selected == self.tokenizer.eos_id:
                    break
            rows.append(result)
        return rows

    def decode(self, ids: list[int], future_times_s=None) -> dict:
        """Parse generated physical controls, retaining missing values as None."""
        tok = self.tokenizer
        ids = [int(i) for i in ids]
        while ids and ids[-1] == tok.pad_id:
            ids.pop()
        failure = {'valid': False, 'mode': None, 'ego_trajectory': [], 'errors': []}
        if not ids or ids[0] not in (tok.token('<SHORT>'), tok.token('<LONG>')):
            failure['errors'] = ['missing SHORT/LONG mode token']
            return failure
        mode = 'short' if ids[0] == tok.token('<SHORT>') else 'long'
        layout, agents, steps = _layout(mode, self.config)
        failure['mode'] = mode
        if len(ids) != len(layout):
            failure['errors'] = [f'incomplete or trailing sequence: expected {len(layout)} tokens, got {len(ids)}']
            return failure
        values = []
        for token_id, field in zip(ids, layout):
            if field in ('position', 'velocity'):
                if token_id == tok.unknown_id:
                    values.append(None)
                else:
                    try:
                        values.append(tok.decode_scalar(token_id, field))
                    except ValueError as error:
                        failure['errors'] = [str(error)]
                        return failure
            elif token_id != tok.token(field):
                failure['errors'] = [f'expected {field}']
                return failure
        states = []
        cursor = 0
        for _ in range(agents):
            row = values[cursor:cursor + 5]
            cursor += 5
            states.append(None if all(v is None for v in row) else dict(zip(('x', 'y', 'vr', 'vx', 'vy'), row)))

        def trajectory():
            nonlocal cursor
            result = []
            for _ in range(steps):
                xy = values[cursor:cursor + 2]
                cursor += 2
                result.append(None if any(v is None for v in xy) else xy)
            return result

        futures = [trajectory() for _ in range(agents)] if mode == 'long' else []
        ego = trajectory()
        times = None
        if future_times_s is not None:
            times = [float(t) for t in future_times_s[:steps]]
            if len(times) != steps or any(t <= 0 or not math.isfinite(t) for t in times) or any(b <= a for a, b in zip(times, times[1:])):
                raise ValueError('future_times_s must provide positive increasing times for all controls')
        return {'valid': True, 'mode': mode, 'road': None, 'road_status': 'unavailable_in_v1_targets',
                'agent_states': states, 'agent_trajectories': futures, 'ego_trajectory': ego,
                'ego_times_s': times,
                'ego_complete': all(point is not None for point in ego), 'token_count': len(ids),
                'errors': [], 'prototype_unvalidated': True}
